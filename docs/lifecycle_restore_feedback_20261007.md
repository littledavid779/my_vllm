# 恢复完成反馈细分：主要不在 Scheduler.update_from_output 内部

## 本轮完成什么

承接 [restore 九段实验](lifecycle_restore_path_results_20261007.md)。该轮发现
worker 观察完成→scheduler 观察完成平均 4.54 ms，但没有将它归因为纯通信开销。

本轮仅补三处默认关闭的时间戳，未修改 admission、promotion、传输顺序或调度顺序。
在 GPU 0 完成 4 runs × 659 requests；149 个实际 restore 请求均采到完整反馈路径。
运行时间约 2026-10-07 13:07–13:11，本机时区。结果：

**worker 到 scheduler 的约 5.09 ms 中，约 3.08 ms 在模型输出尚未准备好阶段，
约 1.94 ms 在输出准备好后等待 engine 消费阶段；scheduler 内部只占约 0.059 ms。**

因此，“把 scheduler.update_from_output 内处理恢复通知的位置提前”不是当前有希望
带来数毫秒收益的主方案。应检查模型输出完成链及 engine 批次消费时机。
这是更具体的定位结果，不是已经实现或验证的性能优化。

## 三个新增时间点

| 字段 | 位置与口径 |
|---|---|
| executor_output_ready_s | UniProcExecutor 中普通结果已返回，或 AsyncModelRunnerOutput.get_output() 返回之后、Future 发布结果之前；不是 DMA 完成时刻，也不是 Future.set_result 的精确时刻 |
| engine_output_consumed_s | EngineCore.step / step_with_batch_queue 已取到最终 model output 后、处理 abort queue 前 |
| scheduler_update_entry_s | Scheduler.update_from_output 入口，在普通请求输出处理之前 |

复用既有 `worker_observed_s` 和 `scheduler_observed_s`，得到四个相邻区间：

`worker observed → output ready → engine consumed → scheduler entry → connector observed`

它们的和必须等于原来 worker→scheduler 区间，不与原区间重复相加。
本轮所有事件实际走 UniProcExecutor / batch_queue。单卡 worker 和 engine 可以处于同一进程，
所以不能将第一段或第二段称为跨进程通信耗时。

输出准备阶段仍可能包含采样、CPU bookkeeping、异步输出线程排队、token D2H 等；
本轮没有把这 3.08 ms 再细分，不能声称它全是采样或全是线程等待。
输出 ready→engine 消费也不等于纯 Future 队列等待，它包含 engine 此时在处理的其他工作。

## 实测分段（事件均值，ms）

| 区间 | 对比对 0，78 次 | 对比对 1，71 次 | 加权合并，149 次 |
|---|---:|---:|---:|
| worker observed→output ready | 3.396 | 2.739 | **3.083** |
| output ready→engine consumed | 1.719 | 2.179 | **1.938** |
| engine consumed→scheduler entry | 0.006 | 0.006 | 0.006 |
| scheduler entry→connector observed | 0.059 | 0.059 | **0.059** |
| worker→scheduler 合计 | 5.180 | 4.984 | **5.086** |
| scheduler observed→first compute | 1.483 | 1.616 | 1.546 |

其他上下文（不是上表附加开销）：

- restore 请求 enqueue→首次计算调度：19.303 / 19.318 ms，合并 19.310 ms。
- host 提交：0.651 / 0.654 ms，合并 0.652 ms。
- submit return→worker observed：9.399 / 9.839 ms，合并 9.609 ms。
- CUDA event elapsed：5.510 / 5.762 ms，合并 5.630 ms，与 host 等待区间重叠。

本轮均值不同于上一轮，不能把 4.54→5.09 ms 或 17.15→19.31 ms 的差值全部归因为
新增观测开销；负载状态、batch 组成和观察开销尚未单独控制。分段占比在两对中的一致性
支持定位，但性能收益仍需 trace-off 对照。

## 总体 TTFT：仍未超过 No-offload

| 对比对 | 策略 / 顺序 | Mean ms | P95 ms | P99 ms |
|---|---|---:|---:|---:|
| 0 | No-offload（先） | 32.741 | 55.173 | 70.811 |
| 0 | Selective（后） | 34.417 | 58.069 | 72.779 |
| 1 | Selective（先） | 37.044 | 61.803 | **131.506** |
| 1 | No-offload（后） | 36.389 | 61.741 | 71.411 |

两对 mean/P95 都较差；第二对 P99 明显退化。保留全部样本，未删尾部点，也未声称
这些尾部点全部由 restore 引起。没有 host 独占预留，不用“系统抖动”直接解释掉负结果。

实际 restore 请求的跨运行同输入差值（Selective − No-offload）为：

| 指标 | 对比对 0 | 对比对 1 |
|---|---:|---:|
| completed prefill tokens | −1134.35 | −1180.61 |
| 等待首次计算调度 ms | +17.871 | +17.716 |
| 首次计算调度→首 token ms | −12.139 | −15.913 |
| scheduler TTFT ms | +5.732 | +1.802 |
| client TTFT ms | +5.274 | +6.564 |

这些是跨运行描述，不是同 GPU/batch 状态的反事实。减少计算量仍然没有稳定转化为 TTFT 优势。

## 下一个最小性能实验

优先候选：**已就绪且含恢复完成通知的队首模型输出，是否应在投递下一批之前先消费。**

依据：当前 EngineCore.step_with_batch_queue 先尝试 schedule/execute 新批次，随后再取
队列尾部的旧结果。观测到“结果已准备好仍等约 1.94 ms”，值得做局部、可关闭的单因素消融，
但这不保证每个结果在新一轮迭代入口已经 ready，必须记录实际可触发的次数。

实施约束：

1. 仅针对单卡 UniProcExecutor 的已就绪队首结果，且包含真实 finished_recving；
   未就绪、普通 decode 输出、其他 executor 默认仍走旧路径。
2. 不越过旧批次、不重排模型结果；保持 abort、失败恢复、抢占、引用计数与 token 占位更新次序。
3. 复用正常 update_from_output，不单独对同一完成通知处理两次。
4. 新开关默认关闭；记录检查次数、命中条件次数、实际提前消费次数，防止“优化没有触发”却比较 TTFT。
5. 开启观测先验证目标区间是否下降，再关闭观测比较 No-offload / 原 Selective / 新 Selective。
   保持完整相同请求集、固定容量、反向顺序和独占目标卡；同时检查吞吐、P99 与 decode ITL。

这项优化最多先针对约 1.94 ms 的可见区间，并不能消除前面约 3.08 ms 的输出准备。
仅按 restore 请求占比估算，其对全局均值的直接影响可能有限，不能提前保证超过 No-offload。
如果没有足够 ready-at-entry 机会，或损害 batch 吞吐，应停止该改法，而非不断调阈值。

更大候选是将恢复完成通知从 token 输出物化链中解耦，但它涉及并发状态和生命周期安全，
不与上述小消融同时实现。本轮未实现这两项性能改动。

## 实现、验证和完整性

- [restore_trace](../vllm/v1/kv_offload/restore_trace.py) 只修改现有 restore timing payload，
  不增加 CUDA 同步或忙轮询；同一 stage 重复写入显式标错，保留原时间戳。
- [UniProcExecutor](../vllm/v1/executor/uniproc_executor.py) 的同步、普通 Future、异步输出
  和列表返回路径都有观测覆盖；关闭开关时保留原 async get_output 提交路径。
- [EngineCore](../vllm/v1/engine/core.py)、[Scheduler](../vllm/v1/core/sched/scheduler.py)
  只加入 stamp，不改变消费顺序。
- [分析器](../scripts/analyze_lifecycle_restore_path.py) 兼容旧九段数据；新 runner 强制
  三个新时间点齐全，且拒绝重复、倒序、跨 host、多 worker、混合版本及错误 executor。
- 共 102 项相关测试通过，其中新增 11 项 executor/engine/stamp 测试和 7 项反馈分析测试。
  初次运行有 5 项因沙箱无法识别 GPU 平台失败，允许访问驱动后对应文件 10 项全部通过。
- Ruff check/format 和 git diff --check 通过；未改依赖或环境，没有提交/覆盖用户其他修改。
- 25 份冻结源码、输入和拟合哈希通过；原始 CSV mean/P50/P95/P99 独立复算一致。
- 2636 条请求路径完整，无抢占/丢失/未完成；149 次恢复与 lifecycle promotion/completion
  request IDs 一一匹配，反馈四段与原 worker→scheduler 区间逐事件相等。
- load blocks 与 H2D bytes 按每块 1,835,008 bytes 独立校验一致。
- GPU 0 共 117 次进程采样无外来 PID；结束时两卡显存 0 MiB、无计算进程，8238 端口关闭。

原始结果：[lifecycle_restore_feedback_gpu0_20261007](../results/lifecycle_restore_feedback_gpu0_20261007)。

- [冻结设计](../results/lifecycle_restore_feedback_gpu0_20261007/manifest.json)
- [总体指标](../results/lifecycle_restore_feedback_gpu0_20261007/evaluation_summary.json)
- [请求分组](../results/lifecycle_restore_feedback_gpu0_20261007/request_analysis.json)
- [第一轮反馈分段](../results/lifecycle_restore_feedback_gpu0_20261007/repeat_0/selective-2g/restore_path_analysis.json)
- [第二轮反馈分段](../results/lifecycle_restore_feedback_gpu0_20261007/repeat_1/selective-2g/restore_path_analysis.json)
