# 恢复结果何时就绪：入口检查过早，主要窗口在 execute_model 调用内

日期：2026-10-12。承接 [四组 ready-first 结果](restore_ready_first_results_20261012.md)。

## 本轮回答的问题

前一轮两组 on 都是零提前消费，不能仅凭 ready→engine 的历史均值认定入口存在可利用机会。本轮只观察原路径，关联同一个 Future 在入口、schedule、execute_model、sample_tokens、结果等待各边界的状态，区分旧 Future 与本轮新建的 Future。

两组各完成 648 请求，共 1296 请求、45 次 RESTORE（42 个带恢复通知的输出批次）、48 次 RECOMPUTE。请求集合、client assignment、固定 R/C 动作及完整恢复时间线审计通过。

## 主要发现

以下按恢复请求事件计数；一个输出批次可能携带多个恢复请求，故不是 45 个独立批次。

| 就绪位置 / 首次观察条件 | Seed 17 | Seed 29 | 合计 | 占 45 次恢复比例 |
| --- | ---: | ---: | ---: | ---: |
| 消费所在循环入口已 ready | 1 | 0 | 1 | 2.2% |
| before_execute 未 ready，after_execute 已 ready | 12 | 14 | 26 | 57.8% |
| 执行后仍未 ready，sample 路径后观察为 ready | 0 | 1 | 1 | 2.2% |
| before_result 未 ready，原结果等待返回后 ready | 7 | 6 | 13 | 28.9% |
| Future 本轮才创建，入队时已 ready，并在本轮消费 | 0 | 4 | 4 | 8.9% |

循环入口的条件分布：40 次对应 Future 未 ready，4 次 Future 尚未创建，只有 1 次已 ready。observe gate 的 eligible 为 1 / 0、consumed 均为 0，与本轮现象相符。

**入口条件稀少得到了新的时序证据；不能将前一轮的零触发直接解释为 offload 未工作或开关没有启用。** 此次是新的运行，不是对前一轮原始 47 次恢复补采时间戳，不能声称逐事件解释了那 47 次。

### 主要等待位于 execute_model 返回之前

对上述 26 次跨越 execute_model 调用才观察到 ready 的请求：

| 观测时间段 | 平均 ms |
| --- | ---: |
| before_execute → after_execute | 5.764 |
| executor 输出打点 → after_execute | 5.113 |
| after_execute → engine 消费打点 | 0.414 |
| executor 输出打点 → engine 消费打点 | 5.527 |

这组事件的后两段相加为 5.527 ms。边界时间戳包含少量观测开销。

代表事件 `chatcmpl-WBAALDw-turn9-bf9172f5`：before_execute 仍 pending，after_execute 才观察到 done；execute 边界间隔 14.912 ms，输出打点到 after_execute 14.813 ms，after_execute 到消费 0.463 ms。

对 13 次在结果等待期间才完成的请求，输出打点到消费平均仅 0.032 ms；4 次本轮新建并消费的请求为 0.038 ms。它们不是当前入口优先消费机制的主要优化对象。

**把检查简单挪到 execute_model 返回之后，并不能回收已经发生在调用内部的那段时间。** 返回之后原路径已经较快地消费结果，仍须测量任何调整的收益与成本。

## 测量边界

- done() 在阶段边界采样，没有注册完成回调，没有额外 Future.result() 或 CUDA 同步。对 False→True 的变化只给出区间，不知道精确完成时刻。
- executor_output_ready_s 在 get_output 后、Future 完成之前打点。5.113 ms 不能全部解释为“Future 已完成但 engine 故意没处理”，其中可能包含异步输出线程返回和 Future 交付的延迟。
- UniProcExecutor 的 non_block 路径先同步调用 run_method 执行 worker 的主机侧代码，再处理异步输出。这是调用期间不能回到 engine 检查点的代码依据；本轮尚未细分调用内部的输入准备、CUDA 提交、同步等待和线程调度。
- 两组 trace-on mean TTFT 为 42.560 / 38.973 ms，P95 为 76.353 / 70.306 ms。这些仅作完整记录，本轮没有性能对照，不与上轮差值宣称优化或观测开销。
- 没有证明发现的时间段可无代价消除，也没有证明可超过 no-offload。

## 实现

- [restore_ready.py](../vllm/v1/kv_offload/restore_ready.py)：新增 RestoreReadinessTrace，活动 Future 最多 64 个、每个最多 128 条边界记录，最多保留 4096 个恢复输出批次。普通输出消费后立即释放记录；上溢和关联错误显式计数。
- [core.py](../vllm/v1/engine/core.py)：在原调用路径前后观察，不重排调度，不提前消费；统计通过已有接口导出。
- 环境开关 `VLLM_EXPERIMENTAL_RESTORE_READINESS_TRACE=1` 要求 ready-first 为 observe，沿用单卡 UniProc / OffloadingConnector 等限制；默认关闭。
- [analyze_restore_readiness.py](../scripts/analyze_restore_readiness.py)：核对 Future 状态单调性、请求集合、输出打点顺序、丢失和上溢，输出逐事件可见区间。
- [run_restore_ready_experiment.py](../scripts/run_restore_ready_experiment.py)：增加 readiness 阶段，固定 seed 17 / 29，各一次 observe。
- [summarize_restore_readiness.py](../scripts/summarize_restore_readiness.py)：汇总事件位置和主机时间段，并复核冻结源文件与输入。

## 验证与运行条件

Qwen3-0.6B，GPU0，GPU 1000 blocks，CPU cache 2 GiB，4 clients，all-block，guard 关闭，APC 和 chunked prefill 开启，runtime seed 0。模型路径与上一轮已迁移路径相同，并在运行前后核对文件哈希。

39 项测试通过：包括旧 Future 在 schedule、execute、result wait 期间完成，新 Future 本轮创建并消费，普通输出记录清理，容量上限，以及分析器的缺失 / 重复通知和时间顺序拒绝路径。既有 ready-first 和恢复反馈测试一并通过；Ruff check / format 与 git diff --check 通过。

运行结束复核 33 个冻结源文件和输入哈希一致。60 次 GPU0 采样未发现外来进程；2 秒采样间隔不能排除瞬时干扰。GPU1 同时有其他用户负载，因此没有整机独占，CPU / 内存等共享干扰仍可能存在。GPU0 和端口 8238 已释放。

## 下一步决策

不扩大 ready-first 性能 sweep，默认保持 off。若继续这一方向，应定位 **execute_model 内部为何占用主机线程，以及输出打点到 Future 可消费之间是否有延迟**，再选择局部优化。

优先考虑按已有主机路径分段，区分输入准备、上一批同步等待、模型提交和输出封装；同步点与线程交付只能靠观测定位，不能把整个区间当成 DMA 或 GIL 开销。若大部分是必要工作且不能重叠，应降低此方向优先级，避免为消除局部等待牺牲批处理效率。

## 结果与复现

- [运行 summary](../results/restore_readiness_gpu0_20261012/summary.json)
- [运行 manifest](../results/restore_readiness_gpu0_20261012/manifest.json)
- [汇总分析](../results/restore_readiness_analysis_20261012/summary.json)
- [45 条事件 CSV](../results/restore_readiness_analysis_20261012/events.csv)
- 每个 arm 的 restore_readiness.json 与 value_policy_stats.json 保留逐 Future 原始边界和关联结果。

```bash
.venv/bin/python scripts/run_restore_ready_experiment.py \
  --source results/promotion_restore_joint_gpu0_20261010 \
  --output results/restore_readiness_gpu0_new_run \
  --phase readiness --gpu 0 --port 8238 \
  --model /hole0/model_info/model_zoo/downloads/Qwen3-0.6B \
  --allow-model-relocation

.venv/bin/python scripts/summarize_restore_readiness.py \
  --source results/restore_readiness_gpu0_new_run \
  --output results/restore_readiness_analysis_new_run
```
