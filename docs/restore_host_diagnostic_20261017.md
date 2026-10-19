# 先拆 5.11 ms，再拆 7.89 ms：主机执行路径诊断

日期：2026-10-17。状态：**观测代码与 CPU 测试完成，GPU 复测尚未运行**。

本轮承接 [恢复就绪位置实验](restore_readiness_diagnostic_20261012.md)，
不改 admission、promotion 或调度顺序。目标是定位可优化环节，而非预先认定
某段等待可以全部消除。最终验收仍是 TTFT / P95 改善，并与 no-offload 比较。

## 1. 两个数字来自不同位置，不能直接相加

| 历史观测 | 样本范围 | 含义与限制 |
| --- | --- | --- |
| 5.113 ms | 10 月 12 日的 26 次恢复请求事件 | 旧输出打点到下一次 execute_model 返回边界；这些 Future 在 before_execute 未完成、after_execute 已完成。不是精确的 Future 完成后等待时间，也不是所有恢复的平均值。 |
| 7.889 ms | 10 月 10 日的 44 次恢复事件 | 传输提交返回到 worker 观察完成；可能混合设备排队、搬运、轮询间隔。不能减去 DMA 均值后就称为轮询开销。 |

第二个数字详见 [联合路径诊断](promotion_restore_joint_diagnostic_20261010.md)。
两轮运行、样本选择不同；本轮只先分析第一段。

## 2. 本轮新增观测

`VLLM_EXPERIMENTAL_RESTORE_HOST_TRACE=1` 默认关闭，且要求 readiness trace
和 ready-first observe。沿用单卡 UniProc、OffloadingConnector 等已有约束。

在 EngineCore 的原 execute_model 调用周围建立线程局部记录，内部打点如下：

| 阶段 | 对应问题 |
| --- | --- |
| executor / worker / runner setup | 是否耗在调用分发和执行准备？ |
| input_prep_event_sync | 是否等待上一批操作释放复用的 CPU 输入缓冲区？这是已有同步点，不是新增同步。 |
| update_states / prepare_inputs | 持久 batch 更新和输入准备占用多少主机时间？ |
| slot_mappings / attention_metadata / preprocess | block 映射、注意力元数据、模型输入预处理是否较重？ |
| kv_start_load / model_forward | KV 启动与模型调用提交各占多少时间？主机模型调用时间不等于 GPU kernel 时间。 |
| kv_wait_for_save / kv_get_finished / kv_stats_and_metadata | 保存等待、完成检查、统计收集是否拖长调用？ |
| postprocess_logits / runner_finalize / executor_output_dispatch | 后处理和返回封装是否占用关键窗口？ |

每个阶段同时记录 `perf_counter`（主机墙钟）和 `thread_time`（当前线程 CPU 时间）。
每次调用最多 96 个边界，溢出时分析失败，不静默丢弃。
线程局部存储避免异步输出线程混入主执行线程的阶段记录。

没有新增 CUDA event、CUDA synchronize、Future.result 调用或提前消费。
异常退出清理观测作用域；默认关闭时不采集时间戳。

## 3. 如何分析

- 仍关联同一个旧 Future，并仅将 `first_done_stage == after_execute` 作为
  5.11 ms 的对应目标组；单独报告样本数量，不能与全部恢复混算。
- 对消费循环内的 host call 做阶段分解，再与“旧输出打点 → after_execute”窗口取交集。
  所有交集加上作用域外的少量边界时间必须等于窗口总长。
- 同时输出完整阶段墙钟和线程 CPU 时间。CPU 时间是完整阶段数值，
  **不按交集比例分摊**，不能拿它与裁剪后的墙钟直接相减。
- 多个恢复请求可能共享一个通知批次；当前汇总按请求加权，不表示各事件独立。
- 输出打点早于 Future 真正完成，不能将交集当成“已经可消费但未消费”的精确时长。
- 低线程 CPU / 高墙钟只提示等待、线程未被调度等可能性；不能单凭它区分
  CUDA 同步、GIL 竞争或操作系统调度。

若同步点占主导，下一步检验依赖与缓冲区生命周期，不能删除同步来追求数字。
若输入准备或提交占主导，先检查能否减少工作量或分离安全的控制路径。
若没有一个稳定主导阶段，或主要是必要工作，应降低该方向优先级。

## 4. 固定实验入口

新增 `--phase host-execute`：固定 assignment seeds 17 / 29，各一次 observe，
每组 648 请求。沿用模型、请求集合、client assignment、固定 R/C 动作、
GPU 1000 blocks、CPU 2 GiB 等配置，并检查代码、输入及模型文件哈希。
沿用 GPU 空闲预检与运行期间外来进程采样，不抢占其他任务。

```bash
.venv/bin/python scripts/run_restore_ready_experiment.py \
  --source results/promotion_restore_joint_gpu0_20261010 \
  --output results/restore_host_gpu0_20261017 \
  --phase host-execute --gpu 0 --port 8238 \
  --model /hole0/model_info/model_zoo/downloads/Qwen3-0.6B \
  --allow-model-relocation
```

这是待运行命令，不是已有结果。输出目录必须不存在，重复运行使用新目录。
每组生成 `restore_host.json`（逐事件和阶段均值）、`restore_readiness.json`、
原始统计及汇总 `summary.json`。本阶段没有 trace-off 对照，不能用 TTFT
与旧实验之差宣称优化或观测开销。

## 5. 验证状态

本轮 51 项 CPU 测试通过，覆盖：默认关闭、跨线程隔离、异常清理、嵌套拒绝、
记录上限、host 记录与原执行路径关联、窗口分段守恒、缺失记录和非法时钟拒绝；
同时回归既有 ready-first、readiness、restore feedback 与联合事件分析。

```bash
.venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_restore_ready.py \
  tests/benchmarks/test_restore_readiness_analysis.py \
  tests/v1/kv_connector/unit/test_restore_feedback_trace.py \
  tests/benchmarks/test_promotion_restore_joint.py \
  tests/v1/kv_connector/unit/test_restore_host_trace.py \
  tests/benchmarks/test_restore_host_analysis.py -q
```

检查时两张 GPU 均为 100% 利用率，因此没有启动新 GPU 运行。
**尚不知道 5.11 ms 的主导阶段，也没有新的 TTFT 优化结论。**

## 6. 后续 7.89 ms 的边界

待第一阶段实测后，再对传输已有的 `end_event.query()` 记录查询时刻与结果，
区分提交后首次轮询间隔、最近一次未完成到首次完成的区间、FIFO 头阻塞等。
不增加忙轮询，不增加同步。首次查询即完成时只能知道完成在查询之前，
不能精确知道设备何时结束。GPU event 的 DMA 时间与主机查询区间需要分别保留。

该第二阶段当前**仅是方案，尚未新增逐传输查询观测，也未运行实验**。

## 代码入口

- [host trace](../vllm/v1/kv_offload/restore_host_trace.py)
- [主机阶段分析](../scripts/analyze_restore_host.py)
- [固定实验 runner](../scripts/run_restore_ready_experiment.py)
- [GPUModelRunner 打点](../vllm/v1/worker/gpu_model_runner.py)
- [KV connector 边界](../vllm/v1/worker/kv_connector_model_runner_mixin.py)
