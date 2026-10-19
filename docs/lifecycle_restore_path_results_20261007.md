# Restore 分段实测：完成反馈与放行路径值得优先检查

后续：[完成反馈细分实测](lifecycle_restore_feedback_20261007.md) 已进一步区分输出准备、
engine 消费等待和 scheduler 内部处理；请不要把本页 4.54 ms 直接称为通信或 scheduler 开销。

## 结论

2026-10-07 12:12–12:16 在 GPU 0 完成 4 组冻结配置实验，共 2636 个请求，
其中 155 次实际恢复。承接 [诊断实现](lifecycle_restore_path_diagnostic_20261006.md) 与
[上一轮请求路径结果](lifecycle_request_path_diagnostic_20261006.md)。

本轮没有修改策略或传输/调度行为。新增的分段表明：

- restore 请求 enqueue→首次计算调度平均 17.153 ms；host 提交仅 0.580 ms。
- worker 已发现恢复完成后，到调度器观察到结果仍有 4.540 ms，之后首次计算调度再等 1.528 ms。
- 两者合计 6.068 ms（上述完整等待的约 35.4%），是值得优先定位的完成反馈/放行区间，
  **不是已经证明可消除的 6 ms，更不是纯 IPC 或某个函数的耗时**。
- 提交返回→worker 发现完成平均 8.464 ms，也占较大部分；其中仍混合传输排队、执行与轮询延迟。
- CUDA event elapsed 平均 5.364 ms，与 host 区间重叠，不能再加到 17.153 ms 上，
  也不能简单用 8.464−5.364 就声称剩余部分全是轮询开销。

因此当前不优先投入大规模 CUDA 小拷贝合并改造，也不再仅调 admission 分数；
先检查恢复完成结果如何随 model output 返回、被 engine 消费并允许 request 重新调度。

## 冻结条件

- Qwen3-0.6B，本地模型；原 128 个 ShareGPT 会话、659 请求、4 clients、32 active conversations。
- GPU KV 1000 blocks；Selective CPU 2 GiB；always_restore；原 admission/cost 参数。
- Guard/probe 关闭；full cost/admission trace 关闭；request-path 和 restore-only trace 开启。
- runtime seed 0；顺序为 No-offload→Selective、Selective→No-offload。
- 使用已诊断过的输入，不是新泛化集；只做两对重复，非性能确认实验。
- GPU 0 运行期间无采样到的外来进程；GPU 1 中途一次采样及结束时空闲，但主机并未独占预留。

## 总体 TTFT：一对改善、一对退化

| 对比对 | 策略 / 执行顺序 | Mean ms | P95 ms | P99 ms |
|---|---|---:|---:|---:|
| 0 | No-offload（先） | 32.913 | 55.972 | 70.511 |
| 0 | Selective（后） | 30.678 | 51.274 | 67.211 |
| 1 | Selective（先） | 34.846 | 58.507 | 73.234 |
| 1 | No-offload（后） | 33.812 | 57.601 | 66.830 |

不能因第一对改善或合并均值改善，就宣称已稳定胜过 No-offload。
第二对 mean/P95/P99 均退化；开启观测的结果也不能与历史 trace-off 数值直接混用。

## 实际恢复请求的 9 段等待

以下均为同一 Selective 运行内实际恢复请求的均值，单位 ms。
合并列按 78/77 条事件加权，不是先对两轮取等权平均。

| 相邻阶段 | 对比对 0（78 次） | 对比对 1（77 次） | 合并（155 次） |
|---|---:|---:|---:|
| enqueue→restore decision | 0.501 | 0.573 | 0.537 |
| decision→allocated | 0.234 | 0.251 | 0.243 |
| allocated→metadata built | 0.309 | 0.521 | 0.415 |
| metadata built→worker received | 0.710 | 0.735 | 0.722 |
| worker received→submit start | 0.081 | 0.167 | 0.124 |
| submit start→submit return | 0.584 | 0.576 | 0.580 |
| submit return→worker observed | 7.791 | 9.146 | 8.464 |
| worker observed→scheduler observed | 4.824 | 4.252 | 4.540 |
| scheduler observed→first compute scheduled | 1.402 | 1.656 | 1.528 |
| **以上区间合计** | **16.437** | **17.878** | **17.153** |
| CUDA event elapsed（重叠，不能相加） | 5.531 | 5.194 | 5.364 |

worker→scheduler 的事件级 P95 分别为 14.067 / 12.969 ms。它不仅影响均值，
也可能影响恢复请求的尾部等待；不过各段的 P95 不可相加来重建整体 P95。

H2D 总量分别为 10,331,095,040 / 9,483,321,344 bytes，分别对应 5630 / 5168
load blocks（每块 1,835,008 bytes），校验一致。本轮没有完整 D2H 字节/成本采样。
旧汇总字段中的 `dma_measurement_available=false` 与 null bytes 仍表示旧 full-cost
账本关闭；新恢复观测的数据在各 Selective 的 `restore_path_analysis.json` 中。
不要将 runner 控制台旧占位的 `d2h_bytes=0` 解读为无 D2H 传输。

## 为什么仍需要优化：减少计算不等于 TTFT 一定降低

下面是 Selective 中实际恢复请求，相对另一运行同一输入 No-offload 的差值。
这不是相同 batch/GPU 状态下的逐事件反事实，不能叫严格 restore-vs-recompute 配对实验。

| 指标，Selective − No-offload | 对比对 0 | 对比对 1 | 合并 |
|---|---:|---:|---:|
| completed prefill tokens | −1159.17 | −1047.68 | −1103.78 |
| enqueue→首次计算调度 ms | +14.917 | +16.624 | +15.765 |
| 首次计算调度→首 token ms | −14.110 | −10.082 | −12.109 |
| scheduler TTFT ms | +0.807 | +6.542 | +3.656 |
| client TTFT ms | −1.230 | +7.522 | +3.118 |

恢复确实减少了已完成的 prompt 工作，但新增等待在两对中均超过首次调度后的时间节省。
第一对客户端仍较好，因为 scheduler 以外残差不同；不能把它直接解释成 GPU 计算收益。
无 external restore 的 1163 个请求，客户端差值合并为 −1.097 ms，分轮为 −2.371 / +0.176 ms，
也不支持将所有退化直接归罪于后台 store。

## 下一步：把完成反馈延迟收窄到可修改的位置

代码依据（不是已实测的子阶段归因）：

1. [worker connector context](../vllm/v1/worker/kv_connector_model_runner_mixin.py) 在
   `_get_kv_connector_output` 的 finally 中调用 `get_finished`，完成信息随模型输出返回，
   而非在 DMA 完成时立即独立通知 scheduler。
2. [EngineCore](../vllm/v1/engine/core.py) 的 batch-queue 路径通过 future.result()
   消费模型结果，再调用 scheduler.update_from_output。
3. [Scheduler](../vllm/v1/core/sched/scheduler.py) 在 update_from_output 后段调用
   `_update_from_kv_xfer_finished`，之后等待请求在下一次调度时检查完成状态并恢复。

最小后续步骤：只给恢复完成批次补充 engine 收到结果、进入 update_from_output、
真正处理 finished_recving 三个边界，分清 4.54 ms 是结果返回/批次消费等待，还是 scheduler
内部处理。若主体在前者，仅把 scheduler 函数向前挪不会解决问题。

随后只选一项单因素优化：

- 若 scheduler 内部处理较长，检验提前处理完成通知的可行性；保持失败加载、abort、
  抢占、block 引用计数和缓存注册的先后依赖，不能直接无条件移动代码。
- 若返回/批次消费较长，检验更及时消费完成结果的路径；不能用全局 synchronize、
  无限忙轮询或破坏批处理来换一部分 TTFT。
- 若提交返回→worker 观察区间仍主导，再补 queue/poll 状态验证，而不是直接把差值当成 DMA 优化空间。

无需再次扩大 admission sweep。验收仍要求关闭诊断后的多次重复、未调参负载以及
对 No-offload 的 mean/P95/P99/吞吐比较。本轮只找到了更具体的候选区间，没有性能改动。

## 完整性与产物

- 24 份冻结源码、输入和成本拟合哈希全部复核通过。
- 4×659 条请求完整；155 条恢复记录与 lifecycle promotion/completion request IDs 一致。
- 每事件只有一条 worker observation；同主机时钟有序；9 段之和等于 request-path 等待。
- 无抢占、无 trace 丢失/未完成；原始 CSV mean/P50/P95/P99 独立复算一致。
- 新 trace 的 decision→scheduler 区间与原 lifecycle wait 口径略有不同；后者额外均值
  0.026 / 0.030 ms，最大差 0.0664 ms，没有把两个不同记录点强行当成完全相同。
- GPU 0 共 115 次进程采样无外来 PID；结束时两卡显存 0 MiB、无计算进程，8238 端口关闭。
- 本轮复跑分析器 12 项测试；实现阶段的 100 项测试及旧模型依赖阻断记录见上文诊断文档。

结果目录：[lifecycle_restore_path_gpu0_20261007](../results/lifecycle_restore_path_gpu0_20261007)。

- [冻结设计](../results/lifecycle_restore_path_gpu0_20261007/manifest.json)
- [总体指标](../results/lifecycle_restore_path_gpu0_20261007/evaluation_summary.json)
- [跨运行请求分组](../results/lifecycle_restore_path_gpu0_20261007/request_analysis.json)
- [第一轮恢复分段](../results/lifecycle_restore_path_gpu0_20261007/repeat_0/selective-2g/restore_path_analysis.json)
- [第二轮恢复分段](../results/lifecycle_restore_path_gpu0_20261007/repeat_1/selective-2g/restore_path_analysis.json)
