# Restore 等待分段诊断：实现完成，GPU 实测待跑

更新：2026-10-07 已完成 4 组 GPU 实测，结果见
[恢复分段实验报告](lifecycle_restore_path_results_20261007.md)。以下保留 10 月 6 日实现与待跑计划快照。

## 目标与当前边界

承接 [请求首 token 路径诊断](lifecycle_request_path_diagnostic_20261006.md)。上一轮
157 个实际恢复请求中，跨运行同输入对照显示：等待首次计算平均增加约 17 ms，
首次计算调度到首 token 平均减少约 11.63 ms。下一步是定位等待发生在哪个环节，
不是继续调整 admission 分数，也不是直接假定 PCIe 小拷贝就是瓶颈。

本轮 **只新增默认关闭的 restore-only 观测及分析**，未改策略参数、传输次序、
调度决策或 CUDA 实现；没有新增 synchronize 或忙轮询。

GPU 0/1 在初始检查时空闲，但测试期间均被其他任务占用。
GPU 0 启动前保护检查拒绝运行（PID 2977872，约 15.9 GiB）；随后 GPU 1 也约
76.6 GiB 占用。因此未启动推理服务、未干预外来任务、没有本轮 GPU 性能数据。
`results/lifecycle_restore_path_gpu0_20261006` 只包含失败启动前的冻结计划及输入，
不是已完成实验。重跑必须使用新目录。

## 记录什么

开关：`VLLM_EXPERIMENTAL_RESTORE_PATH_TRACE=1`。诊断 runner 对应
`--restore-path-trace`，同时要求 `--request-path-trace`。

| 相邻时间点 | 能定位的区间 | 解释限制 |
|---|---|---|
| enqueue → restore decision | 首次处理、lookup 前后等待 | 不是纯队列等待 |
| decision → allocated | 成功分配前的调度工作 | 分配失败重试保留最后一次 decision，早期等待仍在上一段 |
| allocated → metadata built | 准备 load、构建 connector metadata | 包含同轮其他工作 |
| metadata built → worker received | 调度提交到 worker 开始处理传输 | 包含 IPC/排队等，不能直接标作网络时间 |
| worker received → submit start | worker 内提交前等待 | 包含提交先前 deferred stores 及同批其他 loads |
| submit start → submit return | host transfer_async 提交 | 不是纯 Python 开销，也不是纯 DMA |
| submit return → worker observed | 提交返回后到 worker 发现完成 | 包含传输排队、执行及完成轮询延迟，无法仅靠这段进一步拆开 |
| worker observed → scheduler observed | 完成结果返回到调度器处理 | 包含结果传递及调度器处理等待 |
| scheduler observed → first compute scheduled | 恢复完成通知后到首次计算调度 | scheduler observed 不是严格的 request runnable 时间 |

上述 9 段应相加等于同一请求的 enqueue→首次计算调度时间。
host 时钟使用 perf_counter；离线诊断只接受 **同一主机、一个 worker、无抢占、
每请求一次恢复** 的记录。多 worker 不能取不同 worker 时间点的最大值来拼接假路径。

CUDA start/end event 的 elapsed time 复用原实现，另列 `dma_elapsed_ms`。
它位于已设置 stream 依赖之后，不涵盖全部等待；也不是精确的 host DMA 完成时间。
该值与 host 区间重叠，**不参与 9 段求和，不以其差值直接断言软件开销**。
`worker_observed_s` 是 connector 消费完成结果的时刻，可能比 GPU 实际完成晚。

## 实现位置与限制

- [RestorePathTrace](../vllm/v1/kv_offload/restore_trace.py)：有界 active/completed 记录；
  默认 4096 请求，每条最多 16 worker observations；重复、未匹配、溢出、取消显式计数。
  reset 只清除完成记录及计数，不清除仍活跃的请求。
- [OffloadingConnector](../vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py)：
  decision、allocation、metadata、worker submission/completion、scheduler completion 钩子；
  通过原 connector stats 回传时间戳。只给 load 操作附加 restore_timing，store 不附加。
- [分析器](../scripts/analyze_lifecycle_restore_path.py)：逐请求匹配 request-path 记录，
  检查主机一致、单 worker、时间有限且有序、正 transfer bytes、无抢占、无遗漏及分段守恒。
  此处 external-match 请求集合还会由 runner 原有 lifecycle completion 审计交叉验证。
- [runner](../scripts/run_lifecycle_correlated_diagnostic.py)：新参数写入 manifest/spec；
  仍禁止 full cost/admission trace、Guard/probe；新增源码加入哈希冻结。

这仍有少量观测成本，尚未测量 overhead；不得将本轮 trace-on 数据当作 trace-off 性能验收。
分析器是有限范围诊断，不声称支持多机时钟、多次抢占恢复或多 worker 临界路径还原。

## 验证

- 83 项本地相关测试通过（新 collector/worker round-trip、反序列化、默认关闭、
  请求路径守恒、分析器拒绝坏记录及既有 benchmark 回归）。
- 16 项旧 connector promotion/cost/admission/recompute 测试通过。
- 另补 scheduler lookup→allocation→metadata 钩子测试通过；新文件合计 18 项通过，
  其中 17 项包含在上述 83 项中，去重后共 100 项相关测试通过。
- 完整旧 connector 测试尝试中，8 项因远端 `facebook/opt-125m` 配置获取时缺少 SOCKS
  依赖失败，随后停止该测试进程；没有将其算作通过，也未为此改环境/下载模型。
- 新增测试：[collector/接入](../tests/v1/kv_connector/unit/test_restore_path_trace.py)、
  [分析器](../tests/benchmarks/test_lifecycle_restore_path_analysis.py)。

## 待跑固定实验

仍采用 No-offload→Selective、Selective→No-offload，共 4 runs × 659 requests。
Qwen3-0.6B、同一 128 会话、4 clients、32 active conversations、GPU 1000 blocks、
Selective CPU 2 GiB、always_restore、原成本拟合、Guard/probe off。无新策略消融。

先确认目标 GPU 无其他任务，使用新目录：

```bash
.venv/bin/python scripts/run_lifecycle_correlated_diagnostic.py \
  --source results/lifecycle_cost_probe_gpu1_20260917 \
  --output results/lifecycle_restore_path_gpu0_20261006_retry1 \
  --gpu 0 --port 8238 --request-path-trace --restore-path-trace
```

若 GPU 1 空闲而 GPU 0 忙，应改 GPU 参数和输出目录，不抢占其他任务。
每组启动及运行中保留原 GPU 检查。新增输出是每个 Selective 目录的
`restore_path_analysis.json`：逐事件 9 段耗时、DMA elapsed、bytes 和汇总均值。

拿到数据后只选一个主方向：

1. host submit 占比高：再验证 mapping / 逐 block 提交 / 连续块合并。
2. submit return→worker observed 占比高：还需区分 DMA/stream 排队与完成轮询，不能直接改 memcpy。
3. worker→scheduler 或 scheduler→compute 占比高：优先做完成通知/恢复放行路径的单因素消融。
4. DMA 本身已很贵：检验当前 restore/recompute 成本判定。

最终标准不变：选定优化后，在关闭诊断、多次重复和未调参负载上验证 TTFT
是否优于 No-offload，并检查 P95/P99/吞吐；本页只完成定位工具，不提前宣布优化收益。
