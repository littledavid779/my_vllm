# 请求首 token 路径观测与对照诊断

## 当前状态

承接 [精确请求关联实验](lifecycle_correlated_diagnostic_20261004.md)。本轮新增的是 opt-in
观测，不是性能优化；未改 admission/promotion 决策、成本参数、缓存容量或调度顺序。

2026-10-06 初始检查两张 GPU 都有外部计算任务，因此先完成 CPU 测试，没有混跑。
收尾复查时 GPU 0 已空闲，GPU 1 仍有任务；随后在 GPU 0 完成下文预定的 4 组对照，
输出至 `results/lifecycle_request_path_gpu0_20261006`。GPU 1 未被操作。

## 为什么补这些数据

此前已确认实际 restore 与未 restore 请求的跨运行 TTFT 均可能退化，但尚不清楚：

1. 是否 GPU 前缀命中变化，导致部分请求的剩余计算量增加；
2. 计算量相同的请求是否等首次调度更久；
3. 差异是否落在首次计算调度之后或调度器之外。

本轮只补这三个缺口，不提前假定是 PCIe、store 或某个打分公式的瓶颈。

## 观测字段与准确口径

| 字段 | 口径 | 不应解读为 |
|---|---|---|
| enqueued_s | Scheduler.add_request 加入新请求时的 perf_counter | HTTP 请求到达时间 |
| first_compute_s | 首次非零模型计算被调度，在推进 computed tokens 前记录 | GPU kernel 开始时间 |
| first_output_s | 调度器处理并接受首个生成 token 时 | 客户端收到第一个内容 chunk |
| queue_to_first_compute_ms | enqueue 到首次计算调度的墙钟间隔 | 纯队列排队；它可能包含异步 restore 等待和调度 CPU 工作 |
| first_compute_to_output_ms | 首次计算调度到接受首 token | 纯 GPU kernel 执行时间；它包含执行、批处理等待、后续 chunk 调度及处理输出 |
| initial_gpu_hit_tokens | 首次成功分配时采用的本地 GPU 前缀长度 | 所有轮次累计命中或 CPU 恢复长度 |
| initial_external_match_tokens | 首次成功分配时的外部命中长度 | 实际完成恢复总量；恢复仍用现有 completion 记录审计 |
| first_compute_cached_tokens | 第一次安排计算前的 computed tokens，包括本地/外部 KV | 仅 GPU 初始命中 |
| completed_prefill_tokens | 截至首 token，已收到正常 worker 输出的批次所覆盖的 prompt 位置 token 数 | prompt 总长度，或所有 scheduled tokens，或 FLOPs |
| prefill_steps / preemptions | prompt 工作完成批次数 / 首 token 前抢占次数 | decode 步数 |

恢复的 tokens 不增加 completed_prefill_tokens；decode tokens/异步占位 tokens 不计入。
分块 prefill 按批次累加，抢占后对 prompt 的重复计算会重复计数，不强行去重。
保存调度批次的工作量快照，避免异步调度提前推进 request.num_computed_tokens 后误计。

调度器内部有准确的时间分割恒等式：

`scheduler_ttft = queue_to_first_compute + first_compute_to_output`

客户端 TTFT 与 scheduler_ttft 的差只标为 **outside_scheduler 残差**，其中可能包含
API/tokenizer/IPC/输出处理/传输以及客户端首 chunk 指标修正，不能直接叫网络延迟。
组间差值仍是跨运行描述，不是相同运行状态上的反事实，也不是因果开销分解。

## 实现与保护

- [RequestPathTrace](../vllm/v1/core/sched/request_trace.py)：默认容量 4096，记录到首 token 为止；
  活跃请求、完成记录、未回收批次均有限额。溢出、未完成、重复请求、非支持请求显式计数。
- [Scheduler 接入](../vllm/v1/core/sched/scheduler.py)：enqueue、成功分配、调度提交、
  worker 返回、首 token、释放请求六处观测；默认关闭。
- [统计接口](../vllm/v1/engine/core.py)：通过现有 get_value_policy_stats 导出，
  no-offload 无 connector 时也可读取。同样适用于同步/异步 Scheduler。
- 开关：`VLLM_EXPERIMENTAL_REQUEST_PATH_TRACE=1`；本轮 runner 用 `--request-path-trace` 设置，
  并写入每组 spec 与 manifest。不开启时不创建 collector。
- 本次实验只支持普通文本生成；pooling、resumable session、encoder-input 请求标为 unsupported，
  不强行给它们套用此口径。reset 清空已完成记录/计数，但不丢弃仍活跃的请求。
- [离线校验器](../scripts/analyze_lifecycle_request_path.py)：要求每个 CSV 请求恰好对应一条首 token
  记录、无日志丢失/未完成、时间顺序和分割成立、两组服务端实际 prompt 长度一致。
  无抢占时要求 `cached_at_first_compute + completed_prefill = prompt_tokens`。
- 沿用此前 `{conversation_id}-turn{message_count}` 关联，保留原会话 owner 映射；禁用重试。
  CSV 的 input_num_tokens 不含完整 chat template，因此计算守恒使用服务端实际 prompt_tokens。

开启观测有额外时钟/字典操作，尚未测量其性能开销。两组都开启用于诊断，不将所得 TTFT
直接与历史 trace-off 结果混合。后续性能收益必须在关闭观测的条件下单独复验。

## 已完成验证

- 10 项 collector/真实 Scheduler 钩子/统计导出测试通过：分块 prefill、同步/异步调度、
  同步/异步 mock external load、重算重复计数、abort/reset/限额，以及观测开关不改变调度 token 数。
- 56 项分析与相关回归测试通过，其中新增分析校验测试 12 项。
- 这些都是 CPU/模拟 worker 测试，读取本地模型配置，不是 GPU 推理结果。
- Ruff check/format、git diff --check 通过。

测试文件：
[观测测试](../tests/v1/core/test_request_path_trace.py)、
[分析校验测试](../tests/benchmarks/test_lifecycle_request_path_analysis.py)。

## GPU 空闲后的固定计划

沿用两组反向顺序 `N→S、S→N`，4 runs / 2636 requests；原 128 会话、4 clients、32 active
conversations、GPU 1000 blocks、CPU 2 GiB、always_restore，Guard/probe 关闭。
源数据仍是已诊断过的数据，不是新泛化集。这是定位实验，不是性能胜出验收。

确认 GPU 0 空闲后可执行（若空闲的是 GPU 1，应同时改 GPU 参数及新输出目录名称）：

```bash
.venv/bin/python scripts/run_lifecycle_correlated_diagnostic.py \
  --source results/lifecycle_cost_probe_gpu1_20260917 \
  --output results/lifecycle_request_path_gpu0_20261006 \
  --gpu 0 --port 8238 --request-path-trace
```

Runner 每组启动前检查 GPU 和端口，运行中检查外来 GPU 进程；不占用已有任务的卡。
每次执行使用新目录，代码/输入/成本拟合冻结；旧实验目录不覆盖。

预定的第一张结果表按 **实际 restore / 无 external restore × prefill 工作量相同/更多/更少**
分组，同时展示客户端 TTFT、等待首次计算、首次调度后等待及调度器外残差的差值。

- 无恢复且计算量相同仍变慢：优先查等待/共享开销，而不是继续调复用概率。
- 计算量增加：核查 GPU 命中/缓存状态变化，再判断是否策略决策造成。
- 恢复减少计算量，但等首次计算明显增加：优先查恢复路径等待及其调度交互。

只有拿到新实测后才能选择下一项消融。执行结果及边界在下文补充。

## 实测：恢复启动计算前的等待抵消了计算节省

4 runs / 2636 requests 全部完成，约 22:39–22:43（2026-10-06，本机时间），
GPU 0 无外来进程。157 次恢复对应到 157 个具体请求，全部路径时间/token 守恒校验通过。

### 总体指标：本轮较好，但不是稳定胜过 No-offload 的证明

| 对比对 | 策略 | Mean TTFT ms | P95 ms | P99 ms |
|---|---|---:|---:|---:|
| 0 | no-offload（先） | 33.091 | 58.299 | 69.044 |
| 0 | selective（后） | 31.540 | 53.561 | 67.684 |
| 1 | selective（先） | 31.224 | 52.507 | 69.810 |
| 1 | no-offload（后） | 31.385 | 53.340 | 63.911 |

两对 mean/P95 均较低，但第二对 P99 更差。本轮只有两次重复、同一已诊断 workload、
开启了观测，不能推翻此前 trace-off 确认实验的负结果，也不能宣称已满足稳定加速目标。
运行开始时 GPU 1 有任务，结束时为空闲；没有连续记录它的负载，主机不是独占环境。

### 实际恢复请求：同一阶段特征在两对中重复出现

以下均为 **Selective − 同一输入请求的 No-offload** 的组内均值，正值表示更多/更慢。
对应的是跨运行请求，不是同一运行状态上的严格反事实。

| 实际 restore 请求 | 对比对 0（77条） | 对比对 1（80条） |
|---|---:|---:|
| 已完成 prefill tokens 差值 | −1154.48 | −1119.39 |
| 等待首次计算调度的差值 ms | +16.889 | +17.099 |
| 首次计算调度→首 token 差值 ms | −12.231 | −11.061 |
| 调度器内部 TTFT 差值 ms | **+4.658** | **+6.038** |
| 调度器外残差差值 ms | −3.110 | −2.036 |
| 客户端 TTFT 差值 ms | +1.548 | +4.002 |

恢复确实减少了 prompt 位置上的计算任务；但首次计算之前的等待增加更多。
本轮最值得优先处理的是这个 **阶段级** 瓶颈，而不是继续假定 GPU miss 让大多数请求多算了。
这不等于已经定位到某一个函数，更不等于其中 17 ms 全是 PCIe DMA。

Selective 组内自身的观测值：

| 观测，实际恢复请求均值 | 对比对 0 | 对比对 1 |
|---|---:|---:|
| enqueue→首次计算调度 ms | 18.258 | 18.401 |
| promotion decision→scheduler-observed restore complete ms | 16.273 | 16.263 |
| 首次计算调度→首 token ms | 11.627 | 12.223 |

两个等待区间高度相关并且重叠，**不能相加**。本轮没有采集 DMA 分段，不能拿历史
2.595 ms DMA 从本轮 16.27 ms 中相减，就断言剩下全部是同步或软件开销。

### 无 external restore：绝大部分计算量相同，本轮未复现一致退化

| Selective 下未恢复的请求 | 对比对 0 | 对比对 1 |
|---|---:|---:|
| 总数 | 582 | 579 |
| prefill 计算量相同 | 569 | 568 |
| prefill 计算量更多 | 7 | 5 |
| prefill 计算量更少 | 6 | 6 |
| 同计算量子集：客户端 TTFT 差值 ms | −2.050 | −0.613 |
| 同计算量子集：等待首次计算差值 ms | −0.007 | +0.267 |
| 同计算量子集：首次调度后差值 ms | −0.827 | −0.289 |

因此此前“无恢复请求也慢”的现象不能直接升级成“store/metadata 是已证实的主瓶颈”。
这次大多数无恢复请求的输入工作量一致，延迟未退化，仍需承认运行状态/批处理/主机变化。
少数计算量增加的请求确实存在，但不是本轮主要的普遍现象。

## 下一步收窄到什么

优先细分 **restore decision → transfer submit → DMA complete → worker complete →
scheduler ready → 首次计算调度**，只补恢复路径的轻量时间点，不重新开启全量 block trace。
目前已有的 runtime restore wait 应作为总区间校验，避免重复计时。

- 若主要耗在提交/小拷贝组织：验证批量或连续块合并是否减少请求可见等待。
- 若主要耗在完成通知/等待下一轮调度：验证 completion 到 ready 的响应路径。
- 若 DMA 本身成本就高于节省的计算：优先调整 restore/recompute 选择。

这三条是根据细分结果选择的互斥优先方向，不是现在就同时实现三项机制。
本轮没有改优化策略；真正的改动仍须在关闭观测、多次重复及未调参负载上，证明 TTFT
胜过 No-offload，并检查 P99/吞吐是否退化。

## 执行审计与产物

- 22 个冻结源码条目、输入和成本拟合哈希检查通过。
- 4×659 条首 token 记录完整、无重复映射；active/pending/dropped/unsupported/abort 为零。
- 所有请求无抢占；逐请求 `cached + completed_prefill = prompt_tokens`，
  `scheduled_prefill = completed_prefill`；两个时间分段之和等于 scheduler TTFT。
- 客户端 CSV 独立复算 mean/P95 与结果一致；同一逻辑请求的服务端实际 prompt 长度一致。
- 115 次 GPU 0 采样无外来进程；结束时 GPU 0 显存 0 MiB，端口 8238 已关闭。
- 66 项相关测试通过，Ruff check/format、git diff --check 通过。

文件：

- [冻结设计](../results/lifecycle_request_path_gpu0_20261006/manifest.json)
- [逐组指标](../results/lifecycle_request_path_gpu0_20261006/evaluation_summary.json)
- [分组结果与最慢请求](../results/lifecycle_request_path_gpu0_20261006/request_analysis.json)
- [逐请求路径差值 CSV](../results/lifecycle_request_path_gpu0_20261006/paired_requests.csv)
- [独立审计](../results/lifecycle_request_path_gpu0_20261006/audit_summary.json)
