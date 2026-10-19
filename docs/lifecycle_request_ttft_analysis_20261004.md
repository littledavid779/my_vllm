# 逐请求 TTFT 退化分析：先定位，暂不修改策略

## 结论

本轮只离线分析 [18 组确认实验](lifecycle_ttft_confirmation_20261003.md)，没有启动
GPU 实验，没有修改运行时策略。当前最值得优先核查的是长上下文请求的恢复/重算路径，
但现有记录缺少服务端请求到客户端具体轮次的映射，尚不能断定恢复本身造成退化。

主要发现：

1. Selective 相比 no-offload 的平均 TTFT 多 2.565 ms；输入 ≥1024 tokens 的固定分组
   六轮均退化，组内平均多 4.155 ms。
2. Selective 下整个会话都没有 restore 的请求，平均也多 1.609 ms。这排除了
   “所有额外等待都只出现在直接 restore 的请求上”的简单描述，但不证明 store 干扰。
3. 有 restore 的会话包含大量未必发生 restore 的轮次，不能把整组当作恢复请求。
4. 现有模型 estimated_saved_s 不是实测重算反事实；不能用它证明恢复净收益。

## 输入、关联方式与限制

- 源目录：`results/lifecycle_ttft_confirmation_gpu1_20261003`。
- 读取 18 份 request_summary.csv，共 11862 条请求记录；Selective 分别与两个基线
  比较，各形成 6×659=3954 个跨运行请求对。两个比较复用了同一组 Selective 记录，
  不是 7908 个独立 Selective 请求。
- 以 conversation_id + input_num_turns 连接，拒绝重复键、缺失请求和输入长度/client 不一致。
  `input_num_turns` 在此 benchmark 实际是 messages 数量，不是用户轮次序号；首轮取 1，
  后续取 >1。跨运行相同输入不代表相同到达时间、GPU 状态或队列状态。
- CSV request_id 是按客户端 start_time 排序后的数字序号。发送 X-Request-Id 的值仅为
  conversation_id；服务端内部 ID 是 `chatcmpl-{conversation_id}-{8位随机后缀}`。
  读取 SSE 时未将响应 ID 保存到 RequestStats/CSV，现有 promotion/completion 事件又没有
  可关联轮次的时间戳，故本轮不猜测逐请求 restore 归属。
- 核查 benchmark 源文件 SHA256 与原 manifest 一致。代码依据：
  [请求发送和统计](../benchmarks/multi_turn/benchmark_serving_multi_turn.py)、
  [内部 request ID](../vllm/v1/engine/input_processor.py)。
- `approx_cached_percent` = 历史文本 token 占比，**不是实测 GPU/CPU KV 命中率**；不拿它分类。
- promotion/completion 的 (request_id,target) 多重集合与计数器对齐，恢复 block 数对齐；
  event 丢失、pending、incomplete 和 recompute 均要求为零，未知会话 ID 拒绝分析。
  因而可确认“整个会话没有 restore”；有 restore 的会话仍只能标记“具体轮次未知”。

## 相比 No-offload：退化出现在哪些请求

正值 = Selective 更慢。N 包含六次重复，不是独立 workload 样本。
“整体均值贡献” = 本组所有差值之和 / 3954，各个**独立分组维度内部**可加和为 2.565 ms。
这只是均值的算术分解，不是因果开销归因，更不是 P95 分解。

| 分组维度 | 类别 | N | 组内 TTFT 差值 ms | 整体均值贡献 ms |
|---|---|---:|---:|---:|
| 请求阶段 | 首轮 | 768 | +1.549 | +0.301 |
| 请求阶段 | 后续轮次 | 3186 | +2.810 | +2.264 |
| 输入长度 | <128 tokens | 1008 | +2.239 | +0.571 |
| 输入长度 | 128–511 | 1032 | +2.164 | +0.565 |
| 输入长度 | 512–1023 | 1020 | +1.900 | +0.490 |
| 输入长度 | ≥1024 | 894 | +4.155 | +0.939 |
| 会话恢复情况 | 整个会话没有 restore | 2284 | +1.609 | +0.930 |
| 会话恢复情况 | 会话有 restore，具体轮次未知 | 1670 | +3.872 | +1.635 |

后续请求贡献了约 88.3% 的总体均值退化，但也占 80.6% 的请求；不能只看贡献比例判断瓶颈。
输入 ≥1024 tokens 的分组占 22.6% 请求，贡献约 36.6% 的均值退化。

### 六轮一致性

| 分组的 TTFT 差值 ms | R0 | R1 | R2 | R3 | R4 | R5 |
|---|---:|---:|---:|---:|---:|---:|
| 首轮 | −1.674 | −0.672 | +2.947 | +4.349 | +3.616 | +0.727 |
| 后续轮次 | −0.320 | +0.535 | +4.495 | +4.421 | +6.528 | +1.201 |
| 输入 ≥1024 | +0.360 | +2.879 | +4.369 | +6.416 | +8.603 | +2.300 |
| 整个会话没有 restore | −1.300 | −0.247 | +3.483 | +2.325 | +5.090 | +0.095 |
| 会话有 restore，轮次未知 | +0.265 | +1.019 | +5.366 | +7.973 | +6.975 | +2.431 |

这是事后描述，不是显著性检验；六轮来自同一 workload 且共享主机。
小于 128 tokens 分组也六轮均退化（完整数值见 JSON），不能声称只有长请求受损。

## 相比 All-block：改善没有在长请求上明显体现

| 输入长度 | Selective − All-block 平均 TTFT ms |
|---|---:|
| <128 | −2.098 |
| 128–511 | −1.742 |
| 512–1023 | −1.500 |
| ≥1024 | −0.252 |

因此整体胜过 All-block 不能解释为“长请求恢复路径已经优化成功”。

## 尾部样本与恢复观测

每轮单独取 Selective TTFT 最大的 ceil(659×5%)=33 条，六轮合计 198 条：

- 173 条输入 ≥1024 tokens（87.4%）；4 条首轮。
- 22 条属于整个会话没有 restore 的组；其余 176 条属于有 restore 但轮次未知的会话。
- 这是按结果挑出的尾部，**不是恢复请求占比，更不证明这些请求因 restore 而变慢**。
  不能把该子集平均差值当总体收益或 P95 改变量。
- 原始逐条数据、两个策略各自的尾部，以及每轮退化最大的 20 条，都保存在 analysis.json。

| 恢复观测，六轮事件合并 | All-block | Selective |
|---|---:|---:|
| 完成恢复事件 | 372 | 441 |
| 每事件恢复 blocks 均值 | 26.15 | 71.67 |
| scheduler-observed restore wait 均值 | 10.343 ms | 17.491 ms |

事件 request ID 在每轮内部唯一，但具体客户端轮次未知。Selective 恢复的块更多，等待也更长；
事件长度组成不同，**不能用 17.491 vs 10.343 ms 直接证明策略降低了传输效率**。
必须与对应长度和负载的 recompute 比较，本轮没有测量这些事件的另一条分支。

## 下一项验证：先补最小关联，再检验恢复路径

本轮结果不足以选择一个确定的算法修复。下一步优先补最小的观测关联，而不是扩大参数 sweep：

1. 为客户端每个请求保留 conversation_id + message_count 的唯一标识，并将其传入请求头、
   写入 CSV；不要仅依靠 SSE 的外部会话 ID。确保服务端随机后缀后的内部 ID 可逆关联。
2. 保留现有有界 promotion/completion 记录，不恢复全量 block trace。关联完整性必须达 100%，
   日志无丢失；重试与一次请求多个 promotion 要显式处理。
3. 先在同一 workload 小规模复跑 no-offload / selective(always_restore)，对实际恢复请求和
   无恢复请求分别分析。测试分类映射后才启动 GPU，不混合新旧日志作为同一次性能结论。
4. 若退化集中在实际 restore 的长请求，再做固定输入长度/前缀及尽量匹配负载的 restore/recompute
   路径对照；另一次整段运行的 recompute-only 只能作为策略消融，不能宣称逐事件反事实。
5. 若无 restore 请求仍明显退化，再验证写入时机、元数据处理与调度干扰。当前数据不能拆出这些
   开销，不直接启用 Guard 或修改 admission 参数。

以上关联补丁与新 GPU 实验**尚未实施**。这是一项明确的数据缺口，不是发现了确定的性能根因。

## 复现与产物

```bash
.venv/bin/python scripts/analyze_lifecycle_request_ttft.py \
  --source results/lifecycle_ttft_confirmation_gpu1_20261003 \
  --output /tmp/lifecycle_request_ttft_new_analysis
.venv/bin/python -m pytest tests/benchmarks/test_lifecycle_request_ttft.py -q
```

输出目录必须不存在，避免覆盖历史结果。脚本不启动服务、不导入 vLLM runtime、不分配 GPU。

- [分析脚本](../scripts/analyze_lifecycle_request_ttft.py)
- [单元测试](../tests/benchmarks/test_lifecycle_request_ttft.py)
- [完整结果及源数据 SHA256](../results/lifecycle_request_ttft_analysis_20261004/analysis.json)
- [逐请求比较 CSV](../results/lifecycle_request_ttft_analysis_20261004/paired_requests.csv)
- [自动生成的分组表](../results/lifecycle_request_ttft_analysis_20261004/summary.md)

独立检查原始输入哈希未变、两基线各 3954 个请求对、三种分组的贡献均加和为总体均值差。
新增测试覆盖重复/缺失键、输入不一致、无效 TTFT、事件缺失/丢失/未完成、计数不一致、
含连字符的会话 ID、未知会话和不推断具体轮次。
