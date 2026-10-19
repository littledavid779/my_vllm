# 第一版生命周期策略 GPU 功能验证（2026-09-16）

## 结论与证据边界

在 GPU 0 上完成三次独立服务启动、每次 128 请求，共 384 个有效请求记录。
验证了实际 OFFLOAD / DROP、RESTORE / RECOMPUTE、成本完成反馈、Guard 限流与探测。
不是六组正式性能对照，也不是成本校准；不得据此宣称 TTFT 改善或真实负收益保护有效。
三个测试使用人工成本参数，分别为了覆盖不同分支；第二、三轮关闭收益准入过滤。

实现背景：[第一版策略设计](lifecycle_policy_mvp_20260913.md)。

## 环境

- GPU：GPU 0，A100-SXM4-80GB；另一个用户的 vLLM 服务 PID 2362895 仍持有约 73.3 GiB。
- 启动前一次观测利用率为 100%，后续连续五次为空闲；不能证明整个测试期间独占。
- 使用剩余显存，`GPU_MEMORY_UTILIZATION=0.05`；启动后观测约 1519 MiB 剩余。
- 三轮均由实验脚本退出自己的服务。最后 GPU 0 恢复到 75047 MiB 占用，原服务未停止。
- 模型：本地 Qwen3-0.6B，BF16，TP=1；max model length=4096。
- GPU KV：1000 blocks；CPU KV：4 GiB；offloaded block size=16 tokens；GPU/CPU 淘汰均为 LRU。
- 负载：`results/direct_sharegpt_multiturn_20260806/sharegpt_direct_seed0_128.json`。
- 4 clients，max active conversations=32，128 requests，runtime seed=0，闭环发送。
- 使用 `--preserve-dataset-assistant-history`；这是 ShareGPT 内容回放，不是真实生产到达 trace。

## 人工测试参数

所有参数最终保存在各轮 `run_config.json` 的 KV transfer JSON 中。

| 参数 | 准入/恢复验证 | 重算验证 | Guard 验证 |
| --- | ---: | ---: | ---: |
| store 秒/block | 0.002 | 0.002 | 0.002 |
| restore 秒/block | 0.002 | 0.1 | 0.1 |
| recompute 秒/token | 0.001 | 0.000001 | 0.000001 |
| admission_enabled | true | false | false |
| guard_enabled | true | false | true |
| guard_window_s | 30 | 30（未启用） | 0.5 |
| guard_min_samples | 8 | 8（未启用） | 4 |
| ewma_alpha | 0.2 | 0.2 | 1.0 |
| probe_every | 64 | 64（未启用） | 8 |
| probe_max_blocks | 4 | 4（未启用） | 2 |
| max_decision_events | 256 | 4096 | 4096 |

三轮均为 `promotion_policy=cost_aware`，而非用 always_recompute 强制第二、三轮路径。
重算轮通过设置恢复远贵于重算的人工初值覆盖选择分支；这不验证成本预测准确性。
Guard 轮通过暂时允许 all-block admission 和较短窗口制造估计预算持续为负的条件。

## 结果

| 指标 | 准入/恢复验证 | 重算验证 | Guard 验证 |
| --- | ---: | ---: | ---: |
| 有效请求记录 | 128 | 128 | 128 |
| 实际 store blocks | 1450 | 2083 | 143 |
| 拒绝的候选 block 次数 | 1071 | 0 | 3821 |
| 完成的 load blocks | 1285 | 0 | 0 |
| restore 决策 / 完成反馈 | 40 / 40 | 0 / 0 | 0 / 0 |
| recompute 决策 / 完成反馈 | 0 / 0 | 43 / 43 | 26 / 26 |
| 不完整 promotion | 0 | 0 | 0 |
| 最终 pending | 0 | 0 | 0 |
| Guard 切换次数 | 0 | 0 | 2 |
| 最终 Guard 状态 | NORMAL | NORMAL（未启用） | OFF |
| 探测 batch 次数 | 0 | 0 | 32 |

store/load 是累计 block 操作数，不是 unique block 数。
拒绝候选计数可能重复涉及同一个逻辑 block，不能写成“丢弃了 1071 个唯一 KV”。
三组使用不同人工参数和准入开关，其存储差异只作机制观测，不能作为正式收益对照。

Guard 的事件序列为：

1. NORMAL → THROTTLE，窗口估计收益 -0.140 s。
2. THROTTLE → OFF，窗口估计收益 -0.082 s。
3. OFF 下继续执行有限探测，单批选中最多 2 blocks，与配置一致。

这些预算来自人工成本初值，不等于请求真实损失时间。
本轮没有验证 GPU 上 OFF → NORMAL 的恢复过程，也没有验证默认 30 秒窗口的表现。

## 完整性检查

- 每轮 request summary 恰有 128 行，所有行的 output_num_tokens > 0。
- 各次运行脚本退出码为 0；这是生成路径完成检查，不是语义正确性或精确输出一致性验证。
- 各轮 `lcuc_pending=0`、`lcuc_incomplete_promotions=0`。
- 各轮 `offload_cost_events_dropped=0`、`offload_cost_event_unmatched_transfers=0`。
- 第一轮有 184 条 `lcuc_events` 超出默认 256 条有界队列被覆盖；总计数仍在。
  不能把第一轮保留的调试事件当成全量 trace。后两轮上限设为 4096，无覆盖。
- 原始日志和 summary 保留时延，但由于资源共享和人工先验，本报告不将其作为性能证据。

## 原始结果

- [准入/恢复：配置](../results/lifecycle_gpu0_shared_functional_20260916/20260916_001616/seed_0/lru/run_config.json)
- [准入/恢复：策略统计](../results/lifecycle_gpu0_shared_functional_20260916/20260916_001616/seed_0/lru/value_policy_stats.json)
- [重算：配置](../results/lifecycle_gpu0_shared_functional_20260916/recompute/20260916_001824/seed_0/lru/run_config.json)
- [重算：策略统计](../results/lifecycle_gpu0_shared_functional_20260916/recompute/20260916_001824/seed_0/lru/value_policy_stats.json)
- [Guard：配置](../results/lifecycle_gpu0_shared_functional_20260916/guard/20260916_001926/seed_0/lru/run_config.json)
- [Guard：策略统计](../results/lifecycle_gpu0_shared_functional_20260916/guard/20260916_001926/seed_0/lru/value_policy_stats.json)

各目录同时保留 `server.log`、`bench.log`、`request_summary.csv`、`output.json`。
目录中 `lru` 指 GPU 淘汰策略，不表示没有开启生命周期策略。

## 下一步

关键路径已不再停留在 CPU/mock 测试。下一步在可独占 GPU 上：

1. 使用与评测会话分开的输入校准 scheduler-observed restore/recompute 成本；
   明确 store 成本的代理口径，不把 DMA 时长直接视作请求阻塞。
2. 固定校准参数，进行 no-offload / all-block / admission / promotion /
   joint-no-guard / full 六组小规模对照。
3. 正式 trace 增大事件保留上限或分批导出；Guard 评测持续足够长，并单独验证环境改善后的恢复。
4. 分开报告估计预算、真实时延、传输量和决策开销，不能把人工分支覆盖当作算法准确率。
