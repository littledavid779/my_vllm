# Admission 诊断与新增写入成本消融（2026-10-03）

## 目的与范围

承接 [六组消融](lifecycle_decision_ablation_20260917.md)：先定位 selective admission
减少 D2H 却使 TTFT 变差的原因，不同时修改复用概率、前缀收益模型、Guard 或调度。

本轮增加两项默认关闭的能力：

1. `trace_admission`：关联拒绝写入、后续 GPU miss 后的 CPU 查询，以及连续前缀缺口。
2. `new_write_store_cost`：准入评分只对尚未分配 CPU 存储的候选 block 收取 STORE 成本。

这是诊断基础设施与小范围消融，并非已验证的新性能优化。

## 实现

- `OffloadingManager.get_block_residency` 提供无副作用快照：`ready/pending/missing/unknown`。
  LRU/ARC 均支持，reuse wrapper 透传；不调用 touch，不改变替换顺序或统计。
  ARC ghost entries 不属于驻留数据；未实现该接口的 manager 返回 unknown。
- `LifecycleOffloadingManager` 在评分前读取快照；ready/pending 都不需要重复 D2H。
  unknown 保守计费。原公式中的 STORE 项从 `n * Cs` 切换成 `n_new * Cs`。
  `n * Ccap`、收益项、最长正收益前缀规则保持不变。实际事后记账仍按实际新分配量。
- 这**没有**解决既存 block 的收益是否被重复计入，也没有建立完整的边际前缀收益模型。
  因此即使计费修正正确，也不能保证 TTFT 改善。
- connector 使用原有有界 cost-event 流与进程内 block ID，记录 admission 的候选状态、
  选择长度、请求/块位置、原因、实际新分配 ID，以及 allocation failure。
  `trace_admission=true` 必须同时启用 `track_cost_events=true`。
- CPU 查询即使返回零命中也记录 `external_lookup`。整个 GPU miss 后缀的状态为诊断快照；
  原 lookup 仍在第一个缺失/未就绪处停止，后续 ready block 不计作实际命中。
- controller promotion/completion 日志补充 `request_id/target`。

## 分析口径

运行 `scripts/summarize_lifecycle_admission.py VALUE_POLICY_STATS.json`，只读输出 JSON。

它统计：

- 被拒绝的 missing block；未选中的 ready/pending block **不算丢弃**；
- 候选中已有 CPU 分配的 block 次数，衡量旧 STORE 计费涉及的规模；
- 拒绝后出现在其他请求 GPU miss 后缀中的 block（按 request/block 去重）；
- 第一个 missing gap 是否具有更早的拒绝记录；
- 缺口之后的 ready block 数，及紧邻该缺口的连续 ready 段长度；
- 最多 20 个 admission event → lookup event 的关联例子。

后续重新分配存储会清除旧拒绝关联；查询重试不重复累计同一 request/start/gap。
allocation failure 与策略拒绝分开处理。仅统计保留窗口内的关联，不跨统计 reset 重建历史。
旧日志没有这些字段，不能倒推出上述结论；event 丢失或 block-ID tracker 淘汰会报错。

**不能将这些关联称为真实 lost restore、saved time 或 Oracle 收益。**
即使当初接纳，数据也可能在未来需求前被淘汰；补齐一个缺口还可能引发容量竞争。
同一 block 可被不同请求反复计入 ready suffix 统计，不能当 unique blocks。
unknown 状态需要单独检查，不能拿缺失的观测支持完整结论。

## 四组实验

新增 `--suite admission-diagnostic`，所有组开启相同诊断，关闭 Guard/cold probes：

| 组 | Admission | Promotion | STORE 评分 |
|---|---|---|---|
| all-block-control | 全接纳 | always_restore | 原口径 |
| promotion-only | 全接纳 | cost_aware | 原口径 |
| joint-no-probe | 选择性 | cost_aware | 全候选 |
| joint-new-write | 选择性 | cost_aware | 仅新增写入 |

关键对比：joint-new-write vs joint-no-probe，仅一个开关不同。
promotion-only 用于判断 selective admission 相对全接纳的代价。
每组保留原来的完整对话、固定 client assignment、输入 token 一致性和 GPU 占用审计。
新增 residency 接口与分析脚本纳入代码哈希。每组生成 `admission_summary.json`。
使用新的结果目录；旧实验 manifest 与结果不覆盖。

示例（先确认指定 GPU 无其他进程）：

```bash
.venv/bin/python scripts/run_lifecycle_fixed_workload.py \
  --suite admission-diagnostic --gpu 1 --port 8238 --repeats 3 \
  --source results/lifecycle_cost_probe_gpu1_20260917 \
  --output results/lifecycle_admission_diagnostic_gpu1_20261003
```

先做 1 repeat 机制核查，正式结论至少 3 repeats。该数据集已用于诊断，不属于未见验证集。
诊断会增加 O(n) 扫描和日志成本；正式性能验证还需关闭诊断重跑，不能直接与旧结果拼表。
关注 mean/P95 TTFT、实际新写入/恢复 block、拒绝后需求与缺口关联，不能只看 D2H。

## 继续条件

- 若已有 CPU 分配在候选中占比很低，重复 STORE 计费不太可能是主要瓶颈。
- 若拒绝后的第一缺口后常有连续 ready KV，再独立实现 recoverable-prefix marginal admission。
- 若几乎所有拒绝都由冷启动收益先验决定，优先校准需求/成本，不直接提高 prior 美化结果。
- 无论哪种情况，内部 estimated utility 为正不等于端到端性能改善。

## 验证状态

本轮最终回归：99 项策略/校准/CPU manager/benchmark 测试通过，另外 16 项不依赖远程
模型的 connector 测试通过，合计 115 项；Ruff check 与 format check 均通过。
完整 connector 文件中需要远程 `facebook/opt-125m` 的用例在加载配置阶段遇到
`Missing dependencies for SOCKS support`，不能宣称完整集成测试通过。
已完成一轮四组 GPU 诊断，结果与限制见下节；尚未完成多重复性能验证。

回归命令：

```bash
.venv/bin/python -m pytest \
  tests/v1/kv_offload/test_lifecycle_policy.py \
  tests/v1/kv_offload/test_lifecycle_calibration.py \
  tests/v1/kv_offload/test_cpu_manager.py \
  tests/benchmarks/test_lifecycle_admission_summary.py \
  tests/benchmarks/test_lifecycle_fixed_workload.py \
  tests/benchmarks/test_lifecycle_ablation_summary.py \
  tests/benchmarks/test_multiturn_drain.py -q

.venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_offloading_connector.py -q \
  -k 'promotion or lifecycle or admission_trace or cost_events or host_cost or worker_attaches'
```

## 首轮 GPU 诊断结果（2026-10-03）

运行目录：
[lifecycle_admission_diagnostic_gpu1_20261003_smoke](../results/lifecycle_admission_diagnostic_gpu1_20261003_smoke)。
主表来源：[evaluation_summary.json](../results/lifecycle_admission_diagnostic_gpu1_20261003_smoke/evaluation_summary.json)。
输入/代码冻结记录：[manifest.json](../results/lifecycle_admission_diagnostic_gpu1_20261003_smoke/manifest.json)。

GPU 1，Qwen3-0.6B，GPU KV 1000 blocks，CPU tier 4 GiB，block size 16 tokens；
固定 128 会话、659 个 user-turn 请求，4 clients、最多 32 active conversations，runtime seed 0。
使用原冻结校准，无重新拟合；Guard/cold probes 关闭，四组开启相同诊断。
实际运行顺序：all-block-control → joint-new-write → joint-no-probe → promotion-only。

| Policy | Mean TTFT (ms) | P95 TTFT (ms) | Actual store blocks | Completed load blocks | Requests/s |
|---|---:|---:|---:|---:|---:|
| all-block-control | 30.696 | 52.166 | 10998 | 15406 | 31.486 |
| promotion-only | 34.031 | 58.237 | 10997 | 13545 | 30.882 |
| joint-no-probe | 35.847 | 59.622 | 6678 | 6318 | 30.292 |
| joint-new-write | 32.835 | 54.797 | 7097 | 4977 | 30.792 |

stores/loads 是全程累计操作 block 数，不是 unique block 数。
单轮新 joint 相对旧 joint mean TTFT −3.012 ms（−8.40%）；相对 all-block-control
仍慢 6.97%，D2H 少 35.47%。这些是观测差异，**不是计费修正的已证实收益**。

### 1. 有重复计费涉及的候选，但不等于影响实际新写入

| Policy | Candidate blocks | Already allocated candidates | Mixed batches（已有 + missing） |
|---|---:|---:|---:|
| all-block-control | 12868 | 1870 | — |
| joint-no-probe | 22111 | 1076 | 14 |
| joint-new-write | 23418 | 2622 | 36 |

按新版日志的同一状态对选中前缀重打旧 STORE 分数：

`G_old(selected) = G_new(selected) - allocated_in_selected * Cs`

其中 Cs=0.00003428220206612273 s/block，margin=0。仅 6 次选中前缀的 G_old≤0；
但这 6 次 selected 都为 31 blocks，selected_new_blocks=0，stored_block_ids=[]。
对应 admission event IDs：1432、2553、2751、4347、4368、4464。
其余已选中前缀在旧计费下仍为正收益；新版拒绝的前缀在更高的旧计费下也不会获准。

因此，本轮未观察到“扣除重复 STORE 成本直接救回新增写入”的证据。
两条独立轨迹的 stores/loads/TTFT 差异，不能据此归因于该数学修正；
时序、批处理、在线成本反馈也会改变决策。六次纯已有数据的选择仍可能触及控制面行为，
不能据此证明两条系统轨迹完全等价。

### 2. 大量被拒绝的 KV 后来有需求，但没有看到 ready 后缀被缺口阻断

| 指标 | joint-no-probe | joint-new-write |
|---|---:|---:|
| Admission batches | 1872 | 1879 |
| cost_drop batches | 1770 | 1754 |
| Unique rejected missing blocks | 8656 | 8426 |
| 之后出现在其他请求 GPU-miss 后缀中的 rejected unique blocks | 5592 | 5613 |
| 首个 missing gap 关联到此前拒绝 | 367 | 371 |
| 这些缺口之后的 ready CPU blocks | 0 | 0 |

四组所有 external lookup 快照中的 ready_suffix_blocks_behind_gap 均为 0，
candidate/lookup unknown 状态也均为 0。

**当前负载不支持优先做“补一个缺口，解锁已存后缀”的机制。**
更值得诊断的是准入拒绝后续确有需求的 KV：需求概率代理、保留概率与收益成本模型是否过于保守。
“后来有需求”仍不证明当初 offload 在容量和端到端时间上值得；必须保留这个边界。

### 3. Promotion 与收益账本也还没有得到端到端验证

promotion-only 本轮有 284 次 restore 决策、142 次 recompute 决策，mean TTFT 仍高于
always-restore 的 all-block-control。142 是选择次数，不是 recompute 获胜次数。
其内部 estimated net utility 为 +1.573 s，也不能作为实际请求时延改善的保证。

### 审计与限制

- 四组各 659 请求，合计 2636；conversation/turn/client/input token 签名完全一致。
- 所有被 manifest 纳入的代码哈希一致，输入文件哈希一致。
- 122 次 GPU 采样中没有其他进程占用 GPU 1；GPU 0 同时有外部任务，因此不是整机独占。
- store failure、store completion failure、cost-event 丢失、block-ID 淘汰、
  decision-event 丢失、未匹配传输、pending/incomplete promotion 均为 0。
- 完成后 GPU 1 显存为 0 MiB、端口 8238 已释放，未操作 GPU 0 上的任务。
- 只有 1 repeat、1 runtime seed，输入已用于此前诊断；闭环到达时序不完全一致，
  日志可能增加开销，不可与旧无诊断结果直接合并或声称稳定加速。

### 后续优先级调整

1. 保留新增写入计费开关作为一致性修正，但不要将本轮 −8.40% 写成其贡献。
2. 暂缓 ready-suffix 缺口修复；当前没有观测对象支持它。
3. 下一轮优先诊断拒绝概率/成本校准，用固定 always-restore 对照隔离 admission，
   并在同一观测状态做 shadow scoring，避免把独立运行时序差异解释成算法收益。
4. 确认真正改变新增写入与后续有效恢复的机制后，再做至少 3 repeats 及关闭诊断的验证。
