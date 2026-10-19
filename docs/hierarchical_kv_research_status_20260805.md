# 收益感知分层 KV Cache 研究状态

日期：2026-08-05

## 1. 课题定位

暂定题目：

> 面向多轮大模型推理的收益感知分层 KV Cache 管理研究

本课题研究有限 GPU HBM 和 CPU DRAM 容量下的 KV Cache 分层管理。核心问题是：如何结合 KV block 的未来复用价值、传输成本和重新计算成本，在线决定 `KEEP`、`OFFLOAD`、`DROP`、`RESTORE` 和 `RECOMPUTE`，并在外部缓存收益为负时安全退化到 no-offload。

当前课题已从单一的 GPU eviction 打分扩展为分层 KV Cache 控制面，但不将 progressive restore、复杂 scheduler、多机和 SSD 层列为开题阶段的必要贡献。

## 2. 课题演进

2026 年 6 月的工作主要研究 GPU HBM 内部的淘汰顺序，联合考虑：

- block sharing；
- prefix/chain 位置；
- short-term hotness；
- cold/shared 和 private/shared 请求差异。

ShareGPT 真实多轮会话经过 BurstGPT 请求 token 长度匹配后组成的受控并发 workload 表明，sharing/chain 是有用信号，但单独修改 eviction score 在不同压力区间不稳定，hotness 信号也较弱。当前 runner 使用 `REQUEST_RATE=0` 的闭环并发，没有回放 BurstGPT 到达时间。

后续实验表明，HBM 淘汰并不是决策终点：

```text
                           ┌── KEEP in HBM
GPU block value decision ──┤
                           └── EVICT
                                ├── OFFLOAD to DRAM
                                └── DROP

Future request accesses the prefix
                                ├── RESTORE from DRAM
                                └── RECOMPUTE prefill
```

因此，当前研究将早期 sharing、chain 和 hotness 作为复用价值信号，进一步研究 CPU admission、promotion 和负收益保护。

## 3. 统一研究问题

对 block `b` 的净收益可以抽象为：

```text
U(b) = P_reuse(b) * C_saved_compute(b)
       - C_store(b)
       - P_restore(b) * C_restore(b)
       - C_blocking(b)
```

该公式目前是设计目标，不代表系统已经能够精确预测每一项。第一版系统计划使用生命周期统计、分段成本表和在线滑动窗口近似。

系统包含三个核心研究模块：

1. **生命周期观测与成本账本**：记录 create、access、GPU eviction、CPU store/eviction、lookup/load、recompute、传输字节及时间。
2. **收益感知的选择性准入**：GPU 淘汰时在 `OFFLOAD` 和 `DROP` 之间选择。
3. **恢复/重算决策与负收益保护**：CPU 命中时选择 `RESTORE` 或 `RECOMPUTE`；当滑动窗口净收益持续为负时降低 admission rate，必要时关闭 offload。

## 4. 已验证结果

### 4.1 GPU eviction：sharing/chain 有信号，但收益不稳定

128-request、3-seed 的早期结果中，`chain_hybrid` 相对 LRU：

- 平均 TTFT 改善 7.89%；
- P90 TTFT 改善 4.13%；
- 避免淘汰观测到 downstream chain impact 的 blocks。

但在 512-request 高压力环境下，原始 `chain_hybrid` 慢于 LRU。后续 `shared_chain1` 只保护跨请求共享的 chain：

- 平均 TTFT 改善 3.71%；
- P90 TTFT 退化 1.26%。

加入 hit24 tail guard 后，3-seed 聚合结果相对 LRU：

- 平均 TTFT 改善 0.67%；
- P90 TTFT 改善 0.35%；
- seed 1 仍发生退化。

当前解释：sharing/chain 可以作为复用预测信号，但单独的 GPU eviction queue 启发式不足以形成稳定贡献；short-term hotness 不适合作为当前课题的核心信号。

证据：[GPU eviction 3-seed 总结](cachedattention_online_3seed_summary_20260714.md)

### 4.2 不依赖 reset 的 external restore 路径已经打通

自然淘汰 sanity workload 使用一个更长且内容不同的 polluter 将目标请求从 HBM 挤出，没有调用 cache reset。结果为：

- 目标 prompt 为 528 tokens；
- 重访时仅有 1 个 GPU hit block；
- CPU lookup 命中并完成 32-block load；
- CPU 到 GPU 传输了 56 MiB；
- 原始 H2D 时间约 2.51 ms；
- cold 和 restore 请求输出一致；
- sanity report 状态为 `PASS`。

forced-reset 版本没有产生 external load，因此不作为 restore 有效性证据。后续实验均应使用自然 HBM eviction。

证据：[natural-eviction report](../results/external_restore_sanity/20260725_005350/t0_natural_eviction/report.json)

### 4.3 Selective admission 能避免一次性 polluter 污染 CPU

可控 workload 包含会再次访问的目标 A 和不会复用的 polluter P。5 次重复实验结果如下：

| arm | stores | P stores | P dropped | loads | D2H MiB | useful-store | final A median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| all-block-constrained | 146 | 113 | 0 | 0 | 255.50 | 0.00% | 22.90 |
| all-block-large | 146 | 113 | 0 | 33 | 255.50 | 22.60% | 17.81 |
| no-offload | 0 | 0 | 0 | 0 | 0.00 | 0.00% | 23.24 |
| selective-constrained | 33 | 0 | 113 | 33 | 57.75 | 100.00% | 17.89 |
| selective-minimal | 33 | 0 | 113 | 33 | 57.75 | 100.00% | 17.83 |

相对 all-block，selective admission：

- 将 stores 从 146 降至 33，减少 77.4%；
- 将 D2H 从 255.50 MiB 降至 57.75 MiB，减少 77.4%；
- useful-store ratio 从 22.60% 提高到 100%；
- 在 CPU 容量受限时仍保留 A 的 33-block restore，而 all-block 因 pollution 最终 load 为 0；
- 5/5 次实验均通过。

该结果证明了 selective admission 的机制价值，但 workload 是受控构造，尚不能替代真实负载验证。

证据：[selective admission 5-repeat summary](../results/external_restore_admission/20260729_235912/summary.md)

### 4.4 RESTORE 与 RECOMPUTE 存在成本交叉区间

在 Qwen3-0.6B、A100、低 GPU/PCIe 负载下，6 对重复实验得到：

| blocks | paired delta mean | 胜负 | 当前解释 |
|---:|---:|---:|---|
| 16 | -0.48 ms | restore 2 / recompute 4 | recompute 较优 |
| 24 | -0.93 ms | restore 2 / recompute 4 | recompute 较优 |
| 28 | -2.24 ms | restore 1 / recompute 5 | recompute 较优 |
| 30 | -0.90 ms | restore 2 / recompute 4 | recompute 略优但有噪声 |
| 31 | -0.78 ms | restore 3 / recompute 3 | 不确定区间 |
| 33 | +4.38 ms | restore 6 / recompute 0 | restore 稳定较优 |

其中 paired delta 定义为 `recompute latency - restore latency`，正数表示 restore 更快。

固定 33-block 的另一组 5-pair 实验中：

- `always_restore` 的 final A median 为 17.74 ms；
- `always_recompute` 的 final A median 为 22.36 ms；
- restore 快约 4.62 ms。

当前解释：promotion 决策确实有必要，但 30/31 blocks 附近噪声较大，且 28 到 33 blocks 的变化不是简单线性。最终策略需要分段成本模型、负载输入和 deadband，不能把“33 blocks”直接推广为固定阈值。

证据：

- [formal crossover 6-pair summary](../results/external_restore_promotion_length/formal_idle_crossover_6pair_20260801/summary.md)
- [30/31-block fine sweep](../results/external_restore_promotion_length/fine_idle_transition_unique_6pair_20260802/summary.md)
- [33-block restore/recompute 5-pair summary](../results/external_restore_admission/20260801_013440/summary.md)

### 4.5 固定容量的长度匹配闭环 workload 暴露大量无效 offload

在 512 requests、3 seeds、固定 1200 GPU blocks、1 GiB CPU cache 的 ShareGPT 多轮内容、BurstGPT 请求长度匹配闭环 workload 下：

| admission | requested stores | actual stores | CPU evictions | external loads |
|---|---:|---:|---:|---:|
| threshold 0 | 13490.0 | 7700.0 | 7115.0 | 0 |
| threshold 2 | 13488.3 | 3410.7 | 2825.7 | 0 |

Threshold 2 相对 threshold 0：

- actual stores 减少约 55.7%；
- CPU evictions 减少约 60.3%；
- 但仍写入约 3411 个最终没有 load 的 blocks。

这是一个重要的真实会话内容驱动的受控负结果：简单 reuse threshold 可以降低写入量，但尚不能识别真正有收益的 KV；在该资源与并发配置下 CPU tier 没有产生 external reuse。4.6 节的生命周期矩阵进一步表明，该结论不能直接推广到更强会话交错的配置。

目前仍需区分以下原因：

- 会复用的 blocks 一直留在 HBM；
- blocks 在复用前已从 CPU 淘汰；
- 写入 CPU 的主体是一次性 private blocks；
- store、lookup 和会话生命周期之间存在匹配问题；
- 会话 reuse gap 超过 CPU retention window。

证据：

- [fixed-GPU threshold0 3-seed summary](../results/trace_driven_factorial_cpu1g_fixedgpu1200_t0_512/combined_3seed/policy_compare_summary.md)
- [fixed-GPU threshold2 3-seed summary](../results/trace_driven_factorial_cpu1g_fixedgpu1200_t2_512/combined_3seed/policy_compare_summary.md)

### 4.6 生命周期矩阵定位了自然 external reuse 的出现条件

为区分 CPU 容量、HBM 压力和会话交错的影响，在相同 ShareGPT 内容、BurstGPT 长度匹配的闭环并发 workload 上运行了 128-request、单 seed 的首轮诊断矩阵。生命周期统计为 opt-in 聚合统计，`looked after store` 只计算成功写入 DRAM 后、后来真正进入 external maximal-prefix lookup 的唯一 blocks。

| GPU blocks | CPU cache | active sessions | unique stores | GPU evictions | CPU evictions | looked after store | unique loaded | never looked after store |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1200 | 1 GiB | 16 | 1172 | 69 | 588 | 0 | 0 | 1172 |
| 1200 | 4 GiB | 16 | 1174 | 71 | 0 | 0 | 0 | 1174 |
| 600 | 4 GiB | 16 | 1169 | 611 | 0 | 0 | 0 | 1169 |
| 600 | 4 GiB | 32 | 1209 | 1198 | 0 | 520 | 520 | 689 |

前三组依次排除了两个简单解释：把 CPU cache 扩到 4 GiB 并消除 CPU eviction 后仍为零 hit；将 HBM 从 1200 blocks 减半、使 GPU eviction 从 71 增至 611 后仍为零 hit。因此，零 restore 不能只归因于 CPU 容量不足或 HBM 压力不足。

当活跃会话从 16 增至 32 时：

- 35 个 lookup 请求命中 external cache；
- 完成 558 block-load，其中涉及 520 个唯一 blocks；
- 唯一 block useful-store ratio 为 `520 / 1209 = 43.0%`；
- 仍有 689 个已写入 blocks 在本次运行结束前从未进入 external lookup，占 57.0%；
- 平均 GPU prefix hit 从 8.96 blocks 降到 2.27 blocks，说明更强会话交错把一部分可复用前缀推出了 HBM，使其成为 external restore 候选。

这个结果首次在不 reset、不人为重复单个 prompt 的真实多轮内容受控回放中同时观察到 GPU eviction、CPU lookup hit 和 CPU load。它证明自然 external reuse 的确存在，也证明其收益机会强烈依赖并发会话数和 reuse gap。all-block admission 在同一次运行中同时保存了有用和无用 blocks，正好构成 selective admission 与自适应负收益保护的研究对象。

目前该结果仍是单 seed、单模型的机制证据。下一步需要在相同 600-block/4-GiB/32-session 配置上比较 no-offload、all-block 和 selective admission，并进行多 seed 重复；在此之前不能宣称端到端性能收益或可推广性。

证据：

- [1200 blocks / 1 GiB / 16 sessions](../results/trace_driven_lifecycle_cpu1g_fixedgpu1200_t0_128/20260805_012648/policy_compare_summary.md)
- [1200 blocks / 4 GiB / 16 sessions](../results/trace_driven_lifecycle_cpu4g_fixedgpu1200_t0_128/20260805_012802/policy_compare_summary.md)
- [600 blocks / 4 GiB / 16 sessions](../results/trace_driven_lifecycle_cpu4g_fixedgpu600_t0_128/20260805_012911/policy_compare_summary.md)
- [600 blocks / 4 GiB / 32 sessions](../results/trace_driven_lifecycle_cpu4g_fixedgpu600_active32_t0_128/20260805_013653/policy_compare_summary.md)

### 4.7 自然负载 3-seed 对照支持“负收益保护”而非无条件 offload

固定 600 GPU blocks、4 GiB CPU cache 和 32 个活跃会话后，对 no-offload、all-block 和 `store_threshold=2` 进行了 3-seed 对照。三种策略在两张 A100 间轮换；每组每个 seed 运行 128 requests。

| arm | avg TTFT | P90 TTFT | actual stores | completed loads | unique useful-store ratio |
|---|---:|---:|---:|---:|---:|
| no-offload | 23.23 ms | 34.13 ms | 0 | 0 | n/a |
| all-block | 23.97 ms | 34.77 ms | 1205.0 | 559.0 | 42.5% |
| selective threshold 2 | 25.64 ms | 36.96 ms | 551.7 | 58.3 | 10.6% |

机制结果跨 seed 稳定：all-block 每个 seed 都完成 558--561 block loads，并保留约 498--520 个后续真正加载的唯一 blocks；同时平均仍有 693 个已存 blocks 从未进入后续 external lookup。该真实会话内容驱动的受控负载中同时存在值得 offload 和不值得 offload 的 KV。

简单 threshold 2 并不是有效的 selective policy。它使 actual stores 相对 all-block 减少 54.2%，却使 completed loads 减少 89.6%，并且三个 seed 的平均 TTFT 均慢于 no-offload。其原因是该信号必须先观察两次 lookup 才准入，因而会牺牲多轮会话中较早且最有价值的复用机会。

性能结果也暴露了 offload 的负收益风险：all-block 在 seed 0 和 2 快于 no-offload，在 seed 1 明显更慢；3-seed 聚合平均 TTFT 退化 3.17%，P90 退化 1.88%。这说明“restore 比 recompute 快”只是事件级条件，不能推出包含 D2H、H2D、排队和缓存动态后的系统级净收益必然为正。

因此，当前结果不支持无条件 all-block offload，也不支持继续盲目 sweep reuse threshold；它直接支持课题的核心命题：使用 session/turn、sharing/chain 和实测成本识别正收益 offload，并在窗口净收益为负时退化到 no-offload。

证据：[自然负载 3-seed policy comparison](../results/trace_driven_lifecycle_compare_600_active32_3seed_20260805/summary.md)

面向开题报告的完整问题链、边界说明与可直接使用表述见：[自然 external restore 动机实验](natural_external_restore_motivation_20260805.md)。

### 4.8 自然负载 promotion 分解暴露独立运行的计时噪声

在相同 600-block/4-GiB/32-session 配置下新增 all-block + `always_recompute`，即继续执行 CPU store 和 lookup，但 CPU 命中后不执行 H2D load：

| arm | avg TTFT | P90 TTFT | unique CPU hit after store | completed loads |
|---|---:|---:|---:|---:|
| no-offload | 23.23 ms | 34.13 ms | 0 | 0 |
| all-block + restore | 23.97 ms | 34.77 ms | 511.7 | 559.0 |
| all-block + recompute | 25.16 ms | 34.86 ms | 541.3 | 0 |

聚合上 restore 比 recompute 快 1.19 ms，但逐 seed 仍然变号。recompute 的 seed 0 甚至快于 no-offload，尽管它执行了额外 CPU store/lookup；这个现象不能作为 store 收益，而说明不同 server run 之间的调度和批处理噪声不足以支持用 TTFT 均值相减来精确分解 D2H/H2D 成本。

当前能够保留的结论是：自然负载中存在大量 CPU demand，restore/recompute 选择可能影响性能；但最终成本模型必须依赖事件级 timer、相同 trace 的成对回放和负载分桶，不能仅依赖独立运行均值。

证据：[all-block recompute 3-seed summary](../results/trace_driven_lifecycle_compare_600_active32_3seed_20260805/all_block_recompute.md)

### 4.9 已完成未经 BurstGPT 长度匹配的 direct ShareGPT 首轮 3-benchmark-seed 对照

已从 SHA256 校验通过的原始 ShareGPT 数据中使用固定 seed 生成 128-conversation direct workload，不使用 BurstGPT 长度或到达信息。direct 样本包含 3--6 个用户轮次，final prompt tokens 均值为 1237、P90 为 2190、最大为 3935。

固定 1000 GPU blocks、4 GiB CPU cache 和 32 active conversations 后，完成
no-offload、all-block + restore、all-block + recompute 的 benchmark runtime
seeds 0--2 对照。三次均使用同一份 dataset seed0 输入；它们不是三套独立
ShareGPT 抽样：

| arm | avg TTFT | P90 TTFT | completed loads | unique useful-store ratio |
|---|---:|---:|---:|---:|
| no-offload | 28.53 ms | 46.15 ms | 0 | n/a |
| all-block + restore | 30.07 ms | 46.33 ms | 1467.7 | 59.62% |
| all-block + recompute | 32.29 ms | 51.77 ms | 0 | n/a（禁止 load） |

restore 在三个 seed 中都快于 recompute，聚合平均 TTFT 低 2.22 ms；但 all-block + restore 相对 no-offload 的平均 TTFT 仍退化 5.41%。每个 seed 均出现 42--44 次真实 load request，平均每次约 34 blocks，说明 external restore 已稳定迁移到未经原有长度匹配的真实多轮内容样本。与此同时，约 40.38% 的唯一 CPU stores 从未形成后续 external demand，继续支持 selective admission 与负收益保护。

当前仍使用 `REQUEST_RATE=0` 的闭环并发，不能据此宣称真实到达、跨样本或 Agent/RAG 泛化。完整数据来源、抽样偏差、容量校准和结果边界见：[Direct ShareGPT workload 迁移记录](direct_sharegpt_workload_migration_20260806.md)；三策略结果见：[Direct ShareGPT 3-seed 对照](../results/direct_sharegpt_compare_1000_active32_3seed_20260806/summary.md)。

## 5. 当前结论的证据等级

| 结论 | 状态 | 证据等级 |
|---|---|---|
| vLLM external restore 数据路径可用 | 已验证 | 高 |
| 自然 HBM eviction 可以触发 CPU restore | 已验证 | 高 |
| 可控 workload 中 selective admission 能避免 pollution | 已验证 | 较高 |
| 当前空闲环境存在 restore/recompute crossover | 已验证 | 中高 |
| sharing/chain 可作为复用价值信号 | 部分验证 | 中 |
| short-term hotness 是稳定主信号 | 未获得支持 | 低 |
| 当前真实会话内容受控回放中 all-block 有大量无效写入 | 已验证 | 较高 |
| 多会话 trace 在强交错下能自然产生 external restore | 已验证 | 较高（3 seeds） |
| all-block 在同一自然负载中同时写入有用与无用 blocks | 已验证 | 较高（3 seeds） |
| direct ShareGPT 在 1000 blocks 下稳定产生 external restore | 已验证 | 高（3 benchmark seeds + 3 dataset seeds） |
| direct ShareGPT 中 restore 优于 fixed recompute | 已验证 | 高（3 dataset seeds，109 对事件） |
| 简单 lookup-frequency threshold 是有效 selective policy | 未获得支持 | 反例稳定（3 seeds） |
| all-block offload 的系统级净收益稳定为正 | 未获得支持 | 长度匹配负载跨 seed 变号；direct 负载聚合退化 |
| promotion 存在事件级 Oracle gap | 已在当前配置重复验证 | 3 dataset seeds、109 对事件，Oracle 比 always-restore 低 5.53%--9.11% |
| 在线 cost-aware promotion 优于固定策略 | 尚未验证 | 需要跨 seed 和负载上下文 |
| negative-utility guard 能在真实负载安全退化 | 尚未验证 | 无 |
| 策略可以跨模型、硬件和负载泛化 | 尚未验证 | 无 |

## 6. 当前系统实现状态

已实现：

- GPU eviction value-policy 实验框架；
- `store_threshold` selective admission；
- external lookup/store/load 统计；
- opt-in 的 block 生命周期聚合统计，包括 store、CPU eviction、后续 lookup/hit 和 load；
- opt-in 的事件级成本账本，包括 store、lookup、DMA、submit-to-complete、restore wait 和 recompute wait；
- `always_restore` 和 `always_recompute` promotion 基线；
- natural-eviction、admission 和 promotion-length benchmark；
- 多次重复实验的自动汇总脚本；
- restore/recompute 严格事件配对、成本分桶和离线 Oracle 汇总。

尚未实现：

- 在线 `cost_aware` promotion；
- promotion 决策时的 batch、GPU/PCIe 队列等负载上下文；
- restore/recompute 成本模型的在线更新；
- useful-store 和净收益滑动窗口；
- negative-utility circuit breaker；
- 主流 Agent/RAG workload 评测。

相关实现与脚本：

- [OffloadingConnector](../vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py)
- [CPU offload reuse manager](../vllm/v1/kv_offload/reuse_manager.py)
- [CPU offload lifecycle tests](../tests/v1/kv_offload/test_cpu_manager.py)
- [value-policy/lifecycle summarizer](../scripts/summarize_value_policy_runs.py)
- [natural restore benchmark](../benchmarks/external_restore_sanity.py)
- [admission benchmark](../benchmarks/external_restore_admission.py)
- [admission runner](../scripts/run_external_restore_admission.sh)
- [promotion-length runner](../scripts/run_external_restore_promotion_length_sweep.sh)
- [admission summarizer](../scripts/summarize_external_restore_admission.py)
- [promotion-length summarizer](../scripts/summarize_external_restore_promotion_length.py)
- [事件级成本汇总](../scripts/summarize_offload_cost_events.py)
- [restore/recompute 配对与 Oracle](../scripts/compare_offload_promotion_costs.py)
- [ShareGPT 事件级成本账本初步结果](offload_cost_ledger_sharegpt_20260809.md)

## 7. 下一阶段技术路线

### 7.1 已完成：解释 external hit 对资源与会话交错的敏感性

已通过 opt-in 生命周期聚合统计建立第一版事件关系：

```text
block hash
  -> CPU store time
  -> GPU eviction time
  -> CPU eviction time (if any)
  -> next lookup time (if any)
  -> GPU hit / CPU hit / total miss
```

已完成 128 requests、单 seed 的小规模资源矩阵：

| GPU blocks | CPU cache | active conversations | 目的 |
|---:|---:|---:|---|
| 1200 | 1 GiB | 16 | 复现当前配置 |
| 1200 | larger | 16 | 判断 CPU eviction 是否是主因 |
| 600--800 | larger | 16 | 增加自然 HBM pressure |
| 600--800 | larger | 32 | 增加 session reuse gap 和竞争 |

保持真实会话内容、到达顺序和会话内因果顺序，不调用 reset，不人为重复 prompt。

首轮结论是：CPU 容量和单纯 HBM pressure 都不足以产生 external hit；32 个活跃会话带来的更强交错和更长 reuse gap 才使可复用前缀离开 HBM并命中 DRAM。固定该配置的 3-seed 对照也已完成，证明 external restore 和有用/无用 stores 的混合稳定存在，但 all-block 净收益跨 seed 变号，threshold 2 则稳定错杀有用 restore。

### 7.2 进行中：有 external hit 后建立真实事件 Oracle

对每次 CPU hit 记录：

- candidate blocks/tokens；
- restore queue、DMA 和端到端时间；
- batch size、GPU load 和 PCIe load；
- 对应的 recompute time；
- `always_restore`、`always_recompute` 和 Oracle 的事件级 regret。

Oracle 对每个事件使用：

```text
if measured_restore_cost + margin < measured_recompute_cost:
    RESTORE
else:
    RECOMPUTE
```

三数据 seed 第一版已完成：六次 direct ShareGPT 在线运行共产生 125 个
restore 和 123 个 recompute 事件，严格配对 109 个。pooled restore/recompute
平均等待分别为 9.726 和 14.066 ms；85 次 restore 更快，24 次 recompute
更快。逐事件离线 Oracle 为 9.060 ms，比 always-restore 再低 0.666 ms/事件
（6.85%）。各 seed Oracle gap 为 5.53%、6.19%、9.11%，方向一致；但 pooled
block-only 最佳阈值仅节省约 0.051 ms/事件，说明下一步必须补充负载状态，
不能只用长度决策。完整结果见
[ShareGPT 事件级成本账本初步结果](offload_cost_ledger_sharegpt_20260809.md)。

### 7.3 满足门槛后实现在线 cost-aware controller

第一版使用查表或分段线性模型，不使用机器学习：

- 按 candidate blocks、batch size 和 GPU/PCIe load 分桶；
- 使用 EWMA 更新成本；
- 在成本接近时设置 deadband；
- 滑动窗口净收益持续为负时降低 admission rate；
- 必要时自动关闭 CPU offload。

## 8. 最终评测设计

对照组：

1. no-offload；
2. all-block + always-restore；
3. selective + always-restore；
4. selective + always-recompute；
5. selective + cost-aware；
6. offline Oracle。

负载：

- ShareGPT 直接多轮回放，以及当前 BurstGPT 长度匹配闭环负载；
- 工具调用或 Agent workload；
- 长上下文/RAG workload；
- 单轮低复用负对照；
- 混合在线请求。

指标：

- P50/P95/P99 TTFT；
- E2E latency、ITL、throughput 和 SLO attainment；
- external hit requests/blocks；
- restore bytes、queue time、DMA time 和 stall time；
- recomputed tokens；
- useful-store ratio；
- 无效 D2H/H2D bytes；
- CPU eviction/churn；
- 在线决策相对 Oracle 的 regret；
- 控制器自身时间和内存开销。

## 9. Go/No-Go 标准

继续 promotion 方向需要满足：

- 至少一个多轮或 Agent workload 在合理资源配置下自然产生稳定 external hits（当前 direct ShareGPT 已在 3 个独立 dataset seeds 满足）；
- Oracle 相对 `always_restore` 或 no-offload 存在明确收益空间；
- 在线策略取得至少一半 Oracle 收益；
- 控制器开销低于 2%；
- 低复用 workload 相对 no-offload 的退化控制在约 3% 内。

如果只有人工 synthetic workload 才产生 restore，而主流多轮/Agent workload 几乎没有 external hit，则停止将 promotion optimization 作为主要贡献，保留生命周期观测、selective admission 和 negative-utility guard 作为系统核心。

## 10. 开题前十天计划

| 时间 | 任务 | 交付物 |
|---|---|---|
| 第 1--2 天 | 固定课题主线并整理已有证据 | 本文档、三张核心结果图、系统图 |
| 第 3--4 天 | 补齐 block 生命周期观测（已完成聚合版） | 生命周期统计与测试 |
| 第 5 天 | 运行 128-request 小资源矩阵（已完成首轮） | 找到自然 external-hit 配置 |
| 第 6--7 天 | 撰写开题报告 | 背景、现状、问题、方案、可行性、计划 |
| 第 8 天 | 制作答辩 PPT | 课题演进、系统设计、初步结果 |
| 第 9 天 | 模拟答辩 | 风险问题与回答清单 |
| 第 10 天 | 修订与备份 | 最终报告、PPT、原始结果索引 |

开题前不新增 progressive restore、复杂调度、多机、SSD 层或新的 GPU eviction 权重 sweep。

## 11. 开题阶段可使用的核心表述

> 现有 KV Cache 管理通常分别决定 GPU 淘汰、外部存储准入和缓存恢复，容易在低复用或高传输成本下产生大量无效 offload。本课题面向多轮大模型推理，构建收益感知的分层 KV Cache 管理系统，通过生命周期观测、选择性准入、恢复/重算决策和负收益保护，在线决定 KV block 的 KEEP、OFFLOAD、DROP、RESTORE 与 RECOMPUTE，并在不适合外部缓存的负载下安全退化。
