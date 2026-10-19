# 自然多轮负载中的 KV Offload 机会与负收益：动机实验

日期：2026-08-05

## 1. 实验目的

本实验不是为了证明现有 offload 策略已经优于 vLLM 默认策略，而是回答以下三个动机问题：

1. 非人工 reset、非固定 prompt 重放的多轮 trace 中，是否真的存在从 CPU 恢复 KV 的需求？
2. 如果存在，all-block admission 是否会同时产生有效恢复和无效写入？
3. 简单的历史 lookup-frequency threshold 是否足以识别有价值的 blocks？

这些问题决定了是否值得研究 `OFFLOAD/DROP`、`RESTORE/RECOMPUTE` 和 negative-utility guard。如果自然负载中始终没有 external restore，则恢复决策缺少研究对象；如果 all-block 已经稳定优于 no-offload，则负收益保护的必要性也会减弱。

## 2. 实验设置

- 模型：Qwen3-0.6B；
- GPU：两张 A100，不同策略在两张卡之间轮换；
- 会话内容：ShareGPT 多轮对话；
- 会话选择：按 BurstGPT request token trace 的长度，为每项选择 final prompt token 数最接近的 ShareGPT 会话；
- 到达与交错：`REQUEST_RATE=0`、4 clients、固定 active sessions 的闭环并发；不回放 BurstGPT 到达时间；
- 每个会话内部保持因果顺序；
- GPU KV 容量：600 或 1200 blocks；
- CPU KV 容量：1 GiB 或 4 GiB；
- 每次运行：128 requests；
- GPU eviction policy：LRU；
- 不调用 cache reset；
- 不人为重复某个目标 prompt。

生命周期统计只输出聚合计数，不输出 token、请求内容或序列化 block hash。`looked after store` 表示 block 成功写入 CPU 后，后来真正进入 external maximal-prefix lookup；`unique loaded` 表示至少完成过一次 CPU-to-GPU load 的唯一 blocks。

## 3. 动机观察一：GPU eviction 不自动产生 external restore

首轮资源与并发矩阵如下：

| GPU blocks | CPU cache | active sessions | unique stores | GPU evictions | CPU evictions | looked after store | unique loaded |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1200 | 1 GiB | 16 | 1172 | 69 | 588 | 0 | 0 |
| 1200 | 4 GiB | 16 | 1174 | 71 | 0 | 0 | 0 |
| 600 | 4 GiB | 16 | 1169 | 611 | 0 | 0 | 0 |
| 600 | 4 GiB | 32 | 1209 | 1198 | 0 | 520 | 520 |

该矩阵给出三个结论：

1. 将 CPU cache 从 1 GiB 扩大到 4 GiB、消除 CPU eviction 后，16-session 配置仍然没有 external hit，因此零 restore 不能只归因于 CPU 容量不足。
2. 将 GPU KV 容量从 1200 blocks 降到 600 blocks，使 GPU eviction 从约 71 增加到 611 后仍然没有 restore，因此“产生更多 GPU eviction”本身也不充分。
3. 在相同 600-block GPU 和 4-GiB CPU 容量下，将活跃会话从 16 增至 32 后出现 520 个唯一 loaded blocks。更强的会话交错和更长 reuse gap 将未来还会复用的会话前缀推出 HBM，产生了自然 external reuse。

因此，CPU KV tier 的有效流量具有明显的 workload-state dependence：它取决于并发会话数量、复用间隔、HBM retention window 和被淘汰内容是否会再次访问，而不仅是总请求数或 GPU eviction 数。

## 4. 动机观察二：all-block 同时保存有用和无用 KV

固定 600 GPU blocks、4 GiB CPU cache、32 个活跃会话后，all-block admission 的 3-seed 生命周期结果为：

| 指标 | 3-seed 均值或范围 |
|---|---:|
| actual unique stores | 1205.0 |
| completed load blocks | 559.0 |
| unique loaded blocks | 511.7 |
| never looked after store | 693.3 |
| unique useful-store ratio | 42.5% |
| 每个 seed 的 completed loads | 558--561 |

这说明同一个自然多轮负载中同时存在两类 block：

```text
后来离开 HBM 并再次访问
    -> CPU store 提供了跨层复用机会

在观测窗口内从未进入后续 external lookup
    -> CPU store 产生写入和容量占用，但没有 restore 价值
```

因此，`CPU load > 0` 只能证明 external reuse 存在，不能证明 all-block admission 合理。约 42.5% 的唯一 stores 具有事后复用价值，但另外约 57.5% 在本次观测窗口中没有产生 external demand。

## 5. 动机观察三：CPU hit 不等于系统净收益

相同配置下的 3-seed 端到端结果为：

| arm | avg TTFT | P90 TTFT | unique CPU hit after store | completed loads | unique useful-store ratio |
|---|---:|---:|---:|---:|---:|
| no-offload | 23.23 ms | 34.13 ms | 0 | 0 | n/a |
| all-block + restore | 23.97 ms | 34.77 ms | 511.7 | 559.0 | 42.5% |
| all-block + recompute | 25.16 ms | 34.86 ms | 541.3 | 0 | n/a（禁止 load） |
| selective threshold 2 | 25.64 ms | 36.96 ms | 58.3 | 58.3 | 10.6% |

all-block 在 seed 0 和 2 的平均 TTFT 快于 no-offload，在 seed 1 明显更慢；3-seed 聚合平均 TTFT 退化 3.17%，P90 退化 1.88%。因此，“发生了 restore”不能直接推出系统性能更好。

新增的 all-block + always-recompute 保持 CPU store 和 lookup，但命中后不执行 H2D load。其 3-seed 平均 TTFT 为 25.16 ms，比 always-restore 高 1.19 ms；然而逐 seed 的 restore/recompute 胜负仍然变化。特别是 recompute 的 seed 0 甚至快于没有额外 CPU 工作的 no-offload，这不能解释为 store 带来收益，而是说明独立 server run 的调度、批处理和计时噪声足以污染简单减法。

因此，该对照支持“promotion 选择可能有价值”，但尚不能精确分解 `C_store` 和 `C_restore`。下一步应在同一运行中记录事件级 D2H/H2D/queue/stall 时间，或使用成对回放，而不是仅比较不同运行的平均 TTFT。

一个 store 决策的事后净收益应至少包含：

```text
V_store(b) = I_reuse(b) * (C_recompute(b) - C_restore(b))
             - C_store(b)
             - C_blocking(b)
             - C_capacity(b)
```

其中：

- `C_store` 是 GPU-to-CPU 写入成本；
- `C_restore` 包括 lookup、排队、DMA 和同步成本；
- `C_recompute` 是不恢复该 KV 时的 prefill 成本；
- `C_blocking` 表示传输和调度对其他请求的影响；
- `C_capacity` 表示占用有限 CPU cache、挤出其他 block 的机会成本。

此前可控实验还观察到 restore/recompute crossover：16--30 blocks 区间内 recompute 经常不差于 restore，33 blocks 时 restore 才稳定占优。当前自然负载 seed 0 平均每个命中请求加载约 16 blocks，因此其中一部分 CPU hit 可能具有复用意义，却没有正的时延净收益。这是根据现有成本曲线做出的推断，仍需事件级成本实验确认。

### 5.1 Direct ShareGPT 对照增强了 promotion 与 admission 动机

为减少 BurstGPT 长度匹配过程对结论的影响，后续从校验通过的原始 ShareGPT 文件固定抽取 128 个 direct conversations，不使用 BurstGPT 长度或到达信息。在 1000 GPU blocks、4 GiB CPU cache、32 active conversations 下完成 seeds 0--2 对照：

| arm | avg TTFT | P90 TTFT | completed loads | unique useful-store ratio |
|---|---:|---:|---:|---:|
| no-offload | 28.53 ms | 46.15 ms | 0 | n/a |
| all-block + restore | 30.07 ms | 46.33 ms | 1467.7 | 59.62% |
| all-block + recompute | 32.29 ms | 51.77 ms | 0 | n/a（禁止 load） |

direct workload 中每个 seed 有 42--44 次 load request，平均每次约 34 blocks。restore 在三个 seed 中均快于 recompute，聚合平均 TTFT 低 2.22 ms，说明较长 candidate chain 上的 restore 具有更稳定的 promotion 价值。

但 all-block + restore 仍比 no-offload 平均慢 1.54 ms，且 40.38% 的唯一 stores 从未形成后续 external demand。因此，新结果并没有证明 all-block 合理，而是把问题进一步收敛为：保留有正复用价值的 store、丢弃无后续 demand 的 store，并用事件级成本决定 CPU hit 后是否恢复。

证据：[Direct ShareGPT 3-seed 对照](../results/direct_sharegpt_compare_1000_active32_3seed_20260806/summary.md)。

## 6. 动机观察四：简单 frequency threshold 会错杀早期复用

`store_threshold=2` 只在 block 的 lookup observation count 达到 2 后允许写入 CPU。相对 all-block，它：

- 将 actual stores 减少 54.2%；
- 将 completed loads 减少 89.6%；
- 将 unique useful-store ratio 从 42.5% 降到 10.6%；
- 三个 seed 的平均 TTFT 均慢于 no-offload；
- 3-seed 聚合平均 TTFT 相对 no-offload 退化 10.38%。

该策略必须先牺牲较早的复用机会，才能积累足够的 lookup count。这与多轮会话不匹配：一个会话的下一轮访问可能就是最有价值的第一次 external reuse。结果表明，减少 store 数量不等于实现了有效 selective admission。

## 7. 该实验支持的研究问题

四项观察共同形成如下矛盾：

```text
没有足够会话交错
    -> CPU tier 几乎没有有效流量，offload 可能只有成本

出现自然 external reuse
    -> 部分 blocks 值得保留，但 all-block 仍包含大量无效写入

仅按历史 lookup frequency 过滤
    -> 减少写入，却错杀下一轮会话复用

CPU lookup 命中
    -> restore 可能节省计算，也可能比 recompute 更慢
```

由此得到三个具体研究问题：

1. **Selective Admission**：如何结合 session/turn、sharing、chain position 和 reuse gap，在 GPU eviction 时选择 `OFFLOAD` 或 `DROP`？
2. **Cost-Aware Promotion**：CPU 命中后，如何结合候选 block 数、GPU load 和 PCIe load 选择 `RESTORE` 或 `RECOMPUTE`？
3. **Negative-Utility Guard**：当 useful-store ratio 和窗口净收益持续下降时，如何降低 admission rate，并安全退化到 no-offload？

这使研究目标不再是“提高 CPU cache hit rate”，而是：

> 只在跨层 KV 复用产生正系统收益时启用 offload，并避免无效写入、错误恢复和负收益运行区间。

## 8. 该实验没有证明什么

为了避免扩大结论，当前实验尚未证明：

- all-block offload 普遍优于 no-offload；
- 所有 loaded blocks 都应该恢复而不是重新计算；
- 当前 42.5% useful-store ratio 可以推广到其他模型、Agent 或 RAG workload；
- direct ShareGPT 中的 59.6% useful-store ratio 可以推广到其他模型、Agent、RAG 或生产 arrival；
- session/turn、sharing 或 chain 信号已经能够准确预测未来复用；
- proposed online controller 已经取得性能收益。

当前结果适合作为开题中的动机与可行性证据，不应作为最终系统优越性结论。

## 9. 开题报告可直接使用的表述

> 初步实验表明，CPU KV Cache 的有效性对并发会话交错高度敏感：在较低活跃会话数下，即使发生大量 GPU 淘汰，写入 CPU 的 KV 也可能从未被恢复；提高会话交错后，真实多轮 trace 能稳定产生 external restore，但 all-block offload 在长度匹配 workload 中约 57.5%、在 direct ShareGPT 中约 40.4% 的唯一写入没有形成后续 external demand。direct ShareGPT 上 restore 又稳定快于 recompute，却仍未超过 no-offload。这说明分层 KV Cache 需要联合考虑未来复用价值、传输成本与重计算成本，并在收益为负时动态退化。

## 10. 结果索引

- [3-seed policy comparison](../results/trace_driven_lifecycle_compare_600_active32_3seed_20260805/summary.md)
- [Direct ShareGPT 3-seed validation](../results/direct_sharegpt_compare_1000_active32_3seed_20260806/summary.md)
- [all-block aggregate](../results/trace_driven_lifecycle_compare_600_active32_3seed_20260805/all_block.md)
- [all-block recompute aggregate](../results/trace_driven_lifecycle_compare_600_active32_3seed_20260805/all_block_recompute.md)
- [no-offload aggregate](../results/trace_driven_lifecycle_compare_600_active32_3seed_20260805/no_offload.md)
- [threshold-2 aggregate](../results/trace_driven_lifecycle_compare_600_active32_3seed_20260805/selective_t2.md)
- [完整研究状态](hierarchical_kv_research_status_20260805.md)
