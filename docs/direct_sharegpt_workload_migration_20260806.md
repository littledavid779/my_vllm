# Direct ShareGPT Workload 迁移记录

日期：2026-08-06

## 1. 迁移动机

此前使用的 128-conversation workload 保留了真实 ShareGPT 多轮内容，但会根据 BurstGPT request token trace 的长度为每个目标选择 final prompt tokens 最接近的 ShareGPT 会话，并以 `REQUEST_RATE=0` 做闭环并发。它适合机制归因和压力控制，但不能称为 BurstGPT 真实到达 trace。

本轮迁移新增 direct ShareGPT workload：从原始 ShareGPT 文件使用固定 seed 抽样，不使用 BurstGPT 长度或到达信息。该负载用于验证 external reuse 和负收益现象能否离开原有长度匹配过程继续存在。

## 2. 数据来源与完整性

- source: `philschmid/sharegpt-raw`；
- source file: `sharegpt_20230401_clean_lang_split.json`；
- local path: `datasets/authoritative/sharegpt_20230401_clean_lang_split.json`；
- remote file size: 950 MB，local displayed size 907 MiB；
- SHA256: `0b54febbda102b519dd7a8bbe3765d2d73d31826b40faa1981aab746bb119417`；
- local SHA256 与公开源记录一致。

公开来源：<https://huggingface.co/datasets/philschmid/sharegpt-raw/blob/main/sharegpt_20230401_clean_lang_split.json>

## 3. 可复现生成过程

第一步使用仓库已有转换器生成固定候选集合：

```bash
./.venv/bin/python benchmarks/multi_turn/convert_sharegpt_to_openai.py \
  datasets/authoritative/sharegpt_20230401_clean_lang_split.json \
  results/direct_sharegpt_multiturn_20260806/sharegpt_direct_seed0_candidates512.json \
  --seed 0 \
  --max-items 512 \
  --min-turns 6 \
  --max-turns 12 \
  --model /public/model_zoo/Qwen3-0.6B
```

需要透明说明转换器现有语义：

- `min-turns` 和 `max-turns` 计算 message 数而不是 user-turn 数；
- 保留的 message 必须至少包含一个非 ASCII 字符；
- 126032 个原始 items 合并为 61653 个唯一 conversation IDs；
- 2680 个 conversations 通过交替角色、轮数和内容过滤；
- 之后使用 seed 0 随机抽取 512 个候选。

第二步使用 Qwen3 chat template 计算每个用户轮次的真实 prompt tokens，要求：

```text
max prompt tokens <= 4064
reserved output tokens = 32
max model length = 4096
```

512 个候选中 501 个满足长度限制，11 个被排除。最终按照确定性候选顺序选择前 128 个合法 conversations。没有使用 BurstGPT 数据，也没有人为重复目标 prompt。

生成文件：

- [direct 128-conversation workload](../results/direct_sharegpt_multiturn_20260806/sharegpt_direct_seed0_128.json)
- [完整生成元数据](../results/direct_sharegpt_multiturn_20260806/sharegpt_direct_seed0_128.meta.json)
- [512-conversation candidate set](../results/direct_sharegpt_multiturn_20260806/sharegpt_direct_seed0_candidates512.json)

## 4. 与长度匹配 workload 的分布差异

| workload | mean final prompt | P50 | P90 | P99 | max | mean user turns |
|---|---:|---:|---:|---:|---:|---:|
| BurstGPT-length-matched ShareGPT | 709 | 530 | 1479 | 2817 | 2846 | 4.68 |
| direct ShareGPT | 1237 | 1164 | 2190 | 3456 | 3935 | 5.15 |

direct ShareGPT 样本整体更长、轮次略多。因此两个 workload 的绝对 TTFT、GPU eviction 或 external load 数不能直接横向比较。后续只在同一个 direct workload 内比较策略，并将 GPU blocks 视为明确的实验压力参数。

## 5. GPU 容量校准

先使用 all-block + always-restore 做 seed-0 smoke：

| GPU blocks | avg TTFT | GPU evictions | GPU zero-hit ratio | actual stores | completed loads | unique loaded | unique useful-store ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 600 | 28.94 ms | 3299 | 85.94% | 2073 | 1809 | 1241 | 59.9% |
| 1000 | 31.92 ms | 2502 | 65.62% | 2052 | 1425 | 1216 | 59.3% |

600 blocks 产生很强的 HBM 压力，但将容量提高到 1000 blocks 后仍有 1425 个自然 load blocks 和 1216 个唯一 loaded blocks。因此 direct workload 中的 external reuse 不只存在于 600-block 极端压力点。后续首轮策略对照固定使用 1000 blocks，不根据策略结果继续调容量。

这两次独立运行的 TTFT 不能用于判断 600 和 1000 blocks 的性能关系，因为运行时调度噪声和 cache dynamics 同时变化；该表仅用于校准机制压力。

## 6. 1000-block 三 Benchmark Seed 策略对照（固定 Dataset Seed 0）

需要补充说明：本节三次运行都使用
`sharegpt_direct_seed0_128.json`，变化的是 benchmark runtime seed，而不是
ShareGPT converter seed。因此本节证明同一固定 workload 下的运行重复性；
三套独立 dataset seed 的验证见
[事件级成本账本第 8 节](offload_cost_ledger_sharegpt_20260809.md#8-三个独立数据-seed-的复验)。

配置：Qwen3-0.6B、4 GiB CPU cache、32 active conversations、128 requests、LRU。

| arm | avg TTFT | P90 TTFT | CPU lookup-hit blocks | completed loads | unique CPU demand | unique useful-store ratio |
|---|---:|---:|---:|---:|---:|---:|
| no-offload | 28.53 ms | 46.15 ms | 0 | 0 | 0 | n/a |
| all-block + restore | 30.07 ms | 46.33 ms | 1467.7 | 1467.7 | 1229.7 | 59.6% |
| all-block + recompute | 32.29 ms | 51.77 ms | 1514.7 | 0 | 1251.7 | n/a（load 被禁止） |

三 benchmark seed 对照证明：

- 未经过 BurstGPT 长度匹配的 direct ShareGPT 在每个 seed 都产生 42--44 次 restore，平均每次约 34 blocks；
- always-restore 在三个 seed 中都快于 always-recompute，聚合平均 TTFT 低 2.22 ms；
- all-block + restore 的聚合平均 TTFT 仍比 no-offload 高 5.41%，说明 restore 的局部收益没有自动覆盖完整 offload 路径的系统成本；
- 约 59.6% 的 unique stores 后来被加载，约 40.4% 从未形成 external demand，selective admission 因而存在明确优化对象。

该对照尚未证明：

- 每一个 CPU hit 都应该 restore；
- 59.6% useful-store ratio 可以推广到其他模型、并发度或 Agent/RAG workload；
- 当前独立运行的 TTFT 差能够精确分解 store、queue、DMA、stall 和 recompute 成本；
- 当前结果可以代表生产 open-loop arrival。

三组仍是独立 server runs，且只有三个 benchmark seeds。性能方向可以作为动机证据，但最终成本模型仍需事件级 timer 和成对回放。

## 7. 当前可信度与下一步

当前证据等级：

| 结论 | 状态 |
|---|---|
| direct ShareGPT 数据来源与生成可复现 | 已验证 |
| direct ShareGPT 存在自然 external reuse | 已在 3 benchmark seeds 和 3 dataset seeds 验证 |
| 1000 blocks 下 external reuse 仍存在 | 已在 3 benchmark seeds 和 3 dataset seeds 验证 |
| restore 相对 always-recompute 的方向 | 三个 seed 一致，仍需事件级归因 |
| all-block 相对 no-offload 的稳定净收益 | 未获得支持 |
| 真实到达过程 | 不具备，当前仍为闭环并发 |
| Agent/RAG 泛化 | 未验证 |

下一步不再扩大容量或 threshold sweep。先记录事件级 D2H/H2D/queue/stall/recompute 成本，使用真实 lifecycle trace 构建 `OFFLOAD/DROP` 与 `RESTORE/RECOMPUTE` 离线 Oracle。在线策略取得可重复收益后，再扩展到 open-loop arrival 和 Agent/RAG workload。

## 8. 原始结果索引

- [1000-block 3-seed 总结](../results/direct_sharegpt_compare_1000_active32_3seed_20260806/summary.md)
- [1000-block no-offload 3-seed 聚合](../results/direct_sharegpt_compare_1000_active32_3seed_20260806/no_offload.md)
- [1000-block all-block + restore 3-seed 聚合](../results/direct_sharegpt_compare_1000_active32_3seed_20260806/all_block_restore.md)
- [1000-block all-block + recompute 3-seed 聚合](../results/direct_sharegpt_compare_1000_active32_3seed_20260806/all_block_recompute.md)

- [600-block all-block smoke](../results/direct_sharegpt_lifecycle_allblock_cpu4g_600_active32_128/20260806_002636/policy_compare_summary.md)
- [1000-block all-block restore smoke](../results/direct_sharegpt_lifecycle_allblock_cpu4g_1000_active32_128/20260806_002832/policy_compare_summary.md)
- [1000-block no-offload smoke](../results/direct_sharegpt_compare_no_offload_1000_active32_128/20260806_003044/policy_compare_summary.md)
- [1000-block all-block recompute smoke](../results/direct_sharegpt_compare_allblock_recompute_cpu4g_1000_active32_128/20260806_003044/policy_compare_summary.md)
