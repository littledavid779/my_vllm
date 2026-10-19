# 开题 PPT 完整制作大纲：收益感知分层 KV Cache 管理

日期：2026-08-09

建议题目：

> 面向多轮大模型推理的收益感知分层 KV Cache 管理研究

建议副标题：

> 选择性准入、恢复/重算决策与负收益保护

建议规模：正文 16 页，答辩时间 15--18 分钟；附录 10--12 页，用于回答
实验口径、数据可信度和实现细节问题。

本文档不是论文目录，而是逐页 PPT 制作说明。每页均给出：本页目的、核心
结论、推荐版式、必须展示的数据、讲稿要点、前后页过渡及本地证据引用。

---

## 一、整场答辩的唯一主线

全场只讲下面一条逻辑链：

```text
多轮推理产生大量可复用 KV，但 GPU HBM 容量有限
        ↓
早期尝试用 sharing / chain / hotness 改进 GPU eviction
        ↓
发现复用价值信号存在，但单独替换 LRU 的收益随压力变化而不稳定
        ↓
KV 离开 HBM 后仍有两个决策：OFFLOAD / DROP
        ↓
未来 CPU 命中后还有两个决策：RESTORE / RECOMPUTE
        ↓
自然多轮实验发现：
  external reuse 依赖会话交错；
  all-block 同时保存有用与无用 KV；
  restore 通常更快但并非逐事件最优；
  raw DMA 不能代表完整阻塞成本
        ↓
因此提出收益感知分层 KV Cache 控制面：
  生命周期与成本账本
  + 选择性准入
  + 负载感知恢复/重算
  + 负收益保护
```

整场不要把课题讲成“又一个 eviction score”。主语应始终是：

> 在多轮推理中，联合管理 KV 的 GPU 驻留、CPU 准入与恢复方式，并在收益为负
> 时安全退化。

---

## 二、文档与实验引用地图

PPT 正文页脚建议使用短引用编号，例如 `[D1, R3]`；完整路径放在最后一页
“本地证据索引”和答辩附录中。

| 编号 | 文档/结果 | 在 PPT 中的作用 |
| --- | --- | --- |
| D1 | [收益感知分层 KV Cache 研究状态](hierarchical_kv_research_status_20260805.md) | 总体课题定位、证据等级、系统状态和技术路线 |
| D2 | [GPU eviction 三 seed 总结](cachedattention_online_3seed_summary_20260714.md) | 早期 sharing/chain/hotness 结果与课题演进 |
| D3 | [自然 external restore 动机](natural_external_restore_motivation_20260805.md) | restore 路径、生命周期矩阵、负收益逻辑链 |
| D4 | [Direct ShareGPT 迁移记录](direct_sharegpt_workload_migration_20260806.md) | 数据来源、固定 workload、benchmark seed 与 dataset seed 区分 |
| D5 | [ShareGPT 事件级成本账本](offload_cost_ledger_sharegpt_20260809.md) | timer 语义、三 dataset seed、Oracle 与边界 |
| R1 | [Natural restore sanity report](../results/external_restore_sanity/20260725_005350/t0_natural_eviction/report.json) | 证明不 reset 的 HBM→DRAM→HBM 路径可用 |
| R2 | [Selective admission 5-repeat](../results/external_restore_admission/20260729_235912/summary.md) | 受控污染实验，证明 OFFLOAD/DROP 的机制价值 |
| R3 | [Restore/recompute length crossover](../results/external_restore_promotion_length/formal_idle_crossover_6pair_20260801/summary.md) | 受控成本交叉区间 |
| R4 | [Natural workload 3-seed policy comparison](../results/trace_driven_lifecycle_compare_600_active32_3seed_20260805/summary.md) | 真实多轮内容中的 external reuse、all-block 负收益和 threshold2 失败 |
| R5 | [Direct ShareGPT 固定 workload 三 runtime-seed 对照](../results/direct_sharegpt_compare_1000_active32_3seed_20260806/summary.md) | no-offload、restore、recompute 的端到端关系 |
| R6 | [三 dataset seed 事件级结果](../results/direct_sharegpt_cost_ledger_3dataset_seeds_20260809/summary.md) | 109 对事件、Oracle gap、真实内容跨样本重复性 |
| R7 | [跨 seed 机器可读聚合](../results/direct_sharegpt_cost_ledger_3dataset_seeds_20260809/cross_seed_paired_cost_summary.json) | 制图与数值核对的 source of truth |

文档之间的关系：

```text
D2 早期 GPU-only 工作
    └── 解释为什么课题从 eviction score 扩展

D3 自然 restore 动机
    ├── 引用 R1：路径有效
    ├── 引用 R2：selective admission 机制有效
    └── 引用 R4：自然负载中 all-block 净收益不稳定

D4 Direct ShareGPT 数据迁移
    ├── 引用 R5：固定 workload 的策略对照
    └── 为 D5/R6 提供 dataset seed、长度分布和数据来源

D5 事件级成本账本
    ├── 引用 R3：受控 crossover
    ├── 引用 R6：三 dataset seed 结果
    └── 引用 R7：跨 seed Oracle 聚合

D1 汇总 D2--D5，作为课题总状态和开题报告主索引
```

---

## 三、正文逐页制作方案

## 第 1 页：标题页

### 本页目的

让评委立即知道研究对象、场景和三个核心动作。

### 页面内容

- 主标题：面向多轮大模型推理的收益感知分层 KV Cache 管理研究；
- 副标题：选择性准入、恢复/重算决策与负收益保护；
- 姓名、导师、专业、日期；
- 底部小字：基于 vLLM 的系统实现与实验研究。

### 推荐视觉

右侧使用一个极简三层图：

```text
GPU HBM  ← RESTORE / KEEP
   ↕
CPU DRAM ← OFFLOAD
   ↓
DROP / RECOMPUTE
```

### 讲稿要点

“本课题不是单纯提高缓存命中率，而是研究一次 KV 保存、恢复或重算是否真正
产生正收益。”

### 引用

`[D1]`

---

## 第 2 页：研究背景——为什么多轮推理需要分层 KV Cache

### 本页核心结论

多轮会话带来稳定的前缀复用机会，但 HBM 容量增长跟不上并发会话与上下文
长度，系统必须在计算和跨层存储之间选择。

### 推荐版式

左侧画三条多轮 Session 时间线，右侧画 HBM 容量条和 DRAM 层。

```text
Session A: A1 → A2 → A3
Session B: B1 → B2
Session C: C1 → C2 → C3

每个 Session 内保持因果顺序；不同 Session 的 ready request 可交错。
```

### 页面文字控制

只保留三点：

1. KV Cache 避免重复 prefill；
2. 多会话并发导致 HBM 容量竞争；
3. CPU DRAM 容量大，但 D2H/H2D 和阻塞不是免费的。

### 讲稿要点

- “普通单会话轮次是顺序的，调度与缓存竞争来自多个并发会话”；
- “是否 offload 不是命中率问题，而是保存成本与未来节省计算的权衡”。

### 引用

`[D1 §1--3]`

---

## 第 3 页：相关研究与尚未解决的系统缺口

### 本页核心结论

现有工作通常分别优化 GPU eviction、外部 KV 存储、预取或请求调度；但一次
KV 决策的净收益跨越多个阶段，分开优化容易产生负收益。

### 推荐版式

使用四列对照：

| 模块 | 常见问题 | 本课题发现的缺口 |
| --- | --- | --- |
| GPU eviction | 淘汰谁 | 淘汰后是否值得写入 CPU |
| CPU admission | 保存谁 | 写入是否会在 retention window 内复用 |
| Promotion | CPU 命中后加载 | restore 是否比当前负载下重算更快 |
| Scheduler | 运行哪个请求 | restore 阻塞、GPU batch 和 SLO 相互影响 |

底部用红框突出：

> 缺少统一的 `KEEP / OFFLOAD / DROP / RESTORE / RECOMPUTE` 收益控制面。

### 文献处理要求

正式 PPT 制作时，本页应引用原始论文/官方文档，而不是只引用本地讨论。建议
按“hotness-aware caching、agent workflow caching、KV offload、cache-aware
scheduling”四类各选 1--2 篇。相关工作名称和年份需在最终制作前再次核验。

### 讲稿过渡

“我最初也从 eviction queue 开始，但实验表明，仅修改淘汰顺序不能覆盖完整
问题。”

### 引用

课题缺口引用 `[D1 §2--3]`；外部文献待补原始论文引用。

---

## 第 4 页：前期工作——GPU-only 价值淘汰的结果与局限

### 本页核心结论

sharing/chain 能反映复用价值，但收益随压力区间变化，short-term hotness 也未
形成稳定信号；因此研究必须从 GPU eviction 扩展到分层放置。

### 推荐图表

左侧：128-request 分组柱状图；右侧：512-request/guard 结果小表。

#### 必须出现的数据

128-request、3 benchmark seeds：

| policy | avg TTFT | 相对 LRU | P90 TTFT | 相对 LRU |
| --- | ---: | ---: | ---: | ---: |
| LRU | 23.4526 ms | baseline | 33.6252 ms | baseline |
| chain_hybrid | 21.6022 ms | -7.89% | 32.2354 ms | -4.13% |
| hybrid_hot1 | 24.1197 ms | +2.84% | 35.1484 ms | +4.53% |

512-request 改进版本：

- `shared_chain1_hit24` 平均 TTFT 改善 0.67%；
- P90 改善 0.35%；
- seed 1 仍退化。

### 页面结论框

```text
保留：sharing / chain 作为复用价值输入
放弃：将一个加权 eviction score 作为完整课题
```

### 讲稿要点

- 128-request 结果说明信号存在；
- 512-request 和 seed 退化说明单一队列启发式泛化不足；
- 这不是前期工作失败，而是推动问题边界从“淘汰谁”扩展到“淘汰后做什么”。

### 引用

`[D2, D1 §4.1]`

---

## 第 5 页：问题重构——一次 KV 生命周期包含五个动作

### 本页核心结论

GPU 淘汰只是生命周期中的一个节点，完整动作空间需要联合建模。

### 推荐视觉

本页只放一张系统状态机：

```text
                         ┌──────── KEEP ────────┐
                         │                       │
GPU-resident KV ─ eviction decision ─ OFFLOAD → CPU-resident KV
                         │                       │
                         └──────── DROP          ├── RESTORE
                                                 └── RECOMPUTE
```

在图下放统一效用目标：

```text
U(b) = P_reuse(b) · C_saved_compute(b)
       - C_store(b)
       - P_restore(b) · C_restore(b)
       - C_blocking(b)
       - C_capacity_opportunity(b)
```

注明：这是设计目标，当前使用生命周期统计、成本表和在线窗口近似，不宣称
已经精确预测全部项。

### 讲稿过渡

“接下来四组实验分别回答：路径是否真实存在、何时产生 external reuse、
all-block 为什么浪费，以及 restore 是否总比 recompute 好。”

### 引用

`[D1 §2--3]`

---

## 第 6 页：可行性证据——不依赖 reset 的 external restore 已打通

### 本页核心结论

HBM→DRAM→HBM 数据路径、block hash lookup、加载完成和输出正确性已经验证，
后续研究不是优化一条不存在的路径。

### 推荐版式

用横向事件时间线：

```text
A 首次执行并生成 KV
    → 长且不同的 polluter 造成自然 HBM eviction
    → A 重访：仅 1 个 GPU hit block
    → CPU lookup hit
    → 32-block H2D load
    → 输出与 cold run 一致
```

### 必须出现的数据

- 目标 prompt：528 tokens；
- CPU load：32 blocks；
- H2D：56 MiB；
- raw H2D：约 2.51 ms；
- sanity status：PASS；
- forced reset 没有产生 external load，不能作为证据。

### 讲稿要点

强调“自然淘汰”而不是 cache reset。评委若问 restore 是否人为制造，应回答：
polluter 是控制 HBM pressure 的因果实验，但数据路径由自然 eviction 和真实
prefix lookup 触发。

### 引用

`[R1, D3 §4.2, D1 §4.2]`

---

## 第 7 页：问题发现一——external reuse 取决于会话交错，而非只看容量

### 本页核心结论

CPU 容量更大、HBM 淘汰更多，都不自动产生 external reuse；足够强的会话交错
才把可复用前缀推出 HBM，并在下一轮命中 DRAM。

### 推荐图表

使用四组阶梯柱状图，重点画 `unique loaded`，其他指标用表格标注。

| GPU blocks | CPU cache | active sessions | unique stores | GPU evictions | unique loaded |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1200 | 1 GiB | 16 | 1172 | 69 | 0 |
| 1200 | 4 GiB | 16 | 1174 | 71 | 0 |
| 600 | 4 GiB | 16 | 1169 | 611 | 0 |
| 600 | 4 GiB | 32 | 1209 | 1198 | 520 |

图上使用一个醒目标注：

```text
active sessions 16 → 32
unique loaded 0 → 520
```

### 讲稿要点

- 第二行排除“CPU 太小”；
- 第三行排除“HBM pressure 不够”；
- 第四行说明 reuse gap 与会话交错是关键；
- 因此实验负载必须是多会话、会话内顺序、会话间交错。

### 引用

`[D3 §4.6, D1 §4.6]`，原始四组结果路径见 D1 对应小节。

---

## 第 8 页：问题发现二——All-block admission 会被一次性请求污染

### 本页核心结论

在保持同一 33-block 有用 restore 的情况下，选择性准入可以去除 polluter 的
全部写入，证明 `OFFLOAD / DROP` 是独立且有价值的控制动作。

### 推荐图表

左侧画 A/P 请求示意，右侧使用三组柱状图：stores、D2H、useful-store ratio。

```text
A：未来复用 → OFFLOAD
P：一次性 polluter → DROP
```

### 必须出现的数据

| arm | stores | P stores/dropped | loads | D2H | useful-store | final A median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all-block-large | 146 | 113 / 0 | 33 | 255.50 MiB | 22.60% | 17.81 ms |
| selective | 33 | 0 / 113 | 33 | 57.75 MiB | 100% | 17.83 ms |
| all-block-constrained | 146 | 113 / 0 | 0 | 255.50 MiB | 0% | 22.90 ms |

页面右下角突出：

- stores 与 D2H 均减少 77.4%；
- selective 在受限 CPU 下仍保留 A 的 33-block restore；
- 5/5 重复通过。

### 可信度标记

在页脚写：`Controlled causal workload；用于证明机制，不作为真实负载泛化证据。`

### 引用

`[R2, D1 §4.3]`

---

## 第 9 页：问题发现三——真实多轮内容中，有用与无用 offload 同时存在

### 本页核心结论

All-block 可以保留约 500 个有用 blocks，但同时写入约 690 个没有进入后续
external lookup 的 blocks；简单 threshold2 减少写入，却错杀了大部分有用复用。

### 推荐图表

主图使用堆叠柱：每个 arm 的“后续加载”和“未进入后续 external lookup”。
右侧放端到端小表。

#### 机制数据

| arm | unique stores | unique loaded | useful-store ratio | never looked after store |
| --- | ---: | ---: | ---: | ---: |
| all-block | 1205.0 | 511.7 | 42.5% | 693.3 |
| selective-t2 | 551.7 | 58.3 | 10.6% | 493.3 |

#### 端到端数据

| arm | avg TTFT | P90 TTFT | 相对 no-offload avg |
| --- | ---: | ---: | ---: |
| no-offload | 23.2285 ms | 34.1257 ms | baseline |
| all-block | 23.9658 ms | 34.7677 ms | +3.17% |
| all-block-recompute | 25.1602 ms | 34.8599 ms | +8.32% |
| selective-t2 | 25.6398 ms | 36.9631 ms | +10.38% |

### 页面结论框

```text
All-block：有价值，但净收益不稳定
Threshold2：写得更少，却丢掉约 90% restore
需要预测“下一轮是否会复用”，而不是复用后才学习
```

### 讲稿要点

本页是“负收益保护”的核心动机。必须主动说明 all-block 没有稳定优于
no-offload，这恰好说明研究问题真实存在，而不是掩盖负结果。

### 引用

`[R4, D3 §4.7, D1 §4.7]`

---

## 第 10 页：问题发现四——RESTORE / RECOMPUTE 存在交叉，且 DMA 不是完整成本

### 本页核心结论

CPU 命中后不能机械地 restore；短前缀、GPU 负载和传输队列可能使 recompute
更优，而且只测 PCIe DMA 会严重低估完整 restore wait。

### 推荐版式

左侧：受控长度 crossover 折线；右侧：restore 成本分层漏斗。

#### 受控 crossover 数据

`paired delta = recompute - restore`，正数表示 restore 更快：

| blocks | paired delta mean | restore/recompute wins |
| ---: | ---: | ---: |
| 16 | -0.48 ms | 2 / 4 |
| 24 | -0.93 ms | 2 / 4 |
| 28 | -2.24 ms | 1 / 5 |
| 33 | +4.38 ms | 6 / 0 |

#### 真实系统成本层次

| 计时层次 | seed0 restore 均值 |
| --- | ---: |
| CUDA H2D DMA | 2.595 ms |
| submit-to-complete | 5.140 ms |
| decision-to-scheduler-observed restore wait | 10.404 ms |

### 页面结论框

```text
C_restore ≠ bytes / PCIe bandwidth
C_recompute ≠ tokens × constant
两者都依赖 batch、GPU load、transfer queue 和调度观察延迟
```

### 引用

`[R3, D5 §5.1, D1 §4.4]`

---

## 第 11 页：核心实验证据——三套独立 Direct ShareGPT 的事件级成本

### 本页核心结论

真实多轮内容中 restore 总体更快，但约 22% 的严格配对事件由 recompute 获胜；
逐事件 Oracle 存在约 6.85% 的优化空间，而 block-only 规则几乎无法利用该空间。

### 推荐图表

主图：每个 dataset seed 三根柱：always-restore、always-recompute、Oracle。
右侧：胜负饼图或 85/24 大数字。

### 必须出现的数据

| seed | paired | restore wait | recompute wait | restore/recompute wins | Oracle gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 41 | 10.308 ms | 15.672 ms | 32 / 9 | 5.53% |
| 1 | 32 | 9.479 ms | 13.975 ms | 27 / 5 | 6.19% |
| 2 | 36 | 9.283 ms | 12.318 ms | 26 / 10 | 9.11% |
| pooled | 109 | 9.726 ms | 14.066 ms | 85 / 24 | 6.85% |

补充两个结论：

- pooled Oracle：9.060 ms，较 always-restore 低 0.666 ms/事件；
- pooled 最佳 block threshold 只节省 0.051 ms/事件，约 0.52%；各 seed 最佳
  threshold 为 0、4、14。

### 端到端一致性小字

- restore 宏平均 TTFT/P90：26.578/39.111 ms；
- recompute：31.967/47.749 ms；
- 三次 restore 运行 useful-store ratio：3159/5710 = 55.3%，仍有 44.7% 未
  产生 external demand。

### 数据口径标记

- 三个 dataset seed 的 128 conversations 两两仅重叠 7、2、9 个；
- 六次运行均 0 dropped、0 unmatched、0 incomplete；
- 109 对是严格按 conversation、candidate blocks 和本地已计算 tokens 匹配；
- restore/recompute 来自独立在线运行，Oracle 是 in-sample 上界。

### 引用

`[R6, R7, D4, D5 §8]`

---

## 第 12 页：研究问题与三个可检验假设

### 本页核心结论

已有实验已经把“课题是否存在”转化为三个可独立验证的系统假设。

### 页面内容

#### H1：选择性准入

> session/turn reuse、sharing/chain 与实测成本能够减少无效 D2H，同时保留大部分
> 有用 restore。

#### H2：负载感知 promotion

> candidate tokens 与 GPU/transfer queue 状态的成本模型能够优于
> always-restore/always-recompute，并取得至少一半 Oracle gap。

#### H3：负收益保护

> 当滑动窗口净收益为负时，动态降低 admission rate 或关闭 CPU tier，可将低
> 复用负载的退化限制在预设范围内。

### 推荐视觉

三张卡片，每张写“输入信号→动作→评价指标”。

| 假设 | 输入 | 动作 | 主指标 |
| --- | --- | --- | --- |
| H1 | session gap、sharing、chain、历史 reuse | OFFLOAD/DROP | useful-store、D2H、TTFT |
| H2 | blocks/tokens、batch、GPU/PCIe queue | RESTORE/RECOMPUTE | regret、stall、TTFT |
| H3 | 窗口净收益、bytes、saved compute | throttle/disable | 相对 no-offload 退化 |

### 引用

`[D1 §3, §7, §9; D5 §8.4]`

---

## 第 13 页：拟设计的系统架构

### 本页核心结论

系统由观测、决策和执行三个平面组成，三个策略共享同一个成本账本，避免各自
使用互相矛盾的局部指标。

### 推荐架构图

```text
                              ┌──────────────────────────┐
Request / Session metadata ──→│ Block Reuse/Value Model  │
sharing / chain / reuse gap    └────────────┬─────────────┘
                                            │ P_reuse / value
┌──────────────────────────┐                ▼
│ Event-level Cost Ledger  │──────→ Selective Admission
│ store / DMA / queue      │          OFFLOAD / DROP
│ restore / recompute      │                │
│ useful / unused lifetime │                ▼
└────────────┬─────────────┘          CPU KV Manager
             │                              │ CPU hit
             ▼                              ▼
      Negative-Utility Guard ←──── Cost-Aware Promotion
      throttle / disable            RESTORE / RECOMPUTE
             │                              │
             └──────────────→ vLLM Scheduler / GPU Worker
```

### 图中必须明确

- HBM 和 DRAM 是数据层；
- ledger/model/controller 是控制层；
- scheduler 只提供负载上下文，开题阶段不承诺复杂联合调度；
- progressive restore、SSD、多机不属于当前必要范围。

### 已实现/拟实现颜色

- 实线或绿色：事件账本、always baselines、生命周期统计已完成；
- 虚线或橙色：在线 admission、load-aware promotion、guard 待实现。

### 引用

`[D1 §6--7]`；实现入口见附录第 A11 页。

---

## 第 14 页：优化方案与第一版在线算法

### 本页核心结论

第一版使用可解释的查表、EWMA 和 deadband，不直接上机器学习，也不一次加入
复杂 scheduler。

### 方案一：Admission

```text
score_offload(b)
  = predicted_reuse_value
  - measured_store_cost
  - expected_capacity_cost

score > admission_margin → OFFLOAD
otherwise                → DROP
```

信号限制为：session reuse gap、shared count、chain/prefix 位置、历史 reuse、
store bytes/time。

### 方案二：Promotion

```text
if Ĉ_restore(blocks, transfer_queue, GPU_state) + deadband
   < Ĉ_recompute(tokens, batch, GPU_load):
    RESTORE
else:
    RECOMPUTE
```

成本表按 block/token 长度与负载区间分桶，使用 EWMA 在线更新。

### 方案三：Guard

```text
window_net_utility
  = saved_recompute_time
  - store_time
  - restore_stall
  - queue/blocking overhead

持续为负 → 降低 admission rate → 必要时关闭 offload
```

### 页面边界

明确注明：当前完成的是 ledger 和 fixed-policy Oracle；本页三项是开题后的拟实现
方案，不能写成已经取得性能提升。

### 引用

`[D1 §3, §7.3; D5 §7--8]`

---

## 第 15 页：实验设计、评价指标与 Go/No-Go 标准

### 对照组

1. no-offload；
2. all-block + always-restore；
3. all-block + always-recompute；
4. 简单 selective：TTL、2Q/TinyLFU 或 frequency threshold；
5. proposed selective + fixed promotion；
6. proposed selective + cost-aware promotion + guard；
7. offline Oracle。

### Workloads

- 可控 A/P pollution 与 length crossover：机制因果；
- Direct ShareGPT 三 dataset seeds：真实多轮内容；
- BurstGPT-length-matched ShareGPT：压力与消融；
- 后续增加 open-loop arrival；
- 后续增加 Agent/tool-call 或长上下文 RAG；
- 单轮低复用请求作为负对照。

### 指标

| 维度 | 指标 |
| --- | --- |
| 用户性能 | P50/P95/P99 TTFT、ITL、E2E、SLO attainment |
| 系统性能 | throughput、GPU idle、queue/blocking time |
| 分层成本 | D2H/H2D bytes、DMA、restore stall、recomputed tokens |
| 缓存质量 | useful-store、CPU churn、external loads、unused stores |
| 控制器 | Oracle regret、决策开销、内存开销、fallback 次数 |

### Go/No-Go

- 在线策略取得至少一半 promotion Oracle gap；
- 控制器开销低于 2%；
- 低复用 workload 相对 no-offload 的退化控制在约 3%；
- 若 promotion 模型无法稳定超过 always-restore，则将 promotion 降为保护机制，
  主贡献集中在 selective admission 与 negative-utility guard。

### 引用

`[D1 §8--9]`

---

## 第 16 页：创新点、可行性、计划与结束语

### 拟创新点

写“拟”，不要写“已经实现最终创新”。

1. 从单一 eviction queue 扩展为覆盖 `OFFLOAD/DROP/RESTORE/RECOMPUTE` 的
   多动作分层 KV 控制面；
2. 使用事件级真实系统等待而非命中率或 raw PCIe 带宽进行收益核算；
3. 联合多轮会话复用价值与负载状态进行选择性准入和 promotion；
4. 使用 negative-utility guard，使 CPU tier 在不适用负载下安全退化。

### 当前可行性

- 已有 vLLM 原型和 5 个动作所需的关键执行路径；
- external restore、always-recompute 和事件级 timer 已打通；
- 有 synthetic 因果 workload、真实多轮内容和三套独立 dataset seed；
- 两张 A100 足以完成单机 HBM/DRAM 分层研究；
- 当前 109 对事件证明研究对象和 Oracle gap 存在。

### 研究计划

可按学校时间替换月份，内容顺序保持：

| 阶段 | 任务 | 交付物 |
| --- | --- | --- |
| 已完成 | 路径验证、生命周期统计、成本账本、三 seed Oracle | 初步系统与动机实验 |
| 阶段 1 | 补负载上下文，建立离线 admission/promotion Oracle | 成本分桶和可行性边界 |
| 阶段 2 | 在线 selective admission 与 negative guard | 最小控制器原型 |
| 阶段 3 | load-aware promotion 与消融 | 完整四动作策略 |
| 阶段 4 | 多模型、open-loop、Agent/RAG 评测 | 论文、开源代码、学位论文 |

### 最后一页结论

> 前期实验表明，分层 KV Cache 的关键矛盾不是“是否能够 offload”，而是“何时
> offload 的净收益为正”。本课题拟以事件级成本账本为基础，联合优化 CPU 准入
> 和恢复方式，并通过负收益保护保证系统在不适合 external caching 时安全退化。

### 引用

`[D1, D5, R6]`

---

## 四、附录页设计

## A1：术语和统计口径

- HBM block、offloaded block 和 token 数的换算；
- store、lookup、load、restore、recompute 定义；
- useful-store ratio 的分母和截断窗口限制；
- `paired delta = recompute - restore`；
- benchmark seed 与 dataset seed 区别。

引用：`[D4, D5]`。

## A2：Workload 来源与可复现性

- ShareGPT source URL、SHA256；
- converter seed、512 candidates、4064 prompt-token limit；
- 三个 workload SHA256；
- conversation overlap 7/2/9；
- `REQUEST_RATE=0`，不是 BurstGPT 到达回放。

引用：`[D4, R6]`。

## A3：Natural restore sanity 完整过程

展示 A→polluter→A 的 block 生命周期、56 MiB H2D、32 blocks、输出一致性。

引用：`[R1]`。

## A4：四组生命周期矩阵完整表

展示 GPU/CPU capacity、active sessions、GPU/CPU eviction、looked/load 等全部
指标，解释为何 active sessions 是关键变量。

引用：`[D3 §4.6]`。

## A5：Selective admission 五组完整结果

完整列出 all-block-constrained、all-block-large、no-offload、
selective-constrained、selective-minimal 的 5-repeat min/max。

引用：`[R2]`。

## A6：Restore/recompute 长度 sweep

展示 16/24/28/33 blocks 及 30/31 blocks fine sweep；说明这只是空闲负载下的
机制图，不能变成固定生产阈值。

引用：`[R3]` 和
[30/31-block fine sweep](../results/external_restore_promotion_length/fine_idle_transition_unique_6pair_20260802/summary.md)。

## A7：自然多轮三 runtime-seed 完整表

展示 no-offload、all-block、recompute、threshold2 的逐 seed TTFT，并解释
独立 server runs 的计时噪声。

引用：`[R4]`。

## A8：Direct ShareGPT 固定 workload 三 runtime-seed

展示 no-offload 28.53 ms、restore 30.07 ms、recompute 32.29 ms；强调 restore
局部优于 recompute，但 all-block 整体仍比 no-offload 慢 5.41%。

引用：`[R5]`。

## A9：三 dataset seed 全量事件表

展示 41/32/36 配对数、各 seed 阈值、Oracle、TTFT 和 unmatched signatures。

引用：`[R6, R7]`。

## A10：Timer 语义

```text
lookup wall
decision → worker submit
CUDA DMA start → end
submit → worker observes complete
decision → scheduler observes restore complete
decision → target tokens recomputed
```

强调这些时间嵌套，不能用 `restore_wait - DMA` 直接命名某个独立组件。

引用：`[D5 §2, §5.1]`。

## A11：实现位置

- `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`；
- `vllm/v1/kv_offload/reuse_manager.py`；
- `benchmarks/external_restore_admission.py`；
- `benchmarks/external_restore_natural.py`；
- `scripts/summarize_offload_cost_events.py`；
- `scripts/compare_offload_promotion_costs.py`；
- `scripts/aggregate_offload_promotion_costs.py`；
- `scripts/build_direct_sharegpt_workload.py`。

## A12：威胁与边界

- 单模型 Qwen3-0.6B；
- 单机 A100 + CPU DRAM；
- 闭环并发；
- 独立在线 runs 不是同一瞬时状态；
- 109 对只包含严格匹配事件；
- Oracle 和最佳 threshold 是 in-sample；
- 未验证生产 arrival、Agent/RAG 和跨硬件泛化。

---

## 五、建议制作的核心图表

| 图号 | 图表 | 数据源 | 推荐图形 |
| --- | --- | --- | --- |
| F1 | 早期 eviction 策略演进 | D2 | 两组分组柱 + 压力变化箭头 |
| F2 | External reuse 出现条件 | D3 §4.6 | 四组柱状图，突出 active=32 |
| F3 | Selective admission 污染控制 | R2 | stores/D2H 双轴柱 + useful ratio 标注 |
| F4 | Natural workload 有用/无用 store | R4 | stacked bar |
| F5 | Length crossover | R3 | blocks→paired delta 折线，0线分隔 |
| F6 | Restore timer 分层 | D5 §5.1 | 三层漏斗或嵌套条形图 |
| F7 | 三 dataset seed promotion | R6/R7 | restore/recompute/Oracle 分组柱 |
| F8 | Pooled action wins | R7 | 85/24 横向比例条 |
| F9 | 完整系统架构 | D1 §7 | 自绘控制流架构图 |

制图规则：

- 主颜色固定：restore 绿色、recompute 橙色、no-offload 灰色、Oracle 蓝色；
- 所有 TTFT/latency 图统一以 ms 为单位；
- 所有“更好”方向在图注中明确；
- 不在一张图里混合 synthetic 与 direct ShareGPT；
- 不把独立 server run 的 TTFT 差画成精确 D2H/H2D 分解；
- 所有百分比同时标原始分子/分母，避免只展示相对改善。

---

## 六、答辩口径与高风险表述

## 可以说

- external restore 路径和自然 external reuse 已验证；
- all-block 在真实多轮内容中稳定混合有用与无用 writes；
- restore 在当前配置下总体优于 recompute，但约 22% 配对事件由 recompute
  获胜；
- 三 dataset seed 的 promotion Oracle gap 为 5.53%--9.11%；
- simple block threshold 无法稳定逼近 Oracle；
- 当前具备实现在线控制器所需的成本账本和基线。

## 不能说

- “我的最终策略已经稳定优于 no-offload”；
- “44.7% 未恢复 blocks 永远不会复用”；
- “Oracle 6.85% 就等于最终 TTFT 能提升 6.85%”；
- “BurstGPT arrival 已经回放”；
- “当前结果已经代表 Agent/RAG 或生产流量”；
- “2.595 ms DMA 之外的 7.8 ms 全部是 scheduler overhead”；
- “candidate blocks 小于某个固定阈值就一定应该 recompute”。

建议使用的措辞：

```text
初步结果 / 当前配置 / 观测窗口内 / 受控回放 / 机制证据 /
in-sample Oracle / 尚待在线验证 / 尚待跨模型泛化
```

---

## 七、预设答辩问题与回答骨架

### Q1：为什么不继续改 LRU score？

回答：sharing/chain 在 128 requests 有收益，但在 512 requests 和不同 seed 下
不稳定。更关键的是 LRU score 只回答 HBM 淘汰顺序，无法回答是否写 CPU、CPU
命中后是否恢复，以及何时关闭 offload。前期结果因此被保留为 reuse signal，
而不是最终控制面。

### Q2：为什么 all-block 比 no-offload 慢，还要研究 offload？

回答：all-block 同时保存了有用和无用 KV，restore 局部快并不代表包含所有 D2H、
H2D 和 queue 的整体路径有正收益。这个负结果正是 selective admission 和
negative guard 的动机；目标不是证明 offload 永远更快，而是只在收益为正时
启用。

### Q3：Oracle 只有 6.85%，空间是不是太小？

回答：6.85% 是 CPU 命中后的 promotion 上界，不包含 eviction 时避免 44.7%
无需求写入的 admission 空间。课题主线是 admission + promotion + guard；如果
load-aware promotion 无法取得一半 Oracle，promotion 会降为保护机制，主贡献
仍由 admission 与安全退化支撑。

### Q4：真实负载在哪里？

回答：内容来自校验过的原始 ShareGPT，已使用三套独立 converter seed；会话内
因果顺序保留。当前到达是 32-session 闭环并发，因此应称“真实多轮内容的受控
回放”，不能称生产 open-loop trace。open-loop 与 Agent/RAG 是后续外部有效性
评测。

### Q5：Restore/recompute 来自两个 runs，Oracle 可靠吗？

回答：只对 conversation、candidate blocks 和本地已计算 tokens 严格匹配的
109 对事件做比较，未匹配项显式排除；三个独立 dataset seed 方向一致。但两个
runs 的瞬时负载仍不同，所以该 Oracle 是初步 in-sample 上界。下一阶段会记录
running/waiting requests、token budget 和 transfer queue，并使用分桶重复实验。

### Q6：创新点和现有 cache/offload 工作有什么区别？

回答：本课题不只预测 hotness 或 future distance，而是把 CPU admission、真实
restore stall、recompute 和负收益 fallback 放在同一账本中，联合决定多动作。
正式 PPT 的相关工作页必须用原始论文逐项对比，避免只依赖功能名称。

### Q7：两张 A100 是否够？

回答：当前研究目标是单机 HBM/DRAM 分层、事件级成本和在线控制器，不依赖多机
网络。两张 A100 足以完成独占单卡重复、硬件轮换和并发/容量 sweep；多机与 SSD
不属于开题必要范围。

---

## 八、PPT 制作风格和时间控制

- 16:9，浅色背景；
- 中文主字体使用思源黑体/微软雅黑，英文和数字使用 Aptos/Arial；
- 标题不小于 28 pt，正文不小于 18 pt，脚注不小于 10 pt；
- 每页顶部只放一句结论型标题，不使用“实验结果 1”之类空标题；
- 每页最多一个主图、三条结论；
- 数据页讲 50--70 秒，背景页讲 30--45 秒，系统页讲 60--90 秒；
- 16 页正文目标 15--18 分钟；若学校要求 10 分钟，删除第 6 页 sanity 和第
  10 页 controlled crossover，将其移到附录；
- 右下角统一放引用编号；最后一页或附录列本地路径与外部论文引用。

建议结论型页标题示例：

- 不写：“前期实验”；
- 写：“单一 eviction score 的收益随压力变化，不足以覆盖分层决策”；
- 不写：“实验结果”；
- 写：“三套独立 ShareGPT 中，22% CPU 命中事件更适合重算”；
- 不写：“系统设计”；
- 写：“统一成本账本驱动 OFFLOAD/DROP 与 RESTORE/RECOMPUTE”。

---

## 九、制作前最后检查清单

- [ ] 标题中的“收益感知”“分层”“多轮”在问题、方案和实验中均有对应内容；
- [ ] benchmark seed 和 dataset seed 已区分；
- [ ] synthetic、长度匹配内容和 direct ShareGPT 没有混称为同一负载；
- [ ] all-block 相对 no-offload 的负结果没有被隐藏；
- [ ] Oracle 被标为 in-sample upper bound；
- [ ] 已实现与拟实现使用不同颜色/线型；
- [ ] 每个主结论都有一手结果路径；
- [ ] 相关工作使用原始论文/官方资料，而不是二手描述；
- [ ] 正文只保留四到六张核心图，其余放附录；
- [ ] 最终回答始终回到：只在净收益为正时使用 CPU KV tier。

