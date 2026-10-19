# ShareGPT 事件级 KV Offload 成本账本初步结果（2026-08-09）

## 1. 这一步在课题中的位置

当前课题拟研究面向多轮会话推理的层次化 KV Cache 控制面，在线联合决定：

- HBM 淘汰时：`OFFLOAD / DROP`；
- 后续 CPU 命中时：`RESTORE / RECOMPUTE`；
- 当滑动窗口净收益为负时：降低准入率或关闭 offload。

本阶段不是提出最终在线策略，而是建立策略所需的事件级成本账本。此前的
汇总计数可以证明发生了 store/load，但无法回答某一次 CPU 命中究竟应恢复
还是重算，也无法区分纯 DMA 时间与请求真正等待的系统时间。

## 2. 已实现的观测

`OffloadingConnector` 增加了默认关闭的 `track_cost_events`。开启后使用有界
事件队列和临时整数 block ID；不输出 token 内容或原始 block hash。

每个 store 事件记录：

- 请求、申请/实际写入/淘汰的 block 数；
- CPU manager 的 `prepare_store` 时间；
- D2H 字节数、CUDA DMA 时间、提交到完成时间；
- 写入、淘汰的临时 block ID。

每个 promotion 事件记录：

- 请求、CPU 命中的 block/token 数和本地已计算 token 数；
- CPU lookup 时间；
- `RESTORE`：H2D DMA、提交到完成、决策到 scheduler 观察完成的
  `restore_wait`；
- `RECOMPUTE`：决策到目标前缀 token 被实际计算完成的
  `recompute_wait`。

其中 `restore_wait` 和 `recompute_wait` 是当前最接近在线决策目标的同口径
系统等待时间。纯 DMA 只描述传输本身，不能代表请求阻塞成本。

相关实现和工具：

- `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- `scripts/summarize_offload_cost_events.py`
- `scripts/compare_offload_promotion_costs.py`
- `tests/v1/kv_connector/unit/test_offloading_connector.py`

## 3. 实验配置

两组实验使用同一份直接从 ShareGPT 抽取的 128-request 多轮序列：

- 模型：Qwen3-0.6B；
- HBM KV 容量：1000 blocks；
- CPU KV 容量：4 GiB；
- 最大活跃会话：32；
- 输出：16--32 tokens；
- CPU admission：all-block（`store_threshold=0`）；
- promotion policy：分别固定为 `always_restore` 和 `always_recompute`。

结果目录：

- restore：`results/direct_sharegpt_cost_ledger_smoke_20260807/20260807_004412/`
- recompute：`results/direct_sharegpt_recompute_cost_ledger_20260809/20260809_161105/`
- 配对摘要：`results/direct_sharegpt_recompute_cost_ledger_20260809/20260809_161105/paired_restore_recompute_cost_summary.json`

## 4. 数据完整性

| 项目 | always-restore | always-recompute |
| --- | ---: | ---: |
| promotion 事件 | 43 | 44 |
| 有完整目标成本的事件 | 43 | 44 |
| dropped 事件 | 0 | 0 |
| unmatched transfer 更新 | 0 | 0 |
| incomplete recompute | 不适用 | 0 |

两次在线运行因并发时序不同产生了少量状态分叉。按稳定 conversation 前缀、
命中 block 数、本地已计算 token 数和同签名出现次序严格匹配后，共得到 41
对事件；restore 独有 2 个，recompute 独有 3 个。未配对事件被显式保留，未
进入成本差统计。

## 5. 主要结果

### 5.1 Restore 的系统等待不等于纯 DMA

43 个 restore 事件的均值：

| 成本层次 | 均值 |
| --- | ---: |
| H2D CUDA DMA | 2.595 ms |
| 提交到 worker 观察完成 | 5.140 ms |
| promotion 决策到 scheduler 观察完成 | 10.404 ms |

因此，纯 H2D DMA 只约占观测 restore wait 的四分之一。剩余差异包含队列、
依赖、worker 轮询、跨进程传递和 scheduler 再次观察等真实系统因素。三项计时
存在嵌套，不能简单相减后声称得到某一个独立组件，但可以证明只用 raw H2D
建立决策模型会系统性低估 restore 阻塞。

### 5.2 严格配对的 RESTORE / RECOMPUTE 成本

| 指标 | RESTORE | RECOMPUTE |
| --- | ---: | ---: |
| 配对事件数 | 41 | 41 |
| 平均等待 | 10.308 ms | 15.672 ms |
| P50 等待 | 9.560 ms | 15.773 ms |
| P90 等待 | 14.377 ms | 18.270 ms |

平均而言，restore 比 recompute 少等待 5.363 ms。41 对事件中：

- restore 更快：32 次（78.05%）；
- recompute 更快：9 次（21.95%）。

这同时说明两件事：CPU external restore 在真实多轮负载中确实有价值，但固定
`always_restore` 也并非逐事件最优。

### 5.3 当前 Oracle gap 存在，但还不大

在同一批 41 对事件上进行离线反事实计算：

| 策略 | 每事件平均成本 |
| --- | ---: |
| always-restore | 10.308 ms |
| always-recompute | 15.672 ms |
| 逐事件离线 Oracle | 9.738 ms |

Oracle 相比 always-restore 额外节省 0.570 ms/事件，即 5.53%。这是真实可优化
空间的初步证据，但还不足以证明复杂在线算法一定值得实现。

只使用 `candidate_blocks` 的单调阈值规则（小于阈值就重算）在本样本内的最佳
阈值是 0，即退化为 always-restore。虽然 1--8 blocks 区间有 3/4 次重算更快，
其中一次 restore 的收益幅度足以抵消其他三次的小收益。由此可见，不能只按
请求长度或 block 数决策，还需要 GPU/传输队列状态，并应设置收益 margin，
避免为了微小预测收益切换路径。

### 5.4 All-block admission 仍产生明显无效写入

restore 组在保留窗口内：

- unique stored blocks：2050；
- unique restored blocks：1218；
- stored 但未 restored：832；
- store DMA：3.762 GB；
- restore DMA：2.650 GB。

按临时 unique block ID 计算，约 59.4% 的已写入 block 在本次窗口内产生了
restore，约 40.6% 没有产生 restore。后者不能直接等价为永久无用，因为 trace
在 128 请求处截断，但已足以支持继续研究 selective admission，而不是默认将
所有淘汰 block 写入 CPU。

### 5.5 端到端结果只作一致性检查

单次运行的平均/P90 TTFT：

- always-restore：25.353 / 39.721 ms；
- always-recompute：34.282 / 54.491 ms。

方向与事件账本一致，但这两次运行不在同一天，且并发请求顺序出现了分叉，
因此不能把 8.929 ms 的平均差直接解释为 promotion policy 的严格因果收益。
此前三 seed 直接 ShareGPT 对照中，restore 相比 recompute 的平均 TTFT 改善为
2.222 ms（6.88%），该结果更适合作为端到端初步证据。

## 6. 当前能证明与不能证明的内容

已经证明：

1. 直接 ShareGPT 多轮负载可以稳定产生 CPU hit、H2D load 和完整 restore；
2. 在当前模型、容量和并发度下，restore 通常显著快于重算；
3. restore 的主要系统等待不能由 raw DMA 单独解释；
4. 存在少量 recompute 更优事件，同时 all-block admission 存在明显无效写入；
5. 事件级成本账本能够支持离线 Oracle 和后续在线策略评估。

尚未证明：

1. 5.53% 的 promotion Oracle gap 能跨 seed、模型和系统负载稳定存在；
2. 在线算法能在低开销下识别 9 个 recompute 更优事件；
3. 事件配对发生在完全相同的瞬时 GPU/PCIe 状态；
4. 当前结果能推广到更大的模型、不同 PCIe 拓扑和 Agent workload；
5. selective admission 相对 all-block 和 no-offload 的端到端收益。

## 7. 下一步

优先级顺序应为：

1. ~~在同一张独占 GPU 上按 `restore → recompute` 成对运行 3 个数据 seed，
   重复当前完整性检查和严格配对~~（已完成，见第 8 节）；
2. 为 promotion 事件补充决策时的系统上下文，例如 running/waiting request
   数、当前 token budget、活跃 transfer 数和队列深度；
3. 基于事件 trace 构造离线成本模型，比较 always-restore、always-recompute、
   block-only、load-aware 和 Oracle；
4. 将 store 生命周期与 promotion 成本连接，评估完整四动作 Oracle：
   `OFFLOAD / DROP / RESTORE / RECOMPUTE`；
5. 只有当跨 seed Oracle gap 明显且简单基线不能逼近时，再实现在线
   restore-or-recompute；否则将研究重点放在收益空间更大的 selective admission
   与负收益保护。

当前最重要的研究判断是：CPU tier 的价值已经不再悬空，但 promotion 决策的
额外空间可能小于 admission 决策。下一轮实验应同时量化这两个 gap，而不是
直接扩大到 progressive restore 或 scheduler 联合优化。

## 8. 三个独立数据 seed 的复验

### 8.1 Seed 口径纠正与固定输入

此前 `direct_sharegpt_compare_1000_active32_3seed_20260806` 使用的是同一份
`sharegpt_direct_seed0_128.json`，变化的是 benchmark runtime seed。它能反映
同一 workload 在不同客户端分配/调度时序下的重复性，但不能称为三套独立
ShareGPT 抽样。

本轮新增可复现的第二阶段构建脚本
`scripts/build_direct_sharegpt_workload.py`，并生成真正的 dataset seed 1/2。
脚本已在 seed0 上验证：重建的 128-conversation JSON 与现有文件完全一致。

| dataset seed | conversations | mean final prompt | max prompt | 与 seed0/1 的重叠 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 128 | 1237.15 | 3935 | -- |
| 1 | 128 | 1181.00 | 3860 | 与 seed0 重叠 7 |
| 2 | 128 | 1195.14 | 3753 | 与 seed0/1 分别重叠 2/9 |

三套输入各自有 128 个唯一 conversation，分布接近但内容主体不同。SHA256：

- seed0：`43501c376a92de4bc7260f349217bbbc03c5ba4119b2a3bac689929992934296`；
- seed1：`0da77499eeb01bcbb006ca3861692b8a54f4ac4ac749fa3df5583e28c4671193`；
- seed2：`955d911fe0f181f4f46ef6c2ac6ddcfb0092e83e1d27cd3c45e5526a47ffbbaf`。

### 8.2 三数据 seed 事件级结果

所有六次运行均为 0 dropped、0 unmatched transfer、0 incomplete recompute。

| seed | 严格配对 | restore wait | recompute wait | restore/recompute 更快次数 | Oracle 相对 always-restore | 最佳 block 阈值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 41 | 10.308 ms | 15.672 ms | 32 / 9 | 5.53% | 0 |
| 1 | 32 | 9.479 ms | 13.975 ms | 27 / 5 | 6.19% | 4 |
| 2 | 36 | 9.283 ms | 12.318 ms | 26 / 10 | 9.11% | 14 |

跨 seed pooled micro 结果：

- 严格配对事件：109；
- restore 更快 85 次，recompute 更快 24 次；
- always-restore：9.726 ms/事件；
- always-recompute：14.066 ms/事件；
- 逐事件离线 Oracle：9.060 ms/事件；
- Oracle 相比 always-restore 节省 0.666 ms/事件，即 6.85%；
- 三个 seed 的 Oracle 相对收益宏平均为 6.94%，样本标准差为 1.90 个百分点。

pooled 数据上，`candidate_blocks <= threshold` 就重算的最佳 in-sample threshold
为 1，仅比 always-restore 节省 0.051 ms/事件（约 0.52%）。各 seed 的最佳阈值
又分别为 0、4、14，缺乏稳定边界。这进一步证明 block 数只能作为成本模型的
一个输入，不能单独承担在线决策。

### 8.3 端到端与 admission 一致性检查

三个数据 seed 的宏平均 TTFT：

| policy | average TTFT | average P90 TTFT |
| --- | ---: | ---: |
| always-restore | 26.578 ms | 39.111 ms |
| always-recompute | 31.967 ms | 47.749 ms |

restore 在三个数据 seed 中的平均和 P90 TTFT 都优于 recompute，方向与事件账本
一致。与此同时，三次 restore 运行共计（按各运行 unique ID 相加）写入 5710 个
unique blocks，只有 3159 个产生 restore，useful-store ratio 为 55.3%，即 44.7%
没有在观测窗口内产生 external demand。

### 8.4 更新后的研究判断

三数据 seed 已将以下结论从单样本现象提升为重复结果：

1. direct ShareGPT 中 external restore 自然存在；
2. 当前配置下 restore 总体明显优于 recompute；
3. 仍有约 22% 的严格配对事件由 recompute 获胜；
4. promotion Oracle gap 在三个 seed 中均为正，但只有约 5.5%--9.1%；
5. block-only 阈值无法稳定逼近 Oracle；
6. all-block admission 仍产生约 44.7% 未形成恢复需求的写入。

因此下一步不再重复相同配置，而是为 promotion 事件增加决策时的 running/
waiting requests、token budget 和 transfer queue 状态，判断这些信号能否解释
24 个 recompute 胜出事件。若 load-aware 简单模型仍无法取得至少一半 Oracle
收益，应将 promotion 作为系统保护机制而非主要性能贡献，并把主要创新集中在
selective admission 与 negative-utility guard。
