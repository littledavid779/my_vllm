# 第一版生命周期策略：独立校准与六组小规模对照

日期：2026-09-16，23:36–23:44。结果目录：
[lifecycle_calibrated_gpu1_20260916](../results/lifecycle_calibrated_gpu1_20260916)。

## 1. 结论先行

本轮完成 4 次独立服务校准运行和 6 次评测运行，每次 128 请求，共 1280 条有效请求记录。
未改变算法、已冻结参数或预定顺序以改善评测结果。

**没有证明第一版完整策略优于 no-offload。最明确的结果是准入成本模型发生了退化。**

- admission、joint-no-guard、full 的 CPU store 和 load 都是 0。
- promotion-only 确实做出 24 次恢复、19 次重算，全部有完成反馈；但其平均 TTFT
  仍高于 no-offload，P95 也高于 all-block，不能把“两种动作出现”当成“选择正确”。
- full 的平均 TTFT 接近 no-offload，是不写 CPU 的结果，不能归功于 Guard。
- 默认 Guard 未发生状态转换；约 4 秒的评测也不足以验证 30 秒窗口。

## 2. 实验边界与可复现性

- GPU 1：A100-SXM4-80GB，每轮开始前确认无计算进程；运行中约每 2 秒采样。
- 十轮 `gpu_samples.json` 的 `foreign_pids` 均为空。采样不等于绝对无干扰，
  也未隔离整台机器的 CPU、存储等资源。GPU 0 上的既有服务未被修改。
- 最后检查 GPU 1 显存占用 0 MiB、无计算进程；实验端口和服务已退出。
- Qwen3-0.6B、BF16、TP=1、GPU KV 1000 blocks、CPU KV 4 GiB、offloaded block 16 tokens。
- 4096 max model length，4 clients，32 max active conversations，闭环请求，保留 dataset assistant history。
- 校准集 128 会话，与现有 seed 0/1/2 评测集按 ID 和完整消息内容去重；评测使用原 seed 0 内容。
- 校准顺序：runtime seed 0 的 restore → recompute，seed 1 的 recompute → restore。
- 评测顺序预先固定：all-block → admission → joint-no-guard → promotion → no-offload → full。
- 单个评测 runtime seed，每组仅一次；不是多 seed 正式性能结论。

清单、输入与代码哈希见 [manifest.json](../results/lifecycle_calibrated_gpu1_20260916/manifest.json)。
本轮执行前核对了运行脚本、策略、connector 的哈希，与清单一致。

需要特别说明：同一输入文件和 seed 不保证并发闭环的实际请求集合完全相同。
以 `(conversation_id, input_num_turns, input_num_tokens)` 为签名比较多重集合，
all-block / admission / promotion / full 与 no-offload 各有 127/128 条共同记录，
joint-no-guard 为 126/128。因而下面是 arm-level 观察，不是逐请求严格因果比较。

## 3. 独立校准结果

共得到 79 个完整 restore、77 个完整 recompute 和 1399 个完成传输的 store 事件。
拟合只使用校准集，六组评测未反向参与调参。

| 参数 | 冻结值 | 口径 |
| --- | ---: | --- |
| store | 0.587739 ms/block | worker submit-to-complete 总和 / 实际写入 block 总数 |
| restore 固定项 | 8.885299 ms | scheduler wait 的非负仿射最小二乘拟合 |
| restore 斜率 | 0.057392 ms/block | 同上 |
| recompute | 0.022019 ms/token | scheduler wait 的过原点最小二乘拟合 |
| store raw DMA（仅诊断） | 0.073864 ms/block | DMA 总和 / 实际写入 block 总数 |

恢复模型为 `8.885299 + 0.057392 × blocks` ms。
保存成本代理约为 raw DMA/block 的 7.96 倍。两者都不自动等于请求可见的边际阻塞，
尤其 worker 完成时间包括排队/观察等待，各 store 事件的等待区间可能重叠。
本轮没有将代理替换为 DMA 来争取更好的评测结果。

全部校准事件的平均 restore/recompute wait 分别为 10.672 / 13.604 ms。
这两个均值不是严格配对均值，不应直接相减作逐事件收益。
另外保存了严格配对诊断：runtime seed 0 配对 36 组（未匹配 restore/recompute 为 4/2），
seed 1 配对 33 组（未匹配 6/6）。两次 runtime seed 使用同一校准会话集，
不能称为两套独立 dataset seed。

- [冻结参数及样本数](../results/lifecycle_calibrated_gpu1_20260916/calibration_fit.json)
- [runtime seed 0 配对](../results/lifecycle_calibrated_gpu1_20260916/calibration/paired_runtime0.json)
- [runtime seed 1 配对](../results/lifecycle_calibrated_gpu1_20260916/calibration/paired_runtime1.json)

## 4. 六组结果

主表来自 [evaluation_summary.json](../results/lifecycle_calibrated_gpu1_20260916/evaluation_summary.json)。
时间单位 ms；store/load 是累计 block 操作数，不是 unique block 数。

| 策略 | 平均 TTFT | P95 TTFT | store blocks | load blocks | 恢复 / 重算决策 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-offload | 28.620 | 52.576 | 0 | 0 | 不适用 |
| all-block | 30.941 | 57.434 | 2075 | 1497 | 42 / 0 |
| admission-only | 36.687 | 62.454 | 0 | 0 | 0 / 0 |
| promotion-only | 29.957 | 64.083 | 2083 | 1216 | 24 / 19 |
| joint-no-guard | 32.210 | 54.479 | 0 | 0 | 0 / 0 |
| full | 28.724 | 50.524 | 0 | 0 | 0 / 0 |

no-offload 的“不适用”表示没有 CPU-hit promotion 决策，不表示没有 prefill 或重算。

| 策略 | D2H MiB | H2D MiB | 客户端区间吞吐 req/s |
| --- | ---: | ---: | ---: |
| no-offload | 0 | 0 | 33.60 |
| all-block | 3631.25 | 2619.75 | 33.42 |
| admission-only | 0 | 0 | 32.26 |
| promotion-only | 3645.25 | 2128.00 | 32.80 |
| joint-no-guard | 0 | 0 | 32.49 |
| full | 0 | 0 | 33.00 |

此处吞吐为 128 / 请求时间跨度，跨度从最早 `start_time_ms` 到最晚
`start_time_ms + latency_ms`，不含服务启动/编译。它不是最大饱和吞吐。

promotion-only 选择重算的 CPU-hit 候选前缀合计 5120 tokens；这不是全系统全部重算
tokens。它的平均 TTFT 比 all-block 低约 0.984 ms，但 P95 高约 6.649 ms，
相对 no-offload 的平均 TTFT 高约 1.337 ms，不能称为已获得净收益。

admission、joint-no-guard、full 都没有写入，却测到不同 TTFT：这提示应分析运行波动、
请求/调度轨迹与连接器开销，而不能将 full 与 joint 的差别归因于 Guard。

## 5. 为什么准入全部拒绝：可以由公式直接推出

冷启动时，没有 cost bucket 样本，使用冻结初值。每个 block 的重算初值为：

```text
16 × 0.022018834 = 0.352301 ms/block
```

默认 `survival_discount=0.5`，故复用代理 `p <= 0.5`。
即使忽略所有恢复开销，未来单次收益上界也只有：

```text
0.5 × 0.352301 = 0.176151 ms/block
```

小于 store 代理 0.587739 ms/block。因此对任意 n>0 的候选前缀：

```text
G(n) <= n × (0.176151 − 0.587739) < 0
```

在当前初值/默认模型下，拒绝所有候选不是随机失误，而是必然结果。
没有准入 → 没有 CPU hit promotion → 无法获得新的恢复/重算反馈，
也没有已付 store 支出触发负收益 Guard。当前仅在 OFF 状态提供的探测不能打破这个冷启动环。

这不证明保存 KV 永远无收益，证明的是：**当前保存成本代理、单次未来收益模型和冷启动机制
的组合无法产生有效准入。**

## 6. 完整性与尚未验证的内容

- 十轮各 128 条有效请求记录；脚本全部退出成功，未出现 GPU 外来进程采样记录。
- 六组评测均为 0 pending、0 incomplete promotion、0 cost-event dropped、
  0 unmatched transfer、0 store completion failed。
- 新策略的调试事件上限设为 8192，本轮无覆盖。
- promotion-only：24/24 restore 反馈、19/19 recompute 反馈。
- 四个带生命周期控制器的评测组均无 Guard 状态转换。
- `lcuc_decision_time_s` 的累加约 16.7–25.9 ms/组，但它只计部分决策路径，
  不覆盖全部统计、lookup、反馈和连接器开销，不能据此声称系统开销低于某个比例。
- 生成请求完成不等于数值/语义一致性验证。
- full 等零写入组的 `estimated_net_utility_s` 未生成计数，在汇总中为 null，
  不能解读成实测收益等于零或有收益保证。

## 7. 下一步应该收敛，而不是扩大 sweep

1. **先修正保存成本口径。** 区分 DMA、提交到完成、真实关键路径等待以及对其他请求的干扰。
   不应直接把可能重叠的完成等待全额计入逐 block 的请求收益；也不能未经验证直接改用 DMA。
2. **补充 NORMAL 状态下的低预算探测或校准预热。** 避免初值导致全拒绝后永远没有新观测。
   明确探测预算、容量限制和停止条件，独立消融验证。
3. **检查复用价值模型。** 固定 survival 折扣与单次复用假设可能低估共享前缀多次需求；
   需要用生命周期事件检验，而不是调大概率以通过当前测试。
4. **固定实际请求计划再做重复评测。** 消除闭环截断造成的 1–2 条请求差异，
   再比较多 runtime seed 的均值、尾延迟和置信区间。
5. **另设足够长的 Guard 场景。** 当前短 trace 无法证明 30 秒窗口下的限流及恢复有效。

本轮应作为“第一版控制器的成本口径与冷启动问题诊断”，不是正向性能提升实验。
