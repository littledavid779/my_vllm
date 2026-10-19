# 生命周期策略第二轮：直接开销观测与有限冷启动探测

## 目的

上一轮独立校准将 store 的 worker submit-to-complete 时间全额计入成本，
使准入对所有候选均为负，见 [第一轮结果](lifecycle_calibrated_results_20260916.md)。
本轮不修改历史数据，也不把 DMA 时间直接改名为请求等待。
分别修正观测口径、加入可单独关闭的有限探测，再作 2×2 消融。

## 1. 保存时间分层

代码检查确认：D2H 在下一 engine step 开始时提交，传输流先等待生成 KV 的计算流，
worker 又在后续轮询中观察完成。submit-to-complete 不是同步阻塞的同义词。

新增、默认仅在 `track_cost_events=True` 时启用的观测：

| 字段 | 口径 |
| --- | --- |
| `submit_to_complete_s` | 保留旧的提交至 worker 观察完成时间 |
| `dma_time_s` | 保留 CUDA event DMA 时间 |
| `store_host_submit_s` | worker 调用 transfer_async 的 CPU wall time |
| `store_blocking_wait_s` | 抢占时显式 worker.wait 的 CPU wall time 分摊 |
| `store_host_service_s` | scheduler prepare_store wall time + worker 提交及同步等待开销 |

一个抢占 wait 可能等待多个 jobs，仅计一次 elapsed，再按其 block 数分摊，
不把同一等待全额重复记入每个 job。分摊是记账约定，不是逐 job 的因果测量。
多 worker 观测使用最大值而非简单求和；本轮实际验证范围仍为 TP=1。

**直接 host 开销也不是完整 request-visible 边际成本。** 它不涵盖异步 DMA 与计算的
争用、释放 GPU blocks 的延后、CPU cache 容量机会成本、其他请求受到的影响等。
零同步等待不表示 offload 免费。旧 trace 缺少新字段时不按零补齐。

校准入口增加 `--store-cost-source worker-completion|host-service`；默认仍为旧口径。
新口径必须使用新的实验目录和新的观测。校准结果同时保留两种 store/block 初值，
供同一轮内对照，而不是比较不同日期测到的两个数。

## 2. NORMAL 状态的有限冷启动探测

默认 `cold_probe_every=0`，即关闭；不改变已配置的旧策略行为。
开启后，只在以下条件同时满足时绕过收益门槛：

1. Guard 处于 NORMAL，收益准入已开启，当前候选被收益规则全部拒绝。
2. batch 至少有 `cold_probe_min_blocks`，避免只探测很短的零散 decode 增量。
3. 达到 `cold_probe_every` 个合格被拒绝 batch 的频率门槛。
4. 剩余 block 和估计成本预算允许一个至少达到最小长度的前缀。

探测始终选取本次 batch 的连续前缀，不承诺此前所有祖先仍驻留 CPU。
`cold_probe_max_blocks` 控制单批上限；`cold_probe_budget_blocks` 和
`cold_probe_budget_s` 控制本次控制器生命周期内的累计预算。

- 仅实际新预留写入的 blocks 扣预算；已存在或预留失败不扣。
- 成本预算包含配置的 store + capacity 估计，不是实际时间上限。
- 已预留后发生传输失败不退预算，采用保守处理。
- `reset=True` 不重置频率或预算；不能借统计清零无限探索。
- 不绕过 THROTTLE。已有 OFF 恢复探测保持独立；上述总预算只约束 NORMAL 冷启动探测。
- 预算耗尽不会自行补充；这是有限 warmup，不是完整的长期非平稳探索算法。

新增事件 `reason=cold_probe`，单独记录探测次数、实际写入块和累计成本。
被拒绝事件的 `gain_s` 为 null，并记录扫描到的 `max_candidate_gain_s`；
探测事件记录其实际估计负收益，不再用零掩盖绕过门槛的事实。

## 3. 实验设计（获取结果前固定）

与第一轮相同的无会话重复 calibration/evaluation 输入、模型、容量和闭环配置。
先做 4 组新的 all-block restore/recompute 校准，再比较：

| 组 | store 成本 | 收益准入 | 冷启动探测 | Guard |
| --- | --- | --- | --- | --- |
| no-offload | 无 | 无 CPU tier | 无 | 无 |
| all-block | 无收益筛选 | 全部 | 无 | 无 |
| completion-no-probe | worker 完成等待 | 开启 | 关闭 | 关闭 |
| completion-probe | worker 完成等待 | 开启 | 开启 | 关闭 |
| host-no-probe | 直接 host 开销 | 开启 | 关闭 | 关闭 |
| host-probe | 直接 host 开销 | 开启 | 开启 | 关闭 |

四个实验组使用同一套 restore/recompute 初值及 cost-aware promotion，
Guard 一律关闭，以免混淆归因。探测组预先固定：every=8、min=16、max=64 blocks、
总预算=128 blocks / 0.1 s。没有根据评测收益调整。

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_lifecycle_calibrated_smoke.py \
  --gpu 1 --port 8238 --output results/lifecycle_cost_probe_gpu1_20260917 \
  --store-cost-source host-service --evaluation-suite cost-probe-ablation
```

结果清单：[manifest.json](../results/lifecycle_cost_probe_gpu1_20260917/manifest.json)。
相同输入与单 runtime seed 下的闭环截断仍可能造成实际请求集合不同，必须审计；
本轮仍是小规模诊断，不能据单次 TTFT 宣称泛化或显著收益。

## 4. 验证记录

- 92 项 CPU/connector/校准测试通过；Ruff 0.14.0 检查通过。
- 覆盖预算与频率、统计 reset、Guard 优先级、失败预留不扣费、已存在块不重复扣费。
- 用受控时钟验证一次 8 ms 同步等待只分摊为总计 8 ms，而非按 job 数重复记账。
- 用延迟轮询验证 1 s submit-to-complete 不会污染约毫秒级的 host service 时间。
- 2×2 配置测试确保只改变指定成本来源/探测开关，restore/recompute 模型不被改动。

## 5. GPU 结果：机制打通，性能结论仍待复验

2026-09-17 00:18–00:25，GPU 1，Qwen3-0.6B、TP=1、1,000 GPU blocks、
4 GiB CPU tier、32 active conversations / 4 clients；4 组校准 + 6 组评测，
每组 128 requests，共 1,280 requests。旧结果目录未修改。

### 5.1 同一批数据的成本分层

冻结参数见 [calibration_fit.json](../results/lifecycle_cost_probe_gpu1_20260917/calibration_fit.json)。

| 成本观测 / 初值 | 数值 |
| --- | ---: |
| Store worker submit-to-complete / block | 0.584583 ms |
| Store raw DMA / block | 0.073868 ms |
| Store direct host service / block | 0.034282 ms |
| Restore 固定项 | 7.374900 ms |
| Restore 每 block 项 | 0.074951 ms |
| Recompute 每 token 项 | 0.020222 ms |

校准使用 80 个 restore、82 个 recompute 和 1,394 个有效 store 事件。
全部有效 store 均有新增 host 观测字段；另有 7 个 no_transfer 事件，stored_blocks=0，
不用于拟合。校准中显式 preemption blocking wait 均为零，意味着这一项没有被
真实负载激活；其分摊正确性目前由单元测试覆盖。

这些是模型初值，不是逐事件常数，更不是生命周期净收益的测量。
直接 host service 可以小于 DMA，因为两者测的是不同资源上的时间。

### 5.2 六组对照

原始汇总见 [evaluation_summary.json](../results/lifecycle_cost_probe_gpu1_20260917/evaluation_summary.json)。
下表 store/load 为累计 block 传输数量，不是 unique blocks。

| 策略 | 平均 TTFT (ms) | P95 TTFT (ms) | Store blocks | Load blocks | Restore / Recompute 决策 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-offload | 27.028 | 49.771 | 0 | 0 | — |
| all-block | 33.743 | 61.094 | 2,073 | 1,506 | 43 / 0 |
| completion-no-probe | 28.643 | 53.241 | 0 | 0 | 0 / 0 |
| completion-probe | 29.482 | 55.555 | 128 | 76 | 2 / 4 |
| host-no-probe | 30.292 | 53.432 | 1,104 | 283 | 3 / 0 |
| host-probe | 26.528 | 45.351 | 1,155 | 365 | 4 / 4 |

决策数表示 CPU-hit 路径上的选择；no-offload 的零计数不表示没有重算。
这些也不是严格配对的 restore/recompute 胜负事件，不能替代此前 109 paired events。
本轮四个策略组均关闭 Guard；日志仍显示 NORMAL，这是初始状态，不代表保护机制通过验证。

### 5.3 可以支持的机制结论

1. **成本口径改变了准入行为。** completion-no-probe 的 362 次准入全部拒绝；
   host-no-probe 有 15 次正收益准入、347 次拒绝，无需探索也产生真实恢复。
   因此旧口径下的零恢复不能解释成工作负载没有复用。
2. **有限探测打通了“全部拒绝时没有反馈”的路径。** completion-probe 的写入
   全部来自探测，产生 6 次 CPU-hit 决策和 76 blocks 实际恢复，但没有转成盈利准入。
   探测获得信息不等于探测必然有净收益。
3. **预算约束生效。** completion-probe 实际探测 128 blocks，估计成本 74.827 ms；
   host-probe 探测 114 blocks，估计成本 3.908 ms，剩余 14 blocks 小于最小探测批次 16。
   两组均未超过 128 blocks / 100 ms 估计成本预算。host-probe 还有正常收益准入，
   因此不能把其全部 365 load blocks 归因于探测。
4. **不能把少写等同于保留全部有效恢复。** host-probe 相对 all-block 的 D2H
   从 3,803,971,584 降至 2,119,434,240 bytes（−44.28%），但 load blocks 也由
   1,506 降至 365。减少了无效还是有价值的写入，需要进一步逐生命周期分析。

### 5.4 不能支持的性能结论

host-probe 相对 no-offload 的平均 TTFT 低 1.85%、P95 低 8.88%，只是单次观测。
host-no-probe 仍慢于 no-offload，尽管内部 estimated_net_utility 为正；
这再次说明内部成本账本的正值不等于实测端到端收益。

还存在以下限制：

- 仅一个 evaluation runtime seed、一个模型/容量，尚无重复运行和置信区间。
- calibration 与 evaluation 会话 ID / 完整内容无交叉，但 evaluation seed 0 已在
  之前研究中使用；这是迭代诊断集，不是新的未见最终测试集。
- 闭环执行在 128 requests 截断，按 `(conversation_id, input_num_turns, input_num_tokens)`
  比较实际请求多重集合：all-block、completion-no-probe、host-no-probe 相对 no-offload
  各有 1 条替换；两个 probe 组各有 4 条替换。这不是严格同请求、同到达序列回放。
- host 成本缺少异步争用和容量机会成本；不能宣称已经获得完整 request-visible Cs。
- 改进组的 Promotion 反馈仅 8 次，不足以判断在线模型精度和长期收益。
- 采样未发现 GPU 1 的其他计算进程，不等于独占整台机器；CPU/内存等仍可能受其他任务影响。

因此本轮结论是：**新观测和有界探索均已实现且影响真实决策，尚未证明稳定净性能收益。**

### 5.5 审计与复现

```bash
.venv/bin/python scripts/audit_lifecycle_smoke.py \
  results/lifecycle_cost_probe_gpu1_20260917
```

审计已通过：10 组请求完整、GPU 采样 foreign_pids 均为空、成本事件无丢弃/未匹配、
无 pending/incomplete promotion、探测未超预算，三份冻结代码 hash 与 manifest 一致。
另行检查 offload store 预留/完成失败均为零。结束后 GPU 1 和端口 8238 已释放。
审计脚本为只读；请求签名核对不等同于 token 内容或到达顺序一致。

## 6. 下一步（不再根据这次 TTFT 调参）

1. 先固定有限的会话/轮次集合并全部执行完成，消除“谁先完成谁多提交一轮”的截断偏差；
   或使用明确的到达 trace。保留每个会话内部依赖，记录排队和服务时间。
2. 冻结当前参数，对同一工作负载做多次交错顺序重复，再引入独立未见会话作为验证集。
   重点比较 no-offload / all-block / host-no-probe / host-probe，保留旧成本组作诊断。
3. 逐请求/生命周期核对被拒绝前缀、探测块的后续 external demand、实际节省与成本误差，
   区分“少做必要恢复”和“避免负收益恢复”。
4. 只有稳定收益和成本误差范围明确后，再做长时间、负载变化下的 Guard 开关消融。
   本轮不追加调度器或 progressive restore 功能。
