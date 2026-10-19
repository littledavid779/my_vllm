# Admission 容量压力与 TTFT 验证（2026-10-03）

## 初始设计（保留；1 GiB 核查后修订见末节）

目标：判断 selective admission 是否能降低 **P95 TTFT**，而非仅减少写入。
继承 [上一轮诊断](lifecycle_admission_diagnostics_20261003.md)，本轮不修改运行时算法。

- Qwen3-0.6B，GPU 1（启动前检查），GPU KV 1000 blocks；固定 128 会话 / 659 请求，
  4 clients、32 active conversations、runtime seed=0、完整执行、固定 client assignment。
- 校准沿用 `results/lifecycle_cost_probe_gpu1_20260917`，不利用本轮结果调参。
- 五组：no-offload、all-block-1g、selective-1g、all-block-4g、selective-4g。
- 每组 3 repeats，共 15 runs / 9885 requests；不重复设置 no-offload 的 CPU 容量。
- 所有 offload 组使用相同 lifecycle wrapper、always_restore、new_write_store_cost=true、
  trace_admission=true；只改变 admission_enabled 和 CPU capacity。
- Guard/cold probes 关闭。Admission 使用既有成本先验和在线更新，不更改概率或阈值。
- `all-block` 在本文件指带同样观测路径的全接纳对照，不是未包装的原生 offload。
- 预先随机顺序（seed 20260917）；仅将 repeat 0 的 all-block-1g 移至首位作压力核查。
  必须同时产生 CPU eviction 与 completed load 才继续。失败则停止，不事后挑选好看容量。

## 容量选择依据

前次 4 GiB all-block 完成 10998 block stores、8658 CPU block evictions，最终驻留 2340 blocks，
但 aggregate lifecycle 尚无 observed lookup-after-eviction。故 **4 GiB 只是大容量对照，
不等于无淘汰或完全无压力**。

本模型一个 offload block 为 1.75 MiB，1 GiB 可容纳约 585 blocks；完整最大输入少于
4096 tokens / 256 blocks，因此并非设置成连单个长前缀都必然无法容纳。
1 GiB 是容量缩小四倍的预设诊断点，不根据 TTFT 挑选。它是否产生有意义的复用竞争由首组验证。

## 观测与判定

主指标每次运行的 P95 TTFT；同时报告 mean TTFT、吞吐、累计 store/load blocks。
结果用各 repeat 值及均值/样本 SD 展示，不把 3 次重复当成泛化或显著性证明。

- 每个容量下比较 selective − all-block；同时和同 repeat no-offload 对比。
- 不能把 D2H 减少、内部 utility 为正、相比旧 joint 更快当成功。
- CPU 淘汰后出现 missing suffix demand、首个缺口关联到此前 CPU eviction，均按事件关联统计。
  重新分配该 block 清除 eviction 关联；重试去重。这不是反事实 saved time。
- 同时保留拒绝后需求与前缀缺口诊断。若有 CPU eviction demand 但 selective 未改善恢复，
  说明当前选择规则未利用该空间，不能判定空间不存在。
- 现有日志没有完整的普通 prefill 重算等待分解，**不能把总 TTFT 差或丢失的恢复数换算成
  已测量的 recompute stall**。若发现 TTFT 改善，再增加必要归因与关闭诊断复验。

审计：请求/turn/client/input token 一致、无传输失败/日志丢失/ID 淘汰/残留 promotion，
无 recompute 决策/Guard transition/cold probe；每组检查源码/输入/校准哈希，监控外来 GPU 进程。
GPU 0 有其他任务，所以只有目标 GPU 空闲，不声称整机独占。

## 命令与结果

```bash
.venv/bin/python scripts/run_lifecycle_fixed_workload.py \
  --suite admission-capacity --gpu 1 --port 8238 --repeats 3 \
  --source results/lifecycle_cost_probe_gpu1_20260917 \
  --output results/lifecycle_admission_capacity_gpu1_20261003
```

启动前，实验配置与诊断关联单测通过后冻结代码。本节后续追加实际结果，不覆盖失败或负结果。

## 1 GiB 核查与正式配置修订

初始目录 `results/lifecycle_admission_capacity_gpu1_20261003` 保留：

- all-block-1g：659 requests，27647 stores，仅 1 loaded block / 1 restore decision；
  mean TTFT 42.247 ms，P95 67.954 ms。
- 已运行的 no-offload：659 requests，mean TTFT 40.668 ms，P95 66.830 ms。
- 下一组 all-block-4g 启动后被主动中断，不纳入结果；只停止本次实验的进程组。

原“eviction>0 且 load>0”门槛过宽。1 GiB 几乎完全破坏了 external reuse，
不适合本次要检验的“仍有恢复的容量竞争”。这是基于机制流量作出的修订，不基于选择哪组 TTFT 更好。
原计划不再继续或恢复，旧数据不并入正式重复。

正式配置改为 **2 GiB vs 4 GiB**（约 1170 vs 2340 CPU blocks）。
新门槛在 2 GiB 数据产生前固定：all-block-2g 必须有 CPU eviction、至少 100 loaded blocks、
至少 10 restore decisions；不满足则再次停止并报告，不自动继续改容量找收益。
这仅是机制可观测门槛，不是性能成功门槛。其他策略、输入、重复次数均不变。

```bash
.venv/bin/python scripts/run_lifecycle_fixed_workload.py \
  --suite admission-capacity --gpu 1 --port 8238 --repeats 3 \
  --source results/lifecycle_cost_probe_gpu1_20260917 \
  --output results/lifecycle_admission_capacity_gpu1_20261003_2g4g
```

正式运行结束后用 `scripts/summarize_lifecycle_capacity.py` 从原始 CSV/配置/日志独立审计。

## 正式结果：15 组已完成

运行时间约 18:13–18:28（本机时间），15 runs / 9885 requests 全部完成。
来源：
[逐组指标](../results/lifecycle_admission_capacity_gpu1_20261003_2g4g/evaluation_summary.json)、
[独立审计汇总](../results/lifecycle_admission_capacity_gpu1_20261003_2g4g/analysis_summary.json)、
[输入与代码 manifest](../results/lifecycle_admission_capacity_gpu1_20261003_2g4g/manifest.json)。

### TTFT 与吞吐

以下为三次运行的均值；± 是运行间样本标准差，不是置信区间。
P95 是三次 run P95 的平均，不是合并所有请求后的 P95。

| 策略 | Mean TTFT (ms) | P95 TTFT (ms) | 吞吐 (req/s) | 累计 store blocks/run | 累计 load blocks/run |
|---|---:|---:|---:|---:|---:|
| no-offload | 37.281 ± 2.878 | 62.921 ± 3.319 | 30.370 | 0 | 0 |
| all-block-2g | 39.024 ± 1.774 | 67.770 ± 1.620 | 29.704 | 24772.0 | 1526.0 |
| selective-2g | 35.652 ± 3.667 | 60.164 ± 6.486 | 30.447 | 7797.7 | 5237.7 |
| all-block-4g | 31.573 ± 0.439 | 56.013 ± 0.536 | 31.362 | 10994.3 | 15154.3 |
| selective-4g | 35.422 ± 1.845 | 60.388 ± 1.525 | 30.562 | 6851.3 | 6461.3 |

**2 GiB 下出现了候选 TTFT 收益，但尚未证明稳定优于 no-offload。4 GiB 下当前策略明确不如全接纳。**

### 按 repeat 的 P95 差值

负值表示 selective 更快，单位 ms。不是相同状态逐请求配对实验。

| 对比 | Repeat 0 | Repeat 1 | Repeat 2 |
|---|---:|---:|---:|
| selective-2g − all-block-2g | −4.831 | −16.177 | −1.811 |
| selective-2g − no-offload | −3.025 | −8.580 | +3.334 |
| selective-4g − all-block-4g | +3.048 | +5.180 | +4.897 |
| selective-4g − no-offload | −8.089 | −0.263 | +0.755 |

- 2 GiB 相对同容量全接纳，P95 三次都更低，run P95 均值降低约 11.22%。
  但平均 TTFT 第三次略慢 0.364 ms，不能写成所有延迟指标均改善。
- 2 GiB 相对 no-offload，P95 两次更好、一次更差；均值降低 4.38%，波动较大。
  第三次 mean TTFT 也更慢 3.433 ms，吞吐低约 2.66%，不得省略。
- 4 GiB 相对同容量全接纳，P95 三次都更差，均值上升 7.81%；吞吐也三次更低。
- 不能用 selective-2g 与 all-block-4g 作等资源性能胜负判断；4 GiB 全接纳本轮仍更快。

### 机制证据：容量竞争下更多有效恢复，容量宽裕时误拒绝占主导的假设

| 指标（每 run 均值） | all-block-2g | selective-2g | all-block-4g | selective-4g |
|---|---:|---:|---:|---:|
| Completed load blocks | 1526.0 | 5237.7 | 15154.3 | 6461.3 |
| Unique evicted blocks 后续 missing suffix demand | 6820.0 | 984.0 | 0 | 0 |
| 首个缺口关联此前 CPU eviction | 370.3 | 18.3 | 0 | 0 |

2 GiB 的恢复 block 数三次分别从 1478/1604/1496 增至 5137/5273/5303，
均值约 3.43 倍。少写同时有更多实际恢复，因此不只是 D2H 数字变小。
CPU eviction 后的缺失需求也明显减少，与“减少容量污染、保留更多可恢复 KV”的解释一致。
但跨运行缓存状态不同，不能据此量化因果 saved time；selective 仍存在大量拒绝后需求。

4 GiB 两种策略的 observed eviction-after-demand 指标均为 0，而 selective 恢复量大幅减少。
结合其三次 TTFT 回归，这支持“无明显有害容量竞争时，当前准入筛选损害复用”的判断，
不代表所有 4 GiB 工作负载都应该全接纳。

观察到的 mean restore wait 在各组也不同（恢复长度与请求集合不同）：不能将组均值直接当
同一个恢复事件的加速/减速。普通 prefill 的完整重算等待仍未直接测量，不能补造该指标。

### 审计、变更与限制

- 本轮只修改实验配置与只读诊断/审计脚本，没有改运行时准入算法。
- 26 项相关测试通过，Ruff 检查通过。
- 原始 CSV 的请求身份、client assignment、输入 token 签名全部一致；TTFT 重算核对通过。
- 输入、校准、代码哈希一致；cost/decision event 丢失、block-ID 淘汰、传输失败、
  unmatched transfer、pending/incomplete promotion、recompute 决策、Guard transition、probe 均为 0。
- 453 次 GPU 采样未发现目标 GPU 上的外来进程；GPU 0 有其他任务，不是整机独占。
- 结束时 GPU 1 显存 0 MiB，端口 8238 已关闭，未停止 GPU 0 的任务。
- no-offload 的 P95 随 repeat 为 66.742、61.257、60.762 ms；环境/时序波动不能忽略。
  虽然预先随机了顺序，实际三轮 selective-2g 都晚于 all-block-2g，顺序未平衡。
  因此不能把方向一致等同于排除时间漂移后的因果收益。
- 一个已用于诊断的数据集、同一 runtime seed、三次重复；不能宣称泛化、显著性或最终成功。

### 下一步（尚未执行）

先用平衡/反转先后顺序、关闭详细 trace 的对比复核 **2 GiB 的 P95 改善是否仍在**，
保留 no-offload 和全接纳两种基线。若成立，再研究用“淘汰后需求/资源压力”控制准入强度；
不能直接把容量写成固定阈值，也不能宣称这种压力自适应机制已经实现。
4 GiB 的负结果说明 always-on selective 不是正确的默认路径。
