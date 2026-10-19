# 2 GiB TTFT 确认：顺序平衡、关闭详细日志

## 动机与预先固定的设计

承接 [容量实验](lifecycle_admission_capacity_20261003.md)：2 GiB selective 三次 P95
优于同容量全接纳，但对 no-offload 仅两次获胜；运行顺序未平衡，且开启了详细 trace。
本轮不改运行时算法、不改成本参数，只检验这个候选收益是否在更严格对比中保留。

- 三组：no-offload、all-block-2g、selective-2g；每组 6 次，共 18 runs / 11862 requests。
- 使用三个策略的全部 6 种排列，按固定 seed 20261003 打乱排列顺序。
  每个策略在每个位置出现两次，每一对策略两种先后关系各出现三次。
- 固定同一 128 会话 / 659 请求、runtime seed 0、4 clients、32 active conversations、
  Qwen3-0.6B、GPU 1000 blocks、CPU 2 GiB、GPU 1。
- 保持 always_restore、new_write_store_cost=true，Guard/cold probes 关闭。
- `track_cost_events=false`、`trace_admission=false`；bounded controller logs、aggregate
  lifecycle counters 和成本反馈仍保留。因此这是关闭详细 trace，不是绝对零 instrumentation。
- 主指标仍是 run P95 TTFT，同时报告 mean TTFT、吞吐、累计 store/load blocks。
- DMA 字节/时间未采集，应为 null/不可用。复用的底层 runner 控制台可能先打印空事件求和的 0，
  新 runner 在最终 metrics/summary 中明确改为 null；不得使用控制台占位值作传输结论。
- 三组全为新服务进程。只对同一 round 比较，不与前次有 trace 实验直接拼表归因。

## 执行与审计

```bash
.venv/bin/python scripts/run_lifecycle_ttft_confirmation.py \
  --gpu 1 --port 8238 \
  --source results/lifecycle_cost_probe_gpu1_20260917 \
  --output results/lifecycle_ttft_confirmation_gpu1_20261003
```

输入、校准、代码、配置、顺序均写入 manifest。每次检查请求/turn/client/input token 一致性，
store failure、pending/incomplete promotion、Guard/probe/recompute 为零，详细 trace 关闭。
监测 GPU 1 的外来进程，不触碰 GPU 0 的任务。

成功判定不能只看一个平均数：报告六轮相对两个基线的 P95 差值，检查顺序分组、mean TTFT
和吞吐退化情况。即使方向改善，单一已检查数据集和共享主机仍不支持泛化或严格因果保证。
运行顺序平衡减少时间趋势偏差，但不能完全消除瞬时争用；CPU host 并未独占。

本轮运行时算法未更改；历史容量实验代码/结果不覆盖。

## 结果：候选收益未通过对两个基线的确认

18 runs / 11862 requests 全部完成，约 18:40–18:58（本机时间）。
来源：[逐组指标](../results/lifecycle_ttft_confirmation_gpu1_20261003/evaluation_summary.json)、
[汇总](../results/lifecycle_ttft_confirmation_gpu1_20261003/analysis_summary.json)、
[独立审计](../results/lifecycle_ttft_confirmation_gpu1_20261003/audit_summary.json)、
[运行顺序及配置](../results/lifecycle_ttft_confirmation_gpu1_20261003/manifest.json)。

### 六轮汇总

± 是六次 run 之间的样本 SD，不是置信区间；P95 列为 run P95 的均值。
store/load blocks 是各 run 累计操作量的平均，不是 unique blocks。

| 策略 | Mean TTFT (ms) | P95 TTFT (ms) | 吞吐 (req/s) | Store blocks/run | Load blocks/run |
|---|---:|---:|---:|---:|---:|
| no-offload | 32.495 ± 2.823 | 57.966 ± 6.555 | 31.122 | 0 | 0 |
| all-block-2g | 36.493 ± 3.388 | 62.885 ± 4.334 | 30.234 | 24815.8 | 1621.0 |
| selective-2g | 35.060 ± 3.296 | 59.548 ± 4.331 | 30.710 | 7908.0 | 5267.8 |

相对 all-block，selective 的 P95 均值低 5.31%，mean TTFT 低 3.93%，吞吐高 1.57%。
但相对 no-offload，P95 均值高 2.73%，mean TTFT 高 7.89%，吞吐低 1.32%。
**按“降低 TTFT，并胜过 no-offload 与 all-block”这个目标，当前版本仍未达到要求。**

### 每轮 P95 与差值

| Round | no-offload | all-block | selective | selective − no-offload | selective − all-block |
|---|---:|---:|---:|---:|---:|
| 0 | 55.907 | 61.550 | 53.652 | −2.255 | −7.898 |
| 1 | 58.305 | 60.631 | 57.211 | −1.093 | −3.419 |
| 2 | 67.653 | 61.108 | 66.065 | −1.589 | +4.956 |
| 3 | 59.229 | 62.337 | 62.394 | +3.165 | +0.057 |
| 4 | 47.331 | 60.093 | 57.902 | +10.571 | −2.191 |
| 5 | 59.370 | 71.590 | 60.064 | +0.694 | −11.526 |

单位 ms。对 no-offload，P95 3/6 轮更低，mean TTFT 仅 1/6 轮更低。
对 all-block，P95 4/6 轮更低，其中另一次差值 +0.057 ms 接近相同，不能解读为有意义的回归。

### 顺序分组仍有明显差异

| selective 相对基线的 P95 差值均值 (ms) | selective 先运行（3轮） | selective 后运行（3轮） |
|---|---:|---:|
| 对 no-offload | +4.214 | −1.050 |
| 对 all-block | +0.941 | −7.615 |

顺序已平衡，但这些子组表现差异较大。它提示时间变化/系统状态仍值得关注，
**不能据此证明差异全部由顺序引起**，也不能把后运行子组的优势当成稳定算法收益。
同一 round 不是相同请求在相同 runtime 状态上的严格反事实配对。

### 本轮支持与不支持的判断

支持：

- selective 确实少写且恢复更多，容量管理行为发生了实质变化。
- 它相对一个表现较差的 all-block 基线有平均改善，但仍存在运行间回归。
- 增加实际恢复不足以保证 TTFT 更好；no-offload 仍是必须保留的强基线。

不支持：

- “2 GiB 下稳定加速约 11%”：此前三轮有详细 trace 的候选收益没有原样通过本轮确认。
- “压力感知准入一定能解决问题”：该机制尚未实现，当前结果不证明只加一个压力开关就够。
- “恢复一定比重算划算”：本轮固定 always_restore，没有逐事件测量另一分支。
- 将未采集的 DMA 字节/时间、普通 prefill 重算等待解释为零或已测量的节约。

### 审计与工程收尾

- 独立读取原始 CSV，复算 mean/P95，与每组结果一致；所有请求/turn/client/input token 签名一致。
- 六个排列完整，每个策略各位置两次、每对先后关系各三次；18 组配置与 manifest 完全一致。
- 详细 cost/admission trace 关闭；store 失败、incomplete/pending promotion、recompute 决策、
  Guard transition、cold probe、controller event 丢失均为 0。
- 531 次 GPU 采样没有外来 GPU 1 进程；GPU 0 有外部任务，主机并未独占，不能排除 CPU/内存争用。
- 结束时 GPU 1 显存 0 MiB、端口 8238 关闭，没有操作 GPU 0 上的任务。
- 最终相关测试 29 项通过，Ruff check/format 通过。
- 运行中代码哈希始终冻结，独立审计时与 manifest 一致。运行结束后仅将 runner 中两条过长的
  说明字符串拆行以通过 Ruff E501；已验证拆行前后 AST 一致。原版保存在
  [frozen_runner.py](../results/lifecycle_ttft_confirmation_gpu1_20261003/frozen_runner.py)，
  SHA256 与原 manifest 一致。历史审计结果指修正前冻结版本，当前源码哈希会有这个样式差异。

### 下一步（未执行）

不继续扩大容量/阈值 sweep，也不立即加压力控制器。优先在匹配的前缀、上下文长度与负载下，
核查恢复相对重算的请求可见成本，并分清 store/元数据处理/调度等待是否抵消复用收益。
之后只针对被证据支持的瓶颈修改，再以 no-offload 为必要基线复验。
若要给出性能结论，应争取更稳定的主机条件，并增加未用于调参的 workload；不能删去本轮负结果。
