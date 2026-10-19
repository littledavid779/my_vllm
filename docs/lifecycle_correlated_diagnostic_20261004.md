# 最小请求关联补丁与路径诊断

## 预先固定的设计

承接 [离线分析](lifecycle_request_ttft_analysis_20261004.md) 的数据缺口：旧记录只能确认
会话是否恢复，不能确认具体哪个轮次恢复。本轮仅修改观测与实验工具，不修改 admission、
promotion、Guard、GPU/CPU 容量和成本参数。

- Benchmark 新增 opt-in `--correlate-requests` / `MULTITURN_CORRELATE_REQUESTS=1`。
  请求头使用 `{conversation_id}-turn{message_count}`，CSV 新增 external_request_id。
  原数字 request_id 保留；不开启时请求头仍是 conversation_id。
- 一次请求多个 promotion 分别保留，汇总事件数量、blocks、观测 wait。
  禁止 correlated 模式重试；未知 ID、重复 ID、事件丢失/未完成、计数不符均拒绝分析。
- 通过现有 `VLLM_EXPERIMENTAL_GPU_EVICTION_REQUEST_OWNER_MAP` 将新增外部请求 ID 映射回
  `chatcmpl-{conversation_id}`，避免改变 block owner/shared-count 语义。这个映射写入每组 spec。
- no-offload / selective-2g，两组相反顺序：N→S、S→N，共 4 runs / 2636 requests。
  输入仍是原 128 会话 / 659 请求、4 clients、32 active conversations、runtime seed 0。
- Qwen3-0.6B、GPU 1000 blocks、Selective CPU 2 GiB、always_restore；Guard/probe 关闭。
  不打开全量 cost/block/admission trace，只用现有 bounded controller event。
- 新结果独立目录，不与旧 ID/旧观测结果拼接。冻结代码、输入、成本拟合和配置。
  优先审计映射完整性；主机未独占、2 次重复不足以提供稳定性能结论。

## 验证与执行

初始补丁相关 47 项测试通过，Ruff check/format 通过。测试涵盖 opt-in/默认请求头与 CSV、
多轮和多 promotion、未知/缺失/重复 ID、日志丢失，以及保留原会话 owner 的现有函数行为。

```bash
.venv/bin/python scripts/run_lifecycle_correlated_diagnostic.py \
  --source results/lifecycle_cost_probe_gpu1_20260917 \
  --output results/lifecycle_correlated_diagnostic_gpu1_20261004 \
  --gpu 1 --port 8238
```

Runner 每组启动前检查 GPU 与端口；运行时每 2 秒检查 GPU 外部进程，出现外来进程则该组无效。
只清理自己新建的服务进程组。不操作 GPU 0。DMA 未采集，最终 metrics 标为 null。

## 解读边界

- `no_external_restore` 不是“无 GPU 命中”，也不是“选择了 RECOMPUTE”；本轮 always_restore，
  没有记录到 external load 的请求仍可能 GPU hit 或普通 prefill。
- 按 Selective 实际动作分组是处理后分组，只能定位差异，不是严格因果比较。
- 同一会话/轮次的不同运行不是相同排队/批处理状态；不将观测差值称为逐事件 Oracle。
- summed_restore_wait_ms 是对应请求 promotion 的观测等待之和，不是 DMA，不直接当作
  TTFT 的独立可加项，也不是已测量的重算节约。

## 实测结果

4 组 / 2636 请求全部完成，约 20:39–20:43（本机时间）。同一对内比较，不与旧实验拼接。

| 对比对 | 顺序 | 策略 | Mean TTFT ms | P95 TTFT ms | req/s | Restore 请求 |
|---|---|---|---:|---:|---:|---:|
| 0 | 先 | no-offload | 38.723 | 61.278 | 30.261 | 0 |
| 0 | 后 | selective | 40.553 | 71.319 | 29.557 | 64 |
| 1 | 先 | selective | 37.078 | 65.559 | 30.325 | 75 |
| 1 | 后 | no-offload | 31.218 | 55.053 | 31.489 | 0 |

两对均未优于 no-offload，但不能把两次运行当作稳定性能估计。
GPU 0 外部任务在开始时约 56 GiB / 100% 利用率，运行后段读数为空闲；我们没有干预该任务，
也没有连续采样 GPU 0/主机争用。主机负载可能变化，不能将所有差异归因于策略或外部任务。

### 按 Selective 的真实请求动作分组

这里终于是逐请求真实动作，不再是“会话有 restore”。正值表示 Selective 更慢。
对应 no-offload 请求使用同一 conversation_id/message_count/input tokens/client 连接。

| 对比对 | Selective 请求动作 | 请求数 | 同组 no-offload mean ms | Selective mean ms | 组内差值 ms | 全体 mean 差值贡献 ms |
|---|---|---:|---:|---:|---:|---:|
| 0 | 没有 external restore | 595 | 37.220 | 38.924 | +1.704 | +1.539 |
| 0 | 实际 restore | 64 | 52.697 | 55.692 | +2.995 | +0.291 |
| 1 | 没有 external restore | 584 | 29.416 | 34.966 | +5.550 | +4.918 |
| 1 | 实际 restore | 75 | 45.246 | 53.523 | +8.278 | +0.942 |

两对的总均值差分别为 +1.830 / +5.861 ms。约 84% 的算术差值落在无 external restore 的组，
但该组本身占约 89%–90% 的请求，**不能据此认定该路径单位请求开销更大或是因果主瓶颈**。
实际 restore 组的单位请求平均差值更大，两条路径都不能忽略。

两对合并只作描述：139 条恢复请求平均差值 +5.845 ms，1179 条无恢复请求平均差值
+3.609 ms。这些是跨运行 TTFT 差值，绝不是“139 次严格配对的恢复/重算耗时”。

### 长输入与恢复等待

- 实际恢复请求中，输入 ≥1024 tokens 的分别为 61/64、73/75；同输入跨运行 TTFT 平均差值
  分别为 +3.479、+8.402 ms。
- 未恢复的 ≥1024-token 请求也平均慢 +2.964、+5.237 ms。
- 两组 Selective 的 scheduler-observed restore wait 均值分别是 18.501、18.241 ms；
  这是恢复路径观测值，没有相同运行状态下的实际 recompute 反事实。
- 全局 GPU prefix-hit tokens：第一对 no-offload 32384 / selective 31392，第二对
  no-offload 32256 / selective 33328。总量接近不能证明每请求相同，更不能分离各请求剩余
  prefill 工作量与排队时间。

## 本轮回答了什么，下一步只验证什么

确定回答：观测关联补丁工作正常；当前延迟退化不只出现在直接执行恢复的请求。
尚未回答：是否主要由 store、lookup/策略计算、其他请求的传输竞争、GPU KV 状态改变、
服务端排队或共享主机波动造成。也没有证明 restore 对同一状态一定输给 recompute。

因此不直接修改 admission 分数，不宣称下一步仅优化 DMA 就能解决问题。
下一项验证应优先区分**无恢复请求的剩余计算量变化与共享等待开销**：保留本轮精确 ID，
在更稳定的主机条件下记录请求的排队/首次调度时点、GPU 已命中 tokens、实际 prefill tokens，
再决定是否做写入/元数据路径消融。若恢复请求本身仍异常，再做状态尽量匹配的恢复/重算对照。
这些后续观测或消融本轮没有实现、没有自动启动。

## 审计与文件

- 18 个冻结源码条目、输入和成本拟合 SHA256 检查通过；运行中未修改冻结代码。
- 4×659 个不同外部请求 ID，与逻辑轮次一一对应；每组 owner map 与 CSV 全量对应。
- 64+75=139 次 promotion/completion 均完整关联，未观测到一请求多个 promotion；
  分析器仍显式支持多 promotion，测试覆盖，不将它们静默去重。
- 计数、恢复 block 总数、请求集合/client/input token 签名一致；日志丢失/未完成/失败/
  recompute/Guard/probe 均为零；cost/admission 全量 trace 关闭。
- 原始 CSV 独立复算 mean/P95 与结果一致；4 组共 121 次 GPU 1 采样没有外来进程。
- 结束时 GPU 1 显存为 0 MiB，端口 8238 已关闭。GPU 0 外部任务未被操作。
- 最终相关测试 **59 项通过**，Ruff check/format 与 git diff --check 通过。

文件：

- [诊断 runner](../scripts/run_lifecycle_correlated_diagnostic.py)
- [关联与兼容性测试](../tests/benchmarks/test_lifecycle_request_correlation.py)
- [冻结设计](../results/lifecycle_correlated_diagnostic_gpu1_20261004/manifest.json)
- [独立审计](../results/lifecycle_correlated_diagnostic_gpu1_20261004/audit_summary.json)
- [逐组指标](../results/lifecycle_correlated_diagnostic_gpu1_20261004/evaluation_summary.json)
- [逐请求分组及退化最大的请求](../results/lifecycle_correlated_diagnostic_gpu1_20261004/request_analysis.json)
- [逐请求对照 CSV](../results/lifecycle_correlated_diagnostic_gpu1_20261004/paired_requests.csv)
