# Promotion 同口径耗时标签：第一阶段实现与实验

日期：2026-10-09。状态：采集链路和四组校准运行完成；新预测模型、误差保护决策尚未接入。

## 1. 这一步解决什么

延续 [恢复反馈路径诊断](lifecycle_restore_feedback_20261007.md)，先让 RESTORE 与 RECOMPUTE 使用相同的预测目标，再讨论如何选择动作。

旧在线反馈中，RESTORE 结束于 connector 观察到完成，RECOMPUTE 使用计算进度条件。它们不能直接当作两个相同口径的首 token 成本。此次新增独立标签：

```text
T_action = scheduler_first_output - decision_recorded
```

两种动作都结束于 scheduler 处理到首 token 输出；起点是动作选定、特征收集后记录决策的时刻。它不包含此前的排队或动作选择耗时，也不是客户端端到端 TTFT。已有在线 estimator / guard 的学习逻辑没有被替换。

本次只验证数据基础，不宣称 TTFT 改善，也不宣称已超过 no-offload。

## 2. 代码实现

- [promotion_trace.py](../vllm/v1/kv_offload/promotion_trace.py)：默认关闭、容量 4096 的有界决策记录，包含丢弃计数与单调事件 ID。
- [offloading_connector.py](../vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py)：通过 `track_promotion_ttft` 开启；在两类动作处记录 request ID、时间和决策时特征。
- [cpu.py](../vllm/v1/kv_offload/cpu.py)：公开已有计算得到的 offloaded block 字节数，用于统一 blocks / bytes 单位；不改变分配与传输行为。
- [analyze_promotion_ttft_labels.py](../scripts/analyze_promotion_ttft_labels.py)：关联已有 request path，验证时间、动作、前缀、字节数与完整性。重复决策或首 token 前发生抢占的请求显式排除；缺失、溢出和不一致记录报错。
- [run_promotion_ttft_calibration.py](../scripts/run_promotion_ttft_calibration.py)：固定 all-block admission，只切换强制 RESTORE / RECOMPUTE，以 R/C/C/R 顺序采集，不拟合、不推导反事实胜负。

输入特征包括候选 blocks / tokens / bytes、GPU 命中前缀、prompt 长度、两条路径的剩余 prefill、tracked requests，以及 pending load / store requests、blocks 和 bytes。

其中 pending 是 **scheduler 已知但尚未结算的传输工作**，不是硬件 PCIe 队列深度或精确剩余字节；tracked requests 也不是 GPU 利用率或实际 batch token 数。不能在论文中把这些代理量写成硬件实测负载。

## 3. 实验配置与可复现性

- 模型：`/public/model_zoo/Qwen3-0.6B`；GPU0；端口 8238。
- CPU tier：2 GiB，LRU、all-block；admission controller 和 guard 关闭，cold probe 关闭。
- 使用原有 calibration split：128 个会话、648 个请求 / 运行。与 evaluation split 的会话 ID 无交集；但 calibration 并非新采集、从未使用的数据。
- 开启 request path 与新决策 trace；关闭 full cost events、admission trace 和 restore phase trace。
- 冻结 25 个相关代码文件及输入 / fit 的 SHA256；最终复查全部一致。此处旧 fit 仅用于原有配置，动作强制指定，尚未训练新模型。
- 四组 request 集合及 client assignment 一致；实际到达时刻、缓存驻留和执行状态仍可能因动作不同而变化。
- 共 120 次 GPU 采样，未检测到其他进程；2 秒采样不保证排除采样间隙的短暂干扰。
- 运行结束 GPU0 显存归零，端口 8238 已释放。

结果入口：[manifest](../results/promotion_ttft_calibration_gpu0_20261009/manifest.json)、[summary](../results/promotion_ttft_calibration_gpu0_20261009/summary.json)。每组目录中保存 `promotion_ttft_labels.json`、`metrics.json`、`gpu_samples.json` 和原始运行统计。

复现时使用新的输出目录，脚本拒绝覆盖已有结果：

```bash
.venv/bin/python scripts/run_promotion_ttft_calibration.py \
  --source results/lifecycle_cost_probe_gpu1_20260917 \
  --output results/promotion_ttft_calibration_new_run \
  --gpu 0 --port 8238
```

## 4. 采集结果

下表是每个运行自身的观测分布，**不能把行间均值差解释为同一事件的因果收益**。

| 顺序 | 强制动作 | 完成请求 | 有效标签 | 排除标签 | 决策到 scheduler 首 token 均值 | 全部请求客户端 mean TTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | RESTORE | 648 | 49 | 0 | 27.460 ms | 39.629 ms |
| 2 | RECOMPUTE | 648 | 45 | 0 | 23.040 ms | 38.629 ms |
| 3 | RECOMPUTE | 648 | 52 | 0 | 26.543 ms | 42.723 ms |
| 4 | RESTORE | 648 | 43 | 0 | 27.588 ms | 34.822 ms |

合计 2592 个完成请求，189 条有效决策标签：RESTORE 92 条、RECOMPUTE 97 条。其他 2403 个请求没有 CPU 候选决策，不是丢失标签。两次 RESTORE 实际 load 1429 / 1336 blocks；两次 RECOMPUTE load 均为零。

最终另行核对：每个标签对应的 `first_compute_cached_tokens` 等于动作预期；完整前缀命中仍保留最后一个 prompt token 的计算。所有请求的计算量 / 缓存量守恒检查通过。

日志中的旧 `d2h_bytes=0` / `h2d_bytes=0` 不能解释为没有流量：full cost trace 已关闭。最终 per-arm metrics 和 summary 将这些未测字段记为 `null`。

## 5. 能说明什么，不能说明什么

已证明：两种动作能产出相同时间端点、带决策时特征、可关联实际请求的标签；可作为后续成本模型的数据入口。

尚不能证明：RESTORE 或 RECOMPUTE 整体更优、新策略降低 TTFT、超过 no-offload、或同一个事件在哪条路径下更快。即使跨运行 request ID 相同，也不等于 GPU / CPU 缓存和队列状态相同。此处 **189 条标签不是 189 个 paired events**，不能与此前 109 个严格配对事件混用。

主要覆盖缺口：

- 必重算组的 pending load blocks 全为 0；恢复组范围为 0–146。这批数据不能识别 H2D 已有积压时 RECOMPUTE 的成本。
- tracked requests 仅覆盖 0–3，不能据此推广到高并发拥塞状态。
- 总计 189 条标签、重复会话和同一 workload，样本量有限，不适合高维复杂模型；未测 trace 本身的端到端开销。

## 6. 下一步的最小实施顺序

1. 先按会话分组拆分拟合 / 校准数据，检查两个动作的共同状态覆盖；不要把同一会话的多轮和重复运行随机拆到训练与验证两侧。先试小型查表 / 分段模型，报告误差和未覆盖比例，不先拟合复杂模型。
2. 对缺少另一动作样本的状态，补充有界的混合动作校准运行：决策前记录状态，按预先固定的随机方案选 R/C，保持 admission 不变。这能补状态覆盖，但每次仍只有实际动作标签，不产生精确反事实真值。
3. 只有覆盖和误差可接受后，接入候选规则：`T_C_hat - T_R_hat - e_R - e_C > delta` 才恢复。`e_R/e_C` 是校准误差裕量，不冒充形式化安全保证；未覆盖状态单独标记并使用保守回退。
4. 冻结模型和阈值后，在未用于本轮调参的 workload 上比较 no-offload、always-restore 和新策略，优先看端到端 mean / P95 TTFT。旧 evaluation 已用于历史诊断，若要声称未见分布泛化，还需新的会话或负载。
5. Promotion 单独成立后再联动 selective admission。当前即使回退 RECOMPUTE，已发生的 D2H 和缓存状态成本仍在，因此不能保证回退等于 no-offload。

## 7. 验证

针对新 trace、标签解析、已有 restore trace / request path 的单测：33 passed；已有 connector promotion / cost / admission / recompute 回归子集：16 passed。

本次涉及的 7 个 Python 文件通过 Ruff 0.14.0 check / format --check，`git diff --check` 通过。pre-commit 自动安装 Ruff 环境因代理缺少 SOCKS 支持失败；随后使用本机已缓存、与 hooks 指定版本相同的 Ruff 离线检查通过，未将完整 hooks 声称为通过。

```bash
.venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_promotion_ttft_trace.py \
  tests/benchmarks/test_promotion_ttft_labels.py \
  tests/v1/kv_connector/unit/test_restore_path_trace.py \
  tests/benchmarks/test_lifecycle_request_path_analysis.py -q

.venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_offloading_connector.py \
  -q -k 'promotion or cost or admission or recompute'
```
