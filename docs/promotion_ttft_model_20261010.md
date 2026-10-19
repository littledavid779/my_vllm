# Promotion 成本模型与混合动作校准

日期：2026-10-10。

## 当前结论

数据采集和强制动作路径已经运行验证；**新的误差保护型 Promotion 策略尚未接入在线服务，更没有证明 TTFT 超过 no-offload**。本轮是 [同口径标签工作](promotion_ttft_labels_20261009.md) 的下一步，不是性能提升实验。

## 已实现的方法

[fit_promotion_ttft.py](../scripts/fit_promotion_ttft.py) 按会话 ID 的固定 SHA256 分桶，使用约 60% / 20% / 20% 的会话进行拟合 / 误差校准 / 验证。同一会话的所有轮次、动作组、重复运行都留在同一分区；不能随机拆散事件来制造较低的验证误差。

每个动作分别比较固定均值基线与一个小型 ridge 模型：

- 目标：决策记录到 scheduler 首 token 的耗时。
- 特征：candidate tokens、恢复后剩余 prefill、GPU hit tokens、tracked requests、pending load / store blocks。
- 特征先 log1p，再用训练分区均值 / 方差标准化；固定 ridge alpha=10，不根据验证结果调参。
- 误差裕量来自校准分区的 P90 绝对误差，不是严格的置信区间或安全保证。
- 另做粗粒度状态覆盖检查：候选大小、tracked requests、是否有 pending load / store；每种动作至少 3 个训练样本才计入“粗分桶有支持”。这不是完整状态等价的证明。
- 输出只供离线诊断，明确 `deployment_approved=false`；没有让 serving 自动读取这个文件。

## 原有 189 条标签的初步结果

| 动作 | 训练 / 校准 / 验证事件数 | 固定均值验证 MAE | Ridge 验证 MAE | 校准 P90 绝对误差裕量 |
| --- | ---: | ---: | ---: | ---: |
| RESTORE | 61 / 14 / 17 | 10.924 ms | 8.129 ms | 12.517 ms |
| RECOMPUTE | 69 / 10 / 18 | 6.696 ms | 3.559 ms | 10.065 ms |

两种动作的模型均优于固定均值，但 RESTORE 验证误差仍大，且只有 17 个验证事件。不能把“误差变小”写成“策略带来时延收益”。两项独立误差裕量相加约 22.58 ms，也可能使恢复决策过于保守。

完整原始报告：[forced_only.json](../results/promotion_ttft_model_20261009/forced_only.json)。它使用旧四组运行，没有新增 GPU 运行。

覆盖检查发现：pending load 非零时 RESTORE 有 31 条，RECOMPUTE 为 0。强制重算运行不会产生恢复积压，因此只重复 always-recompute 不能自然填补这个状态缺口。

## 补采机制，不是收益策略

新增 [promotion_calibration.py](../vllm/v1/kv_offload/promotion_calibration.py) 与实验开关 `promotion_policy=calibration_mixed`：

1. 实验启动前，为每个稳定 external request ID 用固定随机 seed 独立分配 RESTORE / RECOMPUTE，分配概率各 0.5。
2. 整张 assignment map 写入 manifest 和配置，运行时不根据时延、预测值或未来结果挑动作；同一 request 的重试保持原动作。
3. CPU 候选命中后执行指定动作；实际动作与 map 不符则审计失败。缺少分配的请求直接报错，不悄悄切换默认动作。
4. 要求开启 decision trace，关闭自适应 admission、guard 与选择性 store threshold。原有 always-restore、always-recompute、cost-aware 默认路径不启用此实验开关。

这允许一条运行轨迹中同时有恢复积压和重算事件。它仍不产生同一事件的两条真实反事实标签；请求依赖、共享缓存和被延迟的决策仍需注意，不能据此直接宣称因果胜率。

本轮限定两个 assignment seeds：17、29；不是 17 / 29 轮会话。负载 runtime seed 仍为 0，输入与 10 月 9 日一致，每组 648 请求；固定 all-block、CPU 2 GiB、GPU 1000 blocks、4 clients、本地 Qwen3-0.6B。

```bash
.venv/bin/python scripts/run_promotion_ttft_calibration.py \
  --source results/lifecycle_cost_probe_gpu1_20260917 \
  --output results/promotion_ttft_mixed_gpu0_20261010 \
  --gpu 0 --port 8238 --mixed-seeds 17 29
```

结果入口：[混合运行目录](../results/promotion_ttft_mixed_gpu0_20261010)。脚本拒绝覆盖已有目录，复现应更换输出路径。

## 混合实验结果与最终离线检查

| Assignment seed | 完成请求 | RESTORE 标签 | RECOMPUTE 标签 | Pending load 非零时 R / C |
| --- | ---: | ---: | ---: | ---: |
| 17 | 648 | 20 | 31 | 3 / 7 |
| 29 | 648 | 24 | 21 | 5 / 4 |

新增 1296 个完成请求、96 条有效标签，排除数为 0；实际 load blocks 分别为 599、775。两组实际动作均与冻结 assignment map 一致，实际缓存前缀也符合对应路径。

两次 mean TTFT 为 43.687 / 39.343 ms，仅记录运行分布，不与前一天强制动作运行做性能优劣结论。本轮没有同场 no-offload 对照，且混合策略的目标是采样，不是最小化时延。旧日志的 bytes=0 仍是未开启 full cost trace 的占位，最终 metrics 使用 null。

26 个冻结文件及输入 / fit / heldout SHA256 复查一致；61 次 GPU 采样未发现其他进程。固定 2 秒采样不排除采样间隙短暂干扰。

合并原四组与本轮两组后，共 285 条标签（RESTORE 136、RECOMPUTE 149）；pending load 非零的两类标签为 39 / 11。仅消除了“完全没有”的缺口，尚未覆盖所有拥塞分桶或高并发。

结果：[combined.json](../results/promotion_ttft_model_20261010/combined.json)。模型、特征、alpha 与会话分区规则未根据本轮验证结果调整：

| 动作 | 训练 / 校准 / 验证事件 | 固定均值验证 MAE | Ridge 验证 MAE | 校准 P90 裕量 | 裕量在验证集的覆盖率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RESTORE | 93 / 17 / 26 | 10.212 ms | 7.863 ms | 11.898 ms | 69.2% |
| RECOMPUTE | 103 / 21 / 25 | 7.772 ms | 4.056 ms | 12.393 ms | 96.0% |

注意：新增验证事件改变了验证集合，不能拿这里的 MAE 与旧 189 条数据结果直接宣称模型提升。RESTORE 的校准 P90 裕量在验证集仅覆盖 69.2%，进一步说明它不能当作稳定的置信保证。验证会话数仍少，事件也不独立。

**关键 No-Go 结果：当前“两个动作各自 P90 绝对误差相加”的保护规则不宜直接上线。**

```text
51 条验证事件
  → 45 条在两种动作训练数据的粗分桶内各有至少 3 个样本
  → 仅按 T_C_hat - T_R_hat > 0：7 条建议 RESTORE
  → 再扣除 11.898 + 12.393 = 24.291 ms 裕量：0 条建议 RESTORE
```

这里的 7 / 0 都是离线模型建议，不是真实 restore wins，也不是实际执行统计。只能说明这套模型与误差保护组合在当前支持的验证状态下过度保守；不能说明 restore 本身没有价值。

下一步应先分析预测误差是否主要来自两条路径共有的后缀计算 / 输出等待波动，以及目前缺失的实际 batch 工作量信号。若尝试共享成本项建模，必须先验证哪些成本确实可共享，不能假设恢复与重算的后缀成本严格相同。不要直接调低裕量来追求恢复次数；任何规则修改后都应重新冻结，并使用未用于这轮选择的新数据验收。

复现离线诊断：

```bash
.venv/bin/python scripts/fit_promotion_ttft.py \
  --source results/promotion_ttft_calibration_gpu0_20261009 \
           results/promotion_ttft_mixed_gpu0_20261010 \
  --output results/promotion_ttft_model_new_run/combined.json
```

## 策略何时才算生效

必须分清三个阶段：

- 数据层生效：动作、请求路径与标签一致。
- 决策层生效：在线根据预测净收益和误差裕量实际选择不同路径，而非只用预先分配动作。
- 性能收益成立：冻结规则后，在未用于本轮调参的负载上，重复对比 no-offload / always-restore / 新策略，端到端 TTFT 有可信改善。

本轮最多完成第一阶段和第二阶段的离线准备，不能跳过最后两项验收。旧 evaluation 数据已用于历史诊断，不能把它包装为全新未见负载。

## 测试边界

新分析、混合动作与既有 connector 本地单元测试共 48 项通过。最初两个新增测试因缺少 lifecycle 必填成本参数失败，补齐测试配置后通过。

另 8 项旧 connector 集成测试在读取 `facebook/opt-125m` 配置时因代理缺少 SOCKS 依赖失败；沙箱外重试首项仍是同样问题，未声明全套通过。GPU 校准使用本地 Qwen3 模型，与这一下载路径分开。
