# Promotion 误差定位：优先细查恢复就绪等待

日期：2026-10-10。范围：仅离线分析现有 6 次运行，不重跑 GPU、不修改 serving 策略、不调整模型系数与保护阈值。

## 结论与唯一优先项

**优先把同一轮 mixed calibration 中的 RESTORE 决策与已有 restore 分段 trace 关联，定位“决策→首次计算调度”内部的等待。**

依据不是“DMA 必然慢”，而是：RESTORE 的调度前分段预测 MAE 为 5.919 ms，高于调度后 4.862 ms；调度前平均高估 2.760 ms，且该现象不是只出现在一个运行批次。当前 trace 无法进一步区分提交、完成观察、通知交付与再次调度，因此尚未定位到具体函数或证明可优化收益。

不能据此认定调度后无关，也不能认定重算总体更快。下一步只补这条关联诊断，不同时修改 admission、阈值或 CUDA 传输实现。

## 输入与产物

- 冻结模型：[combined.json](../results/promotion_ttft_model_20261010/combined.json)。沿用会话分组，训练 / 校准 / 验证不串组。
- 原有数据：[强制动作四组](../results/promotion_ttft_calibration_gpu0_20261009/summary.json)、[混合动作两组](../results/promotion_ttft_mixed_gpu0_20261010/summary.json)。共 285 条有效标签。
- 全部验证事件：51 条，RESTORE 26 条、RECOMPUTE 25 条，不删尾部事件。
- [validation_events.csv](../results/promotion_ttft_error_diagnostic_20261010/verified/validation_events.csv)：完整 51 条验证记录，包含请求 / 会话 / 运行、特征、分段时间、预测与误差、训练分桶支持情况、预测收益与保护后收益。
- [all_events.csv](../results/promotion_ttft_error_diagnostic_20261010/verified/all_events.csv)：全部 285 条记录，保留 train / calibrate / validate 标签。
- [summary.json](../results/promotion_ttft_error_diagnostic_20261010/verified/summary.json)：按动作、分区、运行、会话、训练支持分组，包含各动作 Top 5 误差和 SHA256 溯源。
- 诊断脚本：[diagnose_promotion_ttft_errors.py](../scripts/diagnose_promotion_ttft_errors.py)。`verified/` 为最终检查版本；父目录保留首次生成的诊断中间结果。

## 如何归因而不改变模型

实际时间拆成：

```text
T = A + B
A = decision_recorded → first_compute_s
B = first_compute_s → first_output_s
```

`first_compute_s` 是 scheduler 首次为请求安排计算的时刻，不是 GPU 开始执行；B 包含计算、批次等待与输出处理，不能叫纯计算耗时。

诊断使用与原模型完全一致的训练数据、特征变换、标准化和 ridge alpha=10，分别拟合 A 与 B。固定设计矩阵下，线性回归满足两个分段的未截断预测之和等于原总时间预测。脚本验证：

- 原模型可由原训练分区重现，不使用验证标签更新系数。
- 实际 A+B 等于原标签；预测分段之和等于原未截断预测。
- 总带符号误差等于两段误差之和，加上原模型非负截断修正。
- **分段 MAE 不能相加当成总 MAE**，误差可能抵消；summary 同时保留平方误差交叉项。

分段预测只用于描述原模型误差结构，可出现负分段预测，不是新 serving 模型或物理耗时估计器。验证集没有发生总预测截断。

## 1. 误差主要落在哪里

表中带符号误差均为“预测−实际”，正值为高估。

| 动作 | 验证事件 / 会话 | A 实际均值 | B 实际均值 | 总 MAE | A 分段 MAE | B 分段 MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RESTORE | 26 / 12 | 12.532 ms | 11.177 ms | 7.863 ms | 5.919 ms | 4.862 ms |
| RECOMPUTE | 25 / 10 | 1.475 ms | 20.028 ms | 4.056 ms | 0.519 ms | 3.936 ms |

RESTORE 的总预测偏差 +2.278 ms，其中 A 为 +2.760 ms、B 为 −0.482 ms；RECOMPUTE 总偏差约 +0.039 ms。但它们是不同实际动作事件上的偏差，不能直接将偏差差值当成反事实收益校正量。

RESTORE 四个运行分组的 A / B 分段 MAE 分别为：4.930 / 4.786、7.234 / 5.408、4.640 / 4.444、6.537 / 4.308 ms。A 都较大，但后两组只有 5 / 4 条验证事件，不作统计显著性声明。

## 2. 是否只因为少量异常点或无样本状态

不是简单的“缺样本桶解释一切”：

| RESTORE 验证子集 | 事件数 | 总 MAE | A 分段 MAE |
| --- | ---: | ---: | ---: |
| 两个动作各至少 3 条训练样本的粗分桶 | 22 | 7.444 ms | 5.548 ms |
| 不满足上述支持条件 | 4 | 10.165 ms | 7.961 ms |

有支持的分桶仍有明显误差；但粗分桶不代表精确状态覆盖，也不能证明模型特征已经足够。

部分大误差事件（完整数据未删除）：

| 会话 / 运行 | 实际总时间 | 模型预测 | 预测−实际 | 情况 |
| --- | ---: | ---: | ---: | --- |
| KUimWdd / repeat_1 restore | 59.000 ms | 25.458 ms | −33.543 ms | A、B 均被低估；pending load=0 |
| DxpfdxS / mixed_17 | 14.834 ms | 34.708 ms | +19.875 ms | A 高估 13.353 ms；pending load=9 |
| dmMQKJv / mixed_29 | 14.724 ms | 32.024 ms | +17.299 ms | A 高估 9.974 ms；pending load=0 |

因此不能把“pending load 越多一定越慢”当成充分解释，也不能将 59 ms 事件无证据归为系统抖动然后删除。需要同一事件内部的更多时间点才能继续归因。

## 3. 为什么只建议 7/45 条恢复，保护后又归零

51 条验证事件中 45 条满足双动作粗分桶支持。原模型只有 7 条预测 `T_C_hat - T_R_hat > 0`；预测收益范围为 **0.007–1.903 ms**，全部 pending load=0，分属 5 个会话，不是 7 个独立会话。

当前总裕量为 11.898+12.393=24.291 ms，远大于这 7 条预测收益；所以不是阈值稍微偏大一点，而是**当前可辨识收益与预测不确定性不在同一量级**。这些收益仅是模型输出，不代表真实收益也只有不到 2 ms。

同时不能只怪 guard：不加保护时，原模型本身也把大多数验证状态判为重算；模型是否判对尚无同状态双路径真值。RESTORE 校准分区的两段平均误差为 +2.918 / −4.429 ms，存在抵消；验证分区为 +2.760 / −0.482 ms，抵消程度不同。这提供了校准裕量覆盖不稳定的一条描述性线索，不能据此推导严格概率保证。

不做的事：不用验证集偏差直接校正模型；不把 24.291 ms 改成 1 ms 来制造恢复次数；不将跨运行同请求均值差当成逐事件赢家。

## 下一步：一个范围明确的关联 trace 实验

在既有两轮 mixed 配置上保留固定 assignment、all-block 和容量，联合开启 `track_promotion_ttft` 与已有 `VLLM_EXPERIMENTAL_RESTORE_PATH_TRACE`，不增加 full block cost events。

按 request ID 将 A 继续拆成：

```text
决策记录 → 提交 → worker 观察完成
         → 模型输出 ready → engine 消费
         → scheduler 观察完成 → 首次计算调度
```

两个 trace 的决策打点有先后，要保留这段偏移，使相邻主机时间片之和与 A 一致；CUDA elapsed 与主机区间可能重叠，不能再次相加。先检查每个完整 RESTORE 事件的路径、时间顺序、遗漏和重复，再分析分段误差。

现有 runner 明确关闭并拒绝 restore trace，因此下一步需要增加显式诊断模式及相应审计，不能仅绕过检查。该实验只用于定位，不将 trace-on 的 TTFT 与旧 trace-off 结果直接解释为性能差异。

**结束条件：**给出 A 内部哪个子阶段承载主要误差、当前模型缺少哪个可在决策时获得的信号，或者明确现有观测仍不足。根据这次结果，只选一个成本模型修正或运行路径优化。此前 [10 月 7 日反馈诊断](lifecycle_restore_feedback_20261007.md) 可作为候选线索，但不能挪用其均值来分摊这 26 条事件的误差。

## 验证与复现

新增诊断测试及已有标签、拟合、请求路径测试共 **35 passed**；Ruff check 通过。全部计算在 CPU 完成，无 GPU 实验、无部署或 TTFT 提升声明。

```bash
.venv/bin/python scripts/diagnose_promotion_ttft_errors.py \
  --model results/promotion_ttft_model_20261010/combined.json \
  --output results/promotion_ttft_error_diagnostic_new_run

.venv/bin/python -m pytest \
  tests/benchmarks/test_promotion_ttft_errors.py \
  tests/benchmarks/test_promotion_ttft_fit.py \
  tests/benchmarks/test_promotion_ttft_labels.py \
  tests/benchmarks/test_lifecycle_request_path_analysis.py -q
```
