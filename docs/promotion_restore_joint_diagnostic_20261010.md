# 联合恢复分段诊断：提交后观察与结果消费是重点

日期：2026-10-10。状态：两组 GPU0 诊断完成；未修改 admission、promotion 决策、传输顺序或调度顺序，未部署性能优化。

## 主要结果

两次运行共完成 **1296 个请求**，产生 44 次 RESTORE、51 次 RECOMPUTE；95 条有效决策标签，排除数为 0。44 次 RESTORE 都关联到完整的 worker / executor / engine / scheduler 时间线。

**恢复决策到首次计算调度平均 17.866 ms；其中提交返回到 worker 观察完成平均 7.889 ms，输出 ready 后等待 engine 消费平均 3.412 ms。**

这支持优先研究恢复就绪的观察与交付，而非只用 KV 字节数预测恢复成本。但当前仍不能把 7.889 ms 全部归于 DMA、轮询或 PCIe 拥塞，也没有证明修改这些环节一定能使整体 TTFT 超过 no-offload。

## 实施内容

- [run_promotion_ttft_calibration.py](../scripts/run_promotion_ttft_calibration.py) 新增显式 `--restore-diagnostic`，要求同时指定 mixed seeds。旧模式仍默认关闭恢复分段 trace，并检查实际模式是否吻合。
- [analyze_promotion_restore_joint.py](../scripts/analyze_promotion_restore_joint.py) 按 request ID 关联两类 trace，要求单次、未抢占决策，严格检查恢复事件集合、实际传输字节、缓存前缀、反馈时间点与阶段加总。
- [summarize_promotion_restore_joint.py](../scripts/summarize_promotion_restore_joint.py) 汇总阶段均值、波动、尾部和模型误差，沿用旧模型及其训练数据，不用本次标签更新模型。
- 默认运行路径没有性能改动。复用已有计时，未增加同步 CUDA 测量或全 block 事件日志。

## 实验控制与审计

模型 Qwen3-0.6B，GPU0，CPU cache 2 GiB，GPU 1000 blocks，4 clients，每组同样的 648 个请求。assignment seeds 17 / 29、runtime seed 0；all-block、guard 关闭。

两份完整 `run_configurations` 与前一轮 mixed 校准完全一致，仅通过环境开关联合启用已有 restore trace。30 个选定源文件、输入与旧 fit 的哈希在运行后复查一致；heldout 文件未改变。61 次 GPU 采样未发现其他进程，2 秒间隔不保证排除采样间隙的短暂干扰。GPU 和端口 8238 已释放。

| Assignment seed | 请求数 | RESTORE | RECOMPUTE | 完整恢复时间线 | Client mean TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 648 | 20 | 28 | 20 | 39.707 ms |
| 29 | 648 | 24 | 23 | 24 | 36.970 ms |

本轮是 trace-on 诊断，没有 no-offload 对照；不能与上一轮 trace-off 的 TTFT 差值解释成收益或计时开销。日志中旧 bytes=0 字段仍是未测占位，最终 metrics 为 null；单次恢复实际字节由 joint trace 单独核对。

结果：[运行 manifest](../results/promotion_restore_joint_gpu0_20261010/manifest.json)、[运行 summary](../results/promotion_restore_joint_gpu0_20261010/summary.json)、[分段汇总](../results/promotion_restore_joint_analysis_20261010/summary.json)、[44 条恢复记录 CSV](../results/promotion_restore_joint_analysis_20261010/restore_events.csv)。

## 时间分段

两个决策时钟并非同一打点。分析保留 `promotion decision_recorded → restore decision` 的微小偏移，再串联后续相邻主机时刻，保证每个事件的分段之和等于决策到首次计算调度的时间。反馈子阶段取代原 worker→scheduler 总段，不重复累加。

| 主机侧阶段 | Seed 17 均值 | Seed 29 均值 | 合并均值 | 合并标准差 | 合并 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 决策记录→提交开始 | 1.909 | 1.841 | 1.872 | 1.203 | 4.266 |
| Host submit | 0.316 | 0.370 | 0.345 | 0.294 | 0.823 |
| 提交返回→worker 观察完成 | 8.013 | 7.785 | **7.889** | **7.126** | **22.674** |
| worker 观察完成→输出 ready | 2.768 | 2.677 | 2.718 | 1.828 | 5.405 |
| 输出 ready→engine 消费 | 3.422 | 3.403 | **3.412** | **4.261** | **11.240** |
| engine 消费→connector 观察 | 0.070 | 0.083 | 0.077 | 0.066 | 0.174 |
| scheduler 观察完成→首次计算调度 | 1.413 | 1.671 | 1.553 | 0.756 | 3.250 |
| 总计 | 17.911 | 17.828 | **17.866** | 11.270 | — |

单位 ms。P95 与标准差不可按阶段直接相加。

CUDA event elapsed 平均 **2.266 ms**，作为单独诊断量，不加入上表。开始 event 在同方向前序传输依赖之后排入 stream，且 DMA 与 host submit 可能重叠，因此 **7.889−2.266 不能当作精确轮询开销**。

## 与预测误差的关系

复用此前模型原训练分区、标准化与 alpha=10，重现总模型，并获得此前“决策→首次计算调度”的线性分段预测；本轮不重拟合。

- 全部 44 条新恢复记录的调度前 MAE 为 8.069 ms；它们包含旧训练会话，不能作为独立泛化成绩。
- 原验证会话分区中本轮有 12 条恢复记录：调度前 MAE 8.831 ms、实际均值 19.060 ms；样本少且是重复会话，不声称统计显著。
- 提交返回→worker 观察这一段波动最大，与带符号误差（预测−实际）的相关系数约 −0.789；ready→engine 约 −0.631。这是描述性相关，阶段本来就在总耗时内，不能由相关系数推导因果贡献比例。

几个具体事件：

| 会话 / arm | A 实际 | A 预测 | 提交→worker 观察 | CUDA elapsed | ready→engine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gay8UDf / mixed_29 | 45.043 | 12.041 | 33.025 | 1.625 | 2.380 |
| KUimWdd / mixed_17 | 38.095 | 15.084 | 12.526 | 2.140 | 13.196 |
| DCnCnjA / mixed_17 | 4.956 | 20.424 | 2.502 | 1.859 | 0.020 |

第一个事件只有 19 个候选 blocks，决策时 pending load=0、tracked requests=1，仍出现 33 ms 提交后观察间隔。这个计数不能描述执行管线当前所处阶段；不能由 pending load=0 推导设备或 worker 没有等待。

## 对应代码与缺失信号

1. [cpu_gpu.py](../vllm/v1/kv_offload/worker/cpu_gpu.py) 同方向传输等待前一个 event；`get_finished()` 通过 `end_event.query()` 观察完成，不提供精确的主机侧设备完成时间。
2. [kv_connector_model_runner_mixin.py](../vllm/v1/worker/kv_connector_model_runner_mixin.py) 在模型执行 context 的 `finally` 调用 connector `get_finished()`，不是独立于模型调用的连续轮询。需要区分传输尚未完成与模型执行路径尚未再次观察完成，当前时间戳还无法将两者完全分开。
3. [EngineCore.step_with_batch_queue](../vllm/v1/engine/core.py:429) 先尝试 schedule / execute 新批次，再消费队首旧结果。已有模型输出即使携带完成通知，仍可能等待 engine 消费。

因此现有模型缺少的并非又一个静态 block 分数，而可能是**当前模型批次工作量、上次完成检查的时间、输出队列就绪状态**等管线状态。前两项需要额外测量才能证实；队首输出是否 ready 在 engine 当时可直接检查，但它是后续调度时刻的信号，不能回填成此前 promotion 决策时的特征。

## 唯一优先的下一步优化实验

建议先验证一个可关闭的局部改动：**若队首 Future 已就绪且携带恢复完成通知，先按原顺序消费该结果，再进行后续调度。**

选择它而不是直接重构独立传输轮询的原因：ready→engine 的等待在两轮均约 3.4 ms，且有可在本地检查的就绪条件；7.9 ms 更大，但其内部设备等待与检查时机仍混合，直接修改风险和归因难度更高。

约束与验收：

- 默认关闭，仅单卡 UniProcExecutor；不阻塞等待未完成的 Future，不越过队首。
- 保留 abort 处理、原 `update_from_output`、失败传播与引用计数；通知只能消费一次。
- 先记录候选检查次数、满足 ready 条件次数、实际提前消费次数。当前未证明输出在每次循环入口就已经 ready，**3.4 ms 不是可保证回收的收益**。
- 固定动作分配和 all-block，比较原路径 / 新路径，先看目标等待是否缩短及其他阶段是否恶化。
- 最后 trace-off，重复比较 no-offload / 原 offload / 优化 offload 的 mean、P95 TTFT 与吞吐。恢复比例只占全部请求一部分，不能把阶段改善等同于整体改善。

本轮没有实施这个调度改动，也没有修改误差裕量。这是下一轮范围明确的性能实验，而不是已成立的论文贡献。

## 测试与复现

联合时间线、原恢复分析、标签、拟合相关测试 **45 passed**；Ruff check 通过。

```bash
.venv/bin/python scripts/run_promotion_ttft_calibration.py \
  --source results/lifecycle_cost_probe_gpu1_20260917 \
  --output results/promotion_restore_joint_new_run \
  --gpu 0 --port 8238 --mixed-seeds 17 29 --restore-diagnostic

.venv/bin/python scripts/summarize_promotion_restore_joint.py \
  --source results/promotion_restore_joint_new_run \
  --frozen-model results/promotion_ttft_model_20261010/combined.json \
  --output results/promotion_restore_joint_analysis_new_run
```

承接：[前一轮误差归因](promotion_ttft_error_diagnostic_20261010.md)。两份文档分别回答“哪个大阶段难预测”和“恢复等待内部先尝试优化哪里”。
