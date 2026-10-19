# Restore-ready-first：四组诊断完成，on 动作未触发

日期：2026-10-12。承接 [实验设计](restore_ready_first_experiment_20261011.md) 和 [联合恢复诊断](promotion_restore_joint_diagnostic_20261010.md)。

后续更新：[Future 就绪时序诊断](restore_readiness_diagnostic_20261012.md) 已完成，两组新运行中 45 次恢复只有 1 次在入口已就绪，26 次跨越 execute_model 调用才被观察为就绪。下文保留本轮零触发结果。

## 结论

四组各完成 648 个请求，共 2592 个请求。两组 on 的检查次数合计 12,597，eligible 和 consumed 均为 0。因此，**开关和检查代码已运行，但提前消费恢复通知的动作没有执行**。不能将本轮 TTFT 或分段等待的变化归因为该优化动作。

这不是整个 CPU offload 策略失效：两组 on 分别完成 20 / 27 次 RESTORE、593 / 997 个 load blocks，原恢复路径正常。此次也没有修改 admission 或 promotion 动作分配。

## 结果

| Assignment seed | 模式 | 检查次数 | Eligible | Consumed | RESTORE 事件 | Mean TTFT ms | P95 TTFT ms | Ready→engine ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | observe | 6289 | 2 | 0 | 22 | 36.365 | 63.517 | 3.915 |
| 17 | on | 6260 | 0 | 0 | 20 | 37.504 | 62.265 | 1.801 |
| 29 | on | 6337 | 0 | 0 | 27 | 35.900 | 67.618 | 0.464 |
| 29 | observe | 6230 | 0 | 0 | 26 | 39.732 | 70.346 | 3.803 |

按表中顺序运行。Counter 没出现的键按 0 解释。Eligible 是检查次数，RESTORE 是请求事件数，两者不应未经关联就当成同一统计单位。

两组 on：

- 队列为空：169 次；
- 队首 Future 尚未 ready：12,091 次；
- 队首 Future ready：337 次，但全部不携带 finished_recving；
- 不支持的 Future 类型 / failed / eligible / consumed：均为 0。

两组 observe 合计只捕捉到 2 次 eligible，表明这个条件并非绝对不可达，但在当前 workload / 执行路径上很少满足。

## 能证明什么、不能证明什么

当前优化在 engine 循环入口检查队首 Future：只有它已经完成且携带恢复通知，才跳过新 batch 提交、优先消费结果。本轮 on 没有满足该条件，故没有执行所设计的消费顺序调整及相应的跳过 idle sleep 分支。

此前测得的 ready→engine 等待不能视为循环入口已知、可全部回收的时间。结果可能在新 batch 的调度或执行提交期间才就绪；此外 executor 的输出时间戳与 Future.done() 可见时刻也不一定相同。**这些是待验证解释，目前的计数不足以确定就绪发生在哪个阶段。**

虽然 on 的 ready→engine 均值低于 observe，但优化动作没有执行；输入相同也不保证运行时命中集合、批次或线程时序相同。因此不能从这些均值推出收益，更不能声称超过 no-offload。

## 审计与环境

- Qwen3-0.6B；GPU0；GPU 1000 blocks；CPU cache 2 GiB；4 clients；GPU APC 和 chunked prefill 开启；all-block；guard 关闭。
- 固定 assignment seeds 17 / 29，runtime seed 0；每组完整 648 请求；输入与 client assignment 一致，实际决策符合冻结的 R/C map。
- 请求 / 恢复时间线审计通过，共 95 条完整 RESTORE 时间线；本轮是 trace-on 诊断。
- GPU0 共 137 次进程采样，foreign_pids 全部为空，采样间隔 2 秒。GPU1 同时有其他用户任务，因此共享 CPU / DRAM / 系统资源干扰仍可能存在；本轮不宣称整机独占。
- 运行结束后核对 32 个冻结源文件及输入哈希一致。之后仅对 runner 中一条 manifest 说明字符串做 Ruff 要求的源码换行，字符串值和执行行为不变，因此当前 runner 源文件哈希会与运行 manifest 不同。
- runner 在运行前后核对本地模型文件哈希一致。
- 首次运行因原 `/public/model_zoo/Qwen3-0.6B` 路径失效，在模型加载阶段失败；日志保留，未计入有效实验。实际使用 `/hole0/model_info/model_zoo/downloads/Qwen3-0.6B`。
- 新增显式 `--allow-model-relocation`，记录原路径、新路径和模型文件哈希；旧实验没有模型文件哈希，不能声称已验证新旧权重逐字节一致。本轮各组使用相同模型文件。
- 结束时 GPU0 显存为 0 MiB，端口 8238 已释放；GPU1 的 STOP 标记不存在，本轮没有修改他人调度器。

## 决定与下一步

未进入六组 no-offload / off / on 性能确认，因为 on 的实际提前消费为 0，不满足实验设计的门槛。开关保持默认 off。

下一步先做只观察的细粒度时间线：记录循环入口、schedule 前后、execute_model 前后、输出 ready 及消费时刻，并关联同一个队首批次的 Future 状态，区分“入口已就绪”“调度期间就绪”“执行提交期间就绪”“等待结果期间就绪”。需要保留 executor 打点与 Future 完成可见性的区别。

只有找到可在运行时利用的具体窗口后，才选择新的检查点或完成通知机制。不能为了增加触发次数而阻塞等待 Future，也不应直接升级为全局请求重排。

## 数据与复现

- [四组指标](../results/restore_ready_diagnostic_gpu0_20261012_relocated/summary.json)
- [运行 manifest 与模型哈希](../results/restore_ready_diagnostic_gpu0_20261012_relocated/manifest.json)
- [首次失败日志](../results/restore_ready_diagnostic_gpu0_20261012/repeat_0/observe/runner.log)
- 各 arm 下的 `promotion_restore_joint.json`、`promotion_ttft_labels.json`、`gpu_samples.json` 保存原始关联结果和审计记录。

```bash
.venv/bin/python scripts/run_restore_ready_experiment.py \
  --source results/promotion_restore_joint_gpu0_20261010 \
  --output results/restore_ready_diagnostic_gpu0_new_run \
  --phase diagnostic --gpu 0 --port 8238 \
  --model /hole0/model_info/model_zoo/downloads/Qwen3-0.6B \
  --allow-model-relocation
```
