# Restore-ready-first：实现完成，GPU 对比待运行

日期：2026-10-11。承接 [联合恢复诊断](promotion_restore_joint_diagnostic_20261010.md)。

2026-10-12 更新：四组 GPU 诊断已完成；两组 on 均未触发提前消费，未进入性能确认。详见 [结果与下一步](restore_ready_first_results_20261012.md)。下文保留实施时的状态记录。

## 当前状态

已实现默认关闭的 FIFO 队首恢复通知优先消费开关，并完成本地单元 / 调用路径测试。**尚未完成真实 GPU 运行验证，没有新的 TTFT 性能结论。** 本次检查 GPU0 约 71 GiB / 98% 利用率、GPU1 约 25 GiB / 95% 利用率，两卡均在使用，未启动负载或干扰其他进程。该占用只是检查时快照。

## 改动

- [restore_ready.py](../vllm/v1/kv_offload/restore_ready.py)：只检查队首标准 `Future`；未完成、异常、取消、无恢复通知的结果均不提前消费。不扫描后续批次，不调用 CUDA 同步或重新获取异步模型输出。
- [EngineCore](../vllm/v1/engine/core.py)：若条件满足，跳过本轮新 batch 提交，继续进入原来的 FIFO pop / 结果读取 / abort / scheduler update 路径；不是另写一套完成通知处理。随后下一轮正常调度。
- 消费后仍返回 `model_executed=False`，不虚报模型执行；仅对这次确实推进了恢复完成状态的迭代，跳过原先空转路径的 1 ms sleep。否则提前消费会额外引入固定延迟。普通空转仍保持原 sleep。
- [run_lifecycle_calibrated_smoke.py](../scripts/run_lifecycle_calibrated_smoke.py)：新增显式模式参数，并传入子进程环境；未指定时强制 off，避免继承外部实验开关。spec 保存模式，旧 spec 不兼容时拒绝静默复用。
- [run_restore_ready_experiment.py](../scripts/run_restore_ready_experiment.py)：固定 workload、mixed assignment 和容量，先诊断后独立性能比较。

环境开关 `VLLM_EXPERIMENTAL_RESTORE_READY_FIRST`：

| 模式 | 行为 |
| --- | --- |
| off（默认） | 不检查或改变消费顺序 |
| observe | 统计触发条件，但仍按原路径先 dispatch 再消费 |
| on | 队首结果已完成且带 `finished_recving` 时，先消费 |

实验模式只允许单 GPU、精确的 UniProcExecutor 类型、async batch queue、OffloadingConnector、非 speculative 的文本生成；多卡、外部 executor、pooling、encoder / EC 等未覆盖路径拒绝启用。

## 如何证明真正生效

统计通过已有 value-policy stats 导出：

- `restore_ready_first_checks`：检查次数。
- `empty / not_ready / ready / no_restore / failed / unsupported_future`：分支计数。
- `eligible`：队首 ready 且包含恢复通知的次数。
- `consumed`：成功经过原 scheduler update 的提前消费次数。

observe 的 consumed 必须为 0；on 的 consumed 应等于 eligible，且只有 consumed > 0 才能声称优化实际触发。若为 0，不能把 TTFT 波动归于此改动。异常仍由原结果消费路径传播。

## 实验设计

固定源数据 / 配置来自 `results/promotion_restore_joint_gpu0_20261010`：648 请求 / run、4 clients、GPU 1000 blocks、CPU 2 GiB、all-block、guard off、Qwen3-0.6B。seed 17 / 29 是预先指定 R/C 动作的种子，不是改变 workload 或轮次数。

第一阶段 4 次运行，联合 trace 开启：

```text
seed 17：observe → on
seed 29：on → observe
```

检查完整请求集、输入与 client assignment、R/C 动作 map、时间线、触发次数，并观察 ready→engine 等待是否降低。即使相同请求集合，实际 CPU 命中和运行状态也可能因顺序变化而不同，不把事件均值差当成严格配对因果结果。

```bash
.venv/bin/python scripts/run_restore_ready_experiment.py \
  --source results/promotion_restore_joint_gpu0_20261010 \
  --output results/restore_ready_diagnostic_gpu0_new_run \
  --phase diagnostic --gpu 0 --port 8238
```

第一阶段有实际触发、正确性通过后，第二阶段 6 次运行：

```text
seed 17：no-offload → off → on
seed 29：on → off → no-offload
```

关闭 request-path 和 restore-stage 计时；**仍保留 calibration_mixed 所要求的轻量动作记录**，因此不声称完全零观测开销。off / on 两组动作审计相同；no-offload 没有该动作记录，比较时应披露。primary 指标为全请求 mean / P95 / P99 TTFT，辅以请求吞吐、恢复量和实际触发次数。

```bash
.venv/bin/python scripts/run_restore_ready_experiment.py \
  --source results/promotion_restore_joint_gpu0_20261010 \
  --output results/restore_ready_confirmation_gpu0_new_run \
  --phase confirmation --gpu 0 --port 8238
```

脚本拒绝覆盖已存在的输出目录；运行前检查 GPU、端口，运行中采样 GPU 进程。检测到其他任务时该 arm 无效，清理仅针对自己创建的进程组，不终止他人任务。

## 正确性验证范围

新增测试覆盖队首顺序、未就绪不阻塞、observe 不改变行为、普通输出不触发、失败 Future 继续抛出、abort 先于 scheduler update、同一输出不重复消费、消费后下一轮继续调度，以及只对真实进展跳过 idle sleep。已有 executor 输出时间戳与关联 trace 测试一起运行。

这些是 mock / CPU 路径测试，不替代真实异步 GPU 验证。运行结果应保留负结果：即使 ready→engine 降低，吞吐或整体 TTFT 也可能恶化；不能将此前测得的 3.41 ms 当成保证可回收的收益。

本轮相关测试合计 **78 passed**，Ruff check / format --check 与 `git diff --check` 通过。结束前再次检查两卡仍在高负载，GPU 对比实验未启动。
