# 固定完整会话的生命周期策略复验

## 目的与冻结条件

延续 [成本分层与有限探测实验](lifecycle_cost_probe_revision_20260917.md)，
本轮只调整评测方法，不调整 Admission / Promotion 参数，不启用 Guard。
上一轮每客户端请求上限会造成不同策略实际执行不同轮次，不能据单次 TTFT 推断收益。

### 本轮压测修复

- 队列终止标记只表示没有新会话，仍须排空当前 active conversations。
- 允许最后一个 user 消息没有数据集 assistant 答案；模型输出作为该轮结果追加。
- 可选 `--fixed-client-assignment`：第 i 个输入会话固定给第 i modulo 4 个客户端。
  默认共享队列模式保留。固定分配不固定不同客户端之间的全局请求时序。
- shell 的 `COMPLETE_WORKLOAD=1` 同时取消请求上限、开启固定分配；其他模式默认不变。

### 获取结果前确定的配置

- 输入：上一轮相同的 128 个 Direct ShareGPT 会话，共 659 个 user 请求，全部完成。
- 4 clients、32 active conversations，round-robin，会话内部顺序，保留数据集 assistant 历史。
- Qwen3-0.6B，TP=1，1,000 GPU blocks，4 GiB CPU tier，max model len 4,096。
- 输出 min/max tokens 仍为 16/32，不改变上轮设置；上下文输入不依赖本次模型生成内容。
- 成本和探测参数从上一轮 manifest / calibration_fit 冻结导入，不重新拟合。
- 比较 no-offload、all-block、host-no-probe、host-probe；每组 3 次，共 12 组 / 7,908 requests。
- 固定 runtime seed=0，三次重复用于观察运行波动，不冒充三个独立数据集。
- 组顺序由 Random(20260917) 预先打乱，写入 manifest；独立服务进程，无跨组 KV 延续。
- GPU 1 启动前须空闲，采样记录其他 GPU 计算进程，结束只清理自身进程组。

## 审计与结果解释边界

每组完成后自动核对：

1. 659 条成功响应；每个指定 `(conversation_id, input_num_turns, client_id)` 恰好出现一次。
2. 输出包含全部 128 个已完成会话；各组相同会话/轮次的输入 token 数完全一致。
3. 无事件丢弃、未匹配传输、未完成 Promotion、store 失败；探测未超预算。
4. 运行前后的代码 hash 和参数与冻结 manifest 一致。

请求集合相同不代表每个请求的到达时刻、batch、cache state 相同。
这是更公平的闭环端到端比较，不是 strict paired-event 反事实测量，也不是生产到达 trace。
包含会话结束阶段的并发下降；结果代表完成有限会话集合，而非恒定并发稳态吞吐。
输入已在前期研究中使用，因此仍为迭代诊断，不是未见测试集上的泛化验证。
三次重复不足以稳健声称统计显著性；正收益账本也不代替实际 TTFT / 完成时间。

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_lifecycle_fixed_workload.py \
  --gpu 1 --port 8238 \
  --source results/lifecycle_cost_probe_gpu1_20260917 \
  --output results/lifecycle_fixed_workload_gpu1_20260917 --repeats 3
```

运行器支持完全相同配置下跳过已完成组并续跑；代码或参数变化则拒绝续跑。
12 组已全部完成，结果如下。实验目录与全部原始日志保留。

## 测试记录

- 103 项测试通过（新增长尾会话排空、末尾 user、请求上限兼容、重复 ID / 非交替角色、
  审计拒绝缺失请求 / 错误客户端 / 预算超限 / 未完成会话，并运行原策略与 connector 回归）。
- Shell `bash -n`、`git diff --check` 通过。
- Ruff 0.14.0 检查通过，本轮未修改的 `fetch_value_policy_stats` 既有 SIM117 告警
  以单文件忽略项排除，没有进行无关重构；新增文件未使用该忽略。

## 完成结果（2026-09-17）

运行时间约 00:37–00:49。来源：
[冻结清单](../results/lifecycle_fixed_workload_gpu1_20260917/manifest.json)、
[逐组汇总](../results/lifecycle_fixed_workload_gpu1_20260917/evaluation_summary.json)。

### 完整性与公平性审计

12 组均完成 128 个会话 / 659 个请求，共 7,908 个成功请求。
逐组的会话、轮次、客户端分配完全符合预定集合；输入 token 数跨组一致。
冻结代码 hash 验证通过，无 pending/incomplete promotion、store 失败、
成本事件丢弃或未匹配传输；有探测的三组均恰好使用 128 blocks 冷启动预算。
无探测组探测写入均为零。所有组 Guard 关闭。

359 次 GPU 采样中，各组 foreign_pids 均为空。恢复连接后确认实验端口 8238 已关闭；
GPU 1 当时已被后来启动的其他任务使用，其 PID 不在本轮采样中，未对其做任何操作。
GPU 隔离仍只是采样证据，不表示独占整机 CPU / 内存等资源。

### 三次重复汇总

平均 TTFT 为三个运行均值的均值，± 表示三次运行均值的样本标准差，**不是置信区间**。
P95 列为每次运行 P95 的平均值，不是把全部请求混合后计算的 P95。
Store / Load 为每次运行的累计 block 数均值；同一 block 可以被多次恢复，故 load 可大于 store。

| 策略 | 平均 TTFT (ms) | P95 均值 (ms) | 完成请求时间跨度 (s) | 吞吐均值 (req/s) | Store blocks | Load blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no-offload | 35.434 ± 0.428 | 58.538 | 21.425 | 30.758 | 0 | 0 |
| all-block | 30.176 ± 0.802 | 52.478 | 20.787 | 31.704 | 10,995.3 | 15,136.3 |
| host-no-probe | 32.778 ± 2.424 | 56.671 | 21.270 | 30.990 | 7,029.3 | 5,349.7 |
| host-probe | 32.673 ± 2.084 | 57.409 | 21.430 | 30.756 | 7,128.7 | 6,593.7 |

时间跨度从首个请求开始至最后一个请求完成，不包含服务启动；吞吐为 659 / 该跨度。
在该有限会话测试中，有探测策略的 TTFT 改善没有转化为相应的整体吞吐改善。

### 每次运行，避免只展示最好结果

| Repeat | 策略 | 平均 TTFT (ms) | P95 (ms) | Store / Load blocks | Restore / Recompute 决策 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | no-offload | 35.010 | 58.239 | 0 / 0 | — |
| 0 | all-block | 29.411 | 50.029 | 10,997 / 15,187 | 426 / 0 |
| 0 | host-no-probe | 34.433 | 62.328 | 7,100 / 5,735 | 76 / 12 |
| 0 | host-probe | 30.480 | 54.492 | 7,256 / 6,809 | 99 / 15 |
| 1 | no-offload | 35.425 | 58.611 | 0 / 0 | — |
| 1 | all-block | 30.106 | 52.190 | 10,995 / 15,170 | 425 / 0 |
| 1 | host-no-probe | 33.906 | 55.763 | 6,800 / 5,025 | 65 / 18 |
| 1 | host-probe | 34.628 | 60.161 | 7,196 / 6,102 | 98 / 29 |
| 2 | no-offload | 35.866 | 58.763 | 0 / 0 | — |
| 2 | all-block | 31.011 | 55.216 | 10,994 / 15,052 | 422 / 0 |
| 2 | host-no-probe | 29.996 | 51.921 | 7,188 / 5,289 | 68 / 25 |
| 2 | host-probe | 32.911 | 57.575 | 6,934 / 6,870 | 119 / 29 |

这里是动作选择次数，不是两个动作的严格配对胜负次数。

## 当前结论

1. **完整会话下，CPU offload 在这个配置中有价值。** all-block 相对 no-offload
   的平均 TTFT 低 14.84%，三次均更快。它不能推导到所有容量 / 模型 / 到达负载。
   这也不否定之前短窗口的测量，而是限制其结论适用范围：只跑 128 个请求和
   完成 659 个请求覆盖的生命周期不同，且客户端分配方式也改变了。
2. **当前联合策略尚未超过简单全量策略。** host-probe 相对 no-offload 平均 TTFT
   低 7.79%，但相对 all-block 慢 8.28%，三次均慢于同 repeat 的 all-block。
   host-no-probe 平均比 all-block 慢 8.62%（三次中一次更快）。
3. **少传输是真实结果，但不能替代延迟目标。** host-no-probe / host-probe 的 D2H
   相对 all-block 分别少 36.07% / 35.17%；平均累计恢复量也明显下降。
   这不是此前受控实验的“有效恢复不损失”，更不能直接归因为消除了无效写入。
4. **探测提供了反馈，但额外性能收益未建立。** 开启探测相对不开启的平均 TTFT
   只差约 0.105 ms，远小于本轮运行间波动；三次对照只有一次有探测更快。
   其 Promotion 决策更多，却未表现出稳定的端到端优势。
5. **不能凭当前四组确定损失来源。** host 策略同时改变 Admission 和 Promotion，
   因此可能是误拒绝有价值前缀、错误重算选择、在线成本漂移、决策开销或其组合。
   当前数据不支持把原因直接归咎于其中某一个，也不支持只降低准入门槛就能解决。

**研究状态：工程闭环和更公平的完整会话实验已打通；性能目标尚未达成。**
不能把“比 no-offload 快”或“D2H 少 35%”写成策略优于 all-block 的结论。

## 下一步：定位损失，而非继续扩大参数 sweep

在同样完整会话和冻结成本下补齐两组：

| 组 | Admission | Promotion | 用途 |
| --- | --- | --- | --- |
| all-block | 全存 | always restore | 基准 |
| Promotion-only | 全存 | cost-aware | 隔离恢复/重算选择的影响 |
| Admission-only | selective | always restore | 隔离选择性准入的影响 |
| joint-no-probe | selective | cost-aware | 两者组合 |

固定 Guard 关闭、冷启动探测关闭；保留 no-offload 作参考。
现有 `evaluation_config` 的 `promotion` / `admission` 路径已支持这两个消融方向，
但尚未在本轮完整工作负载上运行，不能拿此前短窗口数据补齐该表。

若 Admission-only 损失明显，再检查被丢弃的连续前缀及后续 demand；
若 Promotion-only 损失明显，再检查恢复/重算估计误差及在线修正轨迹。
先分清原因，再修改策略；本轮结束不增加 scheduler / progressive restore，
也不凭这三次运行宣称泛化或统计显著性。
