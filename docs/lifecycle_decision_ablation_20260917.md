# 完整会话下的 Admission / Promotion 双因素消融

## 目的

[上一轮完整会话实验](lifecycle_fixed_workload_20260917.md)中，联合策略的 D2H
比 all-block 少约 35%，但平均 TTFT 仍慢约 8.3%。本轮不调参，区分准入筛选、
恢复/重算选择以及控制器记账路径的影响。

## 结果产生前确定的实验设计

| 组 | 生命周期控制器 | Selective admission | Cost-aware promotion |
| --- | --- | --- | --- |
| no-offload | 无 CPU tier | — | — |
| all-block | 无该控制器，原生 offload 基线 | 关闭 | always restore |
| all-block-control | 开启 | 关闭（全部准入） | 关闭（always restore） |
| admission-only | 开启 | 开启 | 关闭（always restore） |
| promotion-only | 开启 | 关闭（全部准入） | 开启 |
| joint-no-probe | 开启 | 开启 | 开启 |

四个控制器组使用相同代码路径和成本初值，只切换上表两项。
all-block-control 是对照，避免把控制器的附加记账、估计、反馈开销直接归咎于算法选择。
它不是关闭一切控制器开销的“纯空壳”，并且各组运行中模型更新会随实际动作反馈而分化。
所以本轮测量的是完整运行的开关效应，而非严格配对的单事件反事实。

- Guard 和 cold probe 全部关闭；不以此实验验证长期保护或探测效果。
- 128 个相同 ShareGPT 会话，659 个 user 请求，全部执行；固定四客户端分配、
  32 active conversations、round-robin、保留数据集 assistant 历史。
- Qwen3-0.6B / TP=1、1,000 GPU blocks、4 GiB CPU tier、输出 token 限制仍为 16/32。
- 六组各重复三次，共 18 组 / 11,862 requests；runtime seed=0，预先随机组顺序。
- 使用 GPU 0；不沿用 GPU 1 的性能基线，no-offload / all-block 均在本轮重跑。
- 成本初值仍来自上一轮 GPU 1 的独立校准，未针对 GPU 0 重新拟合。
  本轮是冻结参数迁移后的模块诊断，不是 GPU 0 上重新校准后的最优策略评测。
- 实际性能比较仅使用本轮 GPU 0 数据；不直接将跨卡、跨时间结果相减归因。
- 各组用新服务进程；出现外来 GPU 计算进程则该组无效并停止，不清理他人任务。

## 分析规则

以同 repeat 中的 all-block-control 为参考：

- Admission effect：admission-only − all-block-control。
- Promotion effect：promotion-only − all-block-control。
- Joint effect：joint-no-probe − all-block-control。
- Interaction：joint − admission-only − promotion-only + all-block-control。
- Controller-path contrast：all-block-control − 原生 all-block。

上式对平均 TTFT 的正值表示变慢，负值表示变快。三次重复只作诊断，
不能据三点声称显著性或泛化。即使某个模块的端到端效应明显，也仍需事件级检查
来区分误拒绝、成本预测错误、CPU 容量/前缀依赖以及资源争用等具体原因。
四组并非在完全相同的 GPU/CPU 状态上做每一步选择；全局时序仍随策略改变。

## 运行与审计

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_lifecycle_fixed_workload.py \
  --gpu 0 --port 8238 --suite decision-ablation --repeats 3 \
  --source results/lifecycle_cost_probe_gpu1_20260917 \
  --output results/lifecycle_decision_ablation_gpu0_20260917
```

每组核对完整请求集合、固定客户端、跨组输入 token 数；检查 store 失败、
pending/incomplete promotion、成本事件丢失/未匹配、意外 cold probe。
代码、输入文件及成本文件 hash 在每组启动前核对；历史结果不覆盖。

本轮只增加实验套件与配置测试，未修改运行时算法、阈值或成本参数。
已完成 18 组，结果如下。

## 验证工具

- 108 项完整回归测试通过；额外的汇总公式测试 1 项通过。
- Ruff 0.14.0 检查与格式检查通过，`git diff --check` 通过。
- 新增只读汇总工具，重新核对原始 CSV、配置、请求集合、GPU 采样、预算及事件完整性：

```bash
.venv/bin/python scripts/summarize_lifecycle_decision_ablation.py \
  results/lifecycle_decision_ablation_gpu0_20260917
```

未完成时必须显式加 `--allow-partial`；只对完整的六组 repeat 计算消融差值。
汇总给出运行间样本标准差，而不是置信区间；缺失的估计收益不补零。

## 结果：准入是当前优先排查对象

运行跨越 2026-09-17 23:53 至 2026-09-18 00:11。来源：
[manifest](../results/lifecycle_decision_ablation_gpu0_20260917/manifest.json)、
[逐组数据](../results/lifecycle_decision_ablation_gpu0_20260917/evaluation_summary.json)、
[独立复核汇总](../results/lifecycle_decision_ablation_gpu0_20260917/analysis_summary.json)。

18 组 / 11,862 个请求全部完成：每组 128 个会话 / 659 个请求，实际请求集合、
客户端分配和输入 token 数一致。无外来 GPU 计算进程采样、store 失败、
未完成 promotion、成本/决策事件丢失或未匹配传输；cold probe 和 Guard transition 均为零。
代码、输入和成本初值 hash 验证通过。结束后 GPU 0 显存为 0 MiB、端口 8238 已关闭；
GPU 1 的其他任务未被操作。

### 三次运行的均值与波动

± 为三次运行均值的样本标准差，不是置信区间。
P95 列为每次运行 P95 的平均，不是合并请求后的 P95。
累计 store/load blocks 按运行取均值，并非 unique blocks。

| 策略 | 平均 TTFT (ms) | P95 均值 (ms) | 吞吐均值 (req/s) | Store blocks | Load blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-offload | 31.929 ± 0.628 | 54.861 | 31.347 | 0 | 0 |
| all-block | 30.364 ± 0.638 | 52.749 | 31.723 | 10,996.0 | 15,206.0 |
| all-block-control | 29.991 ± 1.072 | 51.606 | 31.691 | 10,993.0 | 15,082.3 |
| admission-only | 35.134 ± 4.346 | 60.849 | 30.661 | 6,604.0 | 6,370.7 |
| promotion-only | 30.259 ± 0.621 | 53.759 | 31.590 | 10,993.0 | 14,049.3 |
| joint-no-probe | 34.132 ± 0.688 | 58.869 | 30.846 | 6,855.0 | 5,507.0 |

all-block 比 no-offload 平均 TTFT 低约 4.90%。联合策略相对 all-block 的 D2H 少约
37.66%，但平均 TTFT 慢约 12.41%；少传输仍未转化为本轮的延迟收益。
Admission-only 的 D2H 少约 39.94%，同样不能视为性能优化成功。

### 每次运行的 TTFT

| 策略 | Repeat 0 | Repeat 1 | Repeat 2 |
| --- | ---: | ---: | ---: |
| no-offload | 32.620 | 31.394 | 31.772 |
| all-block | 31.074 | 29.840 | 30.178 |
| all-block-control | 30.920 | 30.235 | 28.818 |
| admission-only | 32.633 | 32.618 | 40.152 |
| promotion-only | 29.555 | 30.491 | 30.731 |
| joint-no-probe | 33.511 | 34.013 | 34.871 |

单位均为 ms。Admission-only 的第三次运行波动明显，完整保留，未作为异常值剔除。
同轮 no-offload 未明显变慢，但无法凭此排除瞬时 CPU/内存争用等影响。
运行中对设备温度/频率的一次查询发生在组间空闲期，不能用于解释此前波动。

### 双因素差值：哪些结论比较稳定

以下使用同 repeat 的运行均值相减，正值为变慢。**不是相同状态下逐请求配对的因果效应。**

| 对比 | Repeat 0 (ms) | Repeat 1 (ms) | Repeat 2 (ms) | 均值 ± 样本 SD (ms) |
| --- | ---: | ---: | ---: | ---: |
| 控制器路径 − 原生 all-block | −0.154 | +0.395 | −1.360 | −0.373 ± 0.898 |
| Admission-only − 控制组 | +1.712 | +2.383 | +11.335 | +5.143 ± 5.372 |
| Promotion-only − 控制组 | −1.365 | +0.256 | +1.914 | +0.268 ± 1.639 |
| Joint − 控制组 | +2.590 | +3.778 | +6.053 | +4.140 ± 1.760 |
| Joint − Promotion-only | +3.955 | +3.523 | +4.139 | **+3.872 ± 0.317** |
| Joint − Admission-only | +0.878 | +1.396 | −5.282 | −1.003 ± 3.715 |
| Interaction | +2.243 | +1.140 | −7.195 | −1.271 ± 5.160 |

可以支持：

1. **当前准入值得优先排查。** 在保持 cost-aware promotion 的情况下，额外开启
   selective admission 三次均慢约 3.5–4.1 ms，是本轮方向最一致的对比。
   单独开启准入也三次均慢于控制组，但第三次大幅波动使均值不够稳健。
2. **没有证据把几毫秒回归全归咎于控制器记账路径。** 原生与控制器全量恢复组
   差值变号且较小；负均值也不表示控制器记账能“加速”。
3. **Promotion 的稳定增益尚未建立。** 它与 all-block / 控制组接近，
   对控制组只有一次更快，另外两次更慢，不能据第一轮宣称它已经有效。
4. **交互项不稳定。** 不能把前两轮正值包装成已验证的“两个模块存在有害交互”。
   第三轮变号且幅度大；当前只能确定联合实现没有达到性能目标。

### 动作数不是动作正确率

| Promotion-only | Restore 决策 | Recompute 决策 |
| --- | ---: | ---: |
| Repeat 0 | 324 | 102 |
| Repeat 1 | 310 | 114 |
| Repeat 2 | 289 | 134 |

三轮选择了 350 次 recompute，不等于“350 次重算都更优”。本轮未获得这些事件的
两个执行分支的严格配对成本，不能替代此前的 paired-event 证据，也不能直接给出分类准确率。

### 成本账本与端到端排序不一致

all-block-control 三次 estimated_net_utility 分别为 −0.132、−0.178、+0.115 s，
但三次平均 TTFT 都低于同轮 no-offload。
联合策略三次账本都为正（+0.357、+0.391、+0.232 s），却三次都慢于 no-offload。

这不是两个相同反事实状态上的直接测量，因此不能用差值证明某个成本项必然错误。
但它足以说明：**当前估计账本不具备端到端收益保证，不能把其符号直接当作关层依据。**
Guard 在本轮关闭；不应在解决成本口径与校准问题之前宣称其能够保证安全退化。

## 下一步：补足准入损失诊断，不立即降低阈值

1. 固定现有 all-block-control、Promotion-only 作为参考，保留本轮失败结果。
2. 给 Admission 决策增加 request ID、候选 token/block 范围、已存在/新增 blocks、
   选择与拒绝原因；给 Promotion 预测与完成事件增加共同 ID。当前控制器两类事件
   没有共同 request ID，不能依靠事件数组的相邻顺序强行配对。
3. 在新的诊断运行中区分：被拒绝的 KV 后续是否形成 external demand、
   CPU 中是否存在未能加载的后缀、首次缺失/未 ready 的位置、因此重算的前缀规模。
   不把“后来出现需求”直接等同于“当时应该存”，仍需考虑保留容量及成本。
4. 优先检验一个**尚未证实的假设**：目前按单次 store 候选批次定价，
   是否忽略完整可恢复前缀、已有 CPU 内容的边际写入成本和恢复固定开销的共享。
   本地代码可确认 LRU lookup 在首个缺失/未 ready block 处停止；
   但本轮尚未量化前缀缺口造成的损失，不能断言它就是主要原因。
5. 若诊断支持，再实现面向连续可恢复前缀的边际准入价值，并做单独消融。
   若不支持，先修成本估计与运行时观测，不通过扩大 sweep 寻找一个好看的阈值。

本轮未改动线上策略算法；没有把这一工作负载上的结果推广到其他模型、容量或生产到达流。
