# 严格因果执行校准与等延迟对照

本轮完成了严格 token 级因果 mask、检查层前 KV 写回修正和两轮 HTTP 对照。在这批长后缀输入上，严格 mask 比 block mask 的实际任务正确率高 1.04 个百分点，平均 TTFT 低 1.87%；正确率差值区间跨零，不能认定得到稳定精度收益。P95 增加约 5.6%，也未达成尾延迟无回退。

这是固定发布实现的执行校准，不是新的 token 选择算法，也不能当成原论文全部实验的复现。原有 baseline 标签和原工程状态均保留。

## 执行机制与核验

固定 LMCache 0.3.13 / vLLM 0.11.0 / Qwen3-8B，单卡 A100、BF16、等长替换、检查层 1、K 差异 top-k。严格模式保存选中查询的原始位置 p，mask 逐 token 满足 j <= p，使用 FlashInfer 0.3.1 的公开 custom-mask 接口；每次请求仅构建、打包一次 mask，后续层复用。未改变上游选择分数或 RoPE 位置。

GPU primitive 核验覆盖长度 7/67/129、非连续索引、31/32 边界、32 Q heads / 8 KV heads / head size 128。打包位与 mask 完全对应，索引未被修改；与 float32 参考的最大绝对差 0.00837，与完整 causal attention 对应行的差不超过 0.00390625。将未来 K/V 扰动 100 后，过去查询输出差为 0。证据：`results/strict_attention_validation_20261022.json`。

100% 预算控制曾出现 3 个 full 正确、该模式错误的事件。定位到检查层之前已计算的新 K/V 没有写回 connector 最终导出的 GPU 缓冲，尤其分隔符缺口可能保留零值。独立开关将这些全量计算层的 K/V 复制回原缓冲；选点后的分支未改。修正后只剩 1 个上述事件，与原生精确前缀在同一事件上的错误一致；不主张位级一致。证据分别位于 `results/strict_causal_pilot_20261022/` 和 `results/strict_causal_kvwrite_pilot_20261022/`。

下面严格/block 两组均使用相同的检查层前写回修正。block 对照还包含上一轮已记录的索引保护、激活 block mask 及公开稀疏接口适配，因此是本地校准对照，不能称为未经修改的官方发布版。

## 输入、预算冻结与计时

开发小对照有 12 个来源；扩展排除这 12 个来源，保留另外 96 个来源，每类 QA2/QA3/QA9 各 32 个，每来源连续 3 次编辑，共 288 个独立编辑事件。所有样本保留，没有按完整重算是否正确筛选。这些来源来自第二轮已经使用的同一个池，不能称为此前从未评估过的全新测试集。

布局仅 early8k：事实前移到约 4% 位置，后面补充重复的无事实背景，平均总长约 8044 token，后缀约 7666 token。这是 BABILong 来源的编辑衍生压力测试，不是官方 benchmark 分数。

只依据开发集耗时选择 strict 预算：5% / 10% / 15% 分别为 188.31 / 233.66 / 305.86 ms，block 15% 为 236.00 ms。选定 strict 10% 后写入 `calibration.json`，扩展期间没有再调参。该比较回答相近实际延迟下的质量，不能独立分离预算与 mask 的贡献。

两次服务重启后重复 strict/block，运行顺序交换，每次缓存重新建立。native full/prefix 在扩展第一轮新测一次，之后复用的记录不当作新观测。严格/block 各有 576 条观测，但仍只有 96 个独立来源和 288 个独立编辑事件。两轮每个事件的正确/错误判定均无变化。

TTFT 从客户端准备新 B′ 开始，包含 B′ 预请求的完整耗时以及最终请求首个生成 token 前的检索、融合与服务开销。通过 SSE 首个生成 token 计时。单客户端串行请求；没有测并发吞吐。max model length 12288，服务 max num seqs 16 为后端初始化配置，不能理解为实测并发 16。

## 总体结果

| 方法 | 每轮正确数 | 正确率 | 平均 TTFT (ms) | P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| 完整重算 | 140/288 | 48.61% | 718.36 | 779.19 |
| 精确前缀 | 140/288 | 48.61% | 702.58 | 725.99 |
| block mask 15%，校准对照 | 84/288 | 29.17% | 245.79 | 263.18 |
| 严格 mask 10% | 87/288 | 30.21% | 241.20 | 277.83 |

严格/block 的延迟合并两次独立服务运行；正确数每轮均相同。严格相对 block：

- 正确率增加 1.04 个百分点，按来源聚类 bootstrap 1500 次的 95% 区间为 **[-3.82, +5.90] 个百分点**。
- 平均 TTFT 变化 -1.87%，区间 [-2.57%, -1.02%]。
- P95 TTFT 变化 +5.56%，区间 [-3.33%, +9.47%]。

区间只反映本批来源抽样；重复运行不能增加来源数，也不能替代长时间系统噪声评估。未隔离整机 CPU/内存资源。严格 mask 平均耗时约为完整重算的三分之一，但正确率也明显低于完整重算，尚未满足研究目标。

## 错误集中在哪里

| 类别 | 完整重算 | block mask | 严格 mask |
| --- | ---: | ---: | ---: |
| QA2：对象位置、两条支持事实 | 48/96 | 33/96 | 43/96 |
| QA3：三条支持事实 | 8/96 | 12/96 | 6/96 |
| QA9：否定条件 | 84/96 | 39/96 | 38/96 |

QA2 有局部改善，但这属于分项探索，不据此宣布泛化收益。QA3 的 full 也很低，保留该项并明确模型/负载能力边界。QA9 是剩余缺口的主要来源：严格模式每轮有 54 个 full 正确而本方法错误的事件，其中 QA9 为 46 个、QA2 为 6 个、QA3 为 2 个。

例如 `qa9_006_early8k`：旧事实为 John 不再位于 hallway，修改后为 John 位于 hallway；标准答案应从 no 变成 yes，完整重算回答正确，严格/block 均仍回答 no。这是带标准答案的失败，不是仅凭输出不同判错。

严格模式 54 个新增错误中，有 47 个输出等于初始标准答案，48 个等于上一版本标准答案；编辑轮次分布为第 1/2/3 次 26/4/24 个。答案标签重合只能支持旧状态偏置的诊断，不能证明某个具体 KV 是原因，也不能将交替标签下的波动解释为逐轮误差积累。

block 每轮有 68 个 full 正确而它错误的事件，严格模式为 54 个；但 full 错误时 block 有 12 个反向答对，严格仅 1 个。因此新增错误减少 14 个，最终实际正确数仅净增 3 个。保真度改善与任务准确率改善须分别报告。

## 适用边界与下一步依据

当前 C 包含问题文本和 assistant 模板，因此相同问题的旧 KV 也参与复用。固定上游 `examples/blend_kv_v1/blend.py` 的示例把 query 放在单独末段，三次示例请求的 query 文本不同。两种构造存在明确差异，不能据本轮结果断言原论文在原始 RAG 负载上出现同样错误，也不能将重算 query 末段直接包装成新机制。

现阶段值得优先核验的问题是：在严格因果参考下，分别控制问题末段是否全量重算，并让 full、发布版基线和改进版使用相同上下文/问题缓存边界，再判断否定状态变更能否在同一实测预算内修复。只有该基线可比性问题澄清后，才适合提出编辑依赖选点等研究机制。本轮未执行该后续实验，也未宣称获得新算法收益。

三次编辑沿用上游实际缓存保存策略，尚未专门验证每轮把近似 KV 写回 CPU 缓存后的误差累积。自然长文、多模型、真实混合负载、吞吐与线上尾延迟均未在本轮覆盖。

## 重现、结果与回退

分支 `experiment/strict-causal`；执行核心检查点 `0c00d3d`。补丁脚本提供 `--restore`，主实验 driver 在 finally 中恢复 strict/block 与 KV 写回补丁。结束后默认 `verify_runtime.py`、`verify_sources.py`、`parent_guard.py check` 均通过，GPU 1 显存为 0 MiB、无实验进程。

```bash
cd /home/ljc/vllm/reproductions/cacheblend
.venv/bin/python scripts/apply_strict_causal_patch.py
.venv/bin/python scripts/validate_strict_attention.py --gpu 1 --output results/strict_attention_NEW.json
.venv/bin/python scripts/apply_strict_causal_patch.py --restore
.venv/bin/python scripts/run_strict_comparison.py --output results/strict_comparison_NEW --gpu 1
.venv/bin/python scripts/analyze_strict_comparison.py --run results/strict_comparison_NEW
.venv/bin/python scripts/verify_runtime.py
.venv/bin/python scripts/verify_sources.py
.venv/bin/python scripts/parent_guard.py check
.venv/bin/python -m pytest tests -q
.venv/bin/pre-commit run --all-files
```

输入需先按 [协议](STRICT_CAUSAL_PROTOCOL.md) 和锁文件建立。当前 driver 的预算校准还需要本机已保存的 `results/strict_causal_pilot_20261022/`（native 控制）与 `results/strict_causal_kvwrite_pilot_20261022/`（strict 15%）；在新机器复现前应先用 `run_http_pilot.py` 重测这些阶段，不能把未随 Git 提交的数据视为可自动下载的产物。本次原始目录：`results/strict_comparison_20261022/`；内含各阶段完整 manifest、source_snapshot、服务日志、输入、逐请求输出。分析入口为 `strict_summary.json`、`strict_error_cases.json` 和 `strict_report.md`；`analysis_snapshot/` 保存分析程序、本文和 primitive 核验的副本及哈希。原始结果不提交 Git。
