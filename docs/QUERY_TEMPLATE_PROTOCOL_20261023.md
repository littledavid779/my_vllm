# 问题与回答模板：固定预算的严格拆分

目标：查明之前约 17.36 pp 的收益，分别来自问题位置、格式模板位置，还是二者共同作用。实验不预设模板具有独立创新性。

## 区间与五个方案

沿用 Qwen3-8B 的同一份 token-ID 输入，不改文本、模板、标签或顺序。
Q 从最后的 `Question:` 起，到用户结束标记 `<|im_end|>` 之前，含问题文字及其尾部空白，8–11 token。
T 从该 `<|im_end|>` 起直到输入末尾，固定 9 token：

```text
<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

T 叫格式/回答起始模板，包含用户结束标记，不是单指 `assistant` 这个词。Q 和 T 的并集必须精确等于此前 query 组强制的 17–20 token 区间。
逐输入检查 tokenizer、完整问题文字、模板 ID 和所有边界。最终日志逐请求记录 Q/T 每个位置的选择位图，不能只看强制数量。

| 模式 | Q | T | 其他位置 |
| --- | --- | --- | --- |
| topk | 自然入选 | 自然入选 | 原全局 K 差异 top-k，连接此前实验的参照 |
| none | 禁止入选 | 禁止入选 | 从 Q∪T 之外 top-k |
| question | 全部必选 | 禁止入选 | 余量从 Q∪T 之外 top-k |
| template | 禁止入选 | 全部必选 | 余量从 Q∪T 之外 top-k |
| both | 全部必选 | 全部必选 | 余量从 Q∪T 之外 top-k |

增加 none 与原 topk 的区别，是为了防止“只刷新问题”的剩余 top-k 自动选入模板而污染消融。
四格控制严格纳入/排除；原 topk 额外保留，不能把人为排除 Q/T 的 none 冒充原 baseline。
全部模式总选择量为 max(1, floor(N×10%))，必选部分从总预算扣除，位置去重。
both 与之前的 query 选点规则应一致；通过独立 CPU/GPU 检查验证。

检查层为 1，全部模式使用相同严格位置因果 mask 和检查层前 KV 写回修正。
“禁止入选”只作用于检查层及之后的选择性 KV 修复；早层完整计算和 vLLM 正常生成首 token 的流程仍执行。
长度小于 1024 的 B′ 准备请求和无问题标记的合成预热维持原 top-k，不假造问题或模板。
各模式具有相同的边界解析、GPU→CPU 选点审计和日志成本，包含在 TTFT 中。

## 数据与测量

1. strict_holdout_v1：之前出现 17 pp 收益的 96 个来源、early8k 布局、288 个编辑事件。
2. query_fair_v1：上一轮新增的 48 个来源、natural4k 和 early8k 两种布局，共 288 个事件。

两份数据都已用于本地评测，不再称作未见测试集。第二份两个布局共享来源，按来源聚类处理。
每数据集、每方案重新启动服务/缓存两次，顺序 topk/none/question/template/both 与其反序，共 20 个阶段、5760 次测量编辑请求。
不依据第一份数据的答案调整第二份数据的规则，不挑任务或跳过负结果。
数据锁、模型和软件版本保持不变；原生 full/prefix 记录经哈希、配置与事件检查后仅作质量参考，本轮不报告它们的新延迟。

单卡单客户端 HTTP TTFT 包含 B′ 准备整个请求和主请求到首个生成 token；包括检索、搬运、选择、修复、日志和调度。
不含最初缓存播种、分词和首 token 后的解码时间；不测并发吞吐。
报告每组任务正确率、完整重算正确而本组错误及反向情况、平均/P95 TTFT、按任务/布局/编辑次序统计、实际 Q/T 覆盖和两次重复一致性。

## 预先声明的解释

主要比较是 both-question（已有 Q 时加入 T）与 both-template（已有 T 时加入 Q）。
辅助比较包括 question-none、template-none、both-topk、none-topk。
交互项为 accuracy(both)-accuracy(question)-accuracy(template)+accuracy(none)。
置信区间按来源聚类 bootstrap 1500 次，携带编辑/重复/布局；两次重复不能当作独立样本扩容。
本轮为机制筛查，区间未进行多重比较校正，不用于宣称正式等效或发表门槛。

- 若 both 明显优于 question，支持当前负载/实现中模板具有条件性贡献。
- 若 template 已恢复主要收益而 question 没有，应将解释收窄为模板区间刷新，不能继续把收益全部归给问题语义。
- 若 question 已恢复主要收益而 both 没有额外收益，则模板独立贡献缺乏支持。
- 若两部分联合才有效，需要结合交互项和逐事件结果解释。
- 零差异或不显著不证明普遍等效；等预算下强制某部分会挤占其他位置，因此效应是预算重分配下的策略效应。

即使 T 有效，也不证明“模板重算”新颖：需要核查已有系统是否隐含计算了这些位置，并验证其他模型、模板和真实负载。
三次编辑沿用上游缓存保存行为，不证明每轮近似缓存回写后的长期误差积累。

## 保留、排队与复现

数据盘独立仓库新分支 experiment/query-template，旧实验和原 vLLM 工作区保留。
与此前实验共享环境互斥锁，GPU 0/1 连续三次 30 秒无进程、显存≤100 MiB、利用率≤5%后选择一张；阶段间检查，整轮固定同一卡。
每次等待最多 24 小时，任何失败/中断记录原因并逆序恢复补丁，最后核对默认 runtime 和原工程保护记录。
分析自动输出 query_template_report.md、query_template_summary.json 和逐事件 query_template_cases.json。

```bash
.venv/bin/python scripts/run_query_template.py --validate-only --output /tmp/unused
.venv/bin/python scripts/run_query_template.py --output results/query_template_NEW --gpus 0 1
.venv/bin/python scripts/analyze_query_template.py --run results/query_template_NEW
```

命令依赖已有锁定数据、native 质量记录和模型。本协议于运行前冻结，测量结果独立保存在 results 目录。
