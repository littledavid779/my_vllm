# 问题刷新机制验证：预先固定的对照

目的：检验此前末尾 64 token 修复的收益能否由“问题及回答起始模板始终重算”解释。
这是研究方向的第一道验证，不预设固定尾部策略具有新颖性，也不是 ProphetKV 复现。

## 五组及共同执行条件

所有组使用 Qwen3-8B、检查层 1、严格 token 因果 mask、同一检查层前 KV 写回修正。
总选择量固定为 max(1, floor(N * 0.1))。强制部分先扣预算，其余按原 K 差异 top-k 选择，位置不重复。

| 组 | 强制重算区域 |
| --- | --- |
| topk | 无，原全局 K 差异 top-k |
| query | 从最后的 Question: 标记开始，到 assistant/think 模板末尾的完整 token 区间 |
| tail64 | 最后 64 token |
| pre_tail64 | 最后 64 token 前方的连续 64 token，即 [N-128, N-64) |
| random64 | 从 [0, N-64) 中固定种子无放回抽取 64 个位置 |

完整问题组的强制 token 数按实际问题长度确定，**并非固定 64**；总选择量仍与所有组一致。
问题短语和回答起始模板不分开消融，本轮先回答“普通问题刷新能否解释增益”。
pre_tail64 与 random64 的强制区域不含问题；它们的剩余 top-k 仍允许选择问题。
随机种子 20261023，加序列长度；同长度跨版本和重复使用同一随机集合。仅一个随机种子，不能外推随机策略总体。
长度小于 1024 的 B′ 准备请求沿用原 top-k。合成预热没有 Question 标记，query 组不强制位置；其余组执行各自规则。

运行前锁定 tokenizer 标记，并逐输入验证完整问题及模板落在最后 64 token 且预算足够。
运行时五组都解析相同尾部 token、计算同样的日志覆盖信息；这些成本计入客户端 TTFT。
未读取答案或支持事实位置来选择强制 token。GPU 审计检查所有测量输入的尾部签名、总预算、强制数量和问题覆盖。
问题组在检查层之前已经随全部 token 计算，在检查层及后续层始终保留。
它仍会读到其他位置的旧 KV，不能称为整段精确重算。

## 数据、顺序与测量

开发：已有 round2_sparse_probe，12 个来源、两种布局、72 个编辑事件；五组各一次。
主验证：已有 strict_holdout_v1，96 个来源 early8k、288 个编辑事件；五组各两次新服务/新缓存。
这些来源此前已被评估，不称为未见测试集。顺序为 topk/query/tail64/pre_tail64/random64，第二轮反向。
开发后不依据答案调参、不挑任务、不提前因效果差而跳过主验证。
这是单卡单客户端 HTTP 测量；完整 TTFT 包含 B′ 缓存准备和主请求首 token。
旧 full/prefix 记录经输入哈希、配置和事件完整性核对后只作质量参考，不声称新测 native 延迟。
重复不增加独立来源数；主比较为 tail64-query，来源聚类 bootstrap 1500 次。
辅助比较为 tail64-topk/pre_tail64/random64，全部报告，不能择优包装显著性。
按任务、编辑次数、逐事件得失和全量正确率报告；输出不一致不等于答错。

## 继续或调整方向的依据

- 若 query 已恢复之前的准确率收益，固定 64 暂无额外方法优势证据；以后将问题刷新作为共同基线。
- 若 tail64 优于 query，继续拆解额外边界 token、实际重算形状和预算分配，再跨新数据验证。
- 若其他连续区域同样有效，关注连续区域/执行因素；不能把增益都归因于问题语义。
- 小样本不显著并非等效证明。相近 TTFT 必须看实际均值和 P95，不能用同 token 数替代。

## 排队、保留与运行

数据盘独立仓库分支 experiment/query-controls；原工程保护记录不变，旧 query-tail 分支保留。
轮询 GPU 0/1，要求无计算进程、显存不超过 100 MiB、利用率不超过 5%，连续三次 30 秒采样后使用先空闲的一张卡。
整轮固定同一张卡，阶段间再次等待空闲；不终止其他人的进程。每次等待最多 24 小时，超时记录失败。
任务持有独立环境锁；后台日志和 progress.json 保留 PID、GPU 状态、阶段及失败原因。
结束、失败或 SIGINT/SIGTERM 时恢复研究补丁，并检查默认 runtime 与原工程保护记录。
运行完成自动生成逐请求 JSON、选点审计、统计 JSON 和 Markdown 报告；不自动推送或提交结果。

```bash
.venv/bin/python scripts/run_query_controls.py --validate-only --output /tmp/unused
.venv/bin/python scripts/run_query_controls.py --output results/query_controls_NEW --gpus 0 1
.venv/bin/python scripts/analyze_query_controls.py --run results/query_controls_NEW
```

重跑依赖本机已有锁定输入和 native 控制记录；不是仅克隆仓库即可获得全部数据。
目前尚无这一轮 GPU 结果；CPU 测试通过不能当成准确率或延迟改善。
