# 编辑后后缀 KV 复用：CacheBlend 复现与问题刷新实验

目标是在已有方法的基础上，在相近端到端 TTFT 下提高任务质量。这个目录是独立 Git 仓库；现有 `/home/ljc/vllm` 的分支、暂存区、研究改动和 `.venv` 均保留。

## 当前代码与 GitHub 分支（2026-10-24）

GitHub 分支为 [`research/kv-suffix-query-boundary-20261024`](https://github.com/littledavid779/my_vllm/tree/research/kv-suffix-query-boundary-20261024)。分支根目录是这个独立实验工程，保留从首次复现到当前实验的 15 个代码与证据提交；它与同一仓库里的完整 vLLM 工程使用独立历史。此次发布在实验提交之上只更新本 README，不修改实验实现。

| 组成 | 当前使用的版本 |
| --- | --- |
| 正在测量的实验代码 | `73cf1ebf04b77db05c6ce546e06f585c48f13315`；本地分支 `experiment/query-boundary` |
| vLLM | `v0.11.0`，`b8b302cde434df8c9289a2b465406b47ebab1c2d` |
| LMCache / CacheBlend 实现 | `v0.3.13`，`fc031d471a566edb6d49a86c9116cc23cfb04111` |
| 模型 | Qwen3-8B，单卡运行；权重不进入 Git |
| 版本与依赖锁定 | `sources.lock.json`、`requirements.lock` |

当前研究是在固定版本 LMCache 的 K 差异选择器之上，通过独立补丁比较问题计算方式；不能将所有实验组统称为未经修改的官方 CacheBlend。共同后端使用严格原位置因果注意力与检查层前 KV 写回修正；三种近似组分别采用全局 top-k、预算内强制重算问题及模板、问题及模板不入缓存而由 vLLM 正常前向。另设完整重算和精确前缀缓存。各近似组名义预算为 10%，实际计算口径和端到端 TTFT 见协议。

- 当前实验协议：[问题缓存边界与输入表达](docs/QUERY_BOUNDARY_PROTOCOL_20261024.md)。
- 运行和分析入口：`scripts/run_query_boundary.py`、`scripts/analyze_query_boundary.py`。
- 新增机制补丁：`scripts/apply_query_boundary_patch.py`、`patches/lmcache-query-boundary.patch`；只应用到独立 `.venv`，不改写上游源码 checkout。
- 上一轮已完成结果及核验：[问题与模板结果留存](docs/QUERY_TEMPLATE_RESULTS_AND_NEXT_20261024.md)。其中的“下一轮尚未启动”描述的是留存时的状态；当前实验已在 GPU1 启动，发布时仍在测量，尚无正式结论。
- 本次发布前，当前实验的 80 项 CPU 测试通过；三个 100% 预算 GPU smoke 均通过任务正确性和实际计算位置审计。smoke 不作为 10% 预算的质量或性能结果。

代码、补丁、配置、测试、实验协议和已提交的结果摘要进入 GitHub。模型、上游源码 checkout、环境、原始输入和运行日志由 `.gitignore` 排除；原始证据仍在本机数据盘。代码发布不等于原始实验数据的异地备份。

后续开发请在单独目录检出此分支，运行中的工程保持冻结：

```bash
git clone --single-branch --branch research/kv-suffix-query-boundary-20261024 \
  https://github.com/littledavid779/my_vllm.git kv-suffix-research
cd kv-suffix-research
git switch -c experiment/next-study
```

新目录需按下文重建专用环境。当前脚本还依赖本机模型路径、被忽略的输入文件及原工程保护记录；仅克隆代码不能立即重跑全部实验。请先准备输入并核对协议、输入锁文件和本机路径，勿让新实验共享正在使用的可写环境或 KV 缓存。

## 版本与目录

- `sources.lock.json`：官方仓库 URL、tag、commit 和关键算法文件的 SHA256。
- `upstream/lmcache/`：LMCache v0.3.13，包含官方 Qwen3 blending 实现；独立上游 Git checkout。
- `upstream/vllm/`：vLLM v0.11.0 原始源码；独立上游 Git checkout。
- `requirements.in` / `requirements.lock`：直接依赖与完整依赖、下载哈希。
- `.venv/` / `.uv-cache/`：专用环境与下载缓存。
- `patches/`：初始化顺序、token 容器及传输参数的兼容补丁。
- `configs/` / `scripts/`：实验配置及可重跑入口。
- `results/`：原始运行日志与 manifest，不进入 Git。
- `state/parent_before.json`：原工程状态校验，不进入 Git。

上游源码、模型权重、环境和原始结果均不作为普通文件提交到本项目；上游由 commit 与文件哈希锁定。模型只读取 `/hole0/model_info/model_zoo/downloads/Qwen3-8B`，不复制权重。

## 初始官方复现阶段范围

先运行固定版本官方 `examples/blend_kv_v1/blend.py` 的重排文本块流程，验证非前缀复用与检查层选点实际发生。
通过包装器只调整模型路径、上下文上限、单卡显存额度以及结果保存；不替换官方 selector，不接入我们已有的 tail/hybrid 策略。
默认检查层为 1，重算比例为 0.15，CPU 后端、FlashAttention，与该版本官方示例默认设置一致。

这是官方发布实现的运行复现，使用其支持的 Qwen3 模型扩展；不等于已经复现 EuroSys 论文的全部原始模型、数据集和图表。
首个 smoke 的计时是带 DEBUG 日志的一 token 离线生成计时，不作为 HTTP TTFT、任务准确率或优化收益。

## 环境与运行

```bash
cd /home/ljc/vllm/reproductions/cacheblend
uv --cache-dir .uv-cache venv --python /usr/bin/python3.12 .venv
.venv/bin/python scripts/bootstrap_sources.py
uv --cache-dir .uv-cache pip sync --python .venv/bin/python --require-hashes requirements.lock
.venv/bin/python scripts/verify_sources.py
.venv/bin/python scripts/apply_runtime_patch.py
.venv/bin/pre-commit install
.venv/bin/python scripts/run_smoke.py --output results/smoke_NEW
```

已有公开来源负载的 HTTP 管线验证：

```bash
.venv/bin/python scripts/prepare_pilot.py \
  --source /home/ljc/vllm/results/suffix_public_http_corrected_gpu1_20261021/input.json
.venv/bin/python scripts/run_http_pilot.py --output results/http_NEW
```

`prepare_pilot.py` 对已有标准答案负载插入官方分块分隔符，并记录每次新 B′ 的 token；标准答案不变。输入及来源哈希保存在 `data.lock.json`，原始 token 数据保存在忽略的 `data/`。warmup 使用不与测量输入重合的合成块，避免把被测编辑提前变成完整缓存命中。
HTTP 验证逐个启动独立服务，对比完整重算、原生前缀与官方 CacheBlend 15%，客户端通过 SSE 的首个生成 token 计时；不是原论文数据集复现或新的优化结果。
该固定版本的检索遇到首个未缓存块会停止。为了验证真实后缀融合，CacheBlend arm 先单独计算新 B′ 的块缓存，额外请求的完整耗时计入客户端 TTFT；同时启用官方 `save_unfull_chunk` 选项保留完整末段。原生 full/prefix 直接处理最终输入。仅报告主请求耗时会漏掉这项适配成本。

复现初次建立时，`parent_guard.py record` 已保存原工程状态；后续使用 `parent_guard.py check`，不覆盖初始记录。
在新 worktree 中，先用 `parent_guard.py record` 建立该 worktree 的独立状态记录。
`apply_runtime_patch.py` 只允许修改此目录 `.venv` 中三个与固定上游源码一致的目标文件，遇到其他版本立即停止；`--restore` 可恢复其上游原始内容。
兼容修改来自 LMCache 固定版本的示例 README：模型加载后登记模型，再初始化 KV transfer。官方算法文件保持原样。
另有一行版本兼容修改：在 LMCache 的 vLLM 请求适配器中，将 `request.all_token_ids` 转为普通 `list`，使 vLLM 0.11 的只读 `ConstantList` 能通过 msgspec 序列化。token 内容与顺序不变，重算打分与选点不变。
补丁通过原子替换文件应用，避免修改 uv wheel cache 的硬链接；安装缓存保持上游原始内容。
该版本 buffer connector 的两个全注意力传输调用还漏传了 pybind 要求的 `use_mla` 参数；第三个补丁按 C++ 声明的默认语义显式传入 `False`，不修改 CUDA 内核或重算策略。

## Git 与回退

最初在 `baseline/cacheblend` 分支保存复现脚本、版本、配置和兼容补丁，跑通后打带阶段名的 tag。以下 baseline 分支和标签是本地历史检查点；本次只发布上方的研究分支，不自动发布所有本地引用。
已验证的官方示例检查点为 `baseline/official-smoke-20261022`；状态和证据入口见 [本次复现记录](docs/STATUS_20261022.md)。
HTTP 管线检查点为 `baseline/http-pilot-20261022`；已从该检查点预建 `research/edit-aware` 分支，尚未加入算法优化。算法修改保存在独立补丁并记录对应上游 commit，不在现有主工程内做试验。
对照时可用 `git worktree add ../cacheblend-baseline <已验证的baseline-tag>` 创建单独目录；按该目录的锁文件和配置重新建立环境。新目录不共享可写的 Python 环境或 KV 缓存。
不使用 `reset --hard`、`clean` 或覆盖原工作区来回退。本次按用户要求将独立实验历史发布到专用研究分支；原有远端分支、原 vLLM 工作区和运行中实验的分支与提交均保持原样。

独立实验目录可以整体迁往数据盘，并在 `/home/ljc/vllm/reproductions/cacheblend` 保留软链接，使虚拟环境的旧 shebang 和现有入口继续有效。迁移需保留环境与缓存之间的硬链接（如使用 `rsync -aH`），完成内容校验和运行验证后再移除原副本。`parent_guard.py check` 使用初始状态文件内记录的原工程路径，不根据迁移后的物理目录推断；不要重建或覆盖该初始记录。旧结果中保存的绝对路径作为历史证据保留。

## 后续验收

1. 官方 smoke：模型成功加载、重排请求命中、实际执行 top-k 选点，记录真实导入位置和环境。
2. 同框架比较：完整重算、精确前缀与官方 CacheBlend，统一 KV 来源、输入、评分和计时。
3. 先得到 baseline 的质量—TTFT 曲线，再在独立优化分支修改一个机制；相近延迟比较均值和 P95，计入选择开销。
4. 原有人工构造任务仅用于诊断；公开任务与中间编辑衍生任务分开汇报。连续编辑沿用各方法自己的缓存，并增加从精确缓存重新开始的对照。

第二轮已在独立 `experiment/cacheblend-round2` 分支扩展官方预算扫描；输入构建、排除规则和计时口径见 [预定实验协议](docs/ROUND2_PROTOCOL.md)。主实验没有修改官方算法，原有 baseline 标签与研究分支保持原样。
第二轮已完成，见 [结果与后端核验](docs/ROUND2_RESULTS_20261022.md)：主实验保留官方发布配置，额外诊断补丁独立开关并已恢复。没有取得总体准确率改善，尚不能将发布实现的质量缺口归因于论文选择策略。

严格因果校准与两轮等延迟对照已完成，见 [严格因果结果](docs/STRICT_CAUSAL_RESULTS_20261022.md) 和 [协议](docs/STRICT_CAUSAL_PROTOCOL.md)。独立分支 `experiment/strict-causal` 保存可开关的 mask、检查层前 KV 写回诊断及自动恢复脚本；运行后已恢复发布版算法文件。96 来源上的准确率仅增加 1.04 个百分点，区间跨零，尚未证明稳定质量收益。

参考：[CacheBlend 论文](https://arxiv.org/abs/2405.16444v3)、[固定版本 LMCache](https://github.com/LMCache/LMCache/tree/v0.3.13)、[官方兼容说明](https://github.com/LMCache/LMCache/blob/v0.3.13/examples/blend_kv_v1/README.md)。

固定 10% 预算的问题末段 64-token 修复已完成，协议与结果见 [本轮结果](docs/QUERY_TAIL_RESULTS_20261022.md)。配置和对照保存在 `experiment/query-tail` 分支，运行结束后恢复发布版算法。

问题刷新机制对照：见 [预先固定的协议](docs/QUERY_CONTROLS_PROTOCOL_20261023.md)。

共同问题刷新与新来源验证：见 [预先固定的协议](docs/QUERY_FAIR_PROTOCOL_20261023.md)。

问题与模板严格拆分：见 [实验协议](docs/QUERY_TEMPLATE_PROTOCOL_20261023.md)。
