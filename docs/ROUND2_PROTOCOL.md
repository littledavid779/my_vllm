# 第二轮：先检验研究空间

在查看本轮模型输出之前固定以下范围。只运行官方已锁定实现，不加入新的修复机制。

- 公开来源：BABILong QA9、QA2、QA3 各 36 个独立原始样本，共 108 个；跳过上一轮的源索引 0、1。
- 固定选择规则：按源顺序取前 36 个允许单条事实改变答案的样本，选择最早能改变答案的事实；不按模型正确率筛选。全部 300 个官方原始标签先由解释器重放校验。
- 每个来源产生两个布局：`natural4k` 保持原始背景和事实位置；`early8k` 将全部事实按原顺序放在前面，再放 7600 token 的去事实背景。后者是明确的人造位置/长度压力测试，二者分别报告。
- 每条链 3 次等长替换。修订号使相同事实恢复时仍是新的输入，避免重复请求完全命中；每条链的 B、C 带独立标识，避免不同源/布局交叉复用。
- 沿用作者指令与示例、单词答案格式；背景和分隔符改造都使其不再是官方排行榜分数。长背景有重复文本，这一限制随结果报告。
- 完整重算、原生精确前缀、官方 CacheBlend 5% / 15% / 30%，共 3240 次编辑测量；CacheBlend 新 B′ 准备的完整请求耗时计入客户端 TTFT。相同 token-ID / SSE / 评分接口。
- GPU 1、单请求串行、检查层 1、官方默认 FlashAttention 路径、CPU cache、eager；所有方法统一 12288 上下文上限。上游 selector 不变。未声称覆盖官方 sparse 强配置。
- 每个请求测一次，按来源聚类做质量统计；同一来源的两个布局和三次编辑不能当作六个独立样本。延迟是筛查结果，需重复和更强配置才可形成正式性能结论。

主要观察：完整重算答对时 CacheBlend 新增错误的比例、错误涉及的独立来源数、预算上升能否修复、精确前缀在相近质量下的均值与 P95、前移长后缀是否改变优势。
输出不一致另报，不能代替有标准答案的错误。完整重算本身较差的类别全部保留，但不用于证明精度保持。

继续投入的条件：找到跨多个独立来源重复出现、对预算有响应的缓存新增错误，同时至少有一类合理场景的后缀复用延迟优于精确前缀。若只是单个特例，或精确前缀在相关场景始终占优，则不立即设计复杂新方法。
这是一轮方向筛查，不用测试结果反向调整任务、提示或阈值。连续编辑首先遵循上游实际的块保存/跳过逻辑，不将其解释为已经验证了每次修复结果向下一轮传播。

运行入口：

```bash
.venv/bin/python scripts/build_round2.py --output data/round2_screen.json
.venv/bin/python scripts/run_http_pilot.py \
  --input data/round2_screen.json --lock data/round2_screen.lock.json \
  --max-model-len 12288 --port 18265 \
  --arms full prefix cacheblend5 cacheblend15 cacheblend30 \
  --output results/round2_NEW
.venv/bin/python scripts/analyze_round2.py --run results/round2_NEW
```

## 运行中发现的后端因素：追加诊断

默认 FlashAttention 的 `LMCFlashAttnMetadata.update_from_top_indices` 只更新压缩后的 Q 长度；`LMCFlashAttnBackend` 仍传入 `causal=True`，没有传原始离散 Q 位置。已安装 FlashAttention 接口明确规定长短 Q/K 时因果 mask 按右下对齐。
例如长度 5、选择原位置 `[0, 3]`，压缩后两个 Q 会按位置 `[3, 4]` 建 mask，允许访问原位置对应的未来 K。这个后端差异可能影响质量，不能将本轮所有新增错误直接归因于 token 选择策略。

因此在观察到 5% 预算出现多来源错误后，追加官方 FlashInfer sparse 路径小对照。按每类排序最前四个来源固定选取，共 12 来源、24 条链；不依赖其模型输出或是否失败。使用相同输入、同一框架的 full/prefix 控制和官方 sparse 15% 配置。它是探索性后端诊断，不替代预定的 108 来源结果，也不是提出新方法。
官方 sparse 模式由 `VLLM_ATTENTION_BACKEND=FLASHINFER` 和 `LMCACHE_EXTRA_CONFIG={"enable_sparse": true}` 开启；仍使用锁定的 FlashInfer 0.3.1。初始化/JIT 开销单独留档，失败也保留日志。
CPU 调用还确认：sparse metadata 的 `top_indices_slice //=` 会通过视图修改传入的 token 索引。例如输入 `arange(64)` 的第 31、63 个索引变成 0、1；blender 后续使用的索引与该张量共享引用。因此 sparse 原始运行结果也需要结合此实现问题解释，本轮不把修正它视作新研究机制。

```bash
.venv/bin/python scripts/make_backend_probe.py \
  --input data/round2_screen.json --output data/round2_sparse_probe.json
.venv/bin/python scripts/run_http_pilot.py \
  --input data/round2_sparse_probe.json --lock data/round2_sparse_probe.lock.json \
  --max-model-len 12288 --max-num-seqs 16 --port 18265 --attention-backend sparse \
  --arms full prefix cacheblend15 --output results/round2_sparse_NEW
```

vLLM 0.11 FlashInfer 初始化固定构造 9 条 warmup 序列；`max_num_seqs=1` 会在启动时数组广播失败。因此小对照的 full/prefix/sparse 都使用容量 16，但客户端实际并发仍为 1。第一次失败没有测量结果，不计入质量或延迟汇总。

原始 sparse 对照运行成功后，进一步发现 `is_causal=False` 被注释，后续一直走普通右下对齐 causal 分支，并没有执行已构造的 sparse mask。追加同一固定输入的正确性诊断：启用该分支，并将 mask 用的切片改成 `.clone()`，以防原地整除污染 KV 写入位置。CPU 回归检查要求“计划的 block mask 不变、传入索引不变、选择后切换 sparse 路径”。该补丁是 baseline 正确性校正，不能算作新选择算法，也不据此声称超过论文方法。
应用与恢复都仅修改专用 venv，原上游 checkout 不动；运行配置和校验显式区分原始/修正版。
真正运行该分支后，还暴露了旧 `HackBSAWrapper` 与 FlashInfer 0.3.1 的接口/布局不一致：缺少 `workspace_size`，仅补参数后 GQA 布局又触发非法显存访问。最终诊断改用 FlashInfer 自带 `VariableBlockSparseAttentionWrapper`，在入口/出口转换 NHD/HND 布局；block mask、检查层和 top-k 打分仍不变。
这仍不是严格因果的最终 baseline：官方按 32 个选择查询分组近似 mask，组内原始位置可能相距很远；仍需对照逐 token 的严格因果执行。诊断流程没有取得整体质量改善时，不将其作为优化方案提交。

```bash
.venv/bin/python scripts/apply_sparse_index_patch.py
.venv/bin/python scripts/run_http_pilot.py \
  --input data/round2_sparse_probe.json --lock data/round2_sparse_probe.lock.json \
  --max-model-len 12288 --max-num-seqs 16 --port 18265 --attention-backend sparse \
  --sparse-correctness-fix --arms full prefix cacheblend15 --output results/round2_sparse_fixed_NEW
.venv/bin/python scripts/apply_sparse_index_patch.py --restore
.venv/bin/python scripts/verify_runtime.py
```

失败日志依次保存在 `round2_sparse_20261022`、`round2_sparse_fixed_20261022`、`round2_sparse_fixed_workspace_20261022`；最终成功的 block-mask 诊断为 `round2_sparse_public_api_20261022`。最终运行通过 `--controls-from` 复用已完整测量的 FI full/prefix 对照，输入哈希、容量、后端与事件完整性均校验；manifest 保留控制来源及内容哈希，不把复用的 144 条记录当成新测量。
