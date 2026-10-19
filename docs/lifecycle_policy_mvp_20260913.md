# KV 生命周期收益策略：第一版实现

2026-09-16 晚间已完成独立校准与六组小规模对照，发现保存成本代理导致全拒绝退化；
见 [校准与评测结果](lifecycle_calibrated_results_20260916.md)。尚未证明完整策略有净收益。

后续验证：2026-09-16 已完成三轮真实 GPU 功能测试，详见
[GPU 功能验证报告](lifecycle_gpu_functional_20260916.md)。本页的测试状态记录对应 09-13；
后续功能测试仍不等同于正式成本校准或性能对照。

## 1. 范围与当前结论

本版将开题中的四动作落实为 **显式开启的在线实验策略**：

- CPU store 准入：OFFLOAD / DROP。
- CPU hit 后：RESTORE / RECOMPUTE。
- 估计收益反馈：NORMAL → THROTTLE → OFF，并保留有限探测。

不修改默认策略，不加入 progressive restore、GPU 淘汰重排或请求优先级调度。
实现与单元/连接器回归测试不等于性能验证；本版尚未证明在真实负载中优于
no-offload / all-block，也不承诺“无负收益”。

与开题图的一个重要区别：本仓库 OffloadingConnector 在执行过程中主动提交
已计算 block 的 store，**并非只在 GPU eviction 时提交**。
因此第一版的 Decision 1 接在真实 `prepare_store` 路径，而不是虚构一个
GPU eviction 回调。DROP 只表示本次不写 CPU，不会立即删除仍在 GPU 使用的 KV。

设计背景参见 [开题方法修订](proposal_method_revision_20260813.md)。

## 2. 代码位置

| 位置 | 职责 |
| --- | --- |
| `vllm/v1/kv_offload/lifecycle_policy.py` | 配置、复用启发式、成本分桶、两阶段决策、保护与事件 |
| `vllm/v1/kv_offload/reuse_manager.py` | 提取准入选择钩子；保留旧 threshold 策略 |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` | CPU hit 决策、store 准入、完成/取消反馈 |
| `scripts/build_lifecycle_policy_config.py` | 生成实验 arm 的 KV transfer JSON |
| `tests/v1/kv_offload/test_lifecycle_policy.py` | 确定性成本、准入、保护、容量和状态测试 |
| `tests/v1/kv_connector/unit/test_offloading_connector.py` | 连接器路径回归与在线反馈测试 |

未提供 `kv_connector_extra_config.lifecycle_policy` 时保持既有行为。
提供该配置时默认 promotion 为 `cost_aware`；仍可显式指定
`always_restore` / `always_recompute` 做消融。拒绝与 `store_threshold > 1`
叠加，以免两层过滤器混淆归因。

## 3. 已实现的方法

### 3.1 准入：本次候选 batch 的连续前缀

对每个候选前缀 B 估计：

```text
G(B) = mean(p_b) × max(0, Ĉrecompute(B) − Ĉrestore(B))
       − |B| × (Ĉstore_per_block + Ĉcapacity_per_block)
```

选取满足 `G(B) > admission_margin_s` 的最长候选前缀，否则 DROP。
按整个候选前缀估计恢复/重算成本，不把每个 block 当作独立完整 prefill 相加。
这是线性扫描的启发式，不是生命周期最优解，也不保证先前 chunk 或 CPU 中
所有祖先块仍然存在；CPU 淘汰继续由原 LRU/ARC manager 负责。

`p_b` 来自不同 request 对相同 block hash 的历史出现次数和时间衰减：

```text
repeat = 1 − exp(−max(0, count − 1))
recent = exp(−age / reuse_horizon_s)
p_b = survival_discount × [demand_prior + (1 − demand_prior) × repeat × recent]
```

同一 request 的 scheduler retry 不重复增加次数。该量是**未校准的复用代理**，
不是已经证明准确的 external-demand 概率；`survival_discount` 也是固定折扣，
尚未实现会话身份、工具间隔或动态 CPU 存活概率模型。

### 3.2 Promotion：先验 + scheduler wait 在线修正

```text
Ĉrestore = restore_fixed_s + restore_s_per_block × blocks
Ĉrecompute = recompute_s_per_token × tokens

RESTORE iff Ĉrestore + promotion_margin_s < Ĉrecompute
otherwise RECOMPUTE
```

以上为初值。在 `(action, blocks, tokens, active_requests, inflight_transfers)`
的 log2 分桶中，使用实际选择路径的 `observed_wait / prior` 更新 EWMA 倍率。
同桶样本达到 `min_cost_samples` 后才应用修正；缺样本时回到显式先验。
状态计数是 scheduler 代理，不是实时 GPU 利用率或 PCIe 硬件队列测量。

- restore wait：从策略选择到 scheduler 收到 `finished_recving`。
- recompute wait：从选择到观察到 `num_computed_tokens >= target`。
- 这两个量包含排队/调度粒度影响，不等于 raw DMA 或纯 kernel 时间；
  计时从 lookup 后的选择开始，不包含之前的 lookup。
- 请求未达到目标即结束、被抢占或决策目标改变时，丢弃不完整样本。
- 即使 `track_cost_events=False`，策略反馈仍生效。

只观测被选择的路径，会有选择偏差和模型老化问题。
当前没有 Promotion 路径的强制随机探索；不能把估计的另一条路径写成实测反事实。

### 3.3 Protection：估计预算，而非因果净收益保证

新 block 成功预留 CPU store 空间时，扣除先验 store + capacity 成本；
已存在的 block 和失败的预留不重复扣费。实际 store 后续失败不退款，采用保守账本。
restore 完成时，记入 `predicted_recompute_wait − observed_restore_wait`；
选择 recompute 不再记“节省”，之前已经付出的 store 成本仍保留。

窗口达到时间与样本门槛后，根据窗口估计预算的 EWMA 正负逐级改变状态：

| 状态 | 新的 CPU admission |
| --- | --- |
| NORMAL | 按收益准入 |
| THROTTLE | 每 `throttle_every` 个候选 batch 才允许一次 |
| OFF | 普通准入关闭；每 `probe_every` 次最多探测 `probe_max_blocks` |

OFF **不是释放 CPU buffer 或禁止已有 KV 被恢复**，也不是完整退化为 no-offload；
它是关闭普通新写入。保留探测以便有机会恢复，恢复仍依赖探测产生有效访问。
无新请求时不主动探测。窗口检查由后续准入/选择触发，不起后台定时线程。

当前 store 成本固定、capacity 成本默认 0；未测量动态 D2H 阻塞、CPU 容量影子价格
或对其他请求的干扰。长复用间隔可能使支出与回报落在不同窗口。
因此 Guard 是实验性保护启发式，不能证明端到端净收益非负。

## 4. 如何开启与做消融

先在**独立 calibration split** 上测成本，再生成配置。
每 block 的单位指 offloaded block，不是任意 GPU block；block size、模型、
dtype、TP 和硬件必须与校准一致。不要把历史不同配置的均值拼成“准确模型”。

下面仅演示配置生成，数值来自功能测试尺度，**不是 A100 性能参数，不能直接用于报告**：

```bash
source /home/ljc/vllm/.venv/bin/activate
python scripts/build_lifecycle_policy_config.py \
  --arm full --cpu-bytes 1073741824 --block-size 16 \
  --store-ms-per-block 2 --restore-ms-per-block 2 \
  --recompute-ms-per-token 1
```

输出是 `--kv-transfer-config` 可接收的 JSON。校准后，可通过既有脚本的
`KV_TRANSFER_CONFIG_JSON` 环境变量传入，选择 `lru`，保持 GPU KV 策略不变：

```text
KV_TRANSFER_CONFIG_JSON=<生成的完整 JSON>
INPUT_FILE=<固定的 Direct ShareGPT trace>
NUM_GPU_BLOCKS_OVERRIDE=<固定 GPU blocks>
RESULT_ROOT=<本轮独立结果目录>
scripts/run_trace_driven_value_policy_online_sweep.sh lru
```

上面是配置关系示意，不是可直接执行的 shell 命令。实际运行同时固定
GPU_ID、CPU 容量、请求数、并发度、max model length 和 seed。

| 对照 | 生成器 arm | Admission | Promotion | Guard |
| --- | --- | --- | --- | --- |
| no-offload | 不传 KV transfer 配置 | 无 CPU tier | 重算 | 无 |
| all-block | `all-block` | 全部 | always restore | 无 |
| admission-only | `admission` | 收益准入 | always restore | 关闭 |
| promotion-only | `promotion` | 全部 | cost-aware | 关闭 |
| joint-no-guard | `full --disable-guard` | 收益准入 | cost-aware | 关闭 |
| full | `full` | 收益准入 | cost-aware | 开启 |

本版先验与在线参数可在 JSON 的 `lifecycle_policy` 内调整，未知键或非法数值
启动时报错。实验必须保存最终 JSON，而不只保存 arm 名称。

## 5. 观测与测试

已有 `offload_*` 生命周期、字节和成本事件继续保留。新增 `lcuc_*` 包括：

- `restore_decisions` / `recompute_decisions` 与对应 `observations`。
- `estimated_store_charge_s` / `estimated_net_utility_s`。
- `guard_state` / `guard_transitions` / `probe_batches`。
- `decision_time_s`、历史/成本桶大小、`pending`。
- 有界 `events`：准入候选与选中数量、路径预测、完成时间和状态变化。

计数器键带 `lcuc_` 前缀，未发生的事件计数可缺省，消费者应使用 `get(key, 0)`。
`reset=True` 只清区间计数和事件，不清历史模型、保护状态、探测周期或在途观测。
事件队列有界，溢出会计数；它是调试窗口，不是无限全量 trace。

CPU/offline 回归命令：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest \
  tests/v1/kv_offload/test_lifecycle_policy.py \
  tests/v1/kv_offload/test_cpu_manager.py \
  tests/v1/kv_connector/unit/test_offloading_connector.py -q
```

本轮未启动 GPU benchmark，未产生新的 TTFT 或吞吐结论。
最终验证：上述三个完整测试文件共 **73 passed**（15.86 秒，16 条依赖弃用警告）；
配置生成器的 all-block / admission / promotion / full / full-no-guard 均有自动测试。
修改的六个 Python 文件通过 Ruff 0.14.0 检查，`git diff --check` 通过。
仓库 pre-commit Ruff hook 在安装隔离环境依赖时因
`Missing dependencies for SOCKS support` 失败；已用同版本 Ruff 直接验证，
没有将 hook 的环境安装失败记作通过。

下一步先独立校准，再跑相同 trace 的六组小规模 smoke，确认有实际 store/load、
两种 promotion 决策和正常输出；之后才进行多 seed 正式评测。
报告 TTFT、吞吐、D2H/H2D、完成恢复块、重算 tokens 和决策开销，
并将 `estimated_net_utility_s` 与端到端对照结果分开呈现。
