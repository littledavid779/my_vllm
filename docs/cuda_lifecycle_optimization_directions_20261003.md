# CUDA 对分层 KV 生命周期优化的帮助与研究方向

日期：2026-10-03

## 1. 当前判断

CUDA 对当前课题有帮助，但最有价值的切入点是 **KV transfer path**：减少 GPU/CPU block copy 的提交开销、降低传输排队、提高 H2D restore 的请求可见时延，并把 CUDA transfer pressure 纳入生命周期决策。

把 admission 打分本身搬到 GPU 上暂时不是优先方向。当前准入决策发生在 scheduler/worker 控制面，计算量很小；真正需要解释的是为什么纯 DMA 时间明显小于 request-visible restore wait，以及为什么减少写入后端到端 TTFT 仍可能变差。

现有实验已经给出一个明确动机：历史成本账本中，H2D DMA 平均约 2.595 ms，但从 promotion 决策到 scheduler 观察完成平均约 10.404 ms。[ShareGPT 成本账本](offload_cost_ledger_sharegpt_20260809.md) 因此，CUDA 优化应重点观察 `host_submit`、copy queue、worker completion 和 scheduler observation，而不能只看 raw DMA。

## 2. 当前 transfer path 的关键特征

当前 CPU offload 已经具备以下基础：

- CPU KV tensor 在可用时使用 pinned memory；
- GPU→CPU 与 CPU→GPU 使用异步 CUDA stream；
- GPU→CPU transfer 会等待当前 model computation stream；
- transfer 使用 CUDA event 判断完成并记录 transfer time；
- 同一方向的 transfer 通过前一个 event 串行化；
- `swap_blocks` 根据 CPU 上的 block mapping 对每个 block 发起异步 copy。

相关实现：[CPU/GPU offloading handler](../vllm/v1/kv_offload/worker/cpu_gpu.py)、[swap_blocks custom op](../csrc/cache_kernels.cu)。

其中最值得关注的是 `swap_blocks`：当前实现对每个 block、每个 KV tensor 单独调用一次 `cudaMemcpyAsync`，并在 C++ 循环中逐项读取 block mapping。这个路径可能产生大量小 copy submission，导致 DMA 本身之外的 host submit 和 driver 开销变成主要成本。

## 3. 优先级一：合并和批量提交 block copy

### 3.1 连续区间合并

先在 host 侧把连续的源 block 和目标 block 合并。例如：

```text
src: 100, 101, 102, 103
dst: 800, 801, 802, 803
```

可以合并为一个四 block 的 copy。对随机映射仍然保留单 block copy。

需要记录：

- 合并前后的 copy operation 数；
- host submission time；
- CUDA event DMA time；
- submit-to-complete time；
- restore wait；
- 实际有效带宽。

这是低风险、容易回退的第一版优化，也能直接验证“小 DMA 请求数量”是否是瓶颈。

### 3.2 `cudaMemcpyBatchAsync`

对于无法合并的 block，可以测试 CUDA Runtime 提供的 `cudaMemcpyBatchAsync`。该接口允许一次提交多个 pointer-to-pointer copy，整个 batch 按 stream 顺序执行，但 batch 内的 copy 不保证相互顺序，因此不能把有依赖关系的 copy 放在同一个 batch 中。

参考：[CUDA Runtime Memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html)。

实现上可以从 CPU block mapping 构造 source pointer、destination pointer 和 size 数组，然后按 layer/tensor 形成 batch。需要保留旧版逐 copy 路径作为兼容 fallback，因为 vLLM 需要支持多个 CUDA Toolkit 版本。

### 3.3 GPU staging buffer

如果 block 映射高度碎片化，可以评估两阶段路径：

```text
GPU scattered KV blocks
    -> GPU contiguous staging buffer
    -> one large D2H copy
```

restore 时反向执行：

```text
one large H2D copy
    -> GPU scatter into KV blocks
```

它会增加 GPU pack/unpack kernel，但可能减少大量小 DMA submission。该方向只有在 profiling 证明 host submission 明显、DMA 带宽利用率较低时才值得实现。

## 4. 优先级二：H2D restore 与 D2H store 的传输调度

H2D restore 通常直接影响请求 TTFT，D2H store 更多是后台成本。当前 lifecycle policy 已经可以决定是否准入，但还没有显式使用当前 CUDA transfer queue 状态。

可以引入方向感知的 transfer scheduler：

- H2D 队列中有请求可见的 restore 时，限制或延迟新的 D2H store；
- 多个 D2H store 尽量合并后提交；
- H2D 队列为空时再利用空闲带宽执行 D2H；
- 当 transfer backlog 持续为负收益时，向 admission policy 反馈压力。

可以先用 host 侧队列实现，不需要马上修改 CUDA kernel。需要新增的在线状态包括：

- H2D/D2H pending bytes；
- 两个方向的 queue depth；
- 最老 transfer 的等待时间；
- 最近窗口的 submit-to-complete；
- 最近窗口的 `restore_wait / dma_time`；
- GPU 的 `asyncEngineCount` 和实际 H2D/D2H overlap。

CUDA 官方文档说明，pinned host memory、非默认 stream 和支持 concurrent copy/compute 的硬件是实现重叠传输的重要条件；具体并行能力需要结合设备的 copy engine 配置测量。[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

## 5. 优先级三：把 CUDA transfer pressure 纳入 lifecycle utility

当前生命周期策略可抽象为：

```text
U = expected_reuse_benefit
    - store_cost
    - capacity_cost
    - promotion_cost
```

可以把 promotion 和 transfer queue 的实时状态加入成本项：

```text
C_restore_active
    = C_dma
    + C_h2d_queue
    + C_worker_completion
    + C_scheduler_observation

C_store_active
    = C_d2h_service
    + C_d2h_queue
    + C_interference_to_restore
```

随后使用：

```text
U(block_prefix)
    = P(reuse) * max(0, C_recompute - C_restore_active)
    - C_store_active
    - C_capacity
```

这条路线可以形成一个更明确的研究问题：

> KV block 的 offload value 是否应该由未来复用价值和当前 CUDA transfer pressure 共同决定？

它也能直接解释当前观察到的现象：selective 可能减少 D2H，却因为错过有效 reuse 或改变 GPU/CPU 队列状态而没有改善 TTFT。

## 6. 优先级四：减少正常运行中的 CUDA 管理开销

当前 transfer handler 会为 transfer 使用 stream 和 timing event。可以做两个低风险消融：

1. 正常运行使用关闭 timing 的 event，只在 calibration/diagnostic 模式启用 timing event；
2. 每个方向使用持久 stream，保留每个 job 的 completion event，减少 stream 创建、跨 stream wait 和资源池管理开销。

当前同方向 transfer 已经通过前一个 event 串行化，因此需要实测“每个 transfer 一个 stream”和“每个方向一个持久 stream”的差异，不能预先假设多个 stream 会带来并行收益。

CUDA event 文档指出，不需要计时时使用 disable-timing event 可以降低 event 的使用成本。[CUDA Event API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)

## 7. 暂缓方向

### 7.1 CUDA Graphs

CUDA Graphs 适合重复执行且拓扑稳定的操作序列。当前 block mapping、block 数和请求状态经常变化；拓扑变化通常需要重新实例化，只有 kernel 参数或地址变化时 graph update 才更有价值。

后续可以针对常见 block 数建立 graph cache，但当前应先解决 copy submission 和 queue pressure。[CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

### 7.2 Unified Memory / managed memory

当前问题明确需要控制 CPU tier、容量和淘汰语义。Unified Memory 会把迁移行为交给运行时，难以直接表达 `DROP`、`RESTORE`、CPU eviction 和生命周期账本。CUDA 文档也指出，managed-memory prefetch 是否使用 copy engine没有绝对保证；对于大块数据，显式 copy 更容易控制和测量。[Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)

### 7.3 只优化 device-to-shared `memcpy_async`

CUDA device-side `memcpy_async` 主要优化 global memory 到 shared memory 的 kernel 内搬运。它不能直接替代 CPU DRAM 到 GPU 的 H2D DMA，因此不能作为当前外部 KV restore 的首要优化方向。

### 7.4 只把 admission 打分搬到 GPU

当前 admission scoring 是有界的前缀扫描，计算量相对 transfer latency 很小。除非 profiling 证明 scheduler CPU 开销已经显著，否则它不会解决当前 TTFT 和 restore wait 的主要问题。

## 8. 建议的实验顺序

### 实验 A：transfer microbenchmark

使用当前模型的真实 KV tensor shape、dtype、block size 和 block mapping，固定 GPU 与 CPU tier，比较：

1. 当前逐 block `cudaMemcpyAsync`；
2. 连续区间合并；
3. `cudaMemcpyBatchAsync`；
4. staging pack/unpack。

主指标：copy operation 数、host submit、DMA time、submit-to-complete、restore wait、带宽和 GPU kernel interference。

### 实验 B：固定 workload 的 transfer backend 对照

固定 `always_restore`，避免 admission 和 promotion policy 混淆，比较：

- no-offload；
- 原始 all-block；
- CUDA 优化后的 all-block；
- CUDA 优化后的 all-block + transfer priority。

继续使用当前 2 GiB workload 和 no-offload 强基线，至少三次重复。只有当 `restore_wait` 和 TTFT 同方向改善时，才进入更复杂的 lifecycle policy 实验。

### 实验 C：transfer-aware admission

在实验 B 确认 CUDA backend 有收益后，比较：

- 当前 lifecycle policy；
- 加入 H2D queue pressure 的 policy；
- 加入 D2H interference cost 的 policy；
- offline four-action Oracle：`OFFLOAD / DROP / RESTORE / RECOMPUTE`。

记录每次 admission 的 shadow score，确保可以区分“策略真的改变了选择”和“独立运行时序不同”两种情况。

## 9. 成功判据

CUDA 优化不应只以 raw DMA 下降作为成功。至少同时满足：

- host submit time 下降；
- submit-to-complete 或 scheduler-visible restore wait 下降；
- H2D restore 没有明显增加 D2H backlog；
- TTFT/P95 在固定 workload 上改善；
- no-offload 基线没有因为后台 transfer 受到明显影响；
- 输出正确性、block mapping、transfer completion 和 event accounting 保持一致。

当前最值得先做的是 `swap_blocks` 的连续区间合并与 `cudaMemcpyBatchAsync` 对照。它改动边界清楚、容易单独测量，也最可能回答“当前负收益有多少来自 CUDA copy submission overhead”这个核心问题。
