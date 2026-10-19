# 开题 PPT 方法页第二版：生命周期反事实收益控制

日期：2026-08-13

本文件替换第一版总纲中的第 12--16 页。问题动机和前期实验不变，方法表达从
“Ledger + Estimator + Admission + Promotion + Guard”收敛为一个统一控制问题。

## 一句话方法

提出面向 KV 完整生命周期的在线反事实收益控制器：将 store、restore、
recompute 和容量占用换算为 request-visible time cost，在 GPU eviction 时比较
OFFLOAD 与 DROP 的未来净收益，在 CPU hit 时比较 RESTORE 与 RECOMPUTE 的预计
等待成本，并用已实现收益反馈抑制或关闭负收益的 CPU tier。

建议方法名：

Lifecycle Counterfactual Utility Controller（LCUC，生命周期反事实收益控制器）。

方法名只是沟通工具。新意必须由耦合决策、成本语义、Oracle gap 和在线结果
支撑，不能把命名本身当作贡献。

## 第 12 页：统一解法与成本坐标

核心结论：

> 两个决策点解决同一个问题：选择期望 request-visible cost 最低的生命周期路径。

统一成本定义：

    C_s    = runtime 观测到的 CPU store 成本
    C_r    = scheduler-observed restore stall
    C_c    = 相同服务状态下估计的 recompute cost
    C_cap  = CPU 容量占用及驱逐其他 KV 的机会成本

复用模型只输出有效 external demand 概率。sharing、chain、session gap 是
P_demand 的输入，而不是最终 cache value。

已有证据解释为什么必须使用 request-visible cost：

    raw H2D DMA              2.595 ms
    worker completion        5.140 ms
    scheduler-observed wait 10.404 ms

因此不能只使用 bytes / PCIe bandwidth 作为 restore cost。

## 第 13 页：耦合的 Admission 与 Promotion

GPU eviction 时存在两个反事实世界：

    DROP:
      若未来产生 external demand，支付 C_c

    OFFLOAD:
      现在支付 C_s + C_cap
      若未来产生 demand，支付 min(C_r, C_c)

单次未来 demand 近似下：

    G_admit(b)
      = P_demand(b) × [Ĉ_c(b) - min(Ĉ_r(b), Ĉ_c(b))]
        - Ĉ_s(b) - Ĉ_cap(b)

    G_admit > margin  => OFFLOAD
    otherwise         => DROP

CPU hit 后：

    a* = argmin {
        Ĉ_r(blocks, PCIe queue, GPU load),
        Ĉ_c(tokens, batch, GPU load)
    }

关键解释：

- Admission 决定未来是否保留 RESTORE / RECOMPUTE 两个选项；
- Promotion 决定这个未来选项实际值多少钱；
- 因此 admission 不能只是 reuse classifier，CPU hit 也不能固定映射为 restore；
- 两次决策共享同一个成本账本，构成一项 lifecycle decision。

公式边界：

- P_demand 必须包含 KV 在 CPU 中存活到未来需求的概率；
- 多次复用不能只套单次公式，应在 trace replay 中累计每次收益；
- C_cap 是需要定义代理指标并做消融的研究项，不能当作已经精确观测的量。

## 第 14 页：Realized Utility 与安全闭环

架构：

    Runtime observations
      store / stall / recompute / GPU load / PCIe queue
                         |
                     Cost Ledger
                         |
             +-----------+-----------+
             |                       |
       Reuse Estimator          Cost Estimator
       P_demand                 Ĉ_s / Ĉ_r / Ĉ_c / Ĉ_cap
             |                       |
             +-----------+-----------+
                         |
                Lifecycle Controller
                  /                \
          GPU eviction            CPU hit
          OFFLOAD/DROP       RESTORE/RECOMPUTE
                  \                /
                   Realized Utility
                         |
                    Safety Guard
                 NORMAL/THROTTLE/OFF

对已 offload 且生命周期已经关闭的 KV，核算：

    若最终 restore:
      U_i = C_c^cf - C_s^obs - C_r^obs - C_cap^obs

    若 CPU hit 后选择 recompute，或生命周期关闭前始终未产生 demand:
      U_i = -C_s^obs - C_cap^obs

右删失边界：

“当前未复用”不等于“永远不会复用”。只有 CPU eviction、TTL 到期或 trace
结束时才能关闭未复用生命周期；在线 guard 应使用成熟样本或生存概率校正。

Guard：

    Ubar_t = EWMA(U_i)

    持续 Ubar_t < -epsilon:
      NORMAL -> THROTTLE -> OFF

    持续 Ubar_t > +epsilon_prime:
      OFF -> THROTTLE -> NORMAL

使用不同开关阈值、最短驻留时间与 D2H window budget 避免振荡。Throttle 通过
提高 admission margin 和限制 D2H budget 生效，而不是含糊地“少存一点”。

## 第 15 页：Lifecycle Oracle

当前已有：

    Promotion Oracle = 6.85%

它只覆盖 CPU hit 后的 RESTORE / RECOMPUTE，不是完整系统 Oracle。

第一项新工作是在真实未来 trace 上比较：

    L_drop    = future recompute cost
    L_offload = C_s + C_cap + future min(C_r, C_c)
    L_oracle  = min(L_drop, L_offload)

多次复用时累计未来动作，并显式模拟 CPU 容量、block 生存和策略间状态依赖。

需要同时报告：

- event-level cumulative request-visible cost；
- trace replay 下的 Oracle regret；
- 在真实 server 上执行在线策略后的 TTFT/SLO。

不能直接把 event cost Oracle 的百分比称为真实 TTFT 改善。Oracle 首先是方法空间
和策略 regret 的上界；端到端收益必须由在线 server run 验证。

对照：

1. no-offload；
2. all-block + always-restore；
3. simple TTL / 2Q / TinyLFU；
4. online LCUC；
5. Lifecycle Oracle。

Go/No-Go：

- Lifecycle Oracle 相对 no-offload/简单策略存在清晰差距；
- 在线策略获得至少一半 Oracle gap；
- 控制器开销低于 2%；
- 低复用 workload 相对 no-offload 的退化控制在约 3%；
- 若 Lifecycle Oracle 也不能超过 no-offload，则停止复杂性能策略扩展，保留成本
  观测与负收益保护作为工程结果。

## 第 16 页：拟创新与可行性

拟创新只收敛成三条：

1. 生命周期反事实净收益，而非单独的复用概率；
2. 在统一成本空间中耦合 OFFLOAD/DROP 与 RESTORE/RECOMPUTE；
3. 将 CPU tier 视为 conditional resource，用 realized utility、迟滞和 D2H
   budget 实现安全退化。

在系统性相关工作比较完成前，不使用“首次”表述。

当前可行性证据：

- GPU -> CPU -> GPU 与 always-recompute 路径已打通；
- 已有 scheduler-observed timer，而不只 raw DMA；
- 受控 admission 将 D2H 降低 77.4%，且不损失 33-block restore；
- 109 对 ShareGPT 事件存在 85/24 的真实动作 crossover；
- vLLM 中已有 ledger、reuse manager、promotion 与 trace replay 入口。

实施顺序：

1. 完整 Lifecycle Oracle；
2. 分桶成本模型与离线 LCUC replay；
3. 在线 OFFLOAD/DROP + RESTORE/RECOMPUTE；
4. realized-utility guard；
5. 多模型、open-loop、Agent/RAG 外部有效性。

## 两分钟方法讲稿

“我的方法不是简单预测哪个 KV 更热，而是把 KV 离开 GPU 后的完整生命周期
看成一组反事实路径。第一步，我把 store、restore、recompute 和容量代价统一成
请求真正感知到的时间成本。这里不能只看 PCIe，因为实验中 2.595 ms 的 DMA
最终表现为 10.404 ms 的 scheduler wait。

GPU 淘汰时，控制器比较 DROP 与 OFFLOAD。OFFLOAD 的价值不是未来一定 restore，
而是购买一个未来可在 restore 和 recompute 之间取最小值的选择权。因此
admission value 同时依赖复用概率、store cost 和未来 promotion policy。CPU hit
后，控制器再根据当时 GPU、batch 和 PCIe queue 状态选择 restore 或 recompute。

最后，预测可能出错，负载也会变化。因此系统对已经结束的 KV 生命周期核算
realized utility，并用带迟滞的 NORMAL、THROTTLE、OFF 状态机限制 D2H 或关闭
CPU tier。这样 CPU cache 不再是 always-on，而是只在净收益为正时启用。

下一步我会先构建完整 Lifecycle Oracle，量化这套两阶段决策相对 no-offload 和
简单策略到底有多大理论空间，再决定在线控制器值得做到什么复杂度。” 

## 证据引用

- D1：hierarchical_kv_research_status_20260805.md
- D3：natural_external_restore_motivation_20260805.md
- D5：offload_cost_ledger_sharegpt_20260809.md
- R2：external_restore_admission/20260729_235912/summary.md
- R3：external_restore_promotion_length/formal_idle_crossover_6pair_20260801/summary.md
- R4：trace_driven_lifecycle_compare_600_active32_3seed_20260805/summary.md
- R6/R7：direct_sharegpt_cost_ledger_3dataset_seeds_20260809
