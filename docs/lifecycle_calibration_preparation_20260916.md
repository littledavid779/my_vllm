# 独立成本校准与六组对照：准备与暂停记录

**后续更新：** 2026-09-16 晚间已成功续跑，完成四组校准与六组对照。
结果和退化原因见 [独立校准与六组结果](lifecycle_calibrated_results_20260916.md)。
以下暂停信息为当日凌晨的历史记录，不代表当前状态。

## 当前状态

2026-09-16 00:30 检查 GPU 1 时无进程、显存使用 0 MiB。
完成数据准备后，启动前检查发现 GPU 1 已有 PID 3363802（用户 lzq）的
VLLM::EngineCore。00:36 复核显存使用 19049 MiB、利用率 100%。
占用保护在启动我们的服务前报错退出，未进行成本校准，未获得六组性能数据。
没有停止或修改其他用户进程。

## 已完成准备

- 新编排脚本：[run_lifecycle_calibrated_smoke.py](../scripts/run_lifecycle_calibrated_smoke.py)。
- 从已有 dataset seed 1 的 512-conversation 候选中，排除与现有 seed 0/1/2
  评测集 ID 相同或完整消息内容相同的会话，再经过 tokenizer 长度检查，选取 128 个。
- 评测保留原 Direct ShareGPT dataset seed 0 的 128 个会话。
- 校准和评测的 ID、完整会话内容均无重叠；这不意味着公共系统前缀完全无共享。
- 固定模型 Qwen3-0.6B、BF16、TP=1、1000 GPU blocks、4 GiB CPU cache、
  16-token offloaded block、max model length=4096、32 active conversations、4 clients。
- 每次运行 128 请求，闭环；保存 dataset assistant history。
- 校准顺序：runtime seed 0 的 restore/recompute，seed 1 的 recompute/restore。
- 评测顺序在获取成本前由固定随机种子预先确定：all-block、admission、
  joint-no-guard、promotion、no-offload、full。

实验清单与哈希：[manifest.json](../results/lifecycle_calibrated_gpu1_20260916/manifest.json)。
输入：[校准集](../results/lifecycle_calibrated_gpu1_20260916/calibration.json)、
[评测集](../results/lifecycle_calibrated_gpu1_20260916/evaluation.json)。

## 预先固定的拟合方法

- 恢复：对所有完整校准 promotion 事件的 scheduler wait 拟合非负截距/斜率线性模型。
- 重算：对所有完整校准 promotion 事件拟合过原点的每-token 时间模型。
- 保存：所有完整 store 事件的 worker submit-to-complete 总时长 / 实际写入 blocks 总数。
  此值是保存成本代理，不是 request-visible D2H stall；并行传输的事件时长不可解释为
  端到端串行阻塞之和。另存 raw DMA/block 比值，不根据评测效果临时改口径。
- 每类不足 8 个完整事件则停止，不使用人工补值。
- 拟合输出 `calibration_fit.json` 后，六组评测使用固定初值，不用评测结果反向调参。

## 恢复运行

需要先确认 GPU 1 可供本轮独占。已经存在准备目录，不应再次执行默认的 all/prepare。
依次执行：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_lifecycle_calibrated_smoke.py \
  --gpu 1 --port 8238 --output results/lifecycle_calibrated_gpu1_20260916 \
  --phase calibrate

HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_lifecycle_calibrated_smoke.py \
  --gpu 1 --port 8238 --output results/lifecycle_calibrated_gpu1_20260916 \
  --phase evaluate
```

每轮开始前检查 GPU 与端口，运行中每约 2 秒采样 GPU 进程；出现外来进程则不把该轮
记作有效结果。采样不是资源预留，也不能证明不存在短于采样间隔的干扰。
成功的 arm 写入 metrics.json，恢复执行时仅在配置相同的情况下复用已完成结果。

这仍是单模型、单评测 seed 的小规模对照，不能据此声称统计显著性或跨负载泛化。
约数秒的 128-request workload 也不能验证默认 30 秒 Guard 窗口，需要另设长时实验。

## 验证

`tests/v1/kv_offload/test_lifecycle_calibration.py`：7 passed。
新脚本和测试通过 Ruff 0.14.0 静态检查。
数据准备已实际执行；GPU 校准与性能对照尚未开始。
