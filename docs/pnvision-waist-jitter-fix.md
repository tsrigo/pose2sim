# PnVision 腰部抖动复现与修复报告

## 复现对象

- 数据来源：`data/PnVision.tar`，已展开并读取 `data/PnVision/20260507/*/pose-3d/*.trc`。
- 腰部检测点：`Hip`, `RHip`, `LHip`。
- 复现指标：逐点二阶差分 RMS（mm/frame^2）衡量高频抖动；三点两两距离标准差（mm）衡量腰/髋刚体结构是否散架。
- 限制：这里没有外部真值骨架，因此结论基于真实输出 TRC 的时序稳定性和骨盆内部距离一致性，而不是绝对 3D 误差。
- 限制：本次没有重新跑全量 AVI -> RTMLib -> 三角化；`data/PnVision.tar` 包含真实 TRC/日志但不包含中间 pose JSON，所以修复验证从 TRC/filtering 阶段进入真实 `filter_all`。

## 问题证据

- 扫描到 27 个 PnVision TRC；最严重的是 `rec_20260507_135606_高速运动`。
- 最严重 trial 的髋部三点平均抖动为 **17.41 mm/frame^2**。
- 同一 trial 的髋部三点内部距离平均标准差为 **72.57 mm**，说明问题不只是整体平移噪声，而是 `Hip/RHip/LHip` 相对位置在帧间明显变形。

| trial                            |   pelvis_mean_jitter_mm_frame2 |   pairwise_distance_std_mean_mm |
|:---------------------------------|-------------------------------:|--------------------------------:|
| rec_20260507_135606_高速运动     |                        17.4056 |                         72.5703 |
| rec_20260507_114910_zjg_action01 |                        14.011  |                         21.6873 |
| rec_20260430_092226_e76dbb       |                        12.3573 |                         35.8841 |
| rec_20260430_095543_66a1cd       |                        12.1895 |                         58.313  |
| rec_20260507_103904_wxs_action03 |                        11.9745 |                         19.1081 |

![Baseline scan](../figures/pnvision_waist_jitter/fig01_baseline_waist_jitter_scan.png)

## 根因分析

- PnVision 输出的 TRC 已经是 Butterworth 后处理版本，但腰部三点仍有明显内部距离漂移，说明单独对每个 marker 滤波不能保证骨盆局部刚体结构。
- 当前三角化/过滤链路按 marker 独立处理；某一帧、某一相机或某一关键点短暂偏移时，单点重投影误差可能仍然可接受，但 `Hip/RHip/LHip` 的组合形状会抖。
- 腰部视觉抖动的直接表现是骨盆三点相对距离和方向跳动；因此修复点应放在保持局部刚体一致性，而不是只调报告或只画更平滑的曲线。

## 已实现修复

- 在 `Pose2Sim/filtering.py` 新增 `stabilize_rigid_marker_groups_3d`。
- 新增配置项 `filtering.rigid_marker_groups`，可把 `Hip/RHip/LHip` 这类局部刚体 marker 组作为整体稳定。
- 算法步骤：从稳定帧估计组内 3D 模板；逐帧 Kabsch 拟合刚体变换；对变换参数做 centered rolling median；用 `rigid_group_blend` 将刚体结果混回原始轨迹。
- 同时修复 `filter_all(frame_range='all')` 漏掉最后一帧的 off-by-one 问题，保证修复前后都是 578 帧。

推荐配置：

```toml
[filtering]
filter = false
reject_outliers = false
rigid_marker_groups = [
  { name = "pelvis", markers = ["Hip", "RHip", "LHip"] },
]
rigid_group_smoothing_window = 31
rigid_group_blend = 0.7
```

## 修复效果

- 最严重 trial，髋部三点平均抖动：**17.41 -> 10.42 mm/frame^2**（-40.13%）。
- 最严重 trial，髋部内部距离标准差：**72.57 -> 22.16 mm**（-69.46%）。
- 全部 PnVision TRC 内存评估，平均髋部抖动：**5.99 -> 4.24 mm/frame^2**。
- 全部 PnVision TRC 内存评估，平均髋部内部距离标准差：**19.43 -> 6.47 mm**。

![Before after metrics](../figures/pnvision_waist_jitter/fig02_worst_before_after_metrics.png)

![Pairwise distances](../figures/pnvision_waist_jitter/fig03_worst_pairwise_distance_timeseries.png)

![Hip trajectories](../figures/pnvision_waist_jitter/fig04_worst_hip_trajectories.png)

## 原始证据文件

- 基线 TRC：`data/pnvision_waist_jitter/baseline/rec_20260507_135606_b60f4c_baseline.trc`
- 修复后 TRC：`data/pnvision_waist_jitter/rigid_filter/rec_20260507_135606_b60f4c_rigid_filter.trc`
- 真实 `filter_all` 运行输出：`data/pnvision_waist_jitter/run/pose-3d/rec_20260507_135606_0-578_filt_butterworth.trc`
- 全量基线扫描：`data/pnvision_waist_jitter/pnvision_baseline_waist_metrics.csv`
- 全量前后对比：`data/pnvision_waist_jitter/pnvision_all_trials_before_after_metrics.csv`
- 最严重 trial 前后对比：`data/pnvision_waist_jitter/worst_trial_before_after_metrics.csv`
- 距离时间序列：`data/pnvision_waist_jitter/worst_trial_pairwise_distance_timeseries.csv`
