# HZVision 集成与分支维护

`pose2sim-hzvision` 是给 HZVision 调用的 Pose2Sim 长期维护分支。该分支保留 Pose2Sim 原始提交历史，并在此基础上维护 `avi_to_trc` 入口和 PnVision/HZVision 所需的稳定性改动。

## 分支定位

- `master` 或其他 HZVision 主程序分支不直接承载 Pose2Sim 源码。
- `pose2sim-hzvision` 承载完整 Pose2Sim 源码、`avi_to_trc` CLI 和 HZVision 相关算法改动。
- HZVision 主程序通过安装该分支或指定源码路径调用 Pose2Sim，不再手工替换虚拟环境中的 `pose2sim` 文件夹。
- 本分支不提交真实业务数据、临时实验输出、交付 zip 或下载中的视频文件。

## 调用方式

推荐把该分支安装到 HZVision 的运行环境后调用命令行入口：

```bash
avi_to_trc --trial-dir /path/to/Session/Trial_1 --batch-size 16 --overwrite-pose
```

也可以从 Python 中调用：

```python
from Pose2Sim.Utilities.avi_to_trc import avi_to_trc

avi_to_trc(
    trial_dir="/path/to/Session/Trial_1",
    batch_size=16,
    overwrite_pose=True,
)
```

`trial_dir` 需要满足 Pose2Sim trial 结构：trial 下包含 `videos/*.avi`，session 下包含 `calibration/*.toml`，配置文件可以放在 session 或 trial 下。

## 刚体稳定配置

本分支增加了两个可选稳定步骤，默认不启用；只有在 `Config.toml` 中配置 `rigid_marker_groups` 时才会生效。

三角化阶段可配置：

```toml
[triangulation]
rigid_marker_groups = [
  { name = "pelvis", markers = ["Hip", "RHip", "LHip"] }
]
rigid_group_min_markers = 3
rigid_group_smoothing_window = 5
rigid_group_reproj_error_threshold = 15
```

滤波阶段可配置：

```toml
[filtering]
rigid_marker_groups = [
  { name = "pelvis", markers = ["Hip", "RHip", "LHip"] }
]
rigid_group_min_markers = 3
rigid_group_smoothing_window = 31
rigid_group_blend = 0.7
rigid_group_fill_missing = false
```

这些配置用于降低同一刚体局部 marker 的相对抖动，例如 PnVision 腰部 `Hip/RHip/LHip`。如果没有配置，现有 Pose2Sim 行为保持不变。

## 同步官方 Pose2Sim

建议保留两个远端：

```bash
git remote add upstream <official-pose2sim-url>
git remote add hzvision http://platgit.ng.noitom.com.cn/scm/pnvis/hzvision.git
```

同步流程：

```bash
git fetch upstream
git switch pose2sim-hzvision
git merge upstream/main
```

如果官方更新与 HZVision 改动发生冲突，在 `pose2sim-hzvision` 上解决冲突并重新跑 `avi_to_trc` 和刚体稳定性验证。不要把账号、密码或真实数据写入 Git 提交。
