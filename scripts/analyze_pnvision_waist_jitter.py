#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Pose2Sim.common import read_trc
from Pose2Sim.filtering import stabilize_rigid_marker_groups_3d


ROOT = Path(__file__).resolve().parents[1]
PNVISION_DIR = ROOT / "data" / "PnVision" / "20260507"
DATA_DIR = ROOT / "data" / "pnvision_waist_jitter"
FIG_DIR = ROOT / "figures" / "pnvision_waist_jitter"
DOC_PATH = ROOT / "docs" / "pnvision-waist-jitter-fix.md"

GROUP_MARKERS = ["Hip", "RHip", "LHip"]
PAIR_NAMES = [("Hip", "RHip"), ("Hip", "LHip"), ("RHip", "LHip")]
FIX_CONFIG = {
    "filtering": {
        "rigid_marker_groups": [
            {"name": "pelvis", "markers": GROUP_MARKERS},
        ],
        "rigid_group_smoothing_window": 31,
        "rigid_group_blend": 0.7,
    }
}

BASELINE_TRC = DATA_DIR / "baseline" / "rec_20260507_135606_b60f4c_baseline.trc"
RIGID_TRC = DATA_DIR / "rigid_filter" / "rec_20260507_135606_b60f4c_rigid_filter.trc"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def load_marker_data(trc_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], int]:
    coords, frames, time_col, markers, _ = read_trc(trc_path)
    marker_data = {
        marker: coords.iloc[:, marker_id * 3 : (marker_id + 1) * 3].to_numpy(dtype=float)
        for marker_id, marker in enumerate(markers)
    }
    return marker_data, time_col.to_numpy(dtype=float), markers, len(frames)


def second_difference_jitter_mm(points: np.ndarray) -> float:
    valid = np.isfinite(points).all(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < 3:
        return np.nan
    split_points = np.where(np.diff(indices) > 1)[0] + 1
    values = []
    for seq in np.split(indices, split_points):
        if len(seq) < 3:
            continue
        second_diff = np.diff(points[seq], n=2, axis=0)
        values.append(np.sqrt(np.nanmean(np.sum(second_diff * second_diff, axis=1))) * 1000.0)
    return float(np.nanmean(values)) if values else np.nan


def waist_metrics(marker_data: dict[str, np.ndarray], trial: str, method: str, trc_path: Path, frame_count: int) -> dict:
    missing = [marker for marker in GROUP_MARKERS if marker not in marker_data]
    if missing:
        raise ValueError(f"{trc_path} is missing required waist markers: {missing}")

    jitter_by_marker = {
        marker: second_difference_jitter_mm(marker_data[marker])
        for marker in GROUP_MARKERS
    }
    pair_std = {}
    pair_range = {}
    pair_mean = {}
    for marker_a, marker_b in PAIR_NAMES:
        distances_mm = np.linalg.norm(marker_data[marker_a] - marker_data[marker_b], axis=1) * 1000.0
        pair_name = f"{marker_a}-{marker_b}"
        pair_std[pair_name] = float(np.nanstd(distances_mm))
        pair_range[pair_name] = float(np.nanmax(distances_mm) - np.nanmin(distances_mm))
        pair_mean[pair_name] = float(np.nanmean(distances_mm))

    return {
        "trial": trial,
        "method": method,
        "trc_path": str(trc_path.relative_to(ROOT)),
        "frames": frame_count,
        "hip_jitter_mm_frame2": jitter_by_marker["Hip"],
        "rhip_jitter_mm_frame2": jitter_by_marker["RHip"],
        "lhip_jitter_mm_frame2": jitter_by_marker["LHip"],
        "pelvis_mean_jitter_mm_frame2": float(np.nanmean(list(jitter_by_marker.values()))),
        "hip_rhip_distance_mean_mm": pair_mean["Hip-RHip"],
        "hip_lhip_distance_mean_mm": pair_mean["Hip-LHip"],
        "rhip_lhip_distance_mean_mm": pair_mean["RHip-LHip"],
        "hip_rhip_distance_std_mm": pair_std["Hip-RHip"],
        "hip_lhip_distance_std_mm": pair_std["Hip-LHip"],
        "rhip_lhip_distance_std_mm": pair_std["RHip-LHip"],
        "pairwise_distance_std_mean_mm": float(np.nanmean(list(pair_std.values()))),
        "pairwise_distance_range_max_mm": float(np.nanmax(list(pair_range.values()))),
    }


def all_pnvision_trcs() -> list[Path]:
    return sorted(PNVISION_DIR.glob("*/pose-3d/*.trc"))


def scan_baseline_trials() -> pd.DataFrame:
    rows = []
    for trc_path in all_pnvision_trcs():
        marker_data, _, markers, frame_count = load_marker_data(trc_path)
        if not all(marker in markers for marker in GROUP_MARKERS):
            continue
        rows.append(waist_metrics(marker_data, trc_path.parent.parent.name, "baseline", trc_path, frame_count))
    return pd.DataFrame(rows).sort_values("pelvis_mean_jitter_mm_frame2", ascending=False)


def evaluate_in_memory_fix() -> pd.DataFrame:
    rows = []
    for trc_path in all_pnvision_trcs():
        coords, frames, _, markers, _ = read_trc(trc_path)
        if not all(marker in markers for marker in GROUP_MARKERS):
            continue
        baseline_data = {
            marker: coords.iloc[:, marker_id * 3 : (marker_id + 1) * 3].to_numpy(dtype=float)
            for marker_id, marker in enumerate(markers)
        }
        fixed_coords, _ = stabilize_rigid_marker_groups_3d(FIX_CONFIG, coords, markers)
        fixed_data = {
            marker: fixed_coords.iloc[:, marker_id * 3 : (marker_id + 1) * 3].to_numpy(dtype=float)
            for marker_id, marker in enumerate(markers)
        }
        trial = trc_path.parent.parent.name
        rows.append(waist_metrics(baseline_data, trial, "baseline", trc_path, len(frames)))
        rows.append(waist_metrics(fixed_data, trial, "rigid_filter_in_memory", trc_path, len(frames)))
    return pd.DataFrame(rows)


def evaluate_real_output() -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], np.ndarray]:
    rows = []
    data_by_method = {}
    time_s = None
    for method, trc_path in [("baseline", BASELINE_TRC), ("rigid_filter", RIGID_TRC)]:
        marker_data, time_col, _, frame_count = load_marker_data(trc_path)
        rows.append(waist_metrics(marker_data, "rec_20260507_135606_高速运动", method, trc_path, frame_count))
        data_by_method[method] = marker_data
        if time_s is None:
            time_s = time_col
    return pd.DataFrame(rows), data_by_method, time_s


def write_pairwise_timeseries(data_by_method: dict[str, dict[str, np.ndarray]], time_s: np.ndarray) -> pd.DataFrame:
    rows = []
    for method, marker_data in data_by_method.items():
        for marker_a, marker_b in PAIR_NAMES:
            distances_mm = np.linalg.norm(marker_data[marker_a] - marker_data[marker_b], axis=1) * 1000.0
            for frame_id, distance_mm in enumerate(distances_mm):
                rows.append(
                    {
                        "method": method,
                        "frame": frame_id,
                        "time_s": time_s[frame_id],
                        "pair": f"{marker_a}-{marker_b}",
                        "distance_mm": distance_mm,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "worst_trial_pairwise_distance_timeseries.csv", index=False)
    return df


def plot_baseline_scan(baseline_df: pd.DataFrame) -> None:
    plot_df = baseline_df.head(10).iloc[::-1]
    labels = []
    for trial in plot_df["trial"]:
        label = trial.replace("rec_", "")
        label = label.encode("ascii", errors="ignore").decode("ascii").strip("_")
        labels.append(label or "non_ascii_trial")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    axes[0].barh(labels, plot_df["pelvis_mean_jitter_mm_frame2"], color="#c75d4d")
    axes[0].set_xlabel("mm/frame^2")
    axes[0].set_title("Pelvis marker jitter")
    axes[1].barh(labels, plot_df["pairwise_distance_std_mean_mm"], color="#4f7cac")
    axes[1].set_xlabel("mm")
    axes[1].set_title("Internal distance std")
    fig.suptitle("PnVision baseline waist jitter scan: top 10 trials")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_baseline_waist_jitter_scan.png")
    plt.close(fig)


def plot_before_after_bars(before_after_df: pd.DataFrame) -> None:
    plot_df = before_after_df.set_index("method").loc[["baseline", "rigid_filter"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    x = np.arange(3)
    width = 0.36
    marker_cols = ["hip_jitter_mm_frame2", "rhip_jitter_mm_frame2", "lhip_jitter_mm_frame2"]
    for offset, method, color in [(-width / 2, "baseline", "#b45d48"), (width / 2, "rigid_filter", "#4f7cac")]:
        axes[0].bar(x + offset, plot_df.loc[method, marker_cols], width, label=method, color=color)
    axes[0].set_xticks(x, GROUP_MARKERS)
    axes[0].set_ylabel("mm/frame^2")
    axes[0].set_title("Second-difference jitter")
    pair_cols = ["hip_rhip_distance_std_mm", "hip_lhip_distance_std_mm", "rhip_lhip_distance_std_mm"]
    for offset, method, color in [(-width / 2, "baseline", "#b45d48"), (width / 2, "rigid_filter", "#4f7cac")]:
        axes[1].bar(x + offset, plot_df.loc[method, pair_cols], width, label=method, color=color)
    axes[1].set_xticks(x, ["Hip-RHip", "Hip-LHip", "RHip-LHip"], rotation=20, ha="right")
    axes[1].set_ylabel("mm")
    axes[1].set_title("Pairwise distance std")
    axes[0].legend()
    fig.suptitle("Worst PnVision trial before and after rigid waist filtering")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_worst_before_after_metrics.png")
    plt.close(fig)


def plot_pairwise_timeseries(pairwise_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.4), sharex=True)
    for ax, pair in zip(axes, [f"{a}-{b}" for a, b in PAIR_NAMES]):
        for method, color in [("baseline", "#b45d48"), ("rigid_filter", "#4f7cac")]:
            series = pairwise_df[(pairwise_df["pair"] == pair) & (pairwise_df["method"] == method)]
            ax.plot(series["time_s"], series["distance_mm"], label=method, color=color, linewidth=1.1)
        ax.set_ylabel(pair)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Worst trial pelvis internal distances")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_worst_pairwise_distance_timeseries.png")
    plt.close(fig)


def plot_hip_trajectories(data_by_method: dict[str, dict[str, np.ndarray]], time_s: np.ndarray) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.4), sharex=True)
    for ax, marker in zip(axes, GROUP_MARKERS):
        for method, color in [("baseline", "#b45d48"), ("rigid_filter", "#4f7cac")]:
            ax.plot(time_s, data_by_method[method][marker][:, 1], label=method, color=color, linewidth=1.1)
        ax.set_ylabel(f"{marker} Y (m)")
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Worst trial waist marker vertical trajectories")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_worst_hip_trajectories.png")
    plt.close(fig)


def pct_change(before: float, after: float) -> float:
    return (after - before) / before * 100.0


def fmt(value: float) -> str:
    return f"{value:.2f}"


def write_report(baseline_df: pd.DataFrame, all_before_after_df: pd.DataFrame, real_output_df: pd.DataFrame) -> None:
    worst_before = real_output_df[real_output_df["method"] == "baseline"].iloc[0]
    worst_after = real_output_df[real_output_df["method"] == "rigid_filter"].iloc[0]
    aggregate = all_before_after_df.groupby("method").agg(
        pelvis_mean_jitter_mm_frame2=("pelvis_mean_jitter_mm_frame2", "mean"),
        pairwise_distance_std_mean_mm=("pairwise_distance_std_mean_mm", "mean"),
    )
    agg_before = aggregate.loc["baseline"]
    agg_after = aggregate.loc["rigid_filter_in_memory"]
    top_trials = baseline_df.head(5)[
        ["trial", "pelvis_mean_jitter_mm_frame2", "pairwise_distance_std_mean_mm"]
    ].to_markdown(index=False)

    lines = [
        "# PnVision 腰部抖动复现与修复报告",
        "",
        "## 复现对象",
        "",
        "- 数据来源：`data/PnVision.tar`，已展开并读取 `data/PnVision/20260507/*/pose-3d/*.trc`。",
        "- 腰部检测点：`Hip`, `RHip`, `LHip`。",
        "- 复现指标：逐点二阶差分 RMS（mm/frame^2）衡量高频抖动；三点两两距离标准差（mm）衡量腰/髋刚体结构是否散架。",
        "- 限制：这里没有外部真值骨架，因此结论基于真实输出 TRC 的时序稳定性和骨盆内部距离一致性，而不是绝对 3D 误差。",
        "- 限制：本次没有重新跑全量 AVI -> RTMLib -> 三角化；`data/PnVision.tar` 包含真实 TRC/日志但不包含中间 pose JSON，所以修复验证从 TRC/filtering 阶段进入真实 `filter_all`。",
        "",
        "## 问题证据",
        "",
        f"- 扫描到 {len(baseline_df)} 个 PnVision TRC；最严重的是 `rec_20260507_135606_高速运动`。",
        f"- 最严重 trial 的髋部三点平均抖动为 **{fmt(worst_before['pelvis_mean_jitter_mm_frame2'])} mm/frame^2**。",
        f"- 同一 trial 的髋部三点内部距离平均标准差为 **{fmt(worst_before['pairwise_distance_std_mean_mm'])} mm**，说明问题不只是整体平移噪声，而是 `Hip/RHip/LHip` 相对位置在帧间明显变形。",
        "",
        top_trials,
        "",
        "![Baseline scan](../figures/pnvision_waist_jitter/fig01_baseline_waist_jitter_scan.png)",
        "",
        "## 根因分析",
        "",
        "- PnVision 输出的 TRC 已经是 Butterworth 后处理版本，但腰部三点仍有明显内部距离漂移，说明单独对每个 marker 滤波不能保证骨盆局部刚体结构。",
        "- 当前三角化/过滤链路按 marker 独立处理；某一帧、某一相机或某一关键点短暂偏移时，单点重投影误差可能仍然可接受，但 `Hip/RHip/LHip` 的组合形状会抖。",
        "- 腰部视觉抖动的直接表现是骨盆三点相对距离和方向跳动；因此修复点应放在保持局部刚体一致性，而不是只调报告或只画更平滑的曲线。",
        "",
        "## 已实现修复",
        "",
        "- 在 `Pose2Sim/filtering.py` 新增 `stabilize_rigid_marker_groups_3d`。",
        "- 新增配置项 `filtering.rigid_marker_groups`，可把 `Hip/RHip/LHip` 这类局部刚体 marker 组作为整体稳定。",
        "- 算法步骤：从稳定帧估计组内 3D 模板；逐帧 Kabsch 拟合刚体变换；对变换参数做 centered rolling median；用 `rigid_group_blend` 将刚体结果混回原始轨迹。",
        "- 同时修复 `filter_all(frame_range='all')` 漏掉最后一帧的 off-by-one 问题，保证修复前后都是 578 帧。",
        "",
        "推荐配置：",
        "",
        "```toml",
        "[filtering]",
        "filter = false",
        "reject_outliers = false",
        "rigid_marker_groups = [",
        "  { name = \"pelvis\", markers = [\"Hip\", \"RHip\", \"LHip\"] },",
        "]",
        "rigid_group_smoothing_window = 31",
        "rigid_group_blend = 0.7",
        "```",
        "",
        "## 修复效果",
        "",
        f"- 最严重 trial，髋部三点平均抖动：**{fmt(worst_before['pelvis_mean_jitter_mm_frame2'])} -> {fmt(worst_after['pelvis_mean_jitter_mm_frame2'])} mm/frame^2**（{fmt(pct_change(worst_before['pelvis_mean_jitter_mm_frame2'], worst_after['pelvis_mean_jitter_mm_frame2']))}%）。",
        f"- 最严重 trial，髋部内部距离标准差：**{fmt(worst_before['pairwise_distance_std_mean_mm'])} -> {fmt(worst_after['pairwise_distance_std_mean_mm'])} mm**（{fmt(pct_change(worst_before['pairwise_distance_std_mean_mm'], worst_after['pairwise_distance_std_mean_mm']))}%）。",
        f"- 全部 PnVision TRC 内存评估，平均髋部抖动：**{fmt(agg_before['pelvis_mean_jitter_mm_frame2'])} -> {fmt(agg_after['pelvis_mean_jitter_mm_frame2'])} mm/frame^2**。",
        f"- 全部 PnVision TRC 内存评估，平均髋部内部距离标准差：**{fmt(agg_before['pairwise_distance_std_mean_mm'])} -> {fmt(agg_after['pairwise_distance_std_mean_mm'])} mm**。",
        "",
        "![Before after metrics](../figures/pnvision_waist_jitter/fig02_worst_before_after_metrics.png)",
        "",
        "![Pairwise distances](../figures/pnvision_waist_jitter/fig03_worst_pairwise_distance_timeseries.png)",
        "",
        "![Hip trajectories](../figures/pnvision_waist_jitter/fig04_worst_hip_trajectories.png)",
        "",
        "## 原始证据文件",
        "",
        "- 基线 TRC：`data/pnvision_waist_jitter/baseline/rec_20260507_135606_b60f4c_baseline.trc`",
        "- 修复后 TRC：`data/pnvision_waist_jitter/rigid_filter/rec_20260507_135606_b60f4c_rigid_filter.trc`",
        "- 真实 `filter_all` 运行输出：`data/pnvision_waist_jitter/run/pose-3d/rec_20260507_135606_0-578_filt_butterworth.trc`",
        "- 全量基线扫描：`data/pnvision_waist_jitter/pnvision_baseline_waist_metrics.csv`",
        "- 全量前后对比：`data/pnvision_waist_jitter/pnvision_all_trials_before_after_metrics.csv`",
        "- 最严重 trial 前后对比：`data/pnvision_waist_jitter/worst_trial_before_after_metrics.csv`",
        "- 距离时间序列：`data/pnvision_waist_jitter/worst_trial_pairwise_distance_timeseries.csv`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_plot_style()
    baseline_df = scan_baseline_trials()
    all_before_after_df = evaluate_in_memory_fix()
    real_output_df, data_by_method, time_s = evaluate_real_output()
    pairwise_df = write_pairwise_timeseries(data_by_method, time_s)

    baseline_df.to_csv(DATA_DIR / "pnvision_baseline_waist_metrics.csv", index=False)
    all_before_after_df.to_csv(DATA_DIR / "pnvision_all_trials_before_after_metrics.csv", index=False)
    real_output_df.to_csv(DATA_DIR / "worst_trial_before_after_metrics.csv", index=False)

    plot_baseline_scan(baseline_df)
    plot_before_after_bars(real_output_df)
    plot_pairwise_timeseries(pairwise_df)
    plot_hip_trajectories(data_by_method, time_s)
    write_report(baseline_df, all_before_after_df, real_output_df)

    print(f"Wrote {DOC_PATH.relative_to(ROOT)}")
    print(f"Wrote metrics under {DATA_DIR.relative_to(ROOT)}")
    print(f"Wrote figures under {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
