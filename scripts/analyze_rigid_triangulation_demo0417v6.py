#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Pose2Sim.common import read_trc


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "rigid_triangulation_demo0417v6_full"
FIG_DIR = ROOT / "figures" / "rigid_triangulation_demo0417v6_full"
DOC_PATH = ROOT / "docs" / "rigid-triangulation-demo0417v6-full.md"

TRC_PATHS = {
    "baseline": DATA_DIR / "baseline" / "demo0417v6_0-390_baseline.trc",
    "rigid": DATA_DIR / "rigid" / "demo0417v6_0-390_rigid.trc",
}
STRICT_BEFORE_TRC = DATA_DIR / "strict_before" / "demo0417v6_0-390_strict_before.trc"

RIGID_GROUPS = {
    "hip": ["Hip", "LHip", "RHip"],
    "head_face": ["Nose", "LEye", "REye", "LEar", "REar"],
}

JITTER_MARKERS = [
    "Hip",
    "RHip",
    "LHip",
    "Nose",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "RShoulder",
    "LShoulder",
    "RBigToe",
    "LBigToe",
    "RHeel",
    "LHeel",
]


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


def load_marker_data(trc_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    coords, _, time_col, markers, _ = read_trc(trc_path)
    marker_data = {
        marker: coords.iloc[:, marker_id * 3 : (marker_id + 1) * 3].to_numpy(dtype=float)
        for marker_id, marker in enumerate(markers)
    }
    return marker_data, time_col.to_numpy(dtype=float)


def second_difference_jitter_mm(points: np.ndarray) -> float:
    if len(points) < 3:
        return np.nan
    second_diff = np.diff(points, n=2, axis=0)
    return float(np.sqrt(np.nanmean(np.sum(second_diff * second_diff, axis=1)))) * 1000.0


def pairwise_distance_rows(marker_data_by_method: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rows = []
    for group_name, markers in RIGID_GROUPS.items():
        for marker_id, marker_a in enumerate(markers):
            for marker_b in markers[marker_id + 1 :]:
                for method, marker_data in marker_data_by_method.items():
                    distances_mm = np.linalg.norm(marker_data[marker_a] - marker_data[marker_b], axis=1) * 1000.0
                    rows.append(
                        {
                            "group": group_name,
                            "pair": f"{marker_a}-{marker_b}",
                            "method": method,
                            "mean_mm": float(np.nanmean(distances_mm)),
                            "std_mm": float(np.nanstd(distances_mm)),
                            "range_mm": float(np.nanmax(distances_mm) - np.nanmin(distances_mm)),
                        }
                    )
    return rows


def pairwise_change_rows(marker_data_by_method: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rows = []
    baseline_data = marker_data_by_method["baseline"]
    rigid_data = marker_data_by_method["rigid"]
    for group_name, markers in RIGID_GROUPS.items():
        for marker_id, marker_a in enumerate(markers):
            for marker_b in markers[marker_id + 1 :]:
                baseline_distances_mm = np.linalg.norm(baseline_data[marker_a] - baseline_data[marker_b], axis=1) * 1000.0
                rigid_distances_mm = np.linalg.norm(rigid_data[marker_a] - rigid_data[marker_b], axis=1) * 1000.0
                valid = np.isfinite(baseline_distances_mm) & np.isfinite(rigid_distances_mm) & (baseline_distances_mm > 1e-9)
                ratios = np.abs(rigid_distances_mm[valid] - baseline_distances_mm[valid]) / baseline_distances_mm[valid]
                rows.append(
                    {
                        "group": group_name,
                        "pair": f"{marker_a}-{marker_b}",
                        "mean_abs_change_ratio": float(np.nanmean(ratios)) if len(ratios) else np.nan,
                        "p95_abs_change_ratio": float(np.nanpercentile(ratios, 95)) if len(ratios) else np.nan,
                        "max_abs_change_ratio": float(np.nanmax(ratios)) if len(ratios) else np.nan,
                    }
                )
    return rows


def jitter_rows(marker_data_by_method: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rows = []
    for marker in JITTER_MARKERS:
        baseline = second_difference_jitter_mm(marker_data_by_method["baseline"][marker])
        rigid = second_difference_jitter_mm(marker_data_by_method["rigid"][marker])
        rows.append(
            {
                "marker": marker,
                "baseline_jitter_mm": baseline,
                "rigid_jitter_mm": rigid,
                "delta_percent": float((rigid - baseline) / baseline * 100.0) if baseline else np.nan,
            }
        )
    return rows


def displacement_rows(marker_data_by_method: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rows = []
    markers = sorted(set(marker_data_by_method["baseline"]).intersection(marker_data_by_method["rigid"]))
    for marker in markers:
        delta_mm = np.linalg.norm(
            marker_data_by_method["rigid"][marker] - marker_data_by_method["baseline"][marker],
            axis=1,
        ) * 1000.0
        rows.append(
            {
                "marker": marker,
                "mean_delta_mm": float(np.nanmean(delta_mm)),
                "median_delta_mm": float(np.nanmedian(delta_mm)),
                "p95_delta_mm": float(np.nanpercentile(delta_mm, 95)),
                "max_delta_mm": float(np.nanmax(delta_mm)),
            }
        )
    return rows


def parse_acceptance() -> pd.DataFrame:
    log_path = DATA_DIR / "rigid_run.log"
    rows = []
    pattern = re.compile(
        r"Rigid marker group (?P<name>.+?) for person (?P<person>\d+): accepted "
        r"(?P<accepted>\d+)/(?P<total>\d+) frames with mean joint reprojection error (?P<error>[0-9.]+) px"
    )
    applied_pattern = re.compile(r"applied (?P<applied>\d+)/(?P<total>\d+) frames")
    correction_pattern = re.compile(r"mean correction (?P<mean_delta>[0-9.]+) mm, max correction (?P<max_delta>[0-9.]+) mm")
    guard_pattern = re.compile(
        r"correction limit adjusted (?P<limited>\d+) frames; pairwise guard adjusted (?P<guarded>\d+) frames, rejected (?P<rejected>\d+)"
    )
    elapsed_pattern = re.compile(r"elapsed (?P<elapsed>[0-9.]+) s")
    if not log_path.exists():
        return pd.DataFrame(columns=["group", "person", "accepted", "total", "mean_error_px"])
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            applied_match = applied_pattern.search(line)
            correction_match = correction_pattern.search(line)
            guard_match = guard_pattern.search(line)
            elapsed_match = elapsed_pattern.search(line)
            rows.append(
                {
                    "group": match.group("name"),
                    "person": int(match.group("person")),
                    "accepted": int(match.group("accepted")),
                    "total": int(match.group("total")),
                    "mean_error_px": float(match.group("error")),
                    "applied": int(applied_match.group("applied")) if applied_match else np.nan,
                    "mean_delta_mm": float(correction_match.group("mean_delta")) if correction_match else np.nan,
                    "max_delta_mm": float(correction_match.group("max_delta")) if correction_match else np.nan,
                    "correction_limited_frames": int(guard_match.group("limited")) if guard_match else np.nan,
                    "pairwise_guarded_frames": int(guard_match.group("guarded")) if guard_match else np.nan,
                    "pairwise_rejected_frames": int(guard_match.group("rejected")) if guard_match else np.nan,
                    "elapsed_seconds": float(elapsed_match.group("elapsed")) if elapsed_match else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_hip_trajectories(marker_data_by_method: dict[str, dict[str, np.ndarray]], time_s: np.ndarray) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.2), sharex=True)
    markers = ["Hip", "RHip", "LHip"]
    axis_label = "Y-up vertical coordinate (m)"
    for ax, marker in zip(axes, markers):
        ax.plot(time_s, marker_data_by_method["baseline"][marker][:, 1], label="baseline", linewidth=1.2)
        ax.plot(time_s, marker_data_by_method["rigid"][marker][:, 1], label="rigid", linewidth=1.2)
        ax.set_ylabel(marker)
        ax.set_title(f"{marker} vertical trajectory")
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right")
    fig.text(0.01, 0.5, axis_label, rotation=90, va="center")
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    fig.savefig(FIG_DIR / "fig01_hip_trajectories.png")
    plt.close(fig)


def plot_pairwise_stability(pairwise_df: pd.DataFrame) -> None:
    summary = pairwise_df.groupby(["group", "method"], as_index=False)["std_mm"].mean()
    groups = list(RIGID_GROUPS.keys())
    x = np.arange(len(groups))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for offset, method in [(-width / 2, "baseline"), (width / 2, "rigid")]:
        values = [
            float(summary[(summary["group"] == group) & (summary["method"] == method)]["std_mm"].iloc[0])
            for group in groups
        ]
        ax.bar(x + offset, values, width, label=method)
    ax.set_xticks(x, ["Hip rigid group", "Head/face rigid group"])
    ax.set_ylabel("Mean pairwise distance std (mm)")
    ax.set_title("Rigid group internal-distance stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_pairwise_distance_stability.png")
    plt.close(fig)


def plot_jitter(jitter_df: pd.DataFrame) -> None:
    markers = ["Hip", "RHip", "LHip", "RShoulder", "LShoulder", "RBigToe", "LBigToe", "RHeel", "LHeel"]
    plot_df = jitter_df.set_index("marker").loc[markers]
    x = np.arange(len(markers))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(x - width / 2, plot_df["baseline_jitter_mm"], width, label="baseline")
    ax.bar(x + width / 2, plot_df["rigid_jitter_mm"], width, label="rigid")
    ax.set_xticks(x, markers, rotation=35, ha="right")
    ax.set_ylabel("Second-difference jitter RMS (mm/frame²)")
    ax.set_title("Marker jitter before and after rigid triangulation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_marker_jitter.png")
    plt.close(fig)


def plot_distance_timeseries(marker_data_by_method: dict[str, dict[str, np.ndarray]], time_s: np.ndarray) -> None:
    pairs = [("Hip", "LHip"), ("Hip", "RHip"), ("LHip", "RHip")]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(8.4, 6.2), sharex=True)
    for ax, (marker_a, marker_b) in zip(axes, pairs):
        for method in ["baseline", "rigid"]:
            distances_mm = (
                np.linalg.norm(
                    marker_data_by_method[method][marker_a] - marker_data_by_method[method][marker_b],
                    axis=1,
                )
                * 1000.0
            )
            ax.plot(time_s, distances_mm, label=method, linewidth=1.1)
        ax.set_ylabel(f"{marker_a}-{marker_b}\n(mm)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Hip rigid-group internal distances over time")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_hip_pairwise_distance_timeseries.png")
    plt.close(fig)


def plot_face_distance_timeseries(marker_data_by_method: dict[str, dict[str, np.ndarray]], time_s: np.ndarray) -> None:
    pairs = [("Nose", "LEye"), ("Nose", "REye"), ("LEye", "REye"), ("LEar", "REar")]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(8.4, 6.8), sharex=True)
    for ax, (marker_a, marker_b) in zip(axes, pairs):
        for method in ["baseline", "rigid"]:
            distances_mm = (
                np.linalg.norm(
                    marker_data_by_method[method][marker_a] - marker_data_by_method[method][marker_b],
                    axis=1,
                )
                * 1000.0
            )
            ax.plot(time_s, distances_mm, label=method, linewidth=1.1)
        ax.set_ylabel(f"{marker_a}-{marker_b}\n(mm)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Head/face rigid-group internal distances over time")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_face_pairwise_distance_timeseries.png")
    plt.close(fig)


def fmt(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def build_report(
    pairwise_df: pd.DataFrame,
    pairwise_change_df: pd.DataFrame,
    jitter_df: pd.DataFrame,
    acceptance_df: pd.DataFrame,
    displacement_df: pd.DataFrame,
) -> str:
    hip_pairs = pairwise_df[pairwise_df["group"] == "hip"]
    hip_baseline_std = hip_pairs[hip_pairs["method"] == "baseline"]["std_mm"].mean()
    hip_rigid_std = hip_pairs[hip_pairs["method"] == "rigid"]["std_mm"].mean()
    head_pairs = pairwise_df[pairwise_df["group"] == "head_face"]
    head_baseline_std = head_pairs[head_pairs["method"] == "baseline"]["std_mm"].mean()
    head_rigid_std = head_pairs[head_pairs["method"] == "rigid"]["std_mm"].mean()
    hip_pairwise_change = pairwise_change_df[pairwise_change_df["group"] == "hip"]
    hip_max_change_ratio = hip_pairwise_change["max_abs_change_ratio"].max()
    hip_p95_change_ratio = hip_pairwise_change["p95_abs_change_ratio"].max()

    jitter_lookup = jitter_df.set_index("marker")
    hip_jitter_lines = [
        f"- `{marker}`: {fmt(jitter_lookup.loc[marker, 'baseline_jitter_mm'])} -> "
        f"{fmt(jitter_lookup.loc[marker, 'rigid_jitter_mm'])} mm/frame² "
        f"({fmt(jitter_lookup.loc[marker, 'delta_percent'], 1)}%)."
        for marker in ["Hip", "RHip", "LHip"]
    ]

    acceptance_lines = []
    for _, row in acceptance_df.iterrows():
        acceptance_lines.append(
            f"- `{row['group']}`: accepted {int(row['accepted'])}/{int(row['total'])} frames, "
            f"applied {int(row['applied']) if np.isfinite(row.get('applied', np.nan)) else 'nan'}/{int(row['total'])} frames, "
            f"mean joint reprojection error {fmt(row['mean_error_px'], 1)} px, "
            f"mean correction {fmt(row.get('mean_delta_mm', np.nan), 1)} mm, "
            f"max correction {fmt(row.get('max_delta_mm', np.nan), 1)} mm, "
            f"elapsed {fmt(row.get('elapsed_seconds', np.nan), 2)} s."
        )
    if not acceptance_lines:
        acceptance_lines.append("- Rigid acceptance was not found in the log.")

    displacement_lookup = displacement_df.set_index("marker")
    displacement_lines = [
        f"- `{marker}`: median {fmt(displacement_lookup.loc[marker, 'median_delta_mm'])} mm, "
        f"p95 {fmt(displacement_lookup.loc[marker, 'p95_delta_mm'])} mm."
        for marker in ["Hip", "RHip", "LHip", "Nose", "REye", "LEye", "REar", "LEar"]
    ]

    return "\n".join(
        [
            "# demo0417v6 完整真实数据刚体组三角化验证报告",
            "",
            "## 复现内容",
            "",
            "- 数据来自用户提供的 `data/demo0417v6.zip`，不是合成数据。",
            "- 视频帧数：`cam01=467`、`cam02=391`、`cam03=467`、`cam04=467`；为了四路相机同步对比，完整验证使用最短相机范围 `frame_range = [0, 391]`。",
            "- 以 30 fps 计算，本报告覆盖约 13.03 秒、391 帧纯站立视频。",
            "- 使用真实 `Pose2Sim.poseEstimation()` 生成的 2D JSON，再分别运行 baseline `Pose2Sim.triangulation()` 与启用刚体组后的 `Pose2Sim.triangulation()`。",
            "- baseline TRC 和 rigid TRC 使用同一批 2D JSON、同一标定文件、同一帧范围。",
            "",
            "## 根因与改动",
            "",
            "- 原三角化逐个 marker 独立选择相机并独立估计 3D 点，`Hip/LHip/RHip` 与头面部 marker 的相对距离会逐帧变化。",
            "- 修正后的 `triangulation.rigid_marker_groups` 仍先按原流程得到 baseline，再从稳定帧估计组内模板，逐帧联合最小化该组所有有效 2D 重投影误差。",
            "- 输出不再 100% 替换成刚体模板，而是用 `rigid_group_blend` 混回 baseline，并用 `rigid_group_max_correction_m` 与 `rigid_group_max_pairwise_change_ratio` 防止髋部比例被压缩。",
            "- rigid 拟合默认先用全部有效相机，失败时最多排除 1 个相机，避免在差帧上枚举大量相机组合。",
            "- 这次只启用 `['Hip', 'LHip', 'RHip']` 和 `['Nose', 'LEye', 'REye', 'LEar', 'REar']`，没有对肩膀和脚部 marker 加约束。",
            "",
            "## Before / After",
            "",
            f"- 髋部三点平均组内距离标准差：{fmt(hip_baseline_std)} -> {fmt(hip_rigid_std)} mm。",
            f"- 髋部 pairwise 距离相对 baseline 的最大帧级变化：p95 <= {fmt(hip_p95_change_ratio * 100, 1)}%，max <= {fmt(hip_max_change_ratio * 100, 1)}%。",
            f"- 头面部五点平均组内距离标准差：{fmt(head_baseline_std)} -> {fmt(head_rigid_std)} mm。",
            *hip_jitter_lines,
            "",
            "刚体输出相对 baseline 的空间改变量：",
            *displacement_lines,
            "",
            "刚体拟合接受情况：",
            *acceptance_lines,
            "",
            "肩膀和脚部 marker 的抖动数值在本次实验中保持不变，这是预期结果：本次范围没有对这些 marker 施加刚体约束。",
            "",
            "## Figures",
            "",
            "![Hip trajectories](../figures/rigid_triangulation_demo0417v6_full/fig01_hip_trajectories.png)",
            "",
            "![Pairwise stability](../figures/rigid_triangulation_demo0417v6_full/fig02_pairwise_distance_stability.png)",
            "",
            "![Marker jitter](../figures/rigid_triangulation_demo0417v6_full/fig03_marker_jitter.png)",
            "",
            "![Hip distance time series](../figures/rigid_triangulation_demo0417v6_full/fig04_hip_pairwise_distance_timeseries.png)",
            "",
            "![Face distance time series](../figures/rigid_triangulation_demo0417v6_full/fig05_face_pairwise_distance_timeseries.png)",
            "",
            "## Raw Evidence",
            "",
            "- `data/rigid_triangulation_demo0417v6_full/baseline/demo0417v6_0-390_baseline.trc`",
            "- `data/rigid_triangulation_demo0417v6_full/strict_before/demo0417v6_0-390_strict_before.trc`",
            "- `data/rigid_triangulation_demo0417v6_full/rigid/demo0417v6_0-390_rigid.trc`",
            "- `data/rigid_triangulation_demo0417v6_full/rigid_guarded/demo0417v6_0-390_rigid_guarded.trc`",
            "- `data/rigid_triangulation_demo0417v6_full/pairwise_distance_metrics.csv`",
            "- `data/rigid_triangulation_demo0417v6_full/pairwise_distance_change_metrics.csv`",
            "- `data/rigid_triangulation_demo0417v6_full/marker_jitter_metrics.csv`",
            "- `data/rigid_triangulation_demo0417v6_full/marker_displacement_metrics.csv`",
            "- `data/rigid_triangulation_demo0417v6_full/rigid_acceptance_metrics.csv`",
            "- `data/rigid_triangulation_demo0417v6_full/baseline_run.log`",
            "- `data/rigid_triangulation_demo0417v6_full/strict_before/rigid_strict_before.log`",
            "- `data/rigid_triangulation_demo0417v6_full/rigid_run.log`",
            "- `data/rigid_triangulation_demo0417v6_full/rigid_guarded_run.log`",
            "- `data/rigid_triangulation_demo0417v6_full/rigid_guarded_parallel_smoke.log`",
            "",
            "## Limits",
            "",
            "- 本报告验证的是可同步使用的完整 391 帧范围。`cam01/cam03/cam04` 还有额外帧，但 `cam02` 只有 391 帧，因此没有把后续单相机缺帧区间纳入四相机三角化对比。",
            "- 这次没有更新下游 OpenSim marker/IK 设置。",
            "- 如果还要直接降低肩膀或脚部 marker 抖动，需要另行配置肩带/足部刚体或半刚体组，风险和运动学约束需要单独评估。",
            "",
        ]
    )


def main() -> None:
    ensure_dirs()
    set_plot_style()

    marker_data_by_method = {}
    time_s = None
    for method, path in TRC_PATHS.items():
        marker_data_by_method[method], time_s = load_marker_data(path)

    pairwise_df = pd.DataFrame(pairwise_distance_rows(marker_data_by_method))
    pairwise_change_df = pd.DataFrame(pairwise_change_rows(marker_data_by_method))
    jitter_df = pd.DataFrame(jitter_rows(marker_data_by_method))
    acceptance_df = parse_acceptance()
    displacement_df = pd.DataFrame(displacement_rows(marker_data_by_method))

    pairwise_df.to_csv(DATA_DIR / "pairwise_distance_metrics.csv", index=False)
    pairwise_change_df.to_csv(DATA_DIR / "pairwise_distance_change_metrics.csv", index=False)
    jitter_df.to_csv(DATA_DIR / "marker_jitter_metrics.csv", index=False)
    acceptance_df.to_csv(DATA_DIR / "rigid_acceptance_metrics.csv", index=False)
    displacement_df.to_csv(DATA_DIR / "marker_displacement_metrics.csv", index=False)

    plot_hip_trajectories(marker_data_by_method, time_s)
    plot_pairwise_stability(pairwise_df)
    plot_jitter(jitter_df)
    plot_distance_timeseries(marker_data_by_method, time_s)
    plot_face_distance_timeseries(marker_data_by_method, time_s)

    DOC_PATH.write_text(
        build_report(pairwise_df, pairwise_change_df, jitter_df, acceptance_df, displacement_df),
        encoding="utf-8",
    )
    print(f"Wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
