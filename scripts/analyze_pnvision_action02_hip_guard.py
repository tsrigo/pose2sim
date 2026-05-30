#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Pose2Sim.common import read_trc
from Pose2Sim.filtering import filter_all, stabilize_rigid_marker_groups_3d


ROOT = Path(__file__).resolve().parents[1]
PNVISION_DIR = ROOT / "data" / "PnVision" / "20260507"
TARGET_TRIAL = "rec_20260507_134131_lhl_action02"
TARGET_TRC = PNVISION_DIR / TARGET_TRIAL / "pose-3d" / f"{TARGET_TRIAL}.trc"
DATA_DIR = ROOT / "data" / "pnvision_action02_hip_guard"
FIG_DIR = ROOT / "figures" / "pnvision_action02_hip_guard"
DOC_PATH = ROOT / "docs" / "pnvision-action02-hip-guard.md"

GROUP_MARKERS = ["Hip", "RHip", "LHip"]
PAIR_NAMES = [("Hip", "RHip"), ("Hip", "LHip"), ("RHip", "LHip")]
GUARDED_CONFIG = {
    "filtering": {
        "rigid_marker_groups": [
            {"name": "pelvis", "markers": GROUP_MARKERS},
        ],
        "rigid_group_smoothing_window": 31,
        "rigid_group_blend": 0.7,
        "rigid_group_max_correction_m": 0.05,
        "rigid_group_max_pairwise_change_ratio": 0.15,
        "rigid_group_max_correction_step_m": 0.015,
        "rigid_group_fill_missing": False,
    }
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_marker_data(trc_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    coords, frames, time_col, markers, header = read_trc(trc_path)
    missing = [marker for marker in GROUP_MARKERS if marker not in markers]
    if missing:
        raise ValueError(f"{trc_path} is missing required pelvis markers: {missing}")
    return coords, frames, time_col, markers, header


def marker_points(coords: pd.DataFrame, markers: list[str]) -> dict[str, np.ndarray]:
    return {
        marker: coords.iloc[:, markers.index(marker) * 3 : (markers.index(marker) + 1) * 3].to_numpy(dtype=float)
        for marker in GROUP_MARKERS
    }


def second_difference_rms_mm(points: np.ndarray) -> float:
    valid = np.isfinite(points).all(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < 3:
        return np.nan
    chunks = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    values = []
    for chunk in chunks:
        if len(chunk) < 3:
            continue
        diff2 = np.diff(points[chunk], n=2, axis=0)
        values.append(np.sqrt(np.nanmean(np.sum(diff2 * diff2, axis=1))) * 1000.0)
    return float(np.nanmean(values)) if values else np.nan


def pelvis_metrics(coords: pd.DataFrame, markers: list[str]) -> dict[str, float]:
    points = marker_points(coords, markers)
    pair_std = {}
    pair_maxdev = {}
    pair_last30_std = {}
    for marker_a, marker_b in PAIR_NAMES:
        distances_mm = np.linalg.norm(points[marker_a] - points[marker_b], axis=1) * 1000.0
        pair_name = f"{marker_a}-{marker_b}"
        pair_std[pair_name] = float(np.nanstd(distances_mm))
        pair_maxdev[pair_name] = float(np.nanmax(np.abs(distances_mm - np.nanmedian(distances_mm))))
        pair_last30_std[pair_name] = float(np.nanstd(distances_mm[-30:]))

    jitter = {marker: second_difference_rms_mm(values) for marker, values in points.items()}
    return {
        "pair_std_mean_mm": float(np.nanmean(list(pair_std.values()))),
        "pair_maxdev_mean_mm": float(np.nanmean(list(pair_maxdev.values()))),
        "pair_last30_std_mean_mm": float(np.nanmean(list(pair_last30_std.values()))),
        "pelvis_jitter_mean_mm_frame2": float(np.nanmean(list(jitter.values()))),
        "hip_rhip_std_mm": pair_std["Hip-RHip"],
        "hip_lhip_std_mm": pair_std["Hip-LHip"],
        "rhip_lhip_std_mm": pair_std["RHip-LHip"],
    }


def pairwise_timeseries(coords: pd.DataFrame, markers: list[str], method: str, time_col: pd.Series) -> pd.DataFrame:
    points = marker_points(coords, markers)
    rows = []
    for marker_a, marker_b in PAIR_NAMES:
        distances_mm = np.linalg.norm(points[marker_a] - points[marker_b], axis=1) * 1000.0
        for frame_id, distance_mm in enumerate(distances_mm):
            rows.append(
                {
                    "method": method,
                    "frame_index": frame_id,
                    "time_s": float(time_col.iloc[frame_id]),
                    "pair": f"{marker_a}-{marker_b}",
                    "distance_mm": float(distance_mm),
                }
            )
    return pd.DataFrame(rows)


def scan_all_trials() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    stat_rows = []
    for trc_path in sorted(PNVISION_DIR.glob("*/pose-3d/*.trc")):
        coords, _, _, markers, _ = load_marker_data(trc_path)
        fixed_coords, stats = stabilize_rigid_marker_groups_3d(GUARDED_CONFIG, coords, markers)
        trial = trc_path.parent.parent.name
        rows.append({"trial": trial, "method": "baseline", **pelvis_metrics(coords, markers)})
        rows.append({"trial": trial, "method": "guarded_filter", **pelvis_metrics(fixed_coords, markers)})
        if stats:
            stat_rows.append({"trial": trial, **stats[0]})

    metrics_df = pd.DataFrame(rows)
    stats_df = pd.DataFrame(stat_rows)
    metrics_df.to_csv(DATA_DIR / "pnvision_all_trials_guarded_metrics.csv", index=False)
    stats_df.to_csv(DATA_DIR / "pnvision_all_trials_guarded_stats.csv", index=False)
    return metrics_df, stats_df


def run_real_filter_all() -> Path:
    run_dir = DATA_DIR / "real_entrypoint_run"
    pose_dir = run_dir / "pose-3d"
    pose_dir.mkdir(parents=True, exist_ok=True)
    input_path = pose_dir / TARGET_TRC.name
    shutil.copy2(TARGET_TRC, input_path)
    for stale_output in pose_dir.glob("*_filt_butterworth.trc"):
        stale_output.unlink()

    config = {
        "project": {
            "project_dir": str(run_dir),
            "frame_range": "all",
            "frame_rate": 30,
        },
        "filtering": {
            "display_figures": False,
            "save_filt_plots": False,
            "filter": False,
            "reject_outliers": False,
            "type": "butterworth",
            "filter_ik": False,
            "make_c3d": False,
            **GUARDED_CONFIG["filtering"],
        },
    }
    filter_all(config)
    outputs = sorted(pose_dir.glob("*_filt_butterworth.trc"))
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one filtered TRC in {pose_dir}, found {outputs}")
    return outputs[0]


def target_analysis(filtered_trc: Path) -> pd.DataFrame:
    baseline_coords, frames, time_col, markers, _ = load_marker_data(TARGET_TRC)
    guarded_coords, _, _, guarded_markers, _ = load_marker_data(filtered_trc)
    rows = [
        {"trial": TARGET_TRIAL, "method": "baseline", "trc_path": str(TARGET_TRC.relative_to(ROOT)), **pelvis_metrics(baseline_coords, markers)},
        {
            "trial": TARGET_TRIAL,
            "method": "guarded_filter_all",
            "trc_path": str(filtered_trc.relative_to(ROOT)),
            **pelvis_metrics(guarded_coords, guarded_markers),
        },
    ]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(DATA_DIR / "target_before_after_metrics.csv", index=False)

    pairwise_df = pd.concat(
        [
            pairwise_timeseries(baseline_coords, markers, "baseline", time_col),
            pairwise_timeseries(guarded_coords, guarded_markers, "guarded_filter_all", time_col),
        ],
        ignore_index=True,
    )
    pairwise_df.to_csv(DATA_DIR / "target_pairwise_distance_timeseries.csv", index=False)

    baseline_pairwise = pairwise_df[pairwise_df["method"] == "baseline"].copy()
    baseline_pairwise["pair_median_mm"] = baseline_pairwise.groupby("pair")["distance_mm"].transform("median")
    baseline_pairwise["abs_pair_deviation_mm"] = (
        baseline_pairwise["distance_mm"] - baseline_pairwise["pair_median_mm"]
    ).abs()
    score = baseline_pairwise.groupby("frame_index")["abs_pair_deviation_mm"].mean()
    worst_frame_ids = score.sort_values(ascending=False).head(20).index
    worst_frames = pairwise_df[pairwise_df["frame_index"].isin(worst_frame_ids)].copy()
    worst_frames["baseline_frame_score_mm"] = worst_frames["frame_index"].map(score)
    worst_frames.to_csv(DATA_DIR / "target_worst_pairwise_frames.csv", index=False)
    return metrics_df


def plot_target_pairwise() -> None:
    pairwise_df = pd.read_csv(DATA_DIR / "target_pairwise_distance_timeseries.csv")
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.6), sharex=True)
    colors = {"baseline": "#b45d48", "guarded_filter_all": "#4f7cac"}
    for ax, pair in zip(axes, [f"{a}-{b}" for a, b in PAIR_NAMES]):
        for method in ["baseline", "guarded_filter_all"]:
            series = pairwise_df[(pairwise_df["pair"] == pair) & (pairwise_df["method"] == method)]
            ax.plot(series["frame_index"], series["distance_mm"], label=method, color=colors[method], linewidth=1.1)
        ax.axvspan(142, 151, color="#d9b44a", alpha=0.18, linewidth=0)
        ax.set_ylabel(f"{pair} mm")
    axes[-1].set_xlabel("Frame")
    axes[0].legend(loc="upper right")
    fig.suptitle("Target trial pelvis pairwise distances")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "target_pairwise_distances.png", dpi=220)
    plt.close(fig)


def plot_all_trials(metrics_df: pd.DataFrame) -> None:
    pivot = metrics_df.pivot(index="trial", columns="method", values="pair_std_mean_mm")
    pivot = pivot.sort_values("baseline", ascending=False)
    labels = [
        label.replace("rec_", "").encode("ascii", errors="ignore").decode("ascii").strip("_") or "non_ascii_trial"
        for label in pivot.index
    ]
    y = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    ax.barh(y + 0.18, pivot["baseline"], height=0.34, label="baseline", color="#b45d48")
    ax.barh(y - 0.18, pivot["guarded_filter"], height=0.34, label="guarded_filter", color="#4f7cac")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean pairwise distance std (mm)")
    ax.set_title("PnVision pelvis shape stability across all TRCs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "all_trials_pairwise_std.png", dpi=220)
    plt.close(fig)


def write_report(target_df: pd.DataFrame, all_df: pd.DataFrame, stats_df: pd.DataFrame, filtered_trc: Path) -> None:
    target = target_df.set_index("method")
    aggregate = all_df.groupby("method")[["pair_std_mean_mm", "pair_maxdev_mean_mm", "pair_last30_std_mean_mm", "pelvis_jitter_mean_mm_frame2"]].mean()
    target_stats = stats_df[stats_df["trial"] == TARGET_TRIAL].iloc[0].to_dict()
    text = f"""# PnVision action02 hip guard report

## Reproduced issue

- Target: `{TARGET_TRC.relative_to(ROOT)}`.
- The stored filtered TRC does not show a large final-30-frame pelvis-distance spike, but the same trial has a clear pelvis-shape failure around frames 142-151.
- Baseline target pelvis pairwise-distance std: {target.loc['baseline', 'pair_std_mean_mm']:.2f} mm.
- Baseline target mean max distance deviation: {target.loc['baseline', 'pair_maxdev_mean_mm']:.2f} mm.

## Root cause

The pelvis markers are triangulated and filtered as independent points. A short-lived 2D/keypoint or camera-selection error can keep an acceptable per-marker reprojection error while still collapsing the local `Hip/RHip/LHip` shape.

## Code changed

- `Pose2Sim/filtering.py` now guards 3D rigid marker stabilization with:
  - per-marker correction cap: 50 mm by default,
  - pairwise ratio protection for already-stable frames: 15% by default,
  - per-frame correction-vector step cap: 15 mm by default,
  - rejection of rigid corrections that move an outlier frame farther away from the learned template.
- The behavior remains opt-in through `filtering.rigid_marker_groups`; projects without this config keep the previous output path.

## Before / after

Target trial through real `filter_all`:

| metric | baseline | guarded |
|---|---:|---:|
| pairwise distance std mean (mm) | {target.loc['baseline', 'pair_std_mean_mm']:.2f} | {target.loc['guarded_filter_all', 'pair_std_mean_mm']:.2f} |
| pairwise max deviation mean (mm) | {target.loc['baseline', 'pair_maxdev_mean_mm']:.2f} | {target.loc['guarded_filter_all', 'pair_maxdev_mean_mm']:.2f} |
| final-30-frame distance std mean (mm) | {target.loc['baseline', 'pair_last30_std_mean_mm']:.2f} | {target.loc['guarded_filter_all', 'pair_last30_std_mean_mm']:.2f} |
| pelvis jitter mean (mm/frame^2) | {target.loc['baseline', 'pelvis_jitter_mean_mm_frame2']:.2f} | {target.loc['guarded_filter_all', 'pelvis_jitter_mean_mm_frame2']:.2f} |

All 27 PnVision TRCs in-memory:

| metric | baseline | guarded |
|---|---:|---:|
| pairwise distance std mean (mm) | {aggregate.loc['baseline', 'pair_std_mean_mm']:.2f} | {aggregate.loc['guarded_filter', 'pair_std_mean_mm']:.2f} |
| pairwise max deviation mean (mm) | {aggregate.loc['baseline', 'pair_maxdev_mean_mm']:.2f} | {aggregate.loc['guarded_filter', 'pair_maxdev_mean_mm']:.2f} |
| final-30-frame distance std mean (mm) | {aggregate.loc['baseline', 'pair_last30_std_mean_mm']:.2f} | {aggregate.loc['guarded_filter', 'pair_last30_std_mean_mm']:.2f} |
| pelvis jitter mean (mm/frame^2) | {aggregate.loc['baseline', 'pelvis_jitter_mean_mm_frame2']:.2f} | {aggregate.loc['guarded_filter', 'pelvis_jitter_mean_mm_frame2']:.2f} |

Target guard stats: accepted {int(target_stats['accepted'])}/{int(target_stats['total'])}, applied {int(target_stats['applied'])}/{int(target_stats['total'])}, correction-limited frames {int(target_stats['correction_limited'])}, step-limited frames {int(target_stats['step_limited'])}, rejected non-improving pairwise corrections {int(target_stats['pairwise_rejected'])}.

## Figures

![Target pairwise distances](../figures/pnvision_action02_hip_guard/target_pairwise_distances.png)

![All trials pairwise std](../figures/pnvision_action02_hip_guard/all_trials_pairwise_std.png)

## Raw evidence

- Real `filter_all` output: `{filtered_trc.relative_to(ROOT)}`.
- Target metrics: `data/pnvision_action02_hip_guard/target_before_after_metrics.csv`.
- Target pairwise time series: `data/pnvision_action02_hip_guard/target_pairwise_distance_timeseries.csv`.
- Target worst frames: `data/pnvision_action02_hip_guard/target_worst_pairwise_frames.csv`.
- All-trial metrics: `data/pnvision_action02_hip_guard/pnvision_all_trials_guarded_metrics.csv`.
- All-trial guard stats: `data/pnvision_action02_hip_guard/pnvision_all_trials_guarded_stats.csv`.
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure_dirs()
    filtered_trc = run_real_filter_all()
    target_df = target_analysis(filtered_trc)
    all_df, stats_df = scan_all_trials()
    plot_target_pairwise()
    plot_all_trials(all_df)
    write_report(target_df, all_df, stats_df, filtered_trc)
    print(f"Wrote {DOC_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
