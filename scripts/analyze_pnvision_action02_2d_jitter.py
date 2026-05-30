#!/usr/bin/env python
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anytree import RenderTree

from Pose2Sim.skeletons import HALPE_26


ROOT = Path(__file__).resolve().parents[1]
TRIAL = "rec_20260507_134131_lhl_action02"
SOURCE_DIR = ROOT / "data" / "PnVision" / "20260507" / TRIAL
RAW_RUN_DIR = (
    ROOT
    / "data"
    / "pnvision_action02_neck_regression_20260519"
    / "no2d_smoothing"
    / TRIAL
)
SMOOTH_RUN_DIR = ROOT / "data" / "pnvision_action02_rtmpose_smoothing_20260519" / TRIAL
RAW_DETFREQ1_DIR = ROOT / "data" / "pnvision_action02_2d_jitter_detfreq1" / TRIAL
OUT_DIR = ROOT / "data" / "pnvision_action02_2d_jitter_analysis"
FIG_DIR = ROOT / "figures" / "pnvision_action02_2d_jitter"
DOC_PATH = ROOT / "docs" / "pnvision-action02-2d-jitter-analysis.md"

STATIC_RANGE = (0, 60)
STATIC_RANGES = {
    "start_standing": STATIC_RANGE,
    "late_standing": (300, 367),
}
DETECTION_FREQUENCY = 4
LIKELIHOOD_THRESHOLD = 0.3
RUNS = {
    "raw_rtmpose": RAW_RUN_DIR,
    "smoothed_2d": SMOOTH_RUN_DIR,
}
if RAW_DETFREQ1_DIR.exists():
    RUNS["raw_detfreq1"] = RAW_DETFREQ1_DIR
FOCUS_KEYPOINTS = [
    "Neck",
    "Hip",
    "LHip",
    "RHip",
    "LAnkle",
    "RAnkle",
    "LBigToe",
    "RBigToe",
    "LSmallToe",
    "RSmallToe",
    "LHeel",
    "RHeel",
]
OVERLAY_KEYPOINTS = [
    "Neck",
    "Hip",
    "LHip",
    "RHip",
    "LAnkle",
    "RAnkle",
    "LBigToe",
    "RBigToe",
    "LSmallToe",
    "RSmallToe",
    "LHeel",
    "RHeel",
]


def skeleton_names() -> list[str]:
    nodes = [
        node
        for _, _, node in RenderTree(HALPE_26)
        if getattr(node, "id", None) is not None
    ]
    return [node.name for node in sorted(nodes, key=lambda item: item.id)]


KEYPOINT_NAMES = skeleton_names()
KEYPOINT_IDS = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)


def frame_id_from_json(path: Path) -> int:
    match = re.search(r"_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Cannot parse frame id from {path}")
    return int(match.group(1))


def load_pose_jsons(run_name: str, run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pose_dir = run_dir / "pose"
    if not pose_dir.exists():
        raise FileNotFoundError(pose_dir)

    for json_dir in sorted(pose_dir.glob("cam*_json")):
        camera = json_dir.name.replace("_json", "")
        json_files = sorted(json_dir.glob("*.json"), key=frame_id_from_json)
        for json_file in json_files:
            frame = frame_id_from_json(json_file)
            data = json.loads(json_file.read_text(encoding="utf-8"))
            people = data.get("people", [])
            if not people:
                points = np.full((len(KEYPOINT_NAMES), 3), np.nan, dtype=float)
                points[:, 2] = 0.0
            else:
                flat = np.asarray(people[0]["pose_keypoints_2d"], dtype=float)
                points = flat.reshape(-1, 3)
                if len(points) != len(KEYPOINT_NAMES):
                    raise ValueError(
                        f"{json_file} has {len(points)} keypoints, expected {len(KEYPOINT_NAMES)}"
                    )
            for idx, name in enumerate(KEYPOINT_NAMES):
                x, y, score = points[idx]
                rows.append(
                    {
                        "run": run_name,
                        "camera": camera,
                        "frame": frame,
                        "keypoint_id": idx,
                        "keypoint": name,
                        "x_px": x,
                        "y_px": y,
                        "score": score,
                    }
                )
    return pd.DataFrame(rows)


def compute_step_table(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = points.sort_values("frame").groupby(["run", "camera", "keypoint"], sort=False)
    for (run, camera, keypoint), group in grouped:
        group = group.sort_values("frame").reset_index(drop=True)
        xy = group[["x_px", "y_px"]].to_numpy(dtype=float)
        scores = group["score"].to_numpy(dtype=float)
        frames = group["frame"].to_numpy(dtype=int)
        for i in range(1, len(group)):
            if frames[i] != frames[i - 1] + 1:
                continue
            finite = np.isfinite(xy[i]).all() and np.isfinite(xy[i - 1]).all()
            score_min = float(min(scores[i], scores[i - 1]))
            valid = finite and score_min >= LIKELIHOOD_THRESHOLD
            step = float(np.linalg.norm(xy[i] - xy[i - 1])) if finite else math.nan
            rows.append(
                {
                    "run": run,
                    "camera": camera,
                    "keypoint": keypoint,
                    "frame_start": int(frames[i - 1]),
                    "frame_end": int(frames[i]),
                    "step_px": step,
                    "score_min": score_min,
                    "valid": valid,
                    "det_refresh_step": bool(frames[i] % DETECTION_FREQUENCY == 0),
                    "frame_end_mod_det": int(frames[i] % DETECTION_FREQUENCY),
                }
            )
    return pd.DataFrame(rows)


def summarize_static_spread(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for segment, (start, end) in STATIC_RANGES.items():
        subset = points[(points["frame"] >= start) & (points["frame"] <= end)].copy()
        for (run, camera, keypoint), group in subset.groupby(["run", "camera", "keypoint"]):
            valid = (
                np.isfinite(group["x_px"])
                & np.isfinite(group["y_px"])
                & (group["score"] >= LIKELIHOOD_THRESHOLD)
            )
            valid_group = group[valid]
            if len(valid_group) == 0:
                continue
            xy = valid_group[["x_px", "y_px"]].to_numpy(dtype=float)
            center = np.nanmedian(xy, axis=0)
            radial = np.linalg.norm(xy - center, axis=1)
            rows.append(
                {
                    "segment": segment,
                    "run": run,
                    "camera": camera,
                    "keypoint": keypoint,
                    "frames": int(len(valid_group)),
                    "median_score": float(np.nanmedian(valid_group["score"])),
                    "p10_score": float(np.nanpercentile(valid_group["score"], 10)),
                    "x_range_px": float(np.nanmax(xy[:, 0]) - np.nanmin(xy[:, 0])),
                    "y_range_px": float(np.nanmax(xy[:, 1]) - np.nanmin(xy[:, 1])),
                    "radial_p50_px": float(np.nanpercentile(radial, 50)),
                    "radial_p95_px": float(np.nanpercentile(radial, 95)),
                    "radial_max_px": float(np.nanmax(radial)),
                }
            )
    return pd.DataFrame(rows)


def summarize_step_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for segment, (start, end) in STATIC_RANGES.items():
        subset = steps[
            (steps["frame_end"] >= start)
            & (steps["frame_end"] <= end)
            & steps["valid"]
            & np.isfinite(steps["step_px"])
        ]
        for (run, camera, keypoint), group in subset.groupby(["run", "camera", "keypoint"]):
            values = group["step_px"].to_numpy(dtype=float)
            rows.append(
                {
                    "segment": segment,
                    "run": run,
                    "camera": camera,
                    "keypoint": keypoint,
                    "steps": int(len(values)),
                    "step_p50_px": float(np.nanpercentile(values, 50)),
                    "step_p95_px": float(np.nanpercentile(values, 95)),
                    "step_max_px": float(np.nanmax(values)),
                    "step_rms_px": float(np.sqrt(np.nanmean(np.square(values)))),
                    "median_score_min": float(np.nanmedian(group["score_min"])),
                    "p10_score_min": float(np.nanpercentile(group["score_min"], 10)),
                    "large_step_over_5px": float(np.nanmean(values > 5.0)),
                    "large_step_over_10px": float(np.nanmean(values > 10.0)),
                }
            )
    return pd.DataFrame(rows)


def summarize_detector_modulo(steps: pd.DataFrame) -> pd.DataFrame:
    subset = steps[
        (steps["run"] == "raw_rtmpose")
        & (steps["frame_end"] >= STATIC_RANGE[0])
        & (steps["frame_end"] <= STATIC_RANGE[1])
        & steps["valid"]
        & np.isfinite(steps["step_px"])
        & steps["keypoint"].isin(FOCUS_KEYPOINTS)
    ]
    rows = []
    for (keypoint, mod), group in subset.groupby(["keypoint", "frame_end_mod_det"]):
        values = group["step_px"].to_numpy(dtype=float)
        rows.append(
            {
                "keypoint": keypoint,
                "frame_end_mod_det": int(mod),
                "count": int(len(values)),
                "step_p50_px": float(np.nanpercentile(values, 50)),
                "step_p95_px": float(np.nanpercentile(values, 95)),
                "step_mean_px": float(np.nanmean(values)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_focus_metrics(step_metrics: pd.DataFrame, spread_metrics: pd.DataFrame) -> pd.DataFrame:
    step_focus = step_metrics[
        (step_metrics["segment"] == "start_standing")
        & step_metrics["keypoint"].isin(FOCUS_KEYPOINTS)
    ]
    spread_focus = spread_metrics[
        (spread_metrics["segment"] == "start_standing")
        & spread_metrics["keypoint"].isin(FOCUS_KEYPOINTS)
    ]
    agg_step = (
        step_focus.groupby(["run", "keypoint"], as_index=False)
        .agg(
            mean_step_p95_px=("step_p95_px", "mean"),
            max_step_px=("step_max_px", "max"),
            mean_large_step_over_5px=("large_step_over_5px", "mean"),
            p10_score_min=("p10_score_min", "mean"),
        )
        .reset_index(drop=True)
    )
    agg_spread = (
        spread_focus.groupby(["run", "keypoint"], as_index=False)
        .agg(mean_radial_p95_px=("radial_p95_px", "mean"), max_radial_px=("radial_max_px", "max"))
        .reset_index(drop=True)
    )
    return agg_step.merge(agg_spread, on=["run", "keypoint"], how="left")


def plot_focus_steps(focus_metrics: pd.DataFrame) -> None:
    order = FOCUS_KEYPOINTS
    raw = focus_metrics[focus_metrics["run"] == "raw_rtmpose"].set_index("keypoint")
    smooth = focus_metrics[focus_metrics["run"] == "smoothed_2d"].set_index("keypoint")
    x = np.arange(len(order))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.bar(x - width / 2, [raw.loc[k, "mean_step_p95_px"] for k in order], width, label="raw RTMPose", color="#b14b3c")
    ax.bar(x + width / 2, [smooth.loc[k, "mean_step_p95_px"] for k in order], width, label="2D smoothed", color="#2f7e8b")
    ax.set_ylabel("Static segment p95 frame-to-frame step (px)")
    ax.set_xlabel("Keypoint")
    ax.set_title("2D jitter in visually static frames 0-60")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "focus_static_step_p95_by_keypoint.png", dpi=220)
    plt.close(fig)


def plot_detector_modulo(detector_metrics: pd.DataFrame) -> None:
    if detector_metrics.empty:
        return
    aggregated = (
        detector_metrics.groupby("frame_end_mod_det", as_index=False)
        .agg(step_p95_px=("step_p95_px", "mean"), step_mean_px=("step_mean_px", "mean"))
        .sort_values("frame_end_mod_det")
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    labels = [f"{int(mod)}" for mod in aggregated["frame_end_mod_det"]]
    colors = ["#b14b3c" if int(mod) == 0 else "#6687a3" for mod in aggregated["frame_end_mod_det"]]
    ax.bar(labels, aggregated["step_p95_px"], color=colors)
    ax.set_xlabel(f"Frame end modulo det_frequency ({DETECTION_FREQUENCY})")
    ax.set_ylabel("Mean keypoint p95 step (px)")
    ax.set_title("Raw 2D jitter vs detector refresh cadence")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "det_refresh_modulo_static_step.png", dpi=220)
    plt.close(fig)


def plot_detfreq_comparison(focus_metrics: pd.DataFrame) -> None:
    if "raw_detfreq1" not in set(focus_metrics["run"]):
        return
    order = FOCUS_KEYPOINTS
    det4 = focus_metrics[focus_metrics["run"] == "raw_rtmpose"].set_index("keypoint")
    det1 = focus_metrics[focus_metrics["run"] == "raw_detfreq1"].set_index("keypoint")
    x = np.arange(len(order))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.bar(
        x - width / 2,
        [det4.loc[k, "mean_step_p95_px"] for k in order],
        width,
        label="det_frequency=4",
        color="#6687a3",
    )
    ax.bar(
        x + width / 2,
        [det1.loc[k, "mean_step_p95_px"] for k in order],
        width,
        label="det_frequency=1",
        color="#b14b3c",
    )
    ax.set_ylabel("Static segment p95 frame-to-frame step (px)")
    ax.set_xlabel("Keypoint")
    ax.set_title("Detector every frame does not remove raw 2D jitter")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "detfreq1_vs_detfreq4_static_step.png", dpi=220)
    plt.close(fig)


def plot_neck_timeseries(points: pd.DataFrame, step_metrics: pd.DataFrame) -> tuple[str, str]:
    neck = step_metrics[
        (step_metrics["segment"] == "start_standing")
        & (step_metrics["run"] == "raw_rtmpose")
        & (step_metrics["keypoint"] == "Neck")
    ].copy()
    neck = neck.sort_values("step_p95_px", ascending=False)
    camera = str(neck.iloc[0]["camera"]) if len(neck) else "cam01"

    subset = points[
        (points["camera"] == camera)
        & (points["keypoint"] == "Neck")
        & (points["frame"] >= 0)
        & (points["frame"] <= 100)
    ].copy()
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.4), sharex=True)
    colors = {"raw_rtmpose": "#b14b3c", "smoothed_2d": "#2f7e8b"}
    for ax, coord in zip(axes, ["x_px", "y_px"]):
        for run in ["raw_rtmpose", "smoothed_2d"]:
            series = subset[subset["run"] == run].sort_values("frame")
            ax.plot(series["frame"], series[coord], label=run, color=colors[run], linewidth=1.2)
        ax.axvspan(STATIC_RANGE[0], STATIC_RANGE[1], color="#e8e1ca", alpha=0.35)
        ax.set_ylabel(coord)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[-1].set_xlabel("Frame")
    axes[0].set_title(f"Neck 2D coordinate trace, {camera}")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    out_name = "neck_trace_raw_vs_smoothed.png"
    fig.savefig(FIG_DIR / out_name, dpi=220)
    plt.close(fig)
    return camera, out_name


def plot_static_scatter(points: pd.DataFrame, spread_metrics: pd.DataFrame) -> tuple[str, str, str]:
    raw_focus = spread_metrics[
        (spread_metrics["segment"] == "start_standing")
        & (spread_metrics["run"] == "raw_rtmpose")
        & spread_metrics["keypoint"].isin(FOCUS_KEYPOINTS)
    ].sort_values("radial_p95_px", ascending=False)

    examples = []
    if len(raw_focus):
        top = raw_focus.iloc[0]
        examples.append((str(top["camera"]), str(top["keypoint"])))
    neck_focus = raw_focus[raw_focus["keypoint"] == "Neck"].sort_values("radial_p95_px", ascending=False)
    if len(neck_focus):
        top_neck = neck_focus.iloc[0]
        neck_example = (str(top_neck["camera"]), "Neck")
        if neck_example not in examples:
            examples.append(neck_example)
    if not examples:
        examples = [("cam01", "Neck")]

    ncols = len(examples)
    fig, axes = plt.subplots(2, ncols, figsize=(5.4 * ncols, 7.2), squeeze=False)
    colors = {"raw_rtmpose": "#b14b3c", "smoothed_2d": "#2f7e8b"}
    for col, (camera, keypoint) in enumerate(examples):
        for row, run in enumerate(["raw_rtmpose", "smoothed_2d"]):
            ax = axes[row][col]
            series = points[
                (points["run"] == run)
                & (points["camera"] == camera)
                & (points["keypoint"] == keypoint)
                & (points["frame"] >= STATIC_RANGE[0])
                & (points["frame"] <= STATIC_RANGE[1])
                & (points["score"] >= LIKELIHOOD_THRESHOLD)
            ].sort_values("frame")
            ax.plot(series["x_px"], series["y_px"], color=colors[run], linewidth=0.8, alpha=0.55)
            ax.scatter(series["x_px"], series["y_px"], c=series["frame"], cmap="viridis", s=18)
            ax.invert_yaxis()
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"{run}: {camera} {keypoint}")
            ax.set_xlabel("x px")
            ax.set_ylabel("y px")
            ax.grid(color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    out_name = "static_scatter_examples.png"
    fig.savefig(FIG_DIR / out_name, dpi=220)
    plt.close(fig)
    first_camera, first_keypoint = examples[0]
    return first_camera, first_keypoint, out_name


def plot_confidence_vs_step(steps: pd.DataFrame) -> None:
    subset = steps[
        (steps["run"] == "raw_rtmpose")
        & (steps["frame_end"] >= STATIC_RANGE[0])
        & (steps["frame_end"] <= STATIC_RANGE[1])
        & steps["valid"]
        & np.isfinite(steps["step_px"])
        & steps["keypoint"].isin(FOCUS_KEYPOINTS)
    ].copy()
    if subset.empty:
        return
    subset["large_step"] = subset["step_px"] > 5.0
    high_conf_large = subset[subset["large_step"] & (subset["score_min"] >= 0.7)]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(subset["score_min"], subset["step_px"], s=9, alpha=0.28, color="#5d6f8f", label="all focus steps")
    if len(high_conf_large):
        ax.scatter(
            high_conf_large["score_min"],
            high_conf_large["step_px"],
            s=16,
            alpha=0.55,
            color="#b14b3c",
            label=">5 px and score >= 0.7",
        )
    ax.axhline(5.0, color="#b14b3c", linewidth=1.0, linestyle="--")
    ax.axvline(0.7, color="#777777", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Minimum endpoint likelihood for frame step")
    ax.set_ylabel("Frame-to-frame step (px)")
    ax.set_title("Raw static 2D jitter is not fully explained by low likelihood")
    ax.grid(color="#dddddd", linewidth=0.7)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confidence_vs_step_static.png", dpi=220)
    plt.close(fig)


def make_contact_sheets() -> None:
    frames = [0, 30, 60, 90, 120, 180, 240, 300, 367]
    for video_path in sorted(SOURCE_DIR.glob("cam*.avi")):
        cap = cv2.VideoCapture(str(video_path))
        imgs = []
        for frame_id in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, img = cap.read()
            if not ok:
                continue
            img = cv2.resize(img, (248, 325))
            cv2.putText(
                img,
                f"{video_path.stem} f{frame_id}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            imgs.append(img)
        cap.release()
        if imgs:
            sheet = np.concatenate(imgs, axis=1)
            cv2.imwrite(str(FIG_DIR / f"{video_path.stem}_contact_sheet.jpg"), sheet)


def points_by_run_camera(points: pd.DataFrame, run: str, camera: str) -> dict[int, np.ndarray]:
    subset = points[(points["run"] == run) & (points["camera"] == camera)]
    by_frame: dict[int, np.ndarray] = {}
    for frame, group in subset.groupby("frame"):
        arr = np.full((len(KEYPOINT_NAMES), 3), np.nan, dtype=float)
        for _, row in group.iterrows():
            arr[int(row["keypoint_id"])] = [row["x_px"], row["y_px"], row["score"]]
        by_frame[int(frame)] = arr
    return by_frame


def draw_focus_points(frame: np.ndarray, pts: np.ndarray, color: tuple[int, int, int]) -> None:
    for name in OVERLAY_KEYPOINTS:
        idx = KEYPOINT_IDS[name]
        x, y, score = pts[idx]
        if not np.isfinite(x) or not np.isfinite(y) or score < LIKELIHOOD_THRESHOLD:
            continue
        cv2.circle(frame, (int(round(x)), int(round(y))), 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(round(x)), int(round(y))), 9, (0, 0, 0), 1, cv2.LINE_AA)


def make_overlay_video(points: pd.DataFrame, camera: str) -> str:
    video_path = SOURCE_DIR / f"{camera}.avi"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 0.48
    out_size = (int(width * scale) * 2, int(height * scale))
    out_name = f"{camera}_static_raw_vs_smoothed_overlay.mp4"
    writer = cv2.VideoWriter(
        str(FIG_DIR / out_name),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        out_size,
    )
    raw = points_by_run_camera(points, "raw_rtmpose", camera)
    smooth = points_by_run_camera(points, "smoothed_2d", camera)
    for frame_id in range(0, 91):
        ok, frame = cap.read()
        if not ok:
            break
        left = frame.copy()
        right = frame.copy()
        if frame_id in raw:
            draw_focus_points(left, raw[frame_id], (0, 0, 255))
        if frame_id in smooth:
            draw_focus_points(right, smooth[frame_id], (0, 180, 0))
        for panel, label, color in [
            (left, "raw RTMPose JSON", (0, 0, 255)),
            (right, "2D smoothed JSON", (0, 180, 0)),
        ]:
            cv2.putText(panel, f"{label}  frame {frame_id}", (36, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
            cv2.rectangle(panel, (28, 76), (440, 120), (0, 0, 0), -1)
            cv2.putText(panel, "red/green: neck, pelvis, feet", (40, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        joined = np.concatenate([left, right], axis=1)
        joined = cv2.resize(joined, out_size, interpolation=cv2.INTER_AREA)
        writer.write(joined)
    writer.release()
    cap.release()
    return out_name


def markdown_table(df: pd.DataFrame, columns: list[str], n: int = 12) -> str:
    return df.loc[:, columns].head(n).to_markdown(index=False, floatfmt=".2f")


def write_report(
    points: pd.DataFrame,
    step_metrics: pd.DataFrame,
    spread_metrics: pd.DataFrame,
    focus_metrics: pd.DataFrame,
    detector_metrics: pd.DataFrame,
    neck_camera: str,
    scatter_camera: str,
    scatter_keypoint: str,
    overlay_name: str,
) -> None:
    raw_focus = focus_metrics[focus_metrics["run"] == "raw_rtmpose"].copy()
    smooth_focus = focus_metrics[focus_metrics["run"] == "smoothed_2d"].copy()
    merged = raw_focus.merge(
        smooth_focus,
        on="keypoint",
        suffixes=("_raw", "_smooth"),
    )
    merged["step_p95_reduction_pct"] = (
        (merged["mean_step_p95_px_raw"] - merged["mean_step_p95_px_smooth"])
        / merged["mean_step_p95_px_raw"]
        * 100.0
    )
    merged["radial_p95_reduction_pct"] = (
        (merged["mean_radial_p95_px_raw"] - merged["mean_radial_p95_px_smooth"])
        / merged["mean_radial_p95_px_raw"]
        * 100.0
    )
    merged = merged.sort_values("mean_step_p95_px_raw", ascending=False)

    detfreq_table = pd.DataFrame()
    detfreq_text = "未运行 `det_frequency = 1` 对照实验。"
    if "raw_detfreq1" in set(focus_metrics["run"]):
        det4 = focus_metrics[focus_metrics["run"] == "raw_rtmpose"].set_index("keypoint")
        det1 = focus_metrics[focus_metrics["run"] == "raw_detfreq1"].set_index("keypoint")
        detfreq_table = det4.join(det1, lsuffix="_det4", rsuffix="_det1")
        detfreq_table["p95_change_pct"] = (
            (detfreq_table["mean_step_p95_px_det1"] - detfreq_table["mean_step_p95_px_det4"])
            / detfreq_table["mean_step_p95_px_det4"]
            * 100.0
        )
        detfreq_table = detfreq_table.reset_index().sort_values("p95_change_pct", ascending=False)
        worse_count = int((detfreq_table["p95_change_pct"] > 0).sum())
        detfreq_text = (
            f"补充对照把同一段 0-60 帧重跑为 `det_frequency = 1`，"
            f"{worse_count}/{len(detfreq_table)} 个关注点的平均 p95 跳动变大。"
            "因此 4 帧刷新不是主因；每帧重检会带来更多 bbox/crop 变化，"
            "反而可能放大 raw 2D 抖动。"
        )

    top_raw = step_metrics[
        (step_metrics["segment"] == "start_standing")
        & (step_metrics["run"] == "raw_rtmpose")
        & step_metrics["keypoint"].isin(FOCUS_KEYPOINTS)
    ].sort_values("step_p95_px", ascending=False)

    neck_table = step_metrics[
        (step_metrics["segment"] == "start_standing")
        & (step_metrics["run"] == "raw_rtmpose")
        & (step_metrics["keypoint"] == "Neck")
    ].sort_values("step_p95_px", ascending=False)

    conf_subset = steps_for_confidence(points)
    high_conf_large_pct = float(conf_subset["high_conf_large_step"].mean() * 100.0) if len(conf_subset) else 0.0
    large_step_pct = float(conf_subset["large_step"].mean() * 100.0) if len(conf_subset) else 0.0

    det_summary = detector_metrics.groupby("frame_end_mod_det", as_index=False).agg(
        mean_p95_step_px=("step_p95_px", "mean"),
        mean_step_px=("step_mean_px", "mean"),
    )
    det_zero = det_summary[det_summary["frame_end_mod_det"] == 0]
    det_other = det_summary[det_summary["frame_end_mod_det"] != 0]
    det_ratio = math.nan
    if len(det_zero) and len(det_other):
        det_ratio = float(det_zero["mean_p95_step_px"].iloc[0] / det_other["mean_p95_step_px"].mean())
    detfreq_path = RAW_DETFREQ1_DIR.relative_to(ROOT) if RAW_DETFREQ1_DIR.exists() else "not generated"
    detfreq_markdown = (
        markdown_table(
            detfreq_table,
            [
                "keypoint",
                "mean_step_p95_px_det4",
                "mean_step_p95_px_det1",
                "p95_change_pct",
                "mean_radial_p95_px_det4",
                "mean_radial_p95_px_det1",
            ],
            12,
        )
        if len(detfreq_table)
        else ""
    )

    doc = f"""# PnVision action02 2D RTMPose 源头抖动分析

## 范围

试次：`{TRIAL}`。

本次分析使用真实源视频和仓库实际跑出的 RTMPose JSON：

- 源头 raw 2D：`{RAW_RUN_DIR.relative_to(ROOT)}`（`temporal_smoothing = false`）
- 2D 平滑对照：`{SMOOTH_RUN_DIR.relative_to(ROOT)}`（`temporal_smoothing = true`）
- `det_frequency = 1` 小实验：`{detfreq_path}`

主分析段是肉眼静止的开头 {STATIC_RANGE[0]}-{STATIC_RANGE[1]} 帧。contact sheet 显示这段人基本站立，后面才开始走动/转身。

## 汇报用短结论

医生看到的 2D 点抖动不是 TRC 或三维重建才产生的，raw RTMPose JSON 里已经存在。开头站立段里，抖动最大的是脚尖/脚跟等末端点，髋部和颈部也有可见跳动；不少大跳动的置信度仍然很高，所以单靠 likelihood 阈值筛不掉。

根因更接近 top-down 2D 姿态估计本身的逐帧不稳定：每帧都从图像 crop 里独立预测热图峰值，衣服边缘、鞋/地面接触、遮挡和视角会让同一个静止点在像素上来回选峰。2D 时序平滑能显著改善展示效果，但它是显示/质量控制层面的稳定化，不等于模型本身已经可靠。

## 关键证据

raw 2D 在 {STATIC_RANGE[0]}-{STATIC_RANGE[1]} 帧中逐帧位移 p95 最大的点：

{markdown_table(top_raw, ["camera", "keypoint", "step_p50_px", "step_p95_px", "step_max_px", "median_score_min", "large_step_over_5px"], 14)}

Neck 在各相机 raw 2D 里的抖动：

{markdown_table(neck_table, ["camera", "step_p50_px", "step_p95_px", "step_max_px", "median_score_min", "large_step_over_5px"], 8)}

raw 与 2D 平滑后的关注点汇总：

{markdown_table(merged, ["keypoint", "mean_step_p95_px_raw", "mean_step_p95_px_smooth", "step_p95_reduction_pct", "mean_radial_p95_px_raw", "mean_radial_p95_px_smooth", "radial_p95_reduction_pct"], 12)}

`det_frequency` 对照：{detfreq_text}

{detfreq_markdown}

检测刷新节奏检查：当前 `det_frequency = {DETECTION_FREQUENCY}` 时，`frame % {DETECTION_FREQUENCY} == 0` 是检测框刷新边界。刷新边界的平均 p95 位移是非刷新边界的 {det_ratio:.2f}x，整体没有明显的 4 帧周期峰值；个别点位会受 bbox/crop 影响，但不是全局主因。

置信度检查：raw 关注点在 {STATIC_RANGE[0]}-{STATIC_RANGE[1]} 帧内，{large_step_pct:.1f}% 的有效逐帧位移超过 5 px，其中 {high_conf_large_pct:.1f}% 同时满足位移 >5 px 且两端最小 likelihood >=0.7。这说明“大跳动”并不总是低置信度点，医生看到的抖动不能只靠置信度解释。

## 图和视频证据

![Static focus step p95](../figures/pnvision_action02_2d_jitter/focus_static_step_p95_by_keypoint.png)

![Detector refresh cadence](../figures/pnvision_action02_2d_jitter/det_refresh_modulo_static_step.png)

![det_frequency comparison](../figures/pnvision_action02_2d_jitter/detfreq1_vs_detfreq4_static_step.png)

![Neck trace](../figures/pnvision_action02_2d_jitter/neck_trace_raw_vs_smoothed.png)

![Static scatter examples](../figures/pnvision_action02_2d_jitter/static_scatter_examples.png)

![Confidence vs step](../figures/pnvision_action02_2d_jitter/confidence_vs_step_static.png)

代表性 overlay 视频：`figures/pnvision_action02_2d_jitter/{overlay_name}`。左边是 raw RTMPose 关注点，右边是 2D 平滑后关注点，相机 `{scatter_camera}`。

用于确认静止段的 contact sheet：

- `figures/pnvision_action02_2d_jitter/cam01_contact_sheet.jpg`
- `figures/pnvision_action02_2d_jitter/cam02_contact_sheet.jpg`
- `figures/pnvision_action02_2d_jitter/cam03_contact_sheet.jpg`
- `figures/pnvision_action02_2d_jitter/cam04_contact_sheet.jpg`

## 根因判断

1. 这不是 TRC-only artifact。raw JSON 在站立段已经有逐帧 2D 位移，并且能直接画回源视频。
2. `det_frequency = 4` 不是唯一主因。重跑 `det_frequency = 1` 后，多数关注点更抖，说明频繁重检本身也会引入 crop 变化。
3. RTMPose raw 输出没有时间一致性约束。Neck、pelvis、feet 都是逐帧热图选峰，局部视觉证据相似时会在相邻像素峰之间跳。
4. likelihood 不是充分的稳定性指标。一部分明显跳动仍是高置信度，所以低置信度过滤只能覆盖一部分问题。

## 建议的下一步

- 面向医生的 2D 录像默认使用“平滑 + 质量标记”的 overlay；raw overlay 保留为 debug 证据，不直接当稳定测量展示。
- 每个 trial 自动出 2D 抖动质控报告：p95 逐帧位移、静止段散布、高置信大跳动比例、多相机一致性。
- 在 pose 估计前做 bbox/crop 稳定化实验：平滑目标框中心/尺度，或使用跟踪框，再和当前 raw JSON 做同段对照。
- 2D 平滑保持配置化/自适应，避免快速动作被过度平滑。
- 不再用 likelihood 单独说明点位可靠；需要结合时间一致性和多相机几何一致性。

## 限制

当前 JSON 不保存 bbox/crop 参数，所以 bbox 贡献是通过 `det_frequency` 对照和帧序节奏间接判断的。要精确证明 bbox 抖动幅度，需要下一步在 RTMPose 导出阶段记录每帧 bbox center/scale。

## 原始证据文件

- `data/pnvision_action02_2d_jitter_analysis/pose2d_points.csv`
- `data/pnvision_action02_2d_jitter_analysis/pose2d_frame_steps.csv`
- `data/pnvision_action02_2d_jitter_analysis/static_step_metrics.csv`
- `data/pnvision_action02_2d_jitter_analysis/static_spread_metrics.csv`
- `data/pnvision_action02_2d_jitter_analysis/focus_keypoint_summary.csv`
- `data/pnvision_action02_2d_jitter_analysis/detector_modulo_metrics.csv`
- `data/pnvision_action02_2d_jitter_analysis/detfreq1_comparison.csv`
"""
    DOC_PATH.write_text(doc, encoding="utf-8")


def steps_for_confidence(points: pd.DataFrame) -> pd.DataFrame:
    steps = compute_step_table(points)
    subset = steps[
        (steps["run"] == "raw_rtmpose")
        & (steps["frame_end"] >= STATIC_RANGE[0])
        & (steps["frame_end"] <= STATIC_RANGE[1])
        & steps["valid"]
        & np.isfinite(steps["step_px"])
        & steps["keypoint"].isin(FOCUS_KEYPOINTS)
    ].copy()
    if subset.empty:
        return subset
    subset["large_step"] = subset["step_px"] > 5.0
    subset["high_conf_large_step"] = subset["large_step"] & (subset["score_min"] >= 0.7)
    return subset


def main() -> None:
    ensure_dirs()
    make_contact_sheets()

    points = pd.concat(
        [load_pose_jsons(run_name, run_dir) for run_name, run_dir in RUNS.items()],
        ignore_index=True,
    )
    steps = compute_step_table(points)
    spread_metrics = summarize_static_spread(points)
    step_metrics = summarize_step_metrics(steps)
    detector_metrics = summarize_detector_modulo(steps)
    focus_metrics = aggregate_focus_metrics(step_metrics, spread_metrics)

    points.to_csv(OUT_DIR / "pose2d_points.csv", index=False)
    steps.to_csv(OUT_DIR / "pose2d_frame_steps.csv", index=False)
    spread_metrics.to_csv(OUT_DIR / "static_spread_metrics.csv", index=False)
    step_metrics.to_csv(OUT_DIR / "static_step_metrics.csv", index=False)
    detector_metrics.to_csv(OUT_DIR / "detector_modulo_metrics.csv", index=False)
    focus_metrics.to_csv(OUT_DIR / "focus_keypoint_summary.csv", index=False)
    if "raw_detfreq1" in set(focus_metrics["run"]):
        det4 = focus_metrics[focus_metrics["run"] == "raw_rtmpose"].set_index("keypoint")
        det1 = focus_metrics[focus_metrics["run"] == "raw_detfreq1"].set_index("keypoint")
        detfreq = det4.join(det1, lsuffix="_det4", rsuffix="_det1")
        detfreq["p95_change_pct"] = (
            (detfreq["mean_step_p95_px_det1"] - detfreq["mean_step_p95_px_det4"])
            / detfreq["mean_step_p95_px_det4"]
            * 100.0
        )
        detfreq.reset_index().to_csv(OUT_DIR / "detfreq1_comparison.csv", index=False)

    plot_focus_steps(focus_metrics)
    plot_detector_modulo(detector_metrics)
    plot_detfreq_comparison(focus_metrics)
    neck_camera, _ = plot_neck_timeseries(points, step_metrics)
    scatter_camera, scatter_keypoint, _ = plot_static_scatter(points, spread_metrics)
    plot_confidence_vs_step(steps)
    overlay_name = make_overlay_video(points, scatter_camera)

    write_report(
        points=points,
        step_metrics=step_metrics,
        spread_metrics=spread_metrics,
        focus_metrics=focus_metrics,
        detector_metrics=detector_metrics,
        neck_camera=neck_camera,
        scatter_camera=scatter_camera,
        scatter_keypoint=scatter_keypoint,
        overlay_name=overlay_name,
    )

    print(f"Wrote {OUT_DIR.relative_to(ROOT)}")
    print(f"Wrote {FIG_DIR.relative_to(ROOT)}")
    print(f"Wrote {DOC_PATH.relative_to(ROOT)}")
    print(f"Representative static scatter: {scatter_camera} {scatter_keypoint}")
    print(f"Neck trace camera: {neck_camera}")


if __name__ == "__main__":
    main()
