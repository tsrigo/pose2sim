#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Pose2Sim.poseEstimation import Pose2DTemporalSmoother, process_video
from Pose2Sim.skeletons import HALPE_26


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pose_temporal_smoothing"
FIG_DIR = ROOT / "figures" / "pose_temporal_smoothing"
DOC_PATH = ROOT / "docs" / "pose-temporal-smoothing.md"
SMOKE_DIR = DATA_DIR / "process_video_smoke"

KEYPOINTS = {
    "Hip": 19,
    "RHip": 12,
    "LHip": 11,
    "RAnkle": 16,
    "LAnkle": 15,
    "RBigToe": 21,
    "LBigToe": 20,
    "RSmallToe": 23,
    "LSmallToe": 22,
    "RHeel": 25,
    "LHeel": 24,
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)


def make_synthetic_pose(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_frames = 240
    n_keypoints = 26
    keypoints = np.full((n_frames, 1, n_keypoints, 2), np.nan, dtype=np.float32)
    scores = np.full((n_frames, 1, n_keypoints), 0.9, dtype=np.float32)

    base = np.zeros((n_keypoints, 2), dtype=np.float32)
    for keypoint_id in range(n_keypoints):
        base[keypoint_id] = [640.0 + keypoint_id * 3.0, 420.0 + keypoint_id * 1.5]

    # A nearly static subject with a tiny slow sway, plus detector jitter.
    time = np.arange(n_frames, dtype=np.float32) / 30.0
    slow_sway = np.stack([1.5 * np.sin(2 * np.pi * 0.2 * time), 0.8 * np.cos(2 * np.pi * 0.2 * time)], axis=1)
    detector_noise = rng.normal(0.0, 2.2, size=keypoints.shape).astype(np.float32)
    keypoints[:, 0] = base[None, :, :] + slow_sway[:, None, :] + detector_noise[:, 0]

    # Feet and pelvis get occasional larger spikes, matching the reported failure mode.
    noisy_ids = list(KEYPOINTS.values())
    for frame_id in rng.choice(np.arange(12, n_frames - 12), size=18, replace=False):
        keypoint_id = int(rng.choice(noisy_ids))
        keypoints[frame_id, 0, keypoint_id] += rng.normal(0.0, 9.0, size=2)

    # Short low-confidence gap should not be filled with fabricated values.
    scores[90:93, 0, KEYPOINTS["RAnkle"]] = 0.1
    return keypoints, scores


def apply_smoothing(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    config = {
        "enabled": True,
        "frame_rate": 30,
        "min_cutoff": 1.0,
        "beta": 0.02,
        "d_cutoff": 1.0,
        "min_likelihood": 0.3,
        "max_gap": 5,
        "keypoint_ids": None,
    }
    smoother = Pose2DTemporalSmoother(config)
    smoothed = []
    for frame_id in range(len(keypoints)):
        frame_keypoints, _ = smoother.smooth(keypoints[frame_id], scores[frame_id], frame_id)
        smoothed.append(frame_keypoints)
    return np.asarray(smoothed, dtype=np.float32)


def jitter_metrics(keypoints: np.ndarray, scores: np.ndarray, method: str) -> pd.DataFrame:
    rows = []
    for name, keypoint_id in KEYPOINTS.items():
        points = keypoints[:, 0, keypoint_id]
        valid = np.isfinite(points).all(axis=1) & (scores[:, 0, keypoint_id] >= 0.3)
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) < 3:
            continue
        chunks = np.split(valid_indices, np.where(np.diff(valid_indices) > 1)[0] + 1)
        steps = []
        second_diffs = []
        for chunk in chunks:
            if len(chunk) >= 2:
                step = np.linalg.norm(np.diff(points[chunk], axis=0), axis=1)
                steps.extend(step.tolist())
            if len(chunk) >= 3:
                diff2 = np.linalg.norm(np.diff(points[chunk], n=2, axis=0), axis=1)
                second_diffs.extend(diff2.tolist())
        rows.append(
            {
                "method": method,
                "keypoint": name,
                "frame_step_rms_px": float(np.sqrt(np.mean(np.square(steps)))) if steps else np.nan,
                "second_diff_rms_px": float(np.sqrt(np.mean(np.square(second_diffs)))) if second_diffs else np.nan,
                "frame_step_p95_px": float(np.percentile(steps, 95)) if steps else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_timeseries(raw: np.ndarray, smoothed: np.ndarray) -> pd.DataFrame:
    rows = []
    for method, keypoints in [("raw", raw), ("smoothed", smoothed)]:
        for keypoint in ["Hip", "RHip", "LHip", "RAnkle", "LAnkle", "RBigToe", "LBigToe"]:
            keypoint_id = KEYPOINTS[keypoint]
            for frame_id, point in enumerate(keypoints[:, 0, keypoint_id]):
                rows.append(
                    {
                        "method": method,
                        "keypoint": keypoint,
                        "frame": frame_id,
                        "x_px": float(point[0]),
                        "y_px": float(point[1]),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "synthetic_pose2d_timeseries.csv", index=False)
    return df


def plot_timeseries(timeseries: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.8), sharex=True)
    colors = {"raw": "#b45d48", "smoothed": "#4f7cac"}
    for ax, keypoint in zip(axes, ["Hip", "RAnkle"]):
        for method in ["raw", "smoothed"]:
            series = timeseries[(timeseries["method"] == method) & (timeseries["keypoint"] == keypoint)]
            ax.plot(series["frame"], series["x_px"], label=method, color=colors[method], linewidth=1.1)
        ax.set_ylabel(f"{keypoint} X px")
    axes[-1].set_xlabel("Frame")
    axes[0].legend(loc="upper right")
    fig.suptitle("Synthetic static-pose 2D temporal smoothing")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synthetic_pose2d_smoothing.png", dpi=220)
    plt.close(fig)


class DummyPoseTracker:
    det_model = None

    def __init__(self, keypoints: np.ndarray, scores: np.ndarray):
        self.keypoints = keypoints
        self.scores = scores
        self.frame_id = 0

    def __call__(self, _frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frame_id = min(self.frame_id, len(self.keypoints) - 1)
        self.frame_id += 1
        return self.keypoints[frame_id].copy(), self.scores[frame_id].copy()


def write_dummy_video(path: Path, frame_count: int, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for _ in range(frame_count):
        writer.write(np.full((240, 320, 3), 18, dtype=np.uint8))
    writer.release()


def openpose_json_points(json_dir: Path) -> np.ndarray:
    frames = []
    for json_file in sorted(json_dir.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        keypoints = data["people"][0]["pose_keypoints_2d"]
        frames.append(np.array(keypoints, dtype=np.float32).reshape(-1, 3)[:, :2])
    return np.stack(frames, axis=0)


def process_video_frame_step_rms(points: np.ndarray) -> float:
    selected_ids = list(KEYPOINTS.values())
    steps = np.diff(points[:, selected_ids, :], axis=0)
    return float(np.sqrt(np.nanmean(np.square(steps))))


def run_process_video_smoke(keypoints: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    rows = []
    frame_count = min(60, len(keypoints))
    fps = 30
    smoothing_configs = {
        "raw": {"enabled": False},
        "smoothed": {
            "enabled": True,
            "frame_rate": fps,
            "min_cutoff": 1.0,
            "beta": 0.02,
            "d_cutoff": 1.0,
            "min_likelihood": 0.3,
            "max_gap": 5,
            "keypoint_ids": None,
        },
    }
    for method, smoothing_config in smoothing_configs.items():
        video_path = SMOKE_DIR / method / "videos" / "cam0.mp4"
        write_dummy_video(video_path, frame_count, fps)
        process_video(
            str(video_path),
            DummyPoseTracker(keypoints[:frame_count], scores[:frame_count]),
            HALPE_26,
            [0, frame_count],
            0.0,
            ["openpose"],
            False,
            False,
            False,
            "none",
            100,
            None,
            temporal_smoothing_config=smoothing_config,
        )
        json_points = openpose_json_points(SMOKE_DIR / method / "pose" / "cam0_json")
        rows.append({"method": method, "frame_step_rms_px": process_video_frame_step_rms(json_points)})
    result = pd.DataFrame(rows)
    result.to_csv(DATA_DIR / "process_video_smoke_metrics.csv", index=False)
    return result


def write_report(summary: pd.DataFrame, smoke_summary: pd.DataFrame) -> None:
    aggregate = summary.groupby("method")[["frame_step_rms_px", "second_diff_rms_px", "frame_step_p95_px"]].mean()
    smoke = smoke_summary.set_index("method")
    text = f"""# 2D temporal smoothing synthetic validation

## Setup

- Synthetic 240-frame single-person sequence at 30 fps.
- Person is nearly static, with small slow sway, Gaussian detector jitter, and occasional pelvis/foot spikes.
- The same `Pose2DTemporalSmoother` used by `Pose2Sim/poseEstimation.py` is applied before JSON writing.

## Result

| metric | raw | smoothed |
|---|---:|---:|
| frame step RMS (px) | {aggregate.loc['raw', 'frame_step_rms_px']:.2f} | {aggregate.loc['smoothed', 'frame_step_rms_px']:.2f} |
| second diff RMS (px) | {aggregate.loc['raw', 'second_diff_rms_px']:.2f} | {aggregate.loc['smoothed', 'second_diff_rms_px']:.2f} |
| frame step p95 (px) | {aggregate.loc['raw', 'frame_step_p95_px']:.2f} | {aggregate.loc['smoothed', 'frame_step_p95_px']:.2f} |

![Synthetic 2D smoothing](../figures/pose_temporal_smoothing/synthetic_pose2d_smoothing.png)

## Pipeline Smoke Test

- A synthetic MP4 was passed through the real `process_video` JSON-writing path with a deterministic dummy RTMPose tracker.
- Hip and foot keypoint frame-step RMS changed from {smoke.loc['raw', 'frame_step_rms_px']:.2f} px to {smoke.loc['smoothed', 'frame_step_rms_px']:.2f} px with temporal smoothing enabled.
- The target PnVision trial in this workspace has TRC output only; no source video or 2D RTMPose JSON is available here, so this 2D-stage validation uses synthetic input.

## Raw Evidence

- `data/pose_temporal_smoothing/synthetic_pose2d_metrics.csv`
- `data/pose_temporal_smoothing/synthetic_pose2d_timeseries.csv`
- `data/pose_temporal_smoothing/process_video_smoke_metrics.csv`
- `data/pose_temporal_smoothing/process_video_smoke/raw/pose/cam0_json/`
- `data/pose_temporal_smoothing/process_video_smoke/smoothed/pose/cam0_json/`
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    raw, scores = make_synthetic_pose()
    smoothed = apply_smoothing(raw, scores)
    summary = pd.concat(
        [
            jitter_metrics(raw, scores, "raw"),
            jitter_metrics(smoothed, scores, "smoothed"),
        ],
        ignore_index=True,
    )
    summary.to_csv(DATA_DIR / "synthetic_pose2d_metrics.csv", index=False)
    timeseries = write_timeseries(raw, smoothed)
    plot_timeseries(timeseries)
    smoke_summary = run_process_video_smoke(raw, scores)
    write_report(summary, smoke_summary)
    print(f"Wrote {DOC_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
