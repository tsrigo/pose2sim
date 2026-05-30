#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from anytree import RenderTree

from Pose2Sim.skeletons import COCO_25, HALPE_26


ROOT = Path(__file__).resolve().parents[1]
TRIAL = "rec_20260507_134131_lhl_action02"
SOURCE_DIR = ROOT / "data" / "PnVision" / "20260507" / TRIAL
BASELINE_DIR = (
    ROOT
    / "data"
    / "pnvision_action02_2d_smoothing_ablation_20260519"
    / "no2d_smoothing"
    / TRIAL
)
SMOOTH_DIR = (
    ROOT
    / "data"
    / "pnvision_action02_2d_smoothing_ablation_20260519"
    / "with2d_smoothing"
    / TRIAL
)
VITPOSE_DIR = (
    ROOT
    / "data"
    / "pnvision_action02_vitpose_20260520"
    / "full_0_367"
    / TRIAL
)
FIG_DIR = ROOT / "figures" / "pnvision_action02_full_2d_comparison_20260520"
DELIVERABLE_DIR = ROOT / "deliverables" / "pnvision_action02_full_2d_comparison_20260520"

CAMERAS = ["cam01", "cam02", "cam03", "cam04"]
FRAME_START = 0
FRAME_END = 368
LIKELIHOOD_THRESHOLD = 0.3
FOCUS_POINTS = {
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
}


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    trial_dir: Path
    skeleton: object
    color: tuple[int, int, int]


BASELINE = RunSpec(
    key="baseline",
    label="baseline RTMPose",
    trial_dir=BASELINE_DIR,
    skeleton=HALPE_26,
    color=(0, 80, 255),
)
SMOOTH = RunSpec(
    key="with_2d_smoothing",
    label="RTMPose + 2D smoothing",
    trial_dir=SMOOTH_DIR,
    skeleton=HALPE_26,
    color=(40, 210, 40),
)
VITPOSE = RunSpec(
    key="vitpose_s_coco25",
    label="ViTPose-S COCO_25",
    trial_dir=VITPOSE_DIR,
    skeleton=COCO_25,
    color=(255, 170, 0),
)


def skeleton_index(skeleton: object) -> tuple[dict[str, int], list[tuple[int, int]]]:
    name_to_id: dict[str, int] = {}
    edges: list[tuple[int, int]] = []
    for _, _, node in RenderTree(skeleton):
        node_id = getattr(node, "id", None)
        if node_id is not None:
            name_to_id[node.name] = int(node_id)
        parent = getattr(node, "parent", None)
        parent_id = getattr(parent, "id", None) if parent is not None else None
        if node_id is not None and parent_id is not None:
            edges.append((int(parent_id), int(node_id)))
    return name_to_id, edges


def load_points(json_dir: Path, camera: str, frame_id: int, keypoint_count: int) -> np.ndarray:
    json_path = json_dir / f"{camera}_{frame_id:06d}.json"
    points = np.full((keypoint_count, 3), np.nan, dtype=float)
    if not json_path.is_file():
        return points

    data = json.loads(json_path.read_text())
    people = data.get("people", [])
    if not people:
        return points

    flat = people[0].get("pose_keypoints_2d", [])
    if len(flat) < keypoint_count * 3:
        return points

    return np.asarray(flat[: keypoint_count * 3], dtype=float).reshape(keypoint_count, 3)


def valid_point(points: np.ndarray, idx: int) -> bool:
    if idx < 0 or idx >= len(points):
        return False
    x, y, score = points[idx]
    return np.isfinite(x) and np.isfinite(y) and score >= LIKELIHOOD_THRESHOLD


def draw_pose(
    frame: np.ndarray,
    points: np.ndarray,
    name_to_id: dict[str, int],
    edges: list[tuple[int, int]],
    color: tuple[int, int, int],
) -> None:
    for start, end in edges:
        if not (valid_point(points, start) and valid_point(points, end)):
            continue
        x1, y1, _ = points[start]
        x2, y2, _ = points[end]
        cv2.line(
            frame,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            2,
            cv2.LINE_AA,
        )

    for name, idx in name_to_id.items():
        if not valid_point(points, idx):
            continue
        x, y, _ = points[idx]
        radius = 7 if name in FOCUS_POINTS else 4
        thickness = 2 if name in FOCUS_POINTS else 1
        cv2.circle(frame, (int(round(x)), int(round(y))), radius + 2, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.circle(frame, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)


def draw_panel_label(
    frame: np.ndarray,
    label: str,
    frame_id: int,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(frame, (24, 24), (620, 116), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"{label}  frame {frame_id}",
        (40, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "bold points: neck, pelvis, feet",
        (40, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_comparison_video(camera: str, method: RunSpec) -> Path:
    video_path = SOURCE_DIR / f"{camera}.avi"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_scale = 0.50
    panel_size = (int(width * out_scale), int(height * out_scale))
    out_size = (panel_size[0] * 2, panel_size[1])

    out_path = FIG_DIR / f"{camera}_baseline_vs_{method.key}_full_overlay.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        out_size,
    )
    if not writer.isOpened():
        raise ValueError(f"Cannot create video writer: {out_path}")

    baseline_names, baseline_edges = skeleton_index(BASELINE.skeleton)
    method_names, method_edges = skeleton_index(method.skeleton)
    baseline_count = max(baseline_names.values()) + 1
    method_count = max(method_names.values()) + 1
    baseline_json = BASELINE.trial_dir / "pose" / f"{camera}_json"
    method_json = method.trial_dir / "pose" / f"{camera}_json"

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_START)
    written = 0
    for frame_id in range(FRAME_START, FRAME_END):
        ok, frame = cap.read()
        if not ok:
            break

        left = frame.copy()
        right = frame.copy()
        baseline_points = load_points(baseline_json, camera, frame_id, baseline_count)
        method_points = load_points(method_json, camera, frame_id, method_count)

        draw_pose(left, baseline_points, baseline_names, baseline_edges, BASELINE.color)
        draw_pose(right, method_points, method_names, method_edges, method.color)
        draw_panel_label(left, BASELINE.label, frame_id, BASELINE.color)
        draw_panel_label(right, method.label, frame_id, method.color)

        joined = np.concatenate([left, right], axis=1)
        joined = cv2.resize(joined, out_size, interpolation=cv2.INTER_AREA)
        writer.write(joined)
        written += 1

    writer.release()
    cap.release()
    if written != FRAME_END - FRAME_START:
        raise RuntimeError(f"{out_path} wrote {written} frames, expected {FRAME_END - FRAME_START}")
    return out_path


def write_readme(outputs: list[Path]) -> None:
    lines = [
        "# PnVision action02 full 2D comparison videos",
        "",
        f"Source trial: `data/PnVision/20260507/{TRIAL}`",
        "",
        "All videos cover frames 0-367 directly from 2D JSON overlays; no TRC output is used in this step.",
        "",
        "Comparisons:",
        "- left panel: baseline RTMPose `Body_with_feet`, no 2D smoothing",
        "- right panel: either RTMPose with 2D smoothing, or ViTPose-S COCO_25",
        "",
        "Files:",
    ]
    lines.extend(f"- `{path.name}`" for path in sorted(outputs))
    lines.append("")
    (DELIVERABLE_DIR / "README.md").write_text("\n".join(lines))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if DELIVERABLE_DIR.exists():
        shutil.rmtree(DELIVERABLE_DIR)
    DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for method in [SMOOTH, VITPOSE]:
        for camera in CAMERAS:
            outputs.append(make_comparison_video(camera, method))

    for output in outputs:
        shutil.copy2(output, DELIVERABLE_DIR / output.name)
    shutil.copy2(Path(__file__), DELIVERABLE_DIR / Path(__file__).name)
    write_readme(outputs)


if __name__ == "__main__":
    main()
