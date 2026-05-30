#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from anytree import RenderTree

from Pose2Sim.skeletons import COCO_17, HALPE_26


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
YOLO_DIR = ROOT / "data" / "pnvision_action02_yolo26_pose_20260522" / TRIAL
YOLO_ROOT = ROOT / "data" / "pnvision_action02_yolo26_pose_20260522"
FIG_DIR = ROOT / "figures" / "pnvision_action02_yolo26_pose_20260522"
DELIVERABLE_DIR = ROOT / "deliverables" / "pnvision_action02_yolo26_pose_2d_comparison_20260522"

CAMERAS = ["cam01", "cam02", "cam03", "cam04"]
FRAME_START = 0
FRAME_END = 368
LIKELIHOOD_THRESHOLD = 0.3
BASELINE_FOCUS_POINTS = {
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
YOLO_FOCUS_POINTS = {
    "LHip",
    "RHip",
    "LAnkle",
    "RAnkle",
}


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    trial_dir: Path
    skeleton: object
    color: tuple[int, int, int]
    focus_points: set[str]
    note: str


BASELINE = RunSpec(
    key="baseline",
    label="baseline RTMPose / HALPE26",
    trial_dir=BASELINE_DIR,
    skeleton=HALPE_26,
    color=(0, 80, 255),
    focus_points=BASELINE_FOCUS_POINTS,
    note="26 keypoints, including center neck/hip and feet",
)
YOLO = RunSpec(
    key="yolo26n_pose_coco17",
    label="YOLO26n-pose / COCO17",
    trial_dir=YOLO_DIR,
    skeleton=COCO_17,
    color=(255, 170, 0),
    focus_points=YOLO_FOCUS_POINTS,
    note="17 keypoints; no toes, heels, or center neck/hip",
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


def add_coco17_torso_edges(name_to_id: dict[str, int], edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    extra_edges = [
        ("LShoulder", "RShoulder"),
        ("LShoulder", "LHip"),
        ("RShoulder", "RHip"),
        ("LHip", "RHip"),
    ]
    out = list(edges)
    for start, end in extra_edges:
        if start in name_to_id and end in name_to_id:
            out.append((name_to_id[start], name_to_id[end]))
    return out


def draw_pose(
    frame: np.ndarray,
    points: np.ndarray,
    name_to_id: dict[str, int],
    edges: list[tuple[int, int]],
    spec: RunSpec,
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
            spec.color,
            2,
            cv2.LINE_AA,
        )

    for name, idx in name_to_id.items():
        if not valid_point(points, idx):
            continue
        x, y, _ = points[idx]
        radius = 7 if name in spec.focus_points else 4
        thickness = 2 if name in spec.focus_points else 1
        cv2.circle(frame, (int(round(x)), int(round(y))), radius + 2, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.circle(frame, (int(round(x)), int(round(y))), radius, spec.color, -1, cv2.LINE_AA)


def draw_panel_label(
    frame: np.ndarray,
    spec: RunSpec,
    frame_id: int,
) -> None:
    cv2.rectangle(frame, (24, 24), (780, 116), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"{spec.label}  frame {frame_id}",
        (40, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        spec.color,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        spec.note,
        (40, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_comparison_video(camera: str) -> Path:
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

    out_path = FIG_DIR / f"{camera}_baseline_vs_yolo26n_pose_coco17_full_overlay.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        out_size,
    )
    if not writer.isOpened():
        raise ValueError(f"Cannot create video writer: {out_path}")

    baseline_names, baseline_edges = skeleton_index(BASELINE.skeleton)
    yolo_names, yolo_edges = skeleton_index(YOLO.skeleton)
    yolo_edges = add_coco17_torso_edges(yolo_names, yolo_edges)
    baseline_count = max(baseline_names.values()) + 1
    yolo_count = max(yolo_names.values()) + 1
    baseline_json = BASELINE.trial_dir / "pose" / f"{camera}_json"
    yolo_json = YOLO.trial_dir / "pose" / f"{camera}_json"

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_START)
    written = 0
    for frame_id in range(FRAME_START, FRAME_END):
        ok, frame = cap.read()
        if not ok:
            break

        left = frame.copy()
        right = frame.copy()
        baseline_points = load_points(baseline_json, camera, frame_id, baseline_count)
        yolo_points = load_points(yolo_json, camera, frame_id, yolo_count)

        draw_pose(left, baseline_points, baseline_names, baseline_edges, BASELINE)
        draw_pose(right, yolo_points, yolo_names, yolo_edges, YOLO)
        draw_panel_label(left, BASELINE, frame_id)
        draw_panel_label(right, YOLO, frame_id)

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
        "# PnVision action02 YOLO26-pose 2D comparison videos",
        "",
        f"Source trial: `data/PnVision/20260507/{TRIAL}`",
        "",
        "Scope:",
        "- frames 0-367, all four cameras",
        "- left panel: baseline RTMPose `Body_with_feet` / HALPE_26",
        "- right panel: default `yolo26n-pose.pt` exported as OpenPose-style JSON with `COCO_17` keypoints",
        "- generated YOLO JSON is included under `yolo26_openpose_json/pose/`",
        "- if present, the COCO_17 triangulation smoke-test TRC is included under `yolo26_triangulation_smoke/pose-3d/`",
        "",
        "Important limitation:",
        "- Default YOLO26-pose outputs 17 COCO keypoints, not the 26-point foot skeleton used by the baseline.",
        "- Therefore it does not directly provide toes, heels, Head, or explicit center Neck/Hip keypoints.",
        "- The generated JSON can enter downstream code paths that accept `COCO_17`, but it is not marker-identical to HALPE_26.",
        "- With the current `COCO_17` tree, triangulation writes 13 TRC markers because eye/ear keypoints are not included in that tree.",
        "",
        "Quality summary from YOLO inference:",
    ]
    summary_path = YOLO_ROOT / "quality_summary.csv"
    if summary_path.is_file():
        lines.extend(f"    {line}" for line in summary_path.read_text().strip().splitlines())
    lines.extend(
        [
            "",
        "Files:",
        "- `scripts/run_pnvision_action02_yolo26_pose.py`: YOLO26-pose inference to OpenPose-style JSON",
        "- `scripts/make_pnvision_action02_yolo26_2d_comparison_videos.py`: baseline-vs-YOLO video overlay generation",
        "- `yolo26_openpose_json/pose/`: OpenPose-style JSON exported from YOLO26-pose",
        "- `yolo26_triangulation_smoke/`: Config, calibration, and TRC from the COCO_17 downstream smoke test, if generated",
    ]
    )
    lines.extend(f"- `{path.name}`" for path in sorted(outputs))
    lines.append("")
    (DELIVERABLE_DIR / "README.md").write_text("\n".join(lines))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if DELIVERABLE_DIR.exists():
        shutil.rmtree(DELIVERABLE_DIR)
    DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [make_comparison_video(camera) for camera in CAMERAS]

    for output in outputs:
        shutil.copy2(output, DELIVERABLE_DIR / output.name)
    for metadata_name in ["quality_summary.csv", "run_config.json"]:
        src = YOLO_ROOT / metadata_name
        if src.is_file():
            shutil.copy2(src, DELIVERABLE_DIR / metadata_name)
    shutil.copytree(YOLO_DIR / "pose", DELIVERABLE_DIR / "yolo26_openpose_json" / "pose")
    if (YOLO_DIR / "Config.toml").is_file():
        smoke_dir = DELIVERABLE_DIR / "yolo26_triangulation_smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(YOLO_DIR / "Config.toml", smoke_dir / "Config.toml")
        if (YOLO_DIR / "calibration").is_dir():
            shutil.copytree(YOLO_DIR / "calibration", smoke_dir / "calibration")
        if (YOLO_DIR / "pose-3d").is_dir():
            shutil.copytree(YOLO_DIR / "pose-3d", smoke_dir / "pose-3d")
    shutil.copy2(ROOT / "scripts" / "run_pnvision_action02_yolo26_pose.py", DELIVERABLE_DIR / "run_pnvision_action02_yolo26_pose.py")
    shutil.copy2(Path(__file__), DELIVERABLE_DIR / Path(__file__).name)
    write_readme(outputs)


if __name__ == "__main__":
    main()
