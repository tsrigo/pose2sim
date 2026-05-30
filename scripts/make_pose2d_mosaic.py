#!/usr/bin/env python3
"""Create a 2x2 pose-2D mosaic video from Pose2Sim OpenPose JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from Pose2Sim.common import draw_keypts, draw_skel
from Pose2Sim.skeletons import HALPE_26


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    return parser.parse_args()


def load_pose(json_path: Path, keypoint_count: int, score_threshold: float):
    x = np.full(keypoint_count, np.nan, dtype=np.float32)
    y = np.full(keypoint_count, np.nan, dtype=np.float32)
    scores = np.zeros(keypoint_count, dtype=np.float32)

    if not json_path.is_file():
        return x[None, :], y[None, :], scores[None, :]

    people = json.loads(json_path.read_text()).get("people") or []
    if not people:
        return x[None, :], y[None, :], scores[None, :]

    values = people[0].get("pose_keypoints_2d") or []
    for keypoint_id in range(min(keypoint_count, len(values) // 3)):
        px, py, score = values[keypoint_id * 3:keypoint_id * 3 + 3]
        scores[keypoint_id] = score
        if score > score_threshold and not (px == 0 and py == 0):
            x[keypoint_id] = px
            y[keypoint_id] = py

    return x[None, :], y[None, :], scores[None, :]


def annotate(frame, camera_name: str, frame_idx: int):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"{camera_name} F{frame_idx}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_mosaic(trial_dir: Path, output_path: Path, scale: float, score_threshold: float):
    videos = sorted((trial_dir / "videos").glob("*.avi"))
    if len(videos) != 4:
        raise ValueError(f"Expected 4 AVI files under {trial_dir / 'videos'}, found {len(videos)}")

    captures = [cv2.VideoCapture(str(video)) for video in videos]
    try:
        if any(not cap.isOpened() for cap in captures):
            raise RuntimeError("Could not open all input videos")

        frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
        total_frames = min(frame_counts)
        fps = captures[0].get(cv2.CAP_PROP_FPS) or 30.0
        source_width = int(captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        tile_size = (int(round(source_width * scale)), int(round(source_height * scale)))
        mosaic_size = (tile_size[0] * 2, tile_size[1] * 2)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            mosaic_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video writer: {output_path}")

        keypoint_count = 26
        pose_dir = trial_dir / "pose"
        camera_names = [video.stem for video in videos]

        for frame_idx in range(total_frames):
            tiles = []
            for camera_name, cap in zip(camera_names, captures):
                ok, frame = cap.read()
                if not ok:
                    frame = np.zeros((source_height, source_width, 3), dtype=np.uint8)

                json_path = pose_dir / f"{camera_name}_json" / f"{camera_name}_{frame_idx:06d}.json"
                x, y, scores = load_pose(json_path, keypoint_count, score_threshold)
                frame = draw_keypts(frame, x, y, scores, cmap_str="RdYlGn")
                frame = draw_skel(frame, x, y, HALPE_26)
                annotate(frame, camera_name, frame_idx)
                tiles.append(cv2.resize(frame, tile_size, interpolation=cv2.INTER_AREA))

            top = np.hstack(tiles[:2])
            bottom = np.hstack(tiles[2:])
            writer.write(np.vstack([top, bottom]))

        writer.release()
    finally:
        for cap in captures:
            cap.release()


def main():
    args = parse_args()
    make_mosaic(args.trial_dir, args.output, args.scale, args.score_threshold)
    print(args.output)


if __name__ == "__main__":
    main()
