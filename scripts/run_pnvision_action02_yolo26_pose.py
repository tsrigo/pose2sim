#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
TRIAL = "rec_20260507_134131_lhl_action02"
SOURCE_DIR = ROOT / "data" / "PnVision" / "20260507" / TRIAL
OUT_ROOT = ROOT / "data" / "pnvision_action02_yolo26_pose_20260522"
OUT_TRIAL = OUT_ROOT / TRIAL
CAMERAS = ["cam01", "cam02", "cam03", "cam04"]
FRAME_START = 0
FRAME_END = 368
KEYPOINT_COUNT = 17


@dataclass
class CameraStats:
    camera: str
    frames: int = 0
    detections: int = 0
    misses: int = 0
    mean_box_conf: float = float("nan")
    median_kpt_conf: float = float("nan")


def write_openpose_json(path: Path, keypoints: np.ndarray) -> None:
    flat = keypoints.reshape(-1).astype(float).tolist()
    people = []
    if np.isfinite(keypoints[:, :2]).any():
        people.append(
            {
                "person_id": [-1],
                "pose_keypoints_2d": flat,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": [],
            }
        )
    path.write_text(json.dumps({"version": 1.3, "people": people}))


def empty_keypoints() -> np.ndarray:
    return np.full((KEYPOINT_COUNT, 3), np.nan, dtype=float)


def select_person(result) -> tuple[np.ndarray, float, float] | None:
    if result.keypoints is None or result.boxes is None or len(result.keypoints.data) == 0:
        return None

    boxes = result.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=float)
    if len(xyxy) == 0:
        return None

    areas = np.maximum(xyxy[:, 2] - xyxy[:, 0], 0) * np.maximum(xyxy[:, 3] - xyxy[:, 1], 0)
    score = areas * np.maximum(conf, 1e-6)
    idx = int(np.argmax(score))

    keypoints = result.keypoints.data[idx].detach().cpu().numpy().astype(float)
    if keypoints.shape[0] != KEYPOINT_COUNT:
        raise ValueError(f"Expected {KEYPOINT_COUNT} keypoints, got {keypoints.shape[0]}")
    return keypoints, float(conf[idx]), float(np.nanmedian(keypoints[:, 2]))


def process_camera(model: YOLO, camera: str, device: str, imgsz: int, conf: float) -> CameraStats:
    video_path = SOURCE_DIR / f"{camera}.avi"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    json_dir = OUT_TRIAL / "pose" / f"{camera}_json"
    json_dir.mkdir(parents=True, exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_START)

    stats = CameraStats(camera=camera)
    box_confs: list[float] = []
    kpt_confs: list[float] = []

    for frame_id in tqdm(range(FRAME_START, FRAME_END), desc=f"YOLO26 {camera}"):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"{video_path} ended at frame {frame_id}")

        result = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        selected = select_person(result)
        if selected is None:
            stats.misses += 1
            keypoints = empty_keypoints()
        else:
            keypoints, box_conf, kpt_conf = selected
            stats.detections += 1
            box_confs.append(box_conf)
            kpt_confs.append(kpt_conf)

        write_openpose_json(json_dir / f"{camera}_{frame_id:06d}.json", keypoints)
        stats.frames += 1

    cap.release()
    stats.mean_box_conf = float(np.mean(box_confs)) if box_confs else float("nan")
    stats.median_kpt_conf = float(np.median(kpt_confs)) if kpt_confs else float("nan")
    return stats


def write_summary(stats: list[CameraStats], model_name: str, device: str, imgsz: int, conf: float) -> None:
    lines = [
        "camera,frames,detections,misses,mean_box_conf,median_kpt_conf",
        *[
            f"{s.camera},{s.frames},{s.detections},{s.misses},{s.mean_box_conf:.6f},{s.median_kpt_conf:.6f}"
            for s in stats
        ],
    ]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "quality_summary.csv").write_text("\n".join(lines) + "\n")
    (OUT_ROOT / "run_config.json").write_text(
        json.dumps(
            {
                "source_trial": str(SOURCE_DIR.relative_to(ROOT)),
                "output_trial": str(OUT_TRIAL.relative_to(ROOT)),
                "model": model_name,
                "schema": "COCO_17",
                "frames": [FRAME_START, FRAME_END - 1],
                "device": device,
                "imgsz": imgsz,
                "conf": conf,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if OUT_TRIAL.exists() and args.overwrite:
        shutil.rmtree(OUT_TRIAL)
    OUT_TRIAL.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    if getattr(model.model, "kpt_shape", None) != [KEYPOINT_COUNT, 3]:
        raise ValueError(f"{args.model} kpt_shape={getattr(model.model, 'kpt_shape', None)}, expected [17, 3]")

    stats = [process_camera(model, camera, args.device, args.imgsz, args.conf) for camera in CAMERAS]
    write_summary(stats, args.model, args.device, args.imgsz, args.conf)


if __name__ == "__main__":
    main()
