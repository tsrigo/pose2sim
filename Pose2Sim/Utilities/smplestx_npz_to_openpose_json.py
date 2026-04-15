#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import os.path as osp

import numpy as np


CUSTOM_SPINE25_ORDER = [
    "Pelvis",
    "RHip",
    "RKnee",
    "RAnkle",
    "RBigToe",
    "RSmallToe",
    "RHeel",
    "LHip",
    "LKnee",
    "LAnkle",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "Spine_1",
    "Spine_2",
    "Spine_3",
    "Neck",
    "Head",
    "Nose",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SMPLest-X saved NPZ results to OpenPose-style JSON for Pose2Sim.")
    parser.add_argument("--input_npz", required=True, help="Path to smplx_results.npz")
    parser.add_argument("--output_dir", required=True, help="Directory where Pose2Sim-style JSON files will be written")
    parser.add_argument("--person_id", type=int, default=0, help="Person id to export")
    parser.add_argument("--default_conf", type=float, default=1.0, help="Confidence value assigned to exported keypoints")
    return parser.parse_args()


def as_name_to_idx(names):
    return {str(name): idx for idx, name in enumerate(names)}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.input_npz, allow_pickle=True)
    frame_results = data["frame_results"].tolist()
    orig_name_to_idx = as_name_to_idx(data["orig_body_joint_names_22"].tolist())
    reduced_name_to_idx = as_name_to_idx(data["reduced_joint_names_137"].tolist())

    for item in frame_results:
        if int(item["person_id"]) != args.person_id:
            continue

        orig_proj = np.asarray(item["orig_body_joint_proj_22"], dtype=np.float32)
        reduced_proj = np.asarray(item["smplx_joint_proj_137_img"], dtype=np.float32)

        keypoint_map = {
            "Pelvis": orig_proj[orig_name_to_idx["Pelvis"]],
            "RHip": orig_proj[orig_name_to_idx["R_Hip"]],
            "RKnee": orig_proj[orig_name_to_idx["R_Knee"]],
            "RAnkle": orig_proj[orig_name_to_idx["R_Ankle"]],
            "RBigToe": reduced_proj[reduced_name_to_idx["R_Big_toe"]],
            "RSmallToe": reduced_proj[reduced_name_to_idx["R_Small_toe"]],
            "RHeel": reduced_proj[reduced_name_to_idx["R_Heel"]],
            "LHip": orig_proj[orig_name_to_idx["L_Hip"]],
            "LKnee": orig_proj[orig_name_to_idx["L_Knee"]],
            "LAnkle": orig_proj[orig_name_to_idx["L_Ankle"]],
            "LBigToe": reduced_proj[reduced_name_to_idx["L_Big_toe"]],
            "LSmallToe": reduced_proj[reduced_name_to_idx["L_Small_toe"]],
            "LHeel": reduced_proj[reduced_name_to_idx["L_Heel"]],
            "Spine_1": orig_proj[orig_name_to_idx["Spine_1"]],
            "Spine_2": orig_proj[orig_name_to_idx["Spine_2"]],
            "Spine_3": orig_proj[orig_name_to_idx["Spine_3"]],
            "Neck": orig_proj[orig_name_to_idx["Neck"]],
            "Head": orig_proj[orig_name_to_idx["Head"]],
            "Nose": reduced_proj[reduced_name_to_idx["Nose"]],
            "RShoulder": orig_proj[orig_name_to_idx["R_Shoulder"]],
            "RElbow": orig_proj[orig_name_to_idx["R_Elbow"]],
            "RWrist": orig_proj[orig_name_to_idx["R_Wrist"]],
            "LShoulder": orig_proj[orig_name_to_idx["L_Shoulder"]],
            "LElbow": orig_proj[orig_name_to_idx["L_Elbow"]],
            "LWrist": orig_proj[orig_name_to_idx["L_Wrist"]],
        }

        pose_keypoints_2d = []
        for name in CUSTOM_SPINE25_ORDER:
            pt = keypoint_map[name]
            if np.any(~np.isfinite(pt)):
                pose_keypoints_2d.extend([0.0, 0.0, 0.0])
            else:
                pose_keypoints_2d.extend([float(pt[0]), float(pt[1]), float(args.default_conf)])

        openpose_like = {
            "version": 1.3,
            "people": [
                {
                    "person_id": [int(args.person_id)],
                    "pose_keypoints_2d": pose_keypoints_2d,
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": [],
                }
            ],
        }

        out_name = f"{osp.splitext(osp.basename(item['img_path']))[0]}.json"
        with open(osp.join(args.output_dir, out_name), "w") as f:
            json.dump(openpose_like, f)

    print(args.output_dir)


if __name__ == "__main__":
    main()
