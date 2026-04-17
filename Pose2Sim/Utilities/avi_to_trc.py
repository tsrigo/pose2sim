#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## AVI TO TRC                                                            ##
###########################################################################

Run RTMLib pose estimation on synchronized multi-camera AVI videos, export
OpenPose-style JSON detections, then triangulate them into a standard 3D TRC.

The batching is applied to the pose model crops. Detection stays frame-wise,
which matches the shipped YOLOX ONNX detector shape.
'''


## INIT
import argparse
import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
import numpy as np
from anytree import RenderTree
import toml
from tqdm import tqdm

from Pose2Sim.Pose2Sim import recursive_update, setup_logging
from Pose2Sim.common import natural_sort_key
from Pose2Sim.poseEstimation import (
    save_to_openpose,
    setup_backend_device,
    setup_model_class_mode,
)
from Pose2Sim.triangulation import triangulate_all


## AUTHORSHIP INFORMATION
__author__ = "OpenAI"
__copyright__ = "Copyright 2026, Pose2Sim"
__credits__ = ["OpenAI", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')
__status__ = "Development"


## TYPES
@dataclass
class PendingPose:
    frame_idx: int
    image: np.ndarray
    center: np.ndarray
    scale: np.ndarray


@dataclass
class VideoStats:
    total_frames: int = 0
    dropped_frames: int = 0
    detection_misses: int = 0
    pose_batches: int = 0


## HELPERS
def _resolve_config_path(config_path):
    if config_path is None:
        return None

    config_path = Path(config_path).expanduser().resolve()
    if config_path.is_dir():
        config_path = config_path / 'Config.toml'

    if not config_path.is_file():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    return config_path


def load_trial_config(trial_dir, config_path=None):
    '''
    Load a trial config with optional session override merging.
    '''

    trial_dir = Path(trial_dir).expanduser().resolve()
    if not trial_dir.is_dir():
        raise NotADirectoryError(f'Trial directory not found: {trial_dir}')

    config_path = _resolve_config_path(config_path)
    trial_config_path = trial_dir / 'Config.toml'
    session_config_path = trial_dir.parent / 'Config.toml'

    if config_path is not None:
        config_dict = toml.load(config_path)
    elif trial_config_path.is_file() and session_config_path.is_file():
        config_dict = toml.load(session_config_path)
        config_dict = recursive_update(config_dict, toml.load(trial_config_path))
    elif trial_config_path.is_file():
        config_dict = toml.load(trial_config_path)
    elif session_config_path.is_file():
        config_dict = toml.load(session_config_path)
    else:
        raise FileNotFoundError(
            f'No Config.toml found in {trial_dir} nor its parent directory.'
        )

    config_dict.setdefault('project', {})
    config_dict['project']['project_dir'] = str(trial_dir)
    return config_dict


def resolve_session_dir(project_dir):
    project_dir = Path(project_dir).resolve()
    if (project_dir.parent / 'Config.toml').is_file():
        return project_dir.parent
    return project_dir


def find_avi_files(video_dir):
    avi_files = sorted(
        [Path(path) for path in glob.glob(os.path.join(video_dir, '*.avi'))],
        key=lambda path: natural_sort_key(str(path)),
    )
    if not avi_files:
        raise FileNotFoundError(f'No AVI files found in {video_dir}.')
    return avi_files


def ensure_calibration_exists(session_dir):
    calib_dirs = [
        path for path in session_dir.iterdir()
        if path.is_dir() and 'calib' in path.name.lower()
    ]
    if not calib_dirs:
        raise FileNotFoundError(
            f'No calibration directory found in {session_dir}.'
        )

    calib_files = []
    for calib_dir in calib_dirs:
        calib_files.extend(calib_dir.glob('*.toml'))

    if not calib_files:
        raise FileNotFoundError(
            f'No calibration TOML file found under {session_dir}.'
        )


def probe_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'Could not open video: {video_path}')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frame_count <= 0:
        raise ValueError(f'Invalid frame count for video: {video_path}')

    return {
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
    }


def resolve_effective_frame_range(config_dict, video_paths):
    '''
    Clamp the requested frame range to the common valid range across cameras.
    '''

    metadata = [probe_video(video_path) for video_path in video_paths]
    frame_counts = [meta['frame_count'] for meta in metadata]
    common_frame_count = min(frame_counts)
    requested_range = config_dict.get('project', {}).get('frame_range', 'auto')

    if len(set(frame_counts)) > 1:
        logging.warning(
            'Video frame counts differ across cameras (%s). '
            'Clipping to the common valid range of %d frames.',
            frame_counts,
            common_frame_count,
        )

    if requested_range in ('all', 'auto', [], None):
        effective_range = [0, common_frame_count]
    else:
        if not isinstance(requested_range, (list, tuple)) or len(requested_range) != 2:
            raise ValueError(
                f'Invalid frame_range: {requested_range}. Expected [start, end].'
            )
        start = max(0, int(requested_range[0]))
        end = min(int(requested_range[1]), common_frame_count)
        if end <= start:
            raise ValueError(
                f'Invalid effective frame range after clipping: [{start}, {end}].'
            )
        if end != int(requested_range[1]):
            logging.warning(
                'Requested frame_range %s exceeds the common valid range. '
                'Using [%d, %d] instead.',
                requested_range,
                start,
                end,
            )
        effective_range = [start, end]

    config_dict.setdefault('project', {})
    config_dict['project']['frame_range'] = effective_range
    return effective_range, metadata


def ensure_clean_output_dirs(project_dir, video_paths, overwrite_pose):
    '''
    Validate that triangulation will consume the intended JSON folders only.
    '''

    project_dir = Path(project_dir)
    pose_dir = project_dir / 'pose'
    pose_sync_dir = project_dir / 'pose-sync'
    pose_associated_dir = project_dir / 'pose-associated'
    expected_json_dirs = {f'{video_path.stem}_json' for video_path in video_paths}
    existing_pose_json_dirs = set()

    if pose_dir.is_dir():
        existing_pose_json_dirs = {
            path.name for path in pose_dir.iterdir()
            if path.is_dir() and path.name.endswith('_json')
        }

    downstream_dirs_with_json = []
    for downstream_dir in (pose_sync_dir, pose_associated_dir):
        if downstream_dir.is_dir():
            json_dirs = [
                path for path in downstream_dir.iterdir()
                if path.is_dir() and path.name.endswith('_json')
            ]
            if json_dirs:
                downstream_dirs_with_json.append(downstream_dir)

    if overwrite_pose:
        for stale_dir in downstream_dirs_with_json:
            shutil.rmtree(stale_dir)
        if pose_dir.is_dir():
            for json_dir_name in existing_pose_json_dirs:
                shutil.rmtree(pose_dir / json_dir_name)
        return False

    if downstream_dirs_with_json:
        raise FileExistsError(
            'Found existing pose-sync/pose-associated JSON outputs. '
            'Run with --overwrite-pose to regenerate pose data for this CLI.'
        )

    if existing_pose_json_dirs and existing_pose_json_dirs != expected_json_dirs:
        raise FileExistsError(
            'Existing pose JSON folders do not match the AVI set for this trial. '
            'Run with --overwrite-pose to clean and regenerate them.'
        )

    return bool(existing_pose_json_dirs)


def outputs_cover_frame_range(project_dir, video_paths, effective_range):
    pose_dir = Path(project_dir) / 'pose'
    required_count = effective_range[1] - effective_range[0]
    if required_count <= 0:
        return False

    for video_path in video_paths:
        json_dir = pose_dir / f'{video_path.stem}_json'
        if not json_dir.is_dir():
            return False
        json_count = len(list(json_dir.glob('*.json')))
        if json_count < required_count:
            return False
    return True


def instantiate_topdown_solution(config_dict, backend, device):
    pose_model_name = config_dict.get('pose', {}).get('pose_model', 'Body_with_feet')
    mode = config_dict.get('pose', {}).get('mode', 'balanced')
    pose_model_tree, model_class, parsed_mode = setup_model_class_mode(
        pose_model_name,
        mode,
        config_dict,
    )
    solution = model_class(mode=parsed_mode, backend=backend, device=device)

    if getattr(solution, 'one_stage', False) or not hasattr(solution, 'det_model'):
        raise NotImplementedError(
            'avi_to_trc only supports top-down RTMLib models with an explicit '
            'detector and pose model.'
        )

    if not hasattr(solution, 'pose_model') or not hasattr(solution.pose_model, 'preprocess'):
        raise NotImplementedError(
            'The selected RTMLib pose model does not expose the preprocessing '
            'hooks required for batched inference.'
        )

    keypoint_ids = [node.id for _, _, node in RenderTree(pose_model_tree) if node.id is not None]
    if not keypoint_ids:
        raise ValueError('Could not determine any keypoint IDs from the selected pose model.')

    return pose_model_tree, solution.det_model, solution.pose_model, max(keypoint_ids) + 1


def normalize_bboxes(raw_boxes):
    if raw_boxes is None:
        return np.empty((0, 4), dtype=np.float32)
    if isinstance(raw_boxes, tuple):
        raw_boxes = raw_boxes[0]

    boxes = np.asarray(raw_boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    boxes = boxes[:, :4]
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid]


def clamp_bbox(bbox, image_shape):
    height, width = image_shape[:2]
    bbox = np.asarray(bbox, dtype=np.float32).copy()
    bbox[0] = np.clip(bbox[0], 0, width - 1)
    bbox[1] = np.clip(bbox[1], 0, height - 1)
    bbox[2] = np.clip(bbox[2], 0, width - 1)
    bbox[3] = np.clip(bbox[3], 0, height - 1)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def bbox_center(bbox):
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32)


def bbox_area(bbox):
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def select_primary_bbox(boxes, previous_bbox, image_shape):
    boxes = [clamp_bbox(box, image_shape) for box in normalize_bboxes(boxes)]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None

    if previous_bbox is None:
        return max(boxes, key=bbox_area)

    previous_center = bbox_center(previous_bbox)
    image_diagonal = float(np.hypot(image_shape[0], image_shape[1]))
    distance_threshold = max(image_diagonal * 0.25, np.sqrt(max(bbox_area(previous_bbox), 1.0)))

    distances = [float(np.linalg.norm(bbox_center(box) - previous_center)) for box in boxes]
    nearest_idx = int(np.argmin(distances))
    if distances[nearest_idx] <= distance_threshold:
        return boxes[nearest_idx]

    return max(boxes, key=bbox_area)


def run_pose_batch(pose_model, images):
    if not images:
        return None

    batch = np.stack([
        np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
        for image in images
    ], axis=0)

    if pose_model.backend == 'onnxruntime':
        session = pose_model.session
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        return session.run(output_names, {input_name: batch})

    if pose_model.backend == 'openvino':
        try:
            results = pose_model.compiled_model(batch)
            return [results[pose_model.output_layer0], results[pose_model.output_layer1]]
        except Exception:
            logging.warning(
                'OpenVINO batch inference failed; falling back to per-sample pose inference.'
            )

    if pose_model.backend == 'opencv' and batch.shape[0] != 1:
        logging.warning(
            'OpenCV backend does not support batched pose inference here; '
            'falling back to per-sample inference.'
        )

    outputs0, outputs1 = [], []
    for image in images:
        output0, output1 = pose_model.inference(image)
        outputs0.append(output0)
        outputs1.append(output1)

    return [np.concatenate(outputs0, axis=0), np.concatenate(outputs1, axis=0)]


def empty_detection_arrays(kpt_count):
    keypoints = np.empty((0, kpt_count, 2), dtype=np.float32)
    scores = np.empty((0, kpt_count), dtype=np.float32)
    return keypoints, scores


def flush_pose_queue(
    queue,
    pose_model,
    json_dir,
    video_stem,
    average_likelihood_threshold_pose,
    stats,
    kpt_count,
):
    if not queue:
        return

    images = [item.image for item in queue]
    outputs = run_pose_batch(pose_model, images)
    stats.pose_batches += 1

    for batch_idx, item in enumerate(queue):
        sample_outputs = [output[batch_idx:batch_idx + 1] for output in outputs]
        keypoints, scores = pose_model.postprocess(sample_outputs, item.center, item.scale)

        average_score = float(np.nanmean(scores)) if scores.size else np.nan
        if not np.isfinite(average_score) or average_score < average_likelihood_threshold_pose:
            keypoints, scores = empty_detection_arrays(kpt_count)
            stats.dropped_frames += 1

        save_to_openpose(
            str(json_dir / f'{video_stem}_{item.frame_idx:06d}.json'),
            keypoints,
            scores,
        )

    queue.clear()


def process_video(
    video_path,
    det_model,
    pose_model,
    json_dir,
    video_stem,
    effective_range,
    batch_size,
    det_frequency,
    average_likelihood_threshold_pose,
    kpt_count,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'Could not open video: {video_path}')

    cap.set(cv2.CAP_PROP_POS_FRAMES, effective_range[0])
    stats = VideoStats(total_frames=effective_range[1] - effective_range[0])
    queue = []
    previous_bbox = None

    for frame_idx in tqdm(
        range(effective_range[0], effective_range[1]),
        desc=f'Processing {video_path.name}',
    ):
        success, frame = cap.read()
        if not success:
            raise RuntimeError(f'Video ended unexpectedly at frame {frame_idx}: {video_path}')

        should_detect = (
            previous_bbox is None or
            (frame_idx - effective_range[0]) % det_frequency == 0
        )

        bbox = previous_bbox
        if should_detect:
            raw_boxes = det_model(frame)
            bbox = select_primary_bbox(raw_boxes, previous_bbox, frame.shape)
            if bbox is None:
                stats.detection_misses += 1
            previous_bbox = bbox

        if bbox is None:
            flush_pose_queue(
                queue,
                pose_model,
                json_dir,
                video_stem,
                average_likelihood_threshold_pose,
                stats,
                kpt_count,
            )
            keypoints, scores = empty_detection_arrays(kpt_count)
            save_to_openpose(
                str(json_dir / f'{video_path.stem}_{frame_idx:06d}.json'),
                keypoints,
                scores,
            )
            stats.dropped_frames += 1
            continue

        resized_img, center, scale = pose_model.preprocess(frame, bbox.tolist())
        queue.append(PendingPose(frame_idx=frame_idx, image=resized_img, center=center, scale=scale))

        if len(queue) >= batch_size:
            flush_pose_queue(
                queue,
                pose_model,
                json_dir,
                video_stem,
                average_likelihood_threshold_pose,
                stats,
                kpt_count,
            )

    flush_pose_queue(
        queue,
        pose_model,
        json_dir,
        video_stem,
        average_likelihood_threshold_pose,
        stats,
        kpt_count,
    )
    cap.release()

    logging.info(
        '%s: processed %d frames, dropped %d, detection misses %d, pose batches %d.',
        video_path.name,
        stats.total_frames,
        stats.dropped_frames,
        stats.detection_misses,
        stats.pose_batches,
    )
    return stats


def run_batched_pose_export(config_dict, batch_size, backend, device, overwrite_pose):
    project_dir = Path(config_dict.get('project', {}).get('project_dir', '.')).resolve()
    video_dir = project_dir / 'videos'
    pose_dir = project_dir / 'pose'
    pose_dir.mkdir(parents=True, exist_ok=True)

    avi_files = find_avi_files(video_dir)
    effective_range, _ = resolve_effective_frame_range(config_dict, avi_files)
    has_existing_pose_dirs = ensure_clean_output_dirs(project_dir, avi_files, overwrite_pose)

    if has_existing_pose_dirs:
        if outputs_cover_frame_range(project_dir, avi_files, effective_range):
            logging.info(
                'Skipping batched pose export because compatible JSON outputs already exist. '
                'Use --overwrite-pose to regenerate them.'
            )
            return avi_files, effective_range, False
        raise FileExistsError(
            'Existing pose JSON outputs do not fully cover the requested frame range. '
            'Run with --overwrite-pose to regenerate them.'
        )

    average_likelihood_threshold_pose = config_dict.get('pose', {}).get(
        'average_likelihood_threshold_pose',
        0.5,
    )
    det_frequency = int(config_dict.get('pose', {}).get('det_frequency', 4))
    if det_frequency < 1:
        raise ValueError('det_frequency must be an integer greater or equal to 1.')

    _, det_model, pose_model, kpt_count = instantiate_topdown_solution(
        config_dict,
        backend,
        device,
    )

    logging.info(
        'Running batched pose export with backend=%s, device=%s, batch_size=%d, det_frequency=%d.',
        backend,
        device,
        batch_size,
        det_frequency,
    )
    logging.info(
        'Using RTMLib pose model "%s".',
        config_dict.get('pose', {}).get('pose_model', 'Body_with_feet'),
    )

    stats_by_video = {}
    for video_path in avi_files:
        json_dir = pose_dir / f'{video_path.stem}_json'
        json_dir.mkdir(parents=True, exist_ok=True)
        stats_by_video[video_path.name] = process_video(
            video_path=video_path,
            det_model=det_model,
            pose_model=pose_model,
            json_dir=json_dir,
            video_stem=video_path.stem,
            effective_range=effective_range,
            batch_size=batch_size,
            det_frequency=det_frequency,
            average_likelihood_threshold_pose=average_likelihood_threshold_pose,
            kpt_count=kpt_count,
        )

    return avi_files, effective_range, True


def triangulate_trial(config_dict):
    project_dir = Path(config_dict.get('project', {}).get('project_dir', '.')).resolve()
    run_dir = resolve_session_dir(project_dir)
    previous_cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        triangulate_all(config_dict)
    finally:
        os.chdir(previous_cwd)


def avi_to_trc(
    trial_dir,
    config_path=None,
    batch_size=16,
    det_frequency=None,
    overwrite_pose=False,
    backend=None,
    device=None,
):
    '''
    Main callable entrypoint for the AVI -> pose JSON -> 3D TRC workflow.
    '''

    config_dict = load_trial_config(trial_dir, config_path=config_path)
    project_dir = Path(config_dict.get('project', {}).get('project_dir', '.')).resolve()
    session_dir = resolve_session_dir(project_dir)
    ensure_calibration_exists(session_dir)

    use_custom_logging = config_dict.get('logging', {}).get('use_custom_logging', False)
    if not use_custom_logging:
        setup_logging(str(session_dir))

    if config_dict.get('project', {}).get('multi_person', False):
        logging.warning(
            'multi_person=true is not supported by avi_to_trc v1. Forcing single-person mode.'
        )
        config_dict['project']['multi_person'] = False

    output_format = config_dict.get('pose', {}).get('output_format', 'openpose')
    if output_format != 'openpose':
        logging.warning(
            'avi_to_trc requires OpenPose-style JSON for downstream triangulation. '
            'Forcing output_format="openpose".'
        )
        config_dict.setdefault('pose', {})
        config_dict['pose']['output_format'] = 'openpose'

    resolved_backend = backend if backend is not None else config_dict.get('pose', {}).get('backend', 'auto')
    resolved_device = device if device is not None else config_dict.get('pose', {}).get('device', 'auto')
    resolved_backend, resolved_device = setup_backend_device(
        backend=resolved_backend,
        device=resolved_device,
    )
    config_dict.setdefault('pose', {})
    config_dict['pose']['backend'] = resolved_backend
    config_dict['pose']['device'] = resolved_device

    if det_frequency is not None:
        config_dict['pose']['det_frequency'] = int(det_frequency)

    if batch_size < 1:
        raise ValueError('batch_size must be greater or equal to 1.')

    avi_files, effective_range, pose_ran = run_batched_pose_export(
        config_dict=config_dict,
        batch_size=int(batch_size),
        backend=resolved_backend,
        device=resolved_device,
        overwrite_pose=overwrite_pose or config_dict.get('pose', {}).get('overwrite_pose', False),
    )

    successful_streams = sum(
        int((project_dir / 'pose' / f'{video_path.stem}_json').is_dir())
        for video_path in avi_files
    )
    if successful_streams < 2:
        raise RuntimeError(
            'At least two camera JSON streams are required before triangulation.'
        )

    logging.info(
        'Triangulating %d camera streams for frames [%d, %d).',
        successful_streams,
        effective_range[0],
        effective_range[1],
    )
    triangulate_trial(config_dict)
    return {
        'project_dir': str(project_dir),
        'pose_exported': pose_ran,
        'camera_count': successful_streams,
        'frame_range': effective_range,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run batched RTMLib inference on AVI videos and export a 3D TRC.'
    )
    parser.add_argument(
        '--trial-dir',
        required=True,
        help='Trial directory containing videos/*.avi and calibration/config.',
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Optional Config.toml path or directory containing Config.toml.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Pose crop batch size for RTMPose inference.',
    )
    parser.add_argument(
        '--det-frequency',
        type=int,
        default=None,
        help='Run the detector every N frames and reuse the previous box in-between.',
    )
    parser.add_argument(
        '--overwrite-pose',
        action='store_true',
        help='Remove existing pose JSON outputs before regenerating them.',
    )
    parser.add_argument(
        '--backend',
        default=None,
        help='Optional backend override (onnxruntime, openvino, opencv, auto).',
    )
    parser.add_argument(
        '--device',
        default=None,
        help='Optional device override (cpu, cuda, rocm, mps, auto).',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    avi_to_trc(
        trial_dir=args.trial_dir,
        config_path=args.config,
        batch_size=args.batch_size,
        det_frequency=args.det_frequency,
        overwrite_pose=args.overwrite_pose,
        backend=args.backend,
        device=args.device,
    )


if __name__ == '__main__':
    main()
