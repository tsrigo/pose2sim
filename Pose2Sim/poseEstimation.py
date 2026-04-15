#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## POSE ESTIMATION                                                       ##
###########################################################################

    Estimate pose from a video file or a folder of images and 
    write the results to JSON files, videos, and/or images.
    Results can optionally be displayed in real time.

    Supported models: HALPE_26 (default, body and feet), COCO_133 (body, feet, hands), COCO_17 (body)
    Supported modes: lightweight, balanced, performance (edit paths at rtmlib/tools/solutions if you 
    need nother detection or pose models)

    Optionally gives consistent person ID across frames (slower but good for 2D analysis)
    Optionally runs detection every n frames and inbetween tracks points (faster but less accurate).

    If a valid cuda installation is detected, uses the GPU with the ONNXRuntime backend. Otherwise, 
    uses the CPU with the OpenVINO backend.

    INPUTS:
    - videos or image folders from the video directory
    - a Config.toml file

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - Optionally, videos and/or image files with the detected keypoints 
'''


## INIT
import os
import glob
import json
import re
import logging
import ast
from functools import partial
from tqdm import tqdm
from anytree.importer import DictImporter
from anytree import RenderTree
import numpy as np
import cv2

from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body, Hand, Custom, draw_skeleton
from rtmlib.tools.object_detection.post_processings import nms
from Pose2Sim.common import natural_sort_key, sort_people_sports2d, sort_people_deepsort,\
                        colors, thickness, draw_bounding_box, draw_keypts, draw_skel, bbox_xyxy_compute, \
                        get_screen_size, calculate_display_size
from Pose2Sim.skeletons import *

np.set_printoptions(legacy='1.21') # otherwise prints np.float64(3.0) rather than 3.0
import warnings # Silence numpy and CoreML warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", message=".*Input.*has a dynamic shape.*but the runtime shape.*has zero elements.*")

# Not safe, but to be used until OpenMMLab/RTMlib's SSL certificates are updated
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim, David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["HunMin Kim", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def get_keypoint_id(model, keypoint_name, default=0):
    '''
    Resolve a keypoint name to its skeleton id.
    '''

    try:
        return [node.id for _, _, node in RenderTree(model) if node.name == keypoint_name][0]
    except Exception:
        logging.warning(f'{keypoint_name} not found in the current pose model. Falling back to keypoint id {default}.')
        return default


def build_person_filter_config(config_dict, pose_model):
    '''
    Normalize person filtering configuration.
    '''

    filter_cfg = config_dict.get('pose', {}).get('person_filter', {}) or {}
    anchor_keypoint = filter_cfg.get('anchor_keypoint', 'Neck')
    anchor_keypoint_id = None if str(anchor_keypoint).lower() == 'bbox_center' else get_keypoint_id(pose_model, anchor_keypoint, default=0)
    return {
        'enabled': bool(filter_cfg.get('enabled', False)),
        'max_people': max(1, int(filter_cfg.get('max_people', 1))),
        'anchor_keypoint': anchor_keypoint,
        'anchor_keypoint_id': anchor_keypoint_id,
        'min_mean_score': float(filter_cfg.get('min_mean_score', 0.2)),
        'min_bbox_area_px': float(filter_cfg.get('min_bbox_area_px', 0)),
        'max_bbox_area_px': float(filter_cfg.get('max_bbox_area_px', float('inf'))),
        'roi_by_camera': filter_cfg.get('roi_by_camera', {}) or {},
    }


def _source_lookup_keys(source_name):
    stem = os.path.splitext(os.path.basename(source_name))[0]
    prefix = stem.split('_')[0]
    return [stem, prefix]


def _roi_for_source(source_name, roi_by_camera):
    for key in _source_lookup_keys(source_name):
        if key in roi_by_camera:
            return roi_by_camera[key]
    return None


def _anchor_points_from_keypoints(keypoints, bboxes, anchor_keypoint_id):
    if len(keypoints) == 0:
        return np.empty((0, 2))
    centers = np.column_stack(((bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2))
    if anchor_keypoint_id is None:
        return centers
    anchors = keypoints[:, anchor_keypoint_id, :] if anchor_keypoint_id < keypoints.shape[1] else np.full((len(keypoints), 2), np.nan)
    valid_mask = np.all(np.isfinite(anchors), axis=1)
    anchors = anchors.copy()
    anchors[~valid_mask] = centers[~valid_mask]
    return anchors


def _points_in_roi(points, roi):
    if roi is None or len(roi) != 4 or len(points) == 0:
        return np.ones(len(points), dtype=bool)
    x_min, y_min, x_max, y_max = roi
    return (
        (points[:, 0] >= x_min) &
        (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) &
        (points[:, 1] <= y_max)
    )


def filter_person_detections(keypoints, scores, frame_shape, source_name, person_filter_cfg, prev_anchor=None):
    '''
    Apply ROI and single-person filtering after tracking.
    '''

    if not person_filter_cfg.get('enabled', False):
        return keypoints, scores, prev_anchor, None

    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)
    if keypoints.size == 0 or len(keypoints) == 0:
        return keypoints[:0], scores[:0], prev_anchor, {'total': 0, 'kept': 0}

    bboxes = bbox_xyxy_compute(frame_shape, keypoints, padding=0)
    mean_scores = np.nanmean(scores, axis=1)
    bbox_area = np.clip(bboxes[:, 2] - bboxes[:, 0], 0, None) * np.clip(bboxes[:, 3] - bboxes[:, 1], 0, None)
    anchor_points = _anchor_points_from_keypoints(keypoints, bboxes, person_filter_cfg['anchor_keypoint_id'])
    roi = _roi_for_source(source_name, person_filter_cfg.get('roi_by_camera', {}))
    roi_mask = _points_in_roi(anchor_points, roi)

    img_h, img_w = frame_shape[:2]
    bbox_in_image = (
        (bboxes[:, 0] >= 0) &
        (bboxes[:, 1] >= 0) &
        (bboxes[:, 2] <= img_w) &
        (bboxes[:, 3] <= img_h)
    )

    keep_mask = np.isfinite(mean_scores)
    keep_mask &= bbox_in_image
    keep_mask &= mean_scores >= person_filter_cfg['min_mean_score']
    keep_mask &= bbox_area >= person_filter_cfg['min_bbox_area_px']
    keep_mask &= bbox_area <= person_filter_cfg['max_bbox_area_px']
    keep_mask &= roi_mask

    keep_indices = np.where(keep_mask)[0]
    if len(keep_indices) == 0:
        return keypoints[:0], scores[:0], prev_anchor, {'total': len(keypoints), 'kept': 0}

    if prev_anchor is None or not np.all(np.isfinite(prev_anchor)):
        distances = np.full(len(keep_indices), np.inf)
    else:
        distances = np.linalg.norm(anchor_points[keep_indices] - prev_anchor, axis=1)

    order = np.lexsort((
        -mean_scores[keep_indices],
        -bbox_area[keep_indices],
        distances,
    ))
    keep_indices = keep_indices[order][:person_filter_cfg['max_people']]
    filtered_keypoints = keypoints[keep_indices]
    filtered_scores = scores[keep_indices]
    next_anchor = anchor_points[keep_indices[0]] if len(keep_indices) > 0 else prev_anchor

    return filtered_keypoints, filtered_scores, next_anchor, {'total': len(keypoints), 'kept': len(filtered_keypoints)}


def _batch_onnx_inference(session, batch_input):
    '''
    Run ONNX inference with batch_input of shape (N, 3, H, W).
    Bypasses rtmlib's single-frame inference to enable GPU batching.

    Returns list of output arrays, each with batch dimension N.
    '''

    sess_input = {session.get_inputs()[0].name: batch_input}
    sess_output = [out.name for out in session.get_outputs()]
    return session.run(sess_output, sess_input)


def batch_detection(det_model, frames):
    '''
    Run person detection on multiple frames in a single GPU batch.

    INPUTS:
    - det_model: RTMDet model object (from pose_tracker.det_model)
    - frames: list of BGR numpy arrays (raw video frames)

    OUTPUTS:
    - list of per-frame bounding boxes (same format as det_model.__call__)
    '''

    if len(frames) == 0:
        return []

    preprocessed = []
    ratios = []
    for frame in frames:
        img, ratio = det_model.preprocess(frame)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        preprocessed.append(img)
        ratios.append(ratio)

    batch_input = np.stack(preprocessed, axis=0).astype(np.float32)  # (N, 3, 640, 640)
    outputs = _batch_onnx_inference(det_model.session, batch_input)

    # outputs[0] shape: (N, num_anchors, 5) for models with NMS, or (N, num_anchors, 4+classes) without
    results = []
    for i in range(len(frames)):
        # Extract single-frame output and call original postprocess
        frame_output = outputs[0][i:i+1]  # keep batch dim for postprocess compatibility
        bboxes = det_model.postprocess(frame_output, ratios[i])
        results.append(bboxes)

    return results


def batch_pose_topdown(pose_model, frames, per_frame_bboxes, max_batch_crops=128):
    '''
    Run top-down pose estimation on multiple frames' bounding boxes in batched GPU calls.

    INPUTS:
    - pose_model: RTMPose model object (from pose_tracker.pose_model)
    - frames: list of BGR numpy arrays
    - per_frame_bboxes: list of bbox arrays, one per frame
    - max_batch_crops: max crops per ONNX call to limit VRAM usage

    OUTPUTS:
    - list of (keypoints, scores) tuples, one per frame
      keypoints shape: (num_persons, num_keypoints, 2)
      scores shape: (num_persons, num_keypoints)
    '''

    # Collect all crops across all frames
    all_crops = []
    all_centers = []
    all_scales = []
    frame_indices = []  # which frame each crop belongs to

    for fi, (frame, bboxes) in enumerate(zip(frames, per_frame_bboxes)):
        if len(bboxes) == 0:
            bboxes = [[0, 0, frame.shape[1], frame.shape[0]]]
        for bbox in bboxes:
            img, center, scale = pose_model.preprocess(frame, bbox)
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            all_crops.append(img)
            all_centers.append(center)
            all_scales.append(scale)
            frame_indices.append(fi)

    if len(all_crops) == 0:
        return [(np.empty((0, 0, 2)), np.empty((0, 0))) for _ in frames]

    # Run inference in sub-batches to limit VRAM
    all_simcc_x = []
    all_simcc_y = []
    for start in range(0, len(all_crops), max_batch_crops):
        end = min(start + max_batch_crops, len(all_crops))
        batch_input = np.stack(all_crops[start:end], axis=0).astype(np.float32)
        outputs = _batch_onnx_inference(pose_model.session, batch_input)
        all_simcc_x.append(outputs[0])
        all_simcc_y.append(outputs[1])

    simcc_x = np.concatenate(all_simcc_x, axis=0)  # (total_crops, K, Wx)
    simcc_y = np.concatenate(all_simcc_y, axis=0)  # (total_crops, K, Wy)

    # Batch postprocess: get_simcc_maximum handles (N, K, Wx) natively
    from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum
    simcc_split_ratio = 2.0
    locs, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints_all = locs / simcc_split_ratio

    # Rescale each crop's keypoints back to image coordinates
    centers = np.array(all_centers)   # (total_crops, 2)
    scales = np.array(all_scales)     # (total_crops, 2)
    model_input_size = np.array(pose_model.model_input_size)
    keypoints_all = keypoints_all / model_input_size * scales[:, np.newaxis, :]
    keypoints_all = keypoints_all + centers[:, np.newaxis, :] - scales[:, np.newaxis, :] / 2

    # Group by frame
    results = []
    for fi in range(len(frames)):
        mask = [j for j, idx in enumerate(frame_indices) if idx == fi]
        if mask:
            results.append((keypoints_all[mask], scores[mask]))
        else:
            results.append((np.empty((0, keypoints_all.shape[1], 2)), np.empty((0, keypoints_all.shape[1]))))

    return results


def setup_pose_tracker(ModelClass, det_frequency, mode, tracking, backend, device):
    '''
    Set up the RTMLib pose tracker with the appropriate model and backend.
    If CUDA is available, use it with ONNXRuntime backend; else use CPU with openvino

    INPUTS:
    - ModelClass: class. The RTMlib model class to use for pose detection (Body, BodyWithFeet, Wholebody)
    - det_frequency: int. The frequency of pose detection (every N frames)
    - mode: str. The mode of the pose tracker ('lightweight', 'balanced', 'performance')
    - tracking: bool. Whether to track persons across frames with RTMlib tracker
    - backend: str. The backend to use for pose detection (onnxruntime, openvino, opencv)
    - device: str. The device to use for pose detection (cpu, cuda, rocm, mps)

    OUTPUTS:
    - pose_tracker: PoseTracker. The initialized pose tracker object    
    '''

    backend, device = setup_backend_device(backend=backend, device=device)

    # Initialize the pose tracker with Halpe26 model
    pose_tracker = PoseTracker(
        ModelClass,
        det_frequency=det_frequency,
        mode=mode,
        backend=backend,
        device=device,
        tracking=tracking,
        to_openpose=False)
        
    return pose_tracker


def setup_model_class_mode(pose_model, mode, config_dict={}):
    '''
    Set up the pose model class and mode for the pose tracker.
    '''

    if pose_model.upper() in ('HALPE_26', 'BODY_WITH_FEET'):
        model_name = 'HALPE_26'
        ModelClass = BodyWithFeet # 26 keypoints(halpe26)
        logging.info(f"Using HALPE_26 model (body and feet) for pose estimation in {mode} mode.")
    elif pose_model.upper() in ('COCO_133', 'WHOLE_BODY', 'WHOLE_BODY_WRIST'):
        model_name = 'COCO_133'
        ModelClass = Wholebody
        logging.info(f"Using COCO_133 model (body, feet, hands, and face) for pose estimation in {mode} mode.")
    elif pose_model.upper() in ('COCO_17', 'BODY'):
        model_name = 'COCO_17'
        ModelClass = Body
        logging.info(f"Using COCO_17 model (body) for pose estimation in {mode} mode.")
    elif pose_model.upper() =='HAND':
        model_name = 'HAND_21'
        ModelClass = Hand
        logging.info(f"Using HAND_21 model for pose estimation in {mode} mode.")
    elif pose_model.upper() =='FACE':
        model_name = 'FACE_106'
        logging.info(f"Using FACE_106 model for pose estimation in {mode} mode.")
    elif pose_model.upper() =='ANIMAL':
        model_name = 'ANIMAL2D_17'
        logging.info(f"Using ANIMAL2D_17 model for pose estimation in {mode} mode.")
    else:
        model_name = pose_model.upper()
        logging.info(f"Using model {model_name} for pose estimation in {mode} mode.")
    try:
        pose_model = eval(model_name)
    except:
        try: # from Config.toml
            from anytree.importer import DictImporter
            model_name = pose_model.upper()
            pose_model = DictImporter().import_(config_dict.get('pose').get(pose_model)[0])
            if pose_model.id == 'None':
                pose_model.id = None
            logging.info(f"Using model {model_name} for pose estimation.")
        except:
            raise NameError(f'{pose_model} not found in skeletons.py nor in Config.toml')

    # Manually select the models if mode is a dictionary rather than 'lightweight', 'balanced', or 'performance'
    if not mode in ['lightweight', 'balanced', 'performance'] or 'ModelClass' not in locals():
        try:
            from functools import partial
            try:
                mode = ast.literal_eval(mode)
            except: # if within single quotes instead of double quotes when run with sports2d --mode """{dictionary}"""
                mode = mode.strip("'").replace('\n', '').replace(" ", "").replace(",", '", "').replace(":", '":"').replace("{", '{"').replace("}", '"}').replace('":"/',':/').replace('":"\\',':\\')
                mode = re.sub(r'"\[([^"]+)",\s?"([^"]+)\]"', r'[\1,\2]', mode) # changes "[640", "640]" to [640,640]
                mode = json.loads(mode)
            det_class = mode.get('det_class')
            det = mode.get('det_model')
            det_input_size = mode.get('det_input_size')
            pose_class = mode.get('pose_class')
            pose = mode.get('pose_model')
            pose_input_size = mode.get('pose_input_size')

            ModelClass = partial(Custom,
                        det_class=det_class, det=det, det_input_size=det_input_size,
                        pose_class=pose_class, pose=pose, pose_input_size=pose_input_size)
            logging.info(f"Using model {model_name} with the following custom parameters: {mode}.")

            if pose_class == 'RTMO' and model_name != 'COCO_17':
                logging.warning("RTMO currently only supports 'Body' pose_model. Switching to 'Body'.")
                pose_model = eval('COCO_17')
            
        except (json.JSONDecodeError, TypeError):
            logging.warning("Invalid mode. Must be 'lightweight', 'balanced', 'performance', or '''{dictionary}''' of parameters within triple quotes. Make sure input_sizes are within square brackets.")
            logging.warning('Using the default "balanced" mode.')
            mode = 'balanced'

    return pose_model, ModelClass, mode


def setup_backend_device(backend='auto', device='auto'):
    '''
    Set up the backend and device for the pose tracker based on the availability of hardware acceleration.
    TensorRT is not supported by RTMLib yet: https://github.com/Tau-J/rtmlib/issues/12

    If device and backend are not specified, they are automatically set up in the following order of priority:
    1. GPU with CUDA and ONNXRuntime backend (if CUDAExecutionProvider is available)
    2. GPU with ROCm and ONNXRuntime backend (if ROCMExecutionProvider is available, for AMD GPUs)
    3. GPU with MPS or CoreML and ONNXRuntime backend (for macOS systems)
    4. CPU with OpenVINO backend (default fallback)
    '''

    if device!='auto' and backend!='auto':
        device = device.lower()
        backend = backend.lower()

    if device=='auto' or backend=='auto':
        if device=='auto' and backend!='auto' or device!='auto' and backend=='auto':
            logging.warning(f"If you set device or backend to 'auto', you must set the other to 'auto' as well. Both device and backend will be determined automatically.")

        try:
            import torch
            import onnxruntime as ort
            if torch.cuda.is_available() == True and 'CUDAExecutionProvider' in ort.get_available_providers():
                device = 'cuda'
                backend = 'onnxruntime'
                logging.info(f"\nValid CUDA installation found: using ONNXRuntime backend with GPU.")
            elif torch.cuda.is_available() == True and 'ROCMExecutionProvider' in ort.get_available_providers():
                device = 'rocm'
                backend = 'onnxruntime'
                logging.info(f"\nValid ROCM installation found: using ONNXRuntime backend with GPU.")
            else:
                raise 
        except:
            try:
                import onnxruntime as ort
                if 'MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers():
                    device = 'mps'
                    backend = 'onnxruntime'
                    logging.info(f"\nValid MPS installation found: using ONNXRuntime backend with GPU.")
                else:
                    raise
            except:
                device = 'cpu'
                backend = 'openvino'
                logging.info(f"\nNo valid CUDA installation found: using OpenVINO backend with CPU.")
        
    return backend, device


def save_to_openpose(json_file_path, keypoints, scores):
    '''
    Save the keypoints and scores to a JSON file in the OpenPose format

    INPUTS:
    - json_file_path: Path to save the JSON file
    - keypoints: Detected keypoints
    - scores: Confidence scores for each keypoint

    OUTPUTS:
    - JSON file with the detected keypoints and confidence scores in the OpenPose format
    '''

    # Prepare keypoints with confidence scores for JSON output
    nb_detections = len(keypoints)
    # print('results: ', keypoints, scores)
    detections = []
    for i in range(nb_detections): # nb of detected people
        keypoints_with_confidence_i = []
        for kp, score in zip(keypoints[i], scores[i]):
            keypoints_with_confidence_i.extend([kp[0].item(), kp[1].item(), score.item()])
        detections.append({
                    "person_id": [-1],
                    "pose_keypoints_2d": keypoints_with_confidence_i,
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": []
                })
            
    # Create JSON output structure
    json_output = {"version": 1.3, "people": detections}
    
    # Save JSON output for each frame
    json_output_dir = os.path.abspath(os.path.join(json_file_path, '..'))
    if not os.path.isdir(json_output_dir): os.makedirs(json_output_dir)
    with open(json_file_path, 'w') as json_file:
        json.dump(json_output, json_file)


def process_video(video_path, pose_tracker, pose_model, output_format, save_video, save_images, display_detection, frame_range, tracking_mode, max_distance_px, deepsort_tracker, person_filter_cfg):
    '''
    Estimate pose from a video file
    
    INPUTS:
    - video_path: str. Path to the input video file
    - pose_tracker: PoseTracker. Initialized pose tracker object from RTMLib
    - pose_model: str. The pose model to use for pose estimation (HALPE_26, COCO_133, COCO_17)
    - output_format: str. Output format for the pose estimation results ('openpose', 'mmpose', 'deeplabcut')
    - save_video: bool. Whether to save the output video
    - save_images: bool. Whether to save the output images
    - display_detection: bool. Whether to show real-time visualization
    - frame_range: list. Range of frames to process
    - multi_person: bool. Whether to detect multiple people in the video
    - tracking_mode: str. The tracking mode to use for person tracking (deepsort, sports2d)
    - deepsort_tracker: DeepSort tracker object or None

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - if save_video: Video file with the detected keypoints and confidence scores drawn on the frames
    - if save_images: Image files with the detected keypoints and confidence scores drawn on the frames
    '''

    cap = cv2.VideoCapture(video_path)
    cap.read()
    if cap.read()[0] == False:
        raise NameError(f"{video_path} is not a video. Images must be put in one subdirectory per camera.")
    
    pose_dir = os.path.abspath(os.path.join(video_path, '..', '..', 'pose'))
    if not os.path.isdir(pose_dir): os.makedirs(pose_dir)
    video_name_wo_ext = os.path.splitext(os.path.basename(video_path))[0]
    json_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_json')
    output_video_path = os.path.join(pose_dir, f'{video_name_wo_ext}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_img')
    
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get the width and height from the raw video

    if save_video: # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for the output video
        fps = round(cap.get(cv2.CAP_PROP_FPS)) # Get the frame rate from the raw video
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H)) # Create the output video file
        
    if display_detection:
        screen_width, screen_height = get_screen_size()
        display_width, display_height = calculate_display_size(W, H, screen_width, screen_height, margin=50)
        cv2.namedWindow(f"Pose Estimation {os.path.basename(video_path)}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Pose Estimation {os.path.basename(video_path)}", display_width, display_height)

    frame_idx = 0
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_range = [[0,total_frames] if frame_range in ('all', 'auto', []) else frame_range][0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_range[0])
    frame_idx = f_range[0]
    prev_anchor = None
    filter_stats = {'frames': 0, 'detections_before': 0, 'detections_after': 0, 'frames_multi_before': 0, 'frames_multi_after': 0, 'frames_empty_after': 0}

    # Retrieve keypoint names from model
    keypoints_ids = [node.id for _, _, node in RenderTree(pose_model) if node.id!=None]
    kpt_id_max = max(keypoints_ids)+1

    with tqdm(iterable=range(*f_range), desc=f'Processing {os.path.basename(video_path)}') as pbar:
        while cap.isOpened():
            if frame_idx in range(*f_range):
                # print('\nFrame ', frame_idx)
                success, frame = cap.read()
                if not success:
                    break
            
                try: # Frames with no detection cause errors on MacOS CoreMLExecutionProvider
                    # Detect poses
                    keypoints, scores = pose_tracker(frame)

                    # Non maximum suppression (at pose level, not detection, and only using likely keypoints)
                    frame_shape = frame.shape
                    mask_scores = np.mean(scores, axis=1) > 0.2

                    likely_keypoints = np.where(mask_scores[:, np.newaxis, np.newaxis], keypoints, np.nan)
                    likely_scores = np.where(mask_scores[:, np.newaxis], scores, np.nan)
                    likely_bboxes = bbox_xyxy_compute(frame_shape, likely_keypoints, padding=0)
                    score_likely_bboxes = np.nanmean(likely_scores, axis=1)

                    valid_indices = np.where(~np.isnan(score_likely_bboxes))[0]
                    if len(valid_indices) > 0:
                        valid_bboxes = likely_bboxes[valid_indices]
                        valid_scores = score_likely_bboxes[valid_indices]
                        keep_valid = nms(valid_bboxes, valid_scores, nms_thr=0.45)
                        keep = valid_indices[keep_valid]
                    else:
                        keep = []
                    keypoints, scores = likely_keypoints[keep], likely_scores[keep]

                    # Track poses across frames
                    if tracking_mode == 'deepsort':
                        keypoints, scores = sort_people_deepsort(keypoints, scores, deepsort_tracker, frame, frame_idx)
                    if tracking_mode == 'sports2d': 
                        if 'prev_keypoints' not in locals(): 
                            prev_keypoints = keypoints
                        prev_keypoints, keypoints, scores = sort_people_sports2d(prev_keypoints, keypoints, scores=scores, max_dist=max_distance_px)
                    else:
                        pass

                    keypoints, scores, prev_anchor, stats = filter_person_detections(
                        keypoints,
                        scores,
                        frame_shape,
                        video_path,
                        person_filter_cfg,
                        prev_anchor=prev_anchor,
                    )
                    if stats is not None:
                        filter_stats['frames'] += 1
                        filter_stats['detections_before'] += stats['total']
                        filter_stats['detections_after'] += stats['kept']
                        filter_stats['frames_multi_before'] += int(stats['total'] > 1)
                        filter_stats['frames_multi_after'] += int(stats['kept'] > 1)
                        filter_stats['frames_empty_after'] += int(stats['kept'] == 0)

                except:
                    keypoints = np.full((1,kpt_id_max,2), fill_value=np.nan)
                    scores = np.full((1,kpt_id_max), fill_value=np.nan)
                    
                # Save to json
                if 'openpose' in output_format:
                    json_file_path = os.path.join(json_output_dir, f'{video_name_wo_ext}_{frame_idx:06d}.json')
                    save_to_openpose(json_file_path, keypoints, scores)

                # Draw skeleton on the frame
                if display_detection or save_video or save_images:
                    # try:
                    #     # MMPose skeleton
                    #     img_show = frame.copy()
                    #     img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.1) # maybe change this value if 0.1 is too low
                    # except:
                        # Sports2D skeleton
                        valid_X, valid_Y, valid_scores = [], [], []
                        for person_keypoints, person_scores in zip(keypoints, scores):
                            person_X, person_Y = person_keypoints[:, 0], person_keypoints[:, 1]
                            valid_X.append(person_X)
                            valid_Y.append(person_Y)
                            valid_scores.append(person_scores)
                        img_show = frame.copy()
                        img_show = draw_bounding_box(img_show, valid_X, valid_Y, colors=colors, fontSize=2, thickness=thickness)
                        img_show = draw_keypts(img_show, valid_X, valid_Y, valid_scores, cmap_str='RdYlGn')
                        img_show = draw_skel(img_show, valid_X, valid_Y, pose_model)
                
                if display_detection:
                    cv2.imshow(f"Pose Estimation {os.path.basename(video_path)}", img_show)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if save_video:
                    out.write(img_show)

                if save_images:
                    if not os.path.isdir(img_output_dir): os.makedirs(img_output_dir)
                    cv2.imwrite(os.path.join(img_output_dir, f'{video_name_wo_ext}_{frame_idx:06d}.jpg'), img_show)

                frame_idx += 1
                pbar.update(1)

            if frame_idx >= f_range[1]:
                break

    cap.release()
    if save_video:
        out.release()
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if display_detection:
        cv2.destroyAllWindows()
    if person_filter_cfg.get('enabled', False) and filter_stats['frames'] > 0:
        logging.info(
            f'Person filter summary for {os.path.basename(video_path)}: '
            f'mean detections/frame {filter_stats["detections_before"]/filter_stats["frames"]:.2f} -> '
            f'{filter_stats["detections_after"]/filter_stats["frames"]:.2f}, '
            f'frames with >1 detection {filter_stats["frames_multi_before"]} -> {filter_stats["frames_multi_after"]}, '
            f'empty frames after filter {filter_stats["frames_empty_after"]}.'
        )


def process_video_batched(video_path, det_model, pose_model_onnx, pose_model_tree, batch_size, det_frequency,
                          output_format, save_video, save_images, display_detection, frame_range,
                          tracking_mode, max_distance_px, deepsort_tracker, person_filter_cfg):
    '''
    Estimate pose from a video file using GPU batch inference.
    Reads batch_size frames at a time, runs detection and pose estimation in batch,
    then applies NMS/tracking/person_filter sequentially per frame.

    INPUTS:
    - video_path: str. Path to the input video file
    - det_model: RTMDet model object (pose_tracker.det_model) for batch detection
    - pose_model_onnx: RTMPose model object (pose_tracker.pose_model) for batch pose estimation
    - pose_model_tree: anytree model. Skeleton tree for drawing/keypoint info
    - batch_size: int. Number of frames to process in a single GPU batch
    - det_frequency: int. Run detection every N frames, reuse bboxes in between
    - output_format, save_video, save_images, display_detection, frame_range,
      tracking_mode, max_distance_px, deepsort_tracker, person_filter_cfg: same as process_video
    '''

    cap = cv2.VideoCapture(video_path)
    cap.read()
    if cap.read()[0] == False:
        raise NameError(f"{video_path} is not a video. Images must be put in one subdirectory per camera.")

    pose_dir = os.path.abspath(os.path.join(video_path, '..', '..', 'pose'))
    if not os.path.isdir(pose_dir): os.makedirs(pose_dir)
    video_name_wo_ext = os.path.splitext(os.path.basename(video_path))[0]
    json_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_json')
    output_video_path = os.path.join(pose_dir, f'{video_name_wo_ext}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_img')

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    if display_detection:
        screen_width, screen_height = get_screen_size()
        display_width, display_height = calculate_display_size(W, H, screen_width, screen_height, margin=50)
        cv2.namedWindow(f"Pose Estimation {os.path.basename(video_path)}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Pose Estimation {os.path.basename(video_path)}", display_width, display_height)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_range = [[0, total_frames] if frame_range in ('all', 'auto', []) else frame_range][0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_range[0])
    frame_idx = f_range[0]
    prev_anchor = None
    filter_stats = {'frames': 0, 'detections_before': 0, 'detections_after': 0, 'frames_multi_before': 0, 'frames_multi_after': 0, 'frames_empty_after': 0}

    keypoints_ids = [node.id for _, _, node in RenderTree(pose_model_tree) if node.id != None]
    kpt_id_max = max(keypoints_ids) + 1

    cached_bboxes = [[0, 0, W, H]]  # fallback bboxes for det_frequency skip frames
    quit_flag = False

    with tqdm(iterable=range(*f_range), desc=f'Processing {os.path.basename(video_path)} (batch={batch_size})') as pbar:
        while cap.isOpened() and frame_idx < f_range[1] and not quit_flag:
            # 1. Read a batch of frames
            batch_frames = []
            batch_frame_indices = []
            for _ in range(batch_size):
                if frame_idx >= f_range[1]:
                    break
                success, frame = cap.read()
                if not success:
                    break
                batch_frames.append(frame)
                batch_frame_indices.append(frame_idx)
                frame_idx += 1

            if len(batch_frames) == 0:
                break

            # 2. Batch detection: only detect on frames where frame_idx % det_frequency == 0
            det_frame_positions = []  # indices within batch that need detection
            det_frames_list = []
            for i, fidx in enumerate(batch_frame_indices):
                if fidx % det_frequency == 0:
                    det_frame_positions.append(i)
                    det_frames_list.append(batch_frames[i])

            if det_frames_list:
                try:
                    det_results = batch_detection(det_model, det_frames_list)
                except:
                    det_results = [cached_bboxes] * len(det_frames_list)

            # Assign bboxes to each frame in the batch
            per_frame_bboxes = []
            det_result_idx = 0
            for i in range(len(batch_frames)):
                if i in det_frame_positions:
                    bboxes = det_results[det_result_idx]
                    det_result_idx += 1
                    if len(bboxes) > 0:
                        cached_bboxes = bboxes
                    per_frame_bboxes.append(bboxes if len(bboxes) > 0 else cached_bboxes)
                else:
                    per_frame_bboxes.append(cached_bboxes)

            # 3. Batch pose estimation
            try:
                per_frame_results = batch_pose_topdown(pose_model_onnx, batch_frames, per_frame_bboxes)
            except:
                per_frame_results = [
                    (np.full((1, kpt_id_max, 2), fill_value=np.nan), np.full((1, kpt_id_max), fill_value=np.nan))
                    for _ in batch_frames
                ]

            # 4. Sequential post-processing per frame
            for i, frame in enumerate(batch_frames):
                fidx = batch_frame_indices[i]
                keypoints, scores = per_frame_results[i]

                try:
                    # NMS
                    frame_shape = frame.shape
                    if len(keypoints) > 0 and keypoints.size > 0:
                        mask_scores = np.mean(scores, axis=1) > 0.2
                        likely_keypoints = np.where(mask_scores[:, np.newaxis, np.newaxis], keypoints, np.nan)
                        likely_scores = np.where(mask_scores[:, np.newaxis], scores, np.nan)
                        likely_bboxes = bbox_xyxy_compute(frame_shape, likely_keypoints, padding=0)
                        score_likely_bboxes = np.nanmean(likely_scores, axis=1)

                        valid_indices = np.where(~np.isnan(score_likely_bboxes))[0]
                        if len(valid_indices) > 0:
                            valid_bboxes = likely_bboxes[valid_indices]
                            valid_scores_nms = score_likely_bboxes[valid_indices]
                            keep_valid = nms(valid_bboxes, valid_scores_nms, nms_thr=0.45)
                            keep = valid_indices[keep_valid]
                        else:
                            keep = []
                        keypoints, scores = likely_keypoints[keep], likely_scores[keep]

                    # Tracking
                    if tracking_mode == 'deepsort':
                        keypoints, scores = sort_people_deepsort(keypoints, scores, deepsort_tracker, frame, fidx)
                    if tracking_mode == 'sports2d':
                        if 'prev_keypoints' not in locals():
                            prev_keypoints = keypoints
                        prev_keypoints, keypoints, scores = sort_people_sports2d(prev_keypoints, keypoints, scores=scores, max_dist=max_distance_px)

                    # Person filter
                    keypoints, scores, prev_anchor, stats = filter_person_detections(
                        keypoints, scores, frame_shape, video_path, person_filter_cfg, prev_anchor=prev_anchor)
                    if stats is not None:
                        filter_stats['frames'] += 1
                        filter_stats['detections_before'] += stats['total']
                        filter_stats['detections_after'] += stats['kept']
                        filter_stats['frames_multi_before'] += int(stats['total'] > 1)
                        filter_stats['frames_multi_after'] += int(stats['kept'] > 1)
                        filter_stats['frames_empty_after'] += int(stats['kept'] == 0)
                except:
                    keypoints = np.full((1, kpt_id_max, 2), fill_value=np.nan)
                    scores = np.full((1, kpt_id_max), fill_value=np.nan)

                # Save to json
                if 'openpose' in output_format:
                    json_file_path = os.path.join(json_output_dir, f'{video_name_wo_ext}_{fidx:06d}.json')
                    save_to_openpose(json_file_path, keypoints, scores)

                # Draw skeleton
                if display_detection or save_video or save_images:
                    valid_X, valid_Y, valid_scores = [], [], []
                    for person_keypoints, person_scores in zip(keypoints, scores):
                        person_X, person_Y = person_keypoints[:, 0], person_keypoints[:, 1]
                        valid_X.append(person_X)
                        valid_Y.append(person_Y)
                        valid_scores.append(person_scores)
                    img_show = frame.copy()
                    img_show = draw_bounding_box(img_show, valid_X, valid_Y, colors=colors, fontSize=2, thickness=thickness)
                    img_show = draw_keypts(img_show, valid_X, valid_Y, valid_scores, cmap_str='RdYlGn')
                    img_show = draw_skel(img_show, valid_X, valid_Y, pose_model_tree)

                if display_detection:
                    cv2.imshow(f"Pose Estimation {os.path.basename(video_path)}", img_show)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        quit_flag = True
                        break

                if save_video:
                    out.write(img_show)

                if save_images:
                    if not os.path.isdir(img_output_dir): os.makedirs(img_output_dir)
                    cv2.imwrite(os.path.join(img_output_dir, f'{video_name_wo_ext}_{fidx:06d}.jpg'), img_show)

                pbar.update(1)

    cap.release()
    if save_video:
        out.release()
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if display_detection:
        cv2.destroyAllWindows()
    if person_filter_cfg.get('enabled', False) and filter_stats['frames'] > 0:
        logging.info(
            f'Person filter summary for {os.path.basename(video_path)}: '
            f'mean detections/frame {filter_stats["detections_before"]/filter_stats["frames"]:.2f} -> '
            f'{filter_stats["detections_after"]/filter_stats["frames"]:.2f}, '
            f'frames with >1 detection {filter_stats["frames_multi_before"]} -> {filter_stats["frames_multi_after"]}, '
            f'empty frames after filter {filter_stats["frames_empty_after"]}.'
        )


def process_images(image_folder_path, vid_img_extension, pose_tracker, pose_model, output_format, fps, save_video, save_images, display_detection, frame_range, tracking_mode, max_distance_px, deepsort_tracker, person_filter_cfg):
    '''
    Estimate pose estimation from a folder of images

    INPUTS:
    - image_folder_path: str. Path to the input image folder
    - vid_img_extension: str. Extension of the image files
    - pose_tracker: PoseTracker. Initialized pose tracker object from RTMLib
    - pose_model: str. The pose model to use for pose estimation (HALPE_26, COCO_133, COCO_17)
    - output_format: str. Output format for the pose estimation results ('openpose', 'mmpose', 'deeplabcut')
    - save_video: bool. Whether to save the output video
    - save_images: bool. Whether to save the output images
    - display_detection: bool. Whether to show real-time visualization
    - frame_range: list. Range of frames to process
    - tracking_mode: str. The tracking mode to use for person tracking (deepsort, sports2d)
    - deepsort_tracker: DeepSort tracker object or None

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - if save_video: Video file with the detected keypoints and confidence scores drawn on the frames
    - if save_images: Image files with the detected keypoints and confidence scores drawn on the frames
    '''    

    pose_dir = os.path.abspath(os.path.join(image_folder_path, '..', '..', 'pose'))
    if not os.path.isdir(pose_dir): os.makedirs(pose_dir)
    json_output_dir = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_json')
    output_video_path = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_img')

    image_files = glob.glob(os.path.join(image_folder_path, '*'+vid_img_extension))
    sorted(image_files, key=natural_sort_key)

    if save_video: # Set up video writer
        logging.warning('Using default framerate of 60 fps.')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for the output video
        W, H = cv2.imread(image_files[0]).shape[:2][::-1] # Get the width and height from the first image (assuming all images have the same size)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H)) # Create the output video file

    if display_detection:
        screen_width, screen_height = get_screen_size()
        display_width, display_height = calculate_display_size(W, H, screen_width, screen_height, margin=50)
        cv2.namedWindow(f"Pose Estimation {os.path.basename(image_folder_path)}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Pose Estimation {os.path.basename(image_folder_path)}", display_width, display_height)
    
    # Retrieve keypoint names from model
    keypoints_ids = [node.id for _, _, node in RenderTree(pose_model) if node.id!=None]
    kpt_id_max = max(keypoints_ids)+1
    prev_anchor = None
    filter_stats = {'frames': 0, 'detections_before': 0, 'detections_after': 0, 'frames_multi_before': 0, 'frames_multi_after': 0, 'frames_empty_after': 0}
    
    f_range = [[0,len(image_files)] if frame_range in ('all', 'auto', []) else frame_range][0]
    for frame_idx, image_file in enumerate(tqdm(image_files, desc=f'\nProcessing {os.path.basename(img_output_dir)}')):
        if frame_idx in range(*f_range):
            try:
                frame = cv2.imread(image_file)
                frame_idx += 1
            except:
                raise NameError(f"{image_file} is not an image. Videos must be put in the video directory, not in subdirectories.")
            
            try:
                # Detect poses
                keypoints, scores = pose_tracker(frame)

                # Track poses across frames
                if tracking_mode == 'deepsort':
                    keypoints, scores = sort_people_deepsort(keypoints, scores, deepsort_tracker, frame, frame_idx)
                if tracking_mode == 'sports2d': 
                    if 'prev_keypoints' not in locals(): 
                        prev_keypoints = keypoints
                    prev_keypoints, keypoints, scores = sort_people_sports2d(prev_keypoints, keypoints, scores=scores, max_dist=max_distance_px)

                keypoints, scores, prev_anchor, stats = filter_person_detections(
                    keypoints,
                    scores,
                    frame.shape,
                    image_folder_path,
                    person_filter_cfg,
                    prev_anchor=prev_anchor,
                )
                if stats is not None:
                    filter_stats['frames'] += 1
                    filter_stats['detections_before'] += stats['total']
                    filter_stats['detections_after'] += stats['kept']
                    filter_stats['frames_multi_before'] += int(stats['total'] > 1)
                    filter_stats['frames_multi_after'] += int(stats['kept'] > 1)
                    filter_stats['frames_empty_after'] += int(stats['kept'] == 0)
            except:
                keypoints = np.full((1,kpt_id_max,2), fill_value=np.nan)
                scores = np.full((1,kpt_id_max), fill_value=np.nan)

            # Extract frame number from the filename
            if 'openpose' in output_format:
                json_file_path = os.path.join(json_output_dir, f"{os.path.splitext(os.path.basename(image_file))[0]}_{frame_idx:06d}.json")
                save_to_openpose(json_file_path, keypoints, scores)

            # Draw skeleton on the image
            if display_detection or save_video or save_images:
                try:
                    # MMPose skeleton
                    img_show = frame.copy()
                    img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.1) # maybe change this value if 0.1 is too low
                except:
                    # Sports2D skeleton
                    valid_X, valid_Y, valid_scores = [], [], []
                    for person_keypoints, person_scores in zip(keypoints, scores):
                        person_X, person_Y = person_keypoints[:, 0], person_keypoints[:, 1]
                        valid_X.append(person_X)
                        valid_Y.append(person_Y)
                        valid_scores.append(person_scores)
                    img_show = frame.copy()
                    img_show = draw_bounding_box(img_show, valid_X, valid_Y, colors=colors, fontSize=2, thickness=thickness)
                    img_show = draw_keypts(img_show, valid_X, valid_Y, valid_scores, cmap_str='RdYlGn')
                    img_show = draw_skel(img_show, valid_X, valid_Y, pose_model)

            if display_detection:
                cv2.imshow(f"Pose Estimation {os.path.basename(image_folder_path)}", img_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_video:
                out.write(img_show)

            if save_images:
                if not os.path.isdir(img_output_dir): os.makedirs(img_output_dir)
                cv2.imwrite(os.path.join(img_output_dir, f'{os.path.splitext(os.path.basename(image_file))[0]}_{frame_idx:06d}.png'), img_show)

    if save_video:
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if display_detection:
        cv2.destroyAllWindows()
    if person_filter_cfg.get('enabled', False) and filter_stats['frames'] > 0:
        logging.info(
            f'Person filter summary for {os.path.basename(image_folder_path)}: '
            f'mean detections/frame {filter_stats["detections_before"]/filter_stats["frames"]:.2f} -> '
            f'{filter_stats["detections_after"]/filter_stats["frames"]:.2f}, '
            f'frames with >1 detection {filter_stats["frames_multi_before"]} -> {filter_stats["frames_multi_after"]}, '
            f'empty frames after filter {filter_stats["frames_empty_after"]}.'
        )


def estimate_pose_all(config_dict):
    '''
    Estimate pose from a video file or a folder of images and 
    write the results to JSON files, videos, and/or images.
    Results can optionally be displayed in real time.

    Supported models: HALPE_26 (default, body and feet), COCO_133 (body, feet, hands), COCO_17 (body)
    Supported modes: lightweight, balanced, performance (edit paths at rtmlib/tools/solutions if you 
    need nother detection or pose models)

    Optionally gives consistent person ID across frames (slower but good for 2D analysis)
    Optionally runs detection every n frames and inbetween tracks points (faster but less accurate).

    If a valid cuda installation is detected, uses the GPU with the ONNXRuntime backend. Otherwise, 
    uses the CPU with the OpenVINO backend.

    INPUTS:
    - videos or image folders from the video directory
    - a Config.toml file

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - Optionally, videos and/or image files with the detected keypoints 
    '''

    # Read config
    project_dir = config_dict['project']['project_dir']
    # if batch
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    frame_range = config_dict.get('project').get('frame_range')
    multi_person = config_dict.get('project').get('multi_person')
    video_dir = os.path.join(project_dir, 'videos')
    pose_dir = os.path.join(project_dir, 'pose')

    pose_model = config_dict['pose']['pose_model']
    mode = config_dict['pose']['mode'] # lightweight, balanced, performance
    vid_img_extension = config_dict['pose']['vid_img_extension']
    
    output_format = config_dict['pose']['output_format']
    save_video = True if 'to_video' in config_dict['pose']['save_video'] else False
    save_images = True if 'to_images' in config_dict['pose']['save_video'] else False
    display_detection = config_dict['pose']['display_detection']
    overwrite_pose = config_dict['pose']['overwrite_pose']
    det_frequency = config_dict['pose']['det_frequency']
    tracking_mode = config_dict.get('pose').get('tracking_mode')
    max_distance_px = config_dict.get('pose').get('max_distance_px', None)
    if tracking_mode == 'deepsort' and multi_person:
        deepsort_params = config_dict.get('pose').get('deepsort_params')
        try:
            deepsort_params = ast.literal_eval(deepsort_params)
        except: # if within single quotes instead of double quotes when run with sports2d --mode """{dictionary}"""
            deepsort_params = deepsort_params.strip("'").replace('\n', '').replace(" ", "").replace(",", '", "').replace(":", '":"').replace("{", '{"').replace("}", '"}').replace('":"/',':/').replace('":"\\',':\\')
            deepsort_params = re.sub(r'"\[([^"]+)",\s?"([^"]+)\]"', r'[\1,\2]', deepsort_params) # changes "[640", "640]" to [640,640]
            deepsort_params = json.loads(deepsort_params)
        from deep_sort_realtime.deepsort_tracker import DeepSort
        deepsort_tracker = DeepSort(**deepsort_params)
    else:
        deepsort_tracker = None

    backend = config_dict['pose']['backend']
    device = config_dict['pose']['device']
    batch_size = config_dict.get('pose', {}).get('batch_size', 1)

    # Determine frame rate
    video_files = glob.glob(os.path.join(video_dir, '*'+vid_img_extension))
    frame_rate = config_dict.get('project').get('frame_rate')
    if frame_rate == 'auto': 
        try:
            cap = cv2.VideoCapture(video_files[0])
            cap.read()
            if cap.read()[0] == False:
                raise
            frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
        except:
            logging.warning(f'Cannot read video. Frame rate will be set to 60 fps.')
            frame_rate = 30  

    # Set detection frequency
    if det_frequency>1:
        logging.info(f'Inference run only every {det_frequency} frames. Inbetween, pose estimation tracks previously detected points.')
    elif det_frequency==1:
        logging.info(f'Inference run on every single frame.')
    else:
        raise ValueError(f"Invalid det_frequency: {det_frequency}. Must be an integer greater or equal to 1.")

    # Select the appropriate model based on the model_type
    logging.info('\nEstimating pose...')
    pose_model_name = pose_model
    pose_model, ModelClass, mode = setup_model_class_mode(pose_model, mode, config_dict)
    person_filter_cfg = build_person_filter_config(config_dict, pose_model)

    # Select device and backend
    backend, device = setup_backend_device(backend=backend, device=device)

    # Estimate pose
    try:
        pose_listdirs_names = next(os.walk(pose_dir))[1]
        os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
        if not overwrite_pose:
            logging.info('Skipping pose estimation as it has already been done. Set overwrite_pose to true in Config.toml if you want to run it again.')
        else:
            logging.info('Overwriting previous pose estimation. Set overwrite_pose to false in Config.toml if you want to keep the previous results.')
            raise
            
    except:
        # Set up pose tracker
        try:
            pose_tracker = setup_pose_tracker(ModelClass, det_frequency, mode, False, backend, device)
        except:
            logging.error('Error: Pose estimation failed. Check in Config.toml that pose_model and mode are valid.')
            raise ValueError('Error: Pose estimation failed. Check in Config.toml that pose_model and mode are valid.')

        if tracking_mode not in ['deepsort', 'sports2d']:
            logging.warning(f"Tracking mode {tracking_mode} not recognized. Using sports2d method.")
            tracking_mode = 'sports2d'
        logging.info(f'\nPose tracking set up for "{pose_model_name}" model.')
        logging.info(f'Mode: {mode}.')
        logging.info(f'Tracking is performed with {tracking_mode}{"" if not tracking_mode=="deepsort" else f" with parameters: {deepsort_params}"}.\n')

        # Check if batch mode is available
        use_batch = (batch_size > 1 and backend == 'onnxruntime'
                     and pose_tracker.det_model is not None)  # top-down only for now
        if use_batch:
            logging.info(f'GPU batch inference enabled: batch_size={batch_size}.')
        elif batch_size > 1:
            logging.warning(f'batch_size={batch_size} requested but batch inference requires onnxruntime backend '
                          f'with top-down model. Falling back to single-frame processing.')

        video_files = sorted(glob.glob(os.path.join(video_dir, '*'+vid_img_extension)))
        if not len(video_files) == 0:
            # Process video files
            logging.info(f'Found video files with {vid_img_extension} extension.')
            for video_path in video_files:
                pose_tracker.reset()
                if tracking_mode == 'deepsort':
                    deepsort_tracker.tracker.delete_all_tracks()
                if use_batch:
                    process_video_batched(video_path, pose_tracker.det_model, pose_tracker.pose_model, pose_model,
                                         batch_size, det_frequency, output_format, save_video, save_images,
                                         display_detection, frame_range, tracking_mode, max_distance_px,
                                         deepsort_tracker, person_filter_cfg)
                else:
                    process_video(video_path, pose_tracker, pose_model, output_format, save_video, save_images, display_detection, frame_range, tracking_mode, max_distance_px, deepsort_tracker, person_filter_cfg)

        else:
            # Process image folders
            image_folders = sorted([os.path.join(video_dir,f) for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))])
            empty_folders = [folder for folder in image_folders if len(glob.glob(os.path.join(folder, '*'+vid_img_extension)))==0]
            if len(empty_folders) != 0:
                raise NameError(f'No image files with {vid_img_extension} extension found in {empty_folders}.')
            elif len(image_folders) == 0:
                raise NameError(f'No image folders containing files with {vid_img_extension} extension found in {video_dir}.')
            else:
                logging.info(f'Found image folders with {vid_img_extension} extension.')
                for image_folder in image_folders:
                    pose_tracker.reset()
                    image_folder_path = os.path.join(video_dir, image_folder)
                    if tracking_mode == 'deepsort': 
                        deepsort_tracker.tracker.delete_all_tracks()                
                    process_images(image_folder_path, vid_img_extension, pose_tracker, pose_model, output_format, frame_rate, save_video, save_images, display_detection, frame_range, tracking_mode, max_distance_px, deepsort_tracker, person_filter_cfg)
