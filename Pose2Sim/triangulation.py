#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## ROBUST TRIANGULATION  OF 2D COORDINATES                               ##
###########################################################################

This module triangulates 2D json coordinates and builds a .trc file readable 
by OpenSim.

The triangulation is weighted by the likelihood of each detected 2D keypoint 
(if they meet the likelihood threshold). If the reprojection error is above a
threshold, right and left sides are swapped; if it is still above, a camera 
is removed for this point and this frame, until the threshold is met. If more 
cameras are removed than a predefined minimum, triangulation is skipped for 
the point and this frame. 

In the end, missing values are interpolated if the gaps are smaller than a 
threshold. The trial if the person is out of the camera view for a long time. 
The last missing frames are filled with the last valid value, resulting in a
freeze, that might be better than doubtful long interpolations.

In case of multiple subjects detection, make sure you first run the 
personAssociation module. It will then associate people across frames by 
measuring the frame-by-frame distance between them.

INPUTS: 
- a calibration file (.toml extension)
- json files for each camera with only one person of interest
- a Config.toml file
- a skeleton model

OUTPUTS: 
- a .trc file with 3D coordinates in Y-up system coordinates
'''


## INIT
import os
import glob
import fnmatch
import re
import numpy as np
np.set_printoptions(legacy='1.21') # otherwise prints np.float64(3.0) rather than 3.0
import json
import itertools as it
import pandas as pd
import cv2
import toml
from tqdm import tqdm
from collections import Counter
from functools import lru_cache
from anytree import RenderTree
from anytree.importer import DictImporter
import logging

from Pose2Sim.common import retrieve_calib_params, computeP, weighted_triangulation, \
    reprojection, euclidean_distance, sort_people_sports2d, interpolate_zeros_nans, \
    sort_stringlist_by_last_number, zup2yup, convert_to_c3d
from Pose2Sim.skeletons import *


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


STRICT_TRAJECTORY_KEYPOINTS = {
    'Head', 'Nose', 'RWrist', 'LWrist',
    'RBigToe', 'LBigToe', 'RSmallToe', 'LSmallToe', 'RHeel', 'LHeel'
}
DEFAULT_LR_SWAP_KEYPOINTS = {
    'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist',
    'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle'
}
BONE_PAIRS = [
    ('RShoulder', 'RElbow'),
    ('LShoulder', 'LElbow'),
    ('RElbow', 'RWrist'),
    ('LElbow', 'LWrist'),
    ('RHip', 'RKnee'),
    ('LHip', 'LKnee'),
    ('RKnee', 'RAnkle'),
    ('LKnee', 'LAnkle'),
    ('RAnkle', 'RBigToe'),
    ('LAnkle', 'LBigToe'),
]
LEFT_RIGHT_PAIRS = [
    ('RWrist', 'LWrist'),
    ('RElbow', 'LElbow'),
    ('RAnkle', 'LAnkle'),
    ('RBigToe', 'LBigToe'),
    ('RSmallToe', 'LSmallToe'),
    ('RHeel', 'LHeel'),
]


## FUNCTIONS
@lru_cache(maxsize=None)
def cached_camera_combinations(n_cams, nb_cams_off):
    '''
    Cache repeated camera subset enumeration.
    '''

    return tuple(it.combinations(range(n_cams), nb_cams_off))


def resolve_session_dir(project_dir):
    '''
    Resolve the session directory that contains calibration assets.
    '''

    project_dir = os.path.realpath(project_dir)
    parent_dir = os.path.realpath(os.path.join(project_dir, '..'))

    for candidate in (project_dir, parent_dir, os.getcwd()):
        try:
            if any(
                os.path.isdir(os.path.join(candidate, entry)) and 'calib' in entry.lower()
                for entry in os.listdir(candidate)
            ):
                return candidate
        except OSError:
            continue

    return project_dir


def count_persons_in_json(file_path):
    '''
    Count the number of persons in a json file.

    INPUT:
    - file_path: path to the json file

    OUTPUT:
    - int: number of persons in the json file
    '''

    with open(file_path, 'r') as file:
        data = json.load(file)
        return len(data.get('people', []))
    

def indices_of_first_last_non_nan_chunks(series, min_chunk_size=10, chunk_choice_method='largest', trim_output_chunk=True):
    '''
    Find indices of the chunks of at least min_chunk_size consecutive non-NaN values.

    INPUT:
    - series: pandas Series to trim
    - min_chunk_size: minimum size of consecutive non-NaN values to consider (default: 10)
    - chunk_choice_method: 'largest' to return the largest chunk, 'all' to return everything between the first and last non-nan chunk, 
                           'first' to return only the first one, 'last' to return only the last one
    - trim_output_chunk:   if True, the output chunk starts when all values are valid and ends at the first nan
                           else, it starts when at least on value is valid and ends when none is anymore

    OUTPUT:
    - tuple: (start_index, end_index) of the first and last valid chunks
    '''
    
    min_chunk_size = 10 if min_chunk_size == None else min_chunk_size
    non_nan_mask = ~np.isnan(series.values)
    
    # Find runs of consecutive non-NaN values (eg [(8, 15), (16, 17), (19, 26)])
    runs = []
    run_start = None
    for i, bool_val in enumerate(non_nan_mask):
        if bool_val and run_start is None:
            run_start = i
        elif not bool_val and run_start is not None:
            run_end = i
            runs.append((run_start, run_end))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(non_nan_mask)))
    
    # Find runs that have at least min_chunk_size consecutive non-NaN values
    valid_runs = [(start, end) for start, end in runs if end - start >= min_chunk_size]
    if not valid_runs:
        return(0,0)
    
    if chunk_choice_method not in ['largest', 'all', 'first', 'last']:
        chunk_choice_method = 'all'
    if chunk_choice_method == 'largest':
        # Choose the largest chunk
        valid_runs.sort(key=lambda x: x[1] - x[0], reverse=True)
        first_run_start, last_run_end = valid_runs[0]
    elif chunk_choice_method == 'all':
        # Get the start of the first valid run and the end of the last valid run
        first_run_start = valid_runs[0][0]
        last_run_end = valid_runs[-1][1]
    elif chunk_choice_method == 'first':
        # Get the start of the first valid run and the end of that run
        first_run_start, last_run_end = valid_runs[0]
    elif chunk_choice_method == 'last':
        # Get the start of the last valid run and the end of that run
        first_run_start, last_run_end = valid_runs[-1]
    
    # Return the trimmed series
    return first_run_start, last_run_end


def get_triangulation_keypoint_names(config_dict):
    '''
    Retrieve keypoint names in triangulation order.
    '''

    pose_model = config_dict.get('pose').get('pose_model')
    try:
        if pose_model.upper() == 'BODY_WITH_FEET': pose_model = 'HALPE_26'
        elif pose_model.upper() == 'WHOLE_BODY_WRIST': pose_model = 'COCO_133_WRIST'
        elif pose_model.upper() == 'WHOLE_BODY': pose_model = 'COCO_133'
        elif pose_model.upper() == 'BODY': pose_model = 'COCO_17'
        elif pose_model.upper() == 'HAND': pose_model = 'HAND_21'
        elif pose_model.upper() == 'FACE': pose_model = 'FACE_106'
        elif pose_model.upper() == 'ANIMAL': pose_model = 'ANIMAL2D_17'
        model = eval(pose_model)
    except:
        model = import_pose_model_from_config(config_dict, pose_model)

    return [node.name for _, _, node in RenderTree(model) if node.id != None]


def import_pose_model_from_config(config_dict, pose_model):
    '''
    Import a custom skeleton from Config.toml.

    In this environment, TOML array-of-tables syntax (`[[pose.CUSTOM]]`) is
    parsed as a one-element list, while DictImporter expects a single dict root.
    '''

    model_config = config_dict.get('pose').get(pose_model)
    if isinstance(model_config, list):
        if len(model_config) != 1:
            raise ValueError(f'Custom pose model "{pose_model}" must contain exactly one root node.')
        model_config = model_config[0]

    model = DictImporter().import_(model_config)
    if model.id == 'None':
        model.id = None
    return model


def get_keypoint_setting(settings_dict, keypoint_name, default_value):
    '''
    Return a keypoint-specific setting with a global fallback.
    '''

    if not isinstance(settings_dict, dict):
        return default_value

    aliases = [keypoint_name, keypoint_name.lower(), keypoint_name.upper()]
    for alias in aliases:
        if alias in settings_dict:
            return settings_dict.get(alias)
    return default_value


def get_reprojection_threshold(config_dict, keypoint_name):
    '''
    Return keypoint-specific reprojection threshold when configured.
    '''

    default_threshold = config_dict.get('triangulation').get('reproj_error_threshold_triangulation')
    thresholds = config_dict.get('triangulation').get('reproj_error_threshold_by_keypoint', {})
    return float(get_keypoint_setting(thresholds, keypoint_name, default_threshold))


def clean_2d_keypoints(config_dict, keypoints_names, x_files, y_files, likelihood_files, prev_coords=None):
    '''
    Reject 2D detections that are implausible for fixed-camera gait sequences.
    '''

    clean_cfg = config_dict.get('triangulation', {}).get('clean_2d', {})
    if clean_cfg.get('enabled', True) is False:
        return x_files, y_files, likelihood_files

    likelihood_threshold_default = config_dict.get('triangulation').get('likelihood_threshold_triangulation')
    likelihood_thresholds = config_dict.get('triangulation').get('likelihood_threshold_by_keypoint', {})
    jump_thresholds = clean_cfg.get('max_jump_px_by_keypoint', {})
    strict_jump_px = clean_cfg.get('strict_jump_px', 90.0)
    default_jump_px = clean_cfg.get('default_jump_px', 140.0)
    crossing_margin_px = clean_cfg.get('crossing_margin_px', 25.0)
    bone_ratio_min = clean_cfg.get('bone_ratio_min', 0.4)
    bone_ratio_max = clean_cfg.get('bone_ratio_max', 2.2)
    check_left_right_crossing = clean_cfg.get('check_left_right_crossing', True)

    x_clean = np.array(x_files, copy=True)
    y_clean = np.array(y_files, copy=True)
    likelihood_clean = np.array(likelihood_files, copy=True)

    for keypoint_idx, keypoint_name in enumerate(keypoints_names):
        likelihood_threshold = float(get_keypoint_setting(
            likelihood_thresholds,
            keypoint_name,
            likelihood_threshold_default
        ))
        invalid_mask = likelihood_clean[:, keypoint_idx] < likelihood_threshold
        x_clean[invalid_mask, keypoint_idx] = np.nan
        y_clean[invalid_mask, keypoint_idx] = np.nan
        likelihood_clean[invalid_mask, keypoint_idx] = np.nan

    if prev_coords is None:
        return x_clean, y_clean, likelihood_clean

    prev_x, prev_y, _ = prev_coords
    prev_x = np.asarray(prev_x)
    prev_y = np.asarray(prev_y)

    for keypoint_idx, keypoint_name in enumerate(keypoints_names):
        jump_threshold = float(get_keypoint_setting(
            jump_thresholds,
            keypoint_name,
            strict_jump_px if keypoint_name in STRICT_TRAJECTORY_KEYPOINTS else default_jump_px
        ))
        current_valid = np.isfinite(x_clean[:, keypoint_idx]) & np.isfinite(y_clean[:, keypoint_idx])
        previous_valid = np.isfinite(prev_x[:, keypoint_idx]) & np.isfinite(prev_y[:, keypoint_idx])
        valid_mask = current_valid & previous_valid
        if not np.any(valid_mask):
            continue

        dx = x_clean[valid_mask, keypoint_idx] - prev_x[valid_mask, keypoint_idx]
        dy = y_clean[valid_mask, keypoint_idx] - prev_y[valid_mask, keypoint_idx]
        jump_mask = np.hypot(dx, dy) > jump_threshold
        if np.any(jump_mask):
            invalid_idx = np.where(valid_mask)[0][jump_mask]
            x_clean[invalid_idx, keypoint_idx] = np.nan
            y_clean[invalid_idx, keypoint_idx] = np.nan
            likelihood_clean[invalid_idx, keypoint_idx] = np.nan

    if check_left_right_crossing:
        keypoint_to_idx = {name: idx for idx, name in enumerate(keypoints_names)}
        for right_name, left_name in LEFT_RIGHT_PAIRS:
            if right_name not in keypoint_to_idx or left_name not in keypoint_to_idx:
                continue
            right_idx = keypoint_to_idx[right_name]
            left_idx = keypoint_to_idx[left_name]
            for cam_idx in range(x_clean.shape[0]):
                current_valid = np.all(np.isfinite([
                    x_clean[cam_idx, right_idx], x_clean[cam_idx, left_idx],
                    prev_x[cam_idx, right_idx], prev_x[cam_idx, left_idx]
                ]))
                if not current_valid:
                    continue
                previous_dx = prev_x[cam_idx, right_idx] - prev_x[cam_idx, left_idx]
                current_dx = x_clean[cam_idx, right_idx] - x_clean[cam_idx, left_idx]
                if abs(previous_dx) <= crossing_margin_px or abs(current_dx) <= crossing_margin_px:
                    continue
                if np.sign(previous_dx) == np.sign(current_dx):
                    continue

                if np.nan_to_num(likelihood_clean[cam_idx, right_idx], nan=-1.0) <= np.nan_to_num(likelihood_clean[cam_idx, left_idx], nan=-1.0):
                    x_clean[cam_idx, right_idx] = np.nan
                    y_clean[cam_idx, right_idx] = np.nan
                    likelihood_clean[cam_idx, right_idx] = np.nan
                else:
                    x_clean[cam_idx, left_idx] = np.nan
                    y_clean[cam_idx, left_idx] = np.nan
                    likelihood_clean[cam_idx, left_idx] = np.nan

    keypoint_to_idx = {name: idx for idx, name in enumerate(keypoints_names)}
    for point_a, point_b in BONE_PAIRS:
        if point_a not in keypoint_to_idx or point_b not in keypoint_to_idx:
            continue
        idx_a = keypoint_to_idx[point_a]
        idx_b = keypoint_to_idx[point_b]
        for cam_idx in range(x_clean.shape[0]):
            current_valid = np.all(np.isfinite([
                x_clean[cam_idx, idx_a], y_clean[cam_idx, idx_a],
                x_clean[cam_idx, idx_b], y_clean[cam_idx, idx_b],
                prev_x[cam_idx, idx_a], prev_y[cam_idx, idx_a],
                prev_x[cam_idx, idx_b], prev_y[cam_idx, idx_b]
            ]))
            if not current_valid:
                continue

            previous_length = np.hypot(
                prev_x[cam_idx, idx_a] - prev_x[cam_idx, idx_b],
                prev_y[cam_idx, idx_a] - prev_y[cam_idx, idx_b]
            )
            current_length = np.hypot(
                x_clean[cam_idx, idx_a] - x_clean[cam_idx, idx_b],
                y_clean[cam_idx, idx_a] - y_clean[cam_idx, idx_b]
            )
            if previous_length < 1e-6:
                continue
            length_ratio = current_length / previous_length
            if bone_ratio_min <= length_ratio <= bone_ratio_max:
                continue

            if np.nan_to_num(likelihood_clean[cam_idx, idx_a], nan=-1.0) <= np.nan_to_num(likelihood_clean[cam_idx, idx_b], nan=-1.0):
                x_clean[cam_idx, idx_a] = np.nan
                y_clean[cam_idx, idx_a] = np.nan
                likelihood_clean[cam_idx, idx_a] = np.nan
            else:
                x_clean[cam_idx, idx_b] = np.nan
                y_clean[cam_idx, idx_b] = np.nan
                likelihood_clean[cam_idx, idx_b] = np.nan

    return x_clean, y_clean, likelihood_clean


def build_keypoint_neighbors(keypoints_names):
    '''
    Build neighbor lookup for bone consistency checks.
    '''

    neighbors = {name: [] for name in keypoints_names}
    for point_a, point_b in BONE_PAIRS:
        if point_a in neighbors and point_b in neighbors:
            neighbors[point_a].append(point_b)
            neighbors[point_b].append(point_a)
    return neighbors


def update_bone_history(bone_history, keypoints_names, q_points):
    '''
    Keep a short history of valid 3D bone lengths for scoring.
    '''

    if bone_history is None:
        bone_history = {}

    keypoint_to_idx = {name: idx for idx, name in enumerate(keypoints_names)}
    q_points = np.asarray(q_points, dtype=float)
    for point_a, point_b in BONE_PAIRS:
        if point_a not in keypoint_to_idx or point_b not in keypoint_to_idx:
            continue
        q_a = q_points[keypoint_to_idx[point_a]]
        q_b = q_points[keypoint_to_idx[point_b]]
        if not np.all(np.isfinite(q_a)) or not np.all(np.isfinite(q_b)):
            continue

        pair_key = tuple(sorted((point_a, point_b)))
        bone_history.setdefault(pair_key, [])
        bone_history[pair_key].append(float(np.linalg.norm(q_a - q_b)))
        bone_history[pair_key] = bone_history[pair_key][-20:]

    return bone_history


def compute_candidate_score(config_dict, keypoint_name, q_candidate, reprojection_error, nb_cams_excluded,
                            previous_point=None, current_points_3d=None, current_point_names=None,
                            bone_history=None, keypoint_neighbors=None):
    '''
    Combine reprojection, temporal continuity, bone consistency, and camera-drop penalties.
    '''

    temporal_cfg = config_dict.get('triangulation', {}).get('temporal', {})
    bone_cfg = config_dict.get('triangulation', {}).get('bone_consistency', {})

    score = float(reprojection_error)

    if temporal_cfg.get('enabled', True) and previous_point is not None and np.all(np.isfinite(previous_point)) and np.all(np.isfinite(q_candidate)):
        distance_weight = float(temporal_cfg.get('distance_weight', 1.0))
        temporal_scale = float(temporal_cfg.get('distance_scale', 50.0))
        strict_keypoints = set(temporal_cfg.get('keypoints_strict', list(STRICT_TRAJECTORY_KEYPOINTS)))
        strict_multiplier = float(temporal_cfg.get('strict_multiplier', 1.5))
        penalty = np.linalg.norm(np.asarray(q_candidate) - np.asarray(previous_point)) * temporal_scale
        if keypoint_name in strict_keypoints:
            penalty *= strict_multiplier
        score += distance_weight * penalty

    score += float(config_dict.get('triangulation', {}).get('camera_drop_penalty', 0.5)) * nb_cams_excluded

    if not bone_cfg.get('enabled', True):
        return score

    if bone_history is None or current_points_3d is None or current_point_names is None:
        return score

    keypoint_neighbors = keypoint_neighbors or {}
    length_weight = float(bone_cfg.get('length_weight', 1.0))
    reject_ratio = float(bone_cfg.get('reject_if_ratio_exceeds', 1.35))
    length_scale = float(bone_cfg.get('length_scale', 20.0))
    name_to_index = {name: idx for idx, name in enumerate(current_point_names)}
    for neighbor_name in keypoint_neighbors.get(keypoint_name, []):
        neighbor_idx = name_to_index.get(neighbor_name)
        if neighbor_idx is None or neighbor_idx >= len(current_points_3d):
            continue
        neighbor_point = np.asarray(current_points_3d[neighbor_idx], dtype=float)
        if not np.all(np.isfinite(neighbor_point)):
            continue

        pair_key = tuple(sorted((keypoint_name, neighbor_name)))
        reference_lengths = bone_history.get(pair_key, [])
        if len(reference_lengths) == 0:
            continue
        reference_length = float(np.median(reference_lengths))
        if reference_length <= 1e-6:
            continue

        current_length = float(np.linalg.norm(np.asarray(q_candidate) - neighbor_point))
        length_ratio = max(current_length / reference_length, reference_length / max(current_length, 1e-9))
        if length_ratio > reject_ratio:
            return np.inf
        score += length_weight * abs(current_length - reference_length) / reference_length * length_scale

    return score


def make_trc(config_dict, Q, keypoints_names, id_person=-1):
    '''
    Make Opensim compatible trc file from a dataframe with 3D coordinates

    INPUT:
    - config_dict: dictionary of configuration parameters
    - Q: pandas dataframe with 3D coordinates as columns, frame number as rows
    - keypoints_names: list of strings

    OUTPUT:
    - trc file
    '''

    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    multi_person = config_dict.get('project').get('multi_person')
    if multi_person:
        seq_name = f'{os.path.basename(os.path.realpath(project_dir))}_P{id_person}'
    else:
        seq_name = f'{os.path.basename(os.path.realpath(project_dir))}'
    pose3d_dir = os.path.join(project_dir, 'pose-3d')

    # Get frame_rate
    video_dir = os.path.join(project_dir, 'videos')
    vid_img_extension = config_dict['pose']['vid_img_extension']
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

    trc_f = f'{seq_name}_{Q.index[0]}-{Q.index[-1]}.trc'

    #Header
    DataRate = CameraRate = OrigDataRate = frame_rate
    NumFrames = len(Q)
    NumMarkers = len(keypoints_names)
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + trc_f, 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, Q.index[0], NumFrames])),
            'Frame#\tTime\t' + '\t\t\t'.join(keypoints_names) + '\t\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(keypoints_names))]) + '\t']
    
    # Zup to Yup coordinate system
    Q = zup2yup(Q)
    
    #Add Frame# and Time columns
    Q.insert(0, 't', Q.index/ frame_rate)
    # Q = Q.fillna(' ')

    #Write file
    if not os.path.exists(pose3d_dir): os.mkdir(pose3d_dir)
    trc_path = os.path.realpath(os.path.join(pose3d_dir, trc_f))
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        Q.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')

    return trc_path


def retrieve_right_trc_order(trc_paths):
    '''
    Lets the user input which static file correspond to each generated trc file.
    
    INPUT:
    - trc_paths: list of strings
    
    OUTPUT:
    - trc_id: list of integers
    '''
    
    logging.info('\n\nReordering trc file IDs:')
    logging.info(f'\nPlease visualize the generated trc files in Blender or OpenSim.\nTrc files are stored in {os.path.dirname(trc_paths[0])}.\n')
    retry = True
    while retry:
        retry = False
        logging.info('List of trc files:')
        [logging.info(f'#{t_list}: {os.path.basename(trc_list)}') for t_list, trc_list in enumerate(trc_paths)]
        trc_id = []
        for t, trc_p in enumerate(trc_paths):
            logging.info(f'\nStatic trial #{t} corresponds to trc number:')
            trc_id += [input('Enter ID:')]
        
        # Check non int and duplicates
        try:
            trc_id = [int(t) for t in trc_id]
            duplicates_in_input = (len(trc_id) != len(set(trc_id)))
            if duplicates_in_input:
                retry = True
                print('\n\nWARNING: Same ID entered twice: please check IDs again.\n')
        except:
            print('\n\nWARNING: The ID must be an integer: please check IDs again.\n')
            retry = True
    
    return trc_id


def recap_triangulate(config_dict, error, nb_cams_excluded, keypoints_names, cam_excluded_count, interp_frames, non_interp_frames, f_range_trimmed, f_range, trc_paths):
    '''
    Print a message giving statistics on reprojection errors (in pixel and in m)
    as well as the number of cameras that had to be excluded to reach threshold 
    conditions. Also stored in User/logs.txt.

    INPUT:
    - a Config.toml file
    - error: dataframe 
    - nb_cams_excluded: dataframe
    - keypoints_names: list of strings

    OUTPUT:
    - Message in console
    '''

    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    session_dir = resolve_session_dir(project_dir)
    calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, c)) and  'calib' in c.lower()][0]
    calib_files = glob.glob(os.path.join(calib_dir, '*.toml'))
    calib_file = max(calib_files, key=os.path.getctime) # lastly created calibration file
    calib = toml.load(calib_file)
    cal_keys = [c for c in calib.keys() 
            if c not in ['metadata', 'capture_volume', 'charuco', 'checkerboard'] 
            and isinstance(calib[c],dict)]
    cam_names = np.array([calib[c].get('name') if calib[c].get('name') else c for c in cal_keys])
    cam_names = cam_names[list(cam_excluded_count[0].keys())]
    error_threshold_triangulation = config_dict.get('triangulation').get('reproj_error_threshold_triangulation')
    likelihood_threshold = config_dict.get('triangulation').get('likelihood_threshold_triangulation')
    min_chunk_size = config_dict.get('triangulation').get('min_chunk_size', 10)
    show_interp_indices = config_dict.get('triangulation').get('show_interp_indices')
    interpolation_kind = config_dict.get('triangulation').get('interpolation')
    interp_gap_smaller_than = config_dict.get('triangulation').get('interp_if_gap_smaller_than')
    fill_large_gaps_with = config_dict.get('triangulation').get('fill_large_gaps_with')
    make_c3d = config_dict.get('triangulation').get('make_c3d')
    handle_LR_swap = config_dict.get('triangulation').get('handle_LR_swap')
    undistort_points = config_dict.get('triangulation').get('undistort_points')
    
    # Recap
    calib_cam1 = calib[cal_keys[0]]
    fm = calib_cam1['matrix'][0][0]
    Dm = euclidean_distance(calib_cam1['translation'], [0,0,0])

    logging.info('')
    nb_persons_to_detect = len(error)
    for n in range(nb_persons_to_detect):
        first_run_start_min, last_run_end_max = f_range_trimmed[n]
        if last_run_end_max - first_run_start_min <= min_chunk_size:
            continue

        if nb_persons_to_detect > 1:
            logging.info(f'\n\nPARTICIPANT {n}\n')
        
        for idx, name in enumerate(keypoints_names):
            mean_error_keypoint_px = np.around(error[n].iloc[:,idx].mean(), decimals=1) # RMS à la place?
            mean_error_keypoint_m = np.around(mean_error_keypoint_px * Dm / fm, decimals=3)
            mean_cam_excluded_keypoint = np.around(nb_cams_excluded[n].iloc[:,idx].mean(), decimals=2)
            logging.info(f'Mean reprojection error for {name} is {mean_error_keypoint_px} px (~ {mean_error_keypoint_m} m), reached with {mean_cam_excluded_keypoint} excluded cameras. ')
            if show_interp_indices:
                if interpolation_kind != 'none':
                    if len(list(interp_frames[n][idx])) == 0 and len(list(non_interp_frames[n][idx])) == 0:
                        logging.info(f'  No frames needed to be interpolated.')
                    if len(list(interp_frames[n][idx]))>0: 
                        interp_str = str(interp_frames[n][idx]).replace(":", " to ").replace("'", "").replace("]", "").replace("[", "")
                        logging.info(f'  Frames {interp_str} were interpolated.')
                    if len(list(non_interp_frames[n][idx]))>0:
                        noninterp_str = str(non_interp_frames[n][idx]).replace(":", " to ").replace("'", "").replace("]", "").replace("[", "")
                        logging.info(f'  Frames {noninterp_str} were not interpolated.')
                else:
                    logging.info(f'  No frames were interpolated because \'interpolation_kind\' was set to none. ')
        
        mean_error_px = np.around(error[n]['mean'].mean(), decimals=1)
        mean_error_mm = np.around(mean_error_px * Dm / fm *1000, decimals=1)
        mean_cam_excluded = np.around(nb_cams_excluded[n]['mean'].mean(), decimals=2)

        logging.info(f'\n--> Mean reprojection error for all points on frames {f_range_trimmed[n][0]} to {f_range_trimmed[n][1]} is {mean_error_px} px, which roughly corresponds to {mean_error_mm} mm. ')
        logging.info(f'Cameras were excluded if likelihood was below {likelihood_threshold} and if the reprojection error was above {error_threshold_triangulation} px.') 
        if interpolation_kind != 'none':
            logging.info(f'Gaps were interpolated with {interpolation_kind} method if smaller than {interp_gap_smaller_than} frames. Larger gaps were filled with {["the last valid value" if fill_large_gaps_with == "last_value" else "zeros" if fill_large_gaps_with == "zeros" else "NaNs"][0]}.') 
        logging.info(f'In average, {mean_cam_excluded} cameras had to be excluded to reach these thresholds.')
        if len(range(*f_range_trimmed[n])) < len(range(*f_range)):
            logging.warning(f'\nSome frames could not be correctly triangulated: trial trimmed between frames {f_range_trimmed[n]}.\n' +
                         'You might need to tweak the triangulation parameters in Config.toml (for example, try increasing "reproj_error_threshold_triangulation").')
        
        cam_excluded_count[n] = {i: v for i, v in zip(cam_names, cam_excluded_count[n].values())}
        cam_excluded_count[n] = {k: v for k, v in sorted(cam_excluded_count[n].items(), key=lambda item: item[1])[::-1]}
        str_cam_excluded_count = ''
        for i, (k, v) in enumerate(cam_excluded_count[n].items()):
            if i ==0:
                 str_cam_excluded_count += f'Camera {k} was excluded {int(np.round(v*100))}% of the time, '
            elif i == len(cam_excluded_count[n])-1:
                str_cam_excluded_count += f'and Camera {k}: {int(np.round(v*100))}%.'
            else:
                str_cam_excluded_count += f'Camera {k}: {int(np.round(v*100))}%, '
        logging.info(str_cam_excluded_count)
        logging.info(f'3D coordinates are stored at {trc_paths[n]}.')
        
    logging.info('\n\n')
    if make_c3d:
        logging.info('All trc files have been converted to c3d.')
    logging.info(f'Limb swapping was {"handled" if handle_LR_swap else "not handled"}.')
    logging.info(f'Lens distortions were {"taken into account" if undistort_points else "not taken into account"}.')


def triangulation_from_best_cameras(config_dict, coords_2D_kpt, coords_2D_kpt_swapped, projection_matrices, calib_params,
                                    keypoint_name=None, previous_point_3d=None, current_points_3d=None,
                                    current_point_names=None, bone_history=None, keypoint_neighbors=None):
    '''
    Triangulates 2D keypoint coordinates. If reprojection error is above threshold,
    tries swapping left and right sides. If still above, removes a camera until error
    is below threshold unless the number of remaining cameras is below a predefined number.

    1. Creates subset with N cameras excluded 
    2. Tries all possible triangulations
    3. Chooses the one with smallest reprojection error
    If error too big, take off one more camera.
        If then below threshold, retain result.
        If better but still too big, take off one more camera.
    
    INPUTS:
    - a Config.toml file
    - coords_2D_kpt: (x,y,likelihood) * ncams array
    - coords_2D_kpt_swapped: (x,y,likelihood) * ncams array  with left/right swap
    - projection_matrices: list of arrays

    OUTPUTS:
    - Q: array of triangulated point (x,y,z,1.)
    - error_min: float
    - nb_cams_excluded: int
    '''
    
    # Read config_dict
    error_threshold_triangulation = get_reprojection_threshold(config_dict, keypoint_name)
    min_cameras_for_triangulation = config_dict.get('triangulation').get('min_cameras_for_triangulation')
    handle_LR_swap = config_dict.get('triangulation').get('handle_LR_swap')
    allowed_lr_swap = set(config_dict.get('triangulation', {}).get('lr_swap_keypoints', list(DEFAULT_LR_SWAP_KEYPOINTS)))
    if keypoint_name is not None and keypoint_name not in allowed_lr_swap:
        handle_LR_swap = False

    undistort_points = config_dict.get('triangulation').get('undistort_points')
    if undistort_points:
        calib_params_K = calib_params['K']
        calib_params_dist = calib_params['dist']
        calib_params_R = calib_params['R']
        calib_params_T = calib_params['T']

    # Initialize
    x_files, y_files, likelihood_files = coords_2D_kpt
    x_files_swapped, y_files_swapped, likelihood_files_swapped = coords_2D_kpt_swapped
    n_cams = len(x_files)
    error_min = np.inf
    best_score = np.inf
    
    nb_cams_off = 0 # cameras will be taken-off until reprojection error is under threshold
    # print('\n')
    while error_min > error_threshold_triangulation and n_cams - nb_cams_off >= min_cameras_for_triangulation:
        # print("error min ", error_min, "thresh ", error_threshold_triangulation, 'nb_cams_off ', nb_cams_off)
        # Create subsets with "nb_cams_off" cameras excluded
        id_cams_off = cached_camera_combinations(n_cams, nb_cams_off)
        
        if undistort_points:
            calib_params_K_filt = [calib_params_K]*len(id_cams_off)
            calib_params_dist_filt = [calib_params_dist]*len(id_cams_off)
            calib_params_R_filt = [calib_params_R]*len(id_cams_off)
            calib_params_T_filt = [calib_params_T]*len(id_cams_off)
        projection_matrices_filt = [projection_matrices]*len(id_cams_off)

        x_files_filt = np.vstack([x_files.copy()]*len(id_cams_off))
        y_files_filt = np.vstack([y_files.copy()]*len(id_cams_off))
        x_files_swapped_filt = np.vstack([x_files_swapped.copy()]*len(id_cams_off))
        y_files_swapped_filt = np.vstack([y_files_swapped.copy()]*len(id_cams_off))
        likelihood_files_filt = np.vstack([likelihood_files.copy()]*len(id_cams_off))
        
        if nb_cams_off > 0:
            for i in range(len(id_cams_off)):
                excluded_idx = list(id_cams_off[i])
                x_files_filt[i][excluded_idx] = np.nan
                y_files_filt[i][excluded_idx] = np.nan
                x_files_swapped_filt[i][excluded_idx] = np.nan
                y_files_swapped_filt[i][excluded_idx] = np.nan
                likelihood_files_filt[i][excluded_idx] = np.nan
        
        # Excluded cameras index and count
        id_cams_off_tot_new = [np.argwhere(np.isnan(x)).ravel() for x in likelihood_files_filt]
        nb_cams_excluded_filt = [np.count_nonzero(np.nan_to_num(x)==0) for x in likelihood_files_filt] # count nans and zeros
        nb_cams_off_tot = max(nb_cams_excluded_filt)
        # print('likelihood_files_filt ',likelihood_files_filt)
        # print('nb_cams_excluded_filt ', nb_cams_excluded_filt, 'nb_cams_off_tot ', nb_cams_off_tot)
        if nb_cams_off_tot > n_cams - min_cameras_for_triangulation:
            break
        id_cams_off_tot = id_cams_off_tot_new
        
        # print('still in loop')
        if undistort_points:
            calib_params_K_filt = [ [ c[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, c in enumerate(calib_params_K_filt) ]
            calib_params_dist_filt = [ [ c[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, c in enumerate(calib_params_dist_filt) ]
            calib_params_R_filt = [ [ c[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, c in enumerate(calib_params_R_filt) ]
            calib_params_T_filt = [ [ c[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, c in enumerate(calib_params_T_filt) ]
        projection_matrices_filt = [ [ p[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, p in enumerate(projection_matrices_filt) ]
        
        # print('\nnb_cams_off', repr(nb_cams_off), 'nb_cams_excluded', repr(nb_cams_excluded_filt))
        # print('likelihood_files ', repr(likelihood_files))
        # print('y_files ', repr(y_files))
        # print('x_files ', repr(x_files))
        # print('x_files_swapped ', repr(x_files_swapped))
        # print('likelihood_files_filt ', repr(likelihood_files_filt))
        # print('x_files_filt ', repr(x_files_filt))
        # print('id_cams_off_tot ', id_cams_off_tot)
        
        x_files_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(likelihood_files_filt[i][ii]) and not likelihood_files_filt[i][ii]==0. ]) for i,x in enumerate(x_files_filt) ]
        y_files_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(likelihood_files_filt[i][ii]) and not likelihood_files_filt[i][ii]==0. ]) for i,x in enumerate(y_files_filt) ]
        x_files_swapped_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(likelihood_files_filt[i][ii]) and not likelihood_files_filt[i][ii]==0. ]) for i,x in enumerate(x_files_swapped_filt) ]
        y_files_swapped_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(likelihood_files_filt[i][ii]) and not likelihood_files_filt[i][ii]==0. ]) for i,x in enumerate(y_files_swapped_filt) ]
        likelihood_files_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(xx) and not xx==0. ]) for x in likelihood_files_filt ]
        # print('y_files_filt ', repr(y_files_filt))
        # print('x_files_filt ', repr(x_files_filt))
        # Triangulate 2D points
        Q_filt = [weighted_triangulation(projection_matrices_filt[i], x_files_filt[i], y_files_filt[i], likelihood_files_filt[i]) for i in range(len(id_cams_off))]
        
        # Reprojection
        if undistort_points:
            coords_2D_kpt_calc_filt = [np.array([cv2.projectPoints(np.array(Q_filt[i][:-1]), calib_params_R_filt[i][j], calib_params_T_filt[i][j], calib_params_K_filt[i][j], calib_params_dist_filt[i][j])[0].ravel() 
                                        for j in range(n_cams-nb_cams_excluded_filt[i])]) 
                                        for i in range(len(id_cams_off))]
            coords_2D_kpt_calc_filt = [[coords_2D_kpt_calc_filt[i][:,0], coords_2D_kpt_calc_filt[i][:,1]] for i in range(len(id_cams_off))]
        else:
            coords_2D_kpt_calc_filt = [reprojection(projection_matrices_filt[i], Q_filt[i]) for i in range(len(id_cams_off))]
        coords_2D_kpt_calc_filt = np.array(coords_2D_kpt_calc_filt, dtype=object)
        x_calc_filt = coords_2D_kpt_calc_filt[:,0]
        # print('x_calc_filt ', x_calc_filt)
        y_calc_filt = coords_2D_kpt_calc_filt[:,1]
        
        # Reprojection error
        error = []
        for config_off_id in range(len(x_calc_filt)):
            dx = np.asarray(x_files_filt[config_off_id]) - np.asarray(x_calc_filt[config_off_id], dtype=float)
            dy = np.asarray(y_files_filt[config_off_id]) - np.asarray(y_calc_filt[config_off_id], dtype=float)
            error.append(np.mean(np.hypot(dx, dy)))
        # print('error ', error)
            
        # Choosing best triangulation (with min reprojection error)
        # print('\n', error)
        # print('len(error) ', len(error))
        # print('len(x_calc_filt) ', len(x_calc_filt))
        # print('len(likelihood_files_filt) ', len(likelihood_files_filt))
        # print('len(id_cams_off_tot) ', len(id_cams_off_tot))
        # print('min error ', np.nanmin(error))
        # print('argmin error ', np.nanargmin(error))
        scores = [
            compute_candidate_score(
                config_dict,
                keypoint_name,
                Q_filt[idx][:-1],
                error[idx],
                nb_cams_excluded_filt[idx],
                previous_point=previous_point_3d,
                current_points_3d=current_points_3d,
                current_point_names=current_point_names,
                bone_history=bone_history,
                keypoint_neighbors=keypoint_neighbors
            )
            for idx in range(len(error))
        ]
        best_cams = int(np.nanargmin(scores))
        best_score = scores[best_cams]
        error_min = error[best_cams]
        nb_cams_excluded = nb_cams_excluded_filt[best_cams]
        
        Q = Q_filt[best_cams][:-1]


        # Swap left and right sides if reprojection error still too high
        if handle_LR_swap and error_min > error_threshold_triangulation:
            # print('handle')
            n_cams_swapped = 1
            error_off_swap_min = error_min
            while error_off_swap_min > error_threshold_triangulation and n_cams_swapped < (n_cams - nb_cams_off_tot) / 2: # more than half of the cameras switched: may triangulate twice the same side
                # print('SWAP: nb_cams_off ', nb_cams_off, 'n_cams_swapped ', n_cams_swapped, 'nb_cams_off_tot ', nb_cams_off_tot)
                # Create subsets 
                id_cams_swapped = cached_camera_combinations(n_cams-nb_cams_off_tot, n_cams_swapped)
                # print('id_cams_swapped ', id_cams_swapped)
                x_files_filt_off_swap = [[x] * len(id_cams_swapped) for x in x_files_filt]
                y_files_filt_off_swap = [[y] * len(id_cams_swapped) for y in y_files_filt]
                # print('x_files_filt_off_swap ', x_files_filt_off_swap)
                # print('y_files_filt_off_swap ', y_files_filt_off_swap)
                for id_off in range(len(id_cams_off)): # for each configuration with nb_cams_off_tot removed 
                    for id_swapped, config_swapped in enumerate(id_cams_swapped): # for each of these configurations, test all subconfigurations with with n_cams_swapped swapped
                        # print('id_off ', id_off, 'id_swapped ', id_swapped, 'config_swapped ',  config_swapped)
                        x_files_filt_off_swap[id_off][id_swapped][config_swapped] = x_files_swapped_filt[id_off][config_swapped] 
                        y_files_filt_off_swap[id_off][id_swapped][config_swapped] = y_files_swapped_filt[id_off][config_swapped]
                                
                # Triangulate 2D points
                Q_filt_off_swap = np.array([[weighted_triangulation(projection_matrices_filt[id_off], x_files_filt_off_swap[id_off][id_swapped], y_files_filt_off_swap[id_off][id_swapped], likelihood_files_filt[id_off]) 
                                                for id_swapped in range(len(id_cams_swapped))]
                                                for id_off in range(len(id_cams_off))] )
                
                # Reprojection
                if undistort_points:
                    coords_2D_kpt_calc_off_swap = [np.array([[cv2.projectPoints(np.array(Q_filt_off_swap[id_off][id_swapped][:-1]), calib_params_R_filt[id_off][j], calib_params_T_filt[id_off][j], calib_params_K_filt[id_off][j], calib_params_dist_filt[id_off][j])[0].ravel() 
                                                    for j in range(n_cams-nb_cams_off_tot)] 
                                                    for id_swapped in range(len(id_cams_swapped))])
                                                    for id_off in range(len(id_cams_off))]
                    coords_2D_kpt_calc_off_swap = np.array([[[coords_2D_kpt_calc_off_swap[id_off][id_swapped,:,0], coords_2D_kpt_calc_off_swap[id_off][id_swapped,:,1]] 
                                                    for id_swapped in range(len(id_cams_swapped))] 
                                                    for id_off in range(len(id_cams_off))])
                else:
                    coords_2D_kpt_calc_off_swap = [np.array([reprojection(projection_matrices_filt[id_off], Q_filt_off_swap[id_off][id_swapped]) 
                                                    for id_swapped in range(len(id_cams_swapped))])
                                                    for id_off in range(len(id_cams_off))]
                # print(repr(coords_2D_kpt_calc_off_swap))
                x_calc_off_swap = [c[:,0] for c in coords_2D_kpt_calc_off_swap]
                y_calc_off_swap = [c[:,1] for c in coords_2D_kpt_calc_off_swap]
                
                # Reprojection error
                # print('x_files_filt_off_swap ', x_files_filt_off_swap)
                # print('x_calc_off_swap ', x_calc_off_swap)
                error_off_swap = []
                for id_off in range(len(id_cams_off)):
                    error_percam = []
                    for id_swapped, config_swapped in enumerate(id_cams_swapped):
                        dx = np.asarray(x_files_filt_off_swap[id_off][id_swapped]) - np.asarray(x_calc_off_swap[id_off][id_swapped], dtype=float)
                        dy = np.asarray(y_files_filt_off_swap[id_off][id_swapped]) - np.asarray(y_calc_off_swap[id_off][id_swapped], dtype=float)
                        error_percam.append(np.mean(np.hypot(dx, dy)))
                    error_off_swap.append(error_percam)
                error_off_swap = np.array(error_off_swap)
                score_off_swap = np.full(error_off_swap.shape, np.inf, dtype=float)
                for id_off in range(len(id_cams_off)):
                    for id_swapped in range(len(id_cams_swapped)):
                        score_off_swap[id_off, id_swapped] = compute_candidate_score(
                            config_dict,
                            keypoint_name,
                            Q_filt_off_swap[id_off][id_swapped][:-1],
                            error_off_swap[id_off, id_swapped],
                            nb_cams_excluded_filt[id_off],
                            previous_point=previous_point_3d,
                            current_points_3d=current_points_3d,
                            current_point_names=current_point_names,
                            bone_history=bone_history,
                            keypoint_neighbors=keypoint_neighbors
                        )

                best_off_swap_config = np.unravel_index(np.argmin(score_off_swap), score_off_swap.shape)
                error_off_swap_min = error_off_swap[best_off_swap_config]
                best_off_swap_score = score_off_swap[best_off_swap_config]
                
                id_off_cams = best_off_swap_config[0]
                id_swapped_cams = id_cams_swapped[best_off_swap_config[1]]
                Q_best = Q_filt_off_swap[best_off_swap_config][:-1]

                n_cams_swapped += 1

            if best_off_swap_score < best_score:
                best_score = best_off_swap_score
                error_min = error_off_swap_min
                best_cams = id_off_cams
                Q = Q_best
        
        # print(error_min)
        
        nb_cams_off += 1
    
    # Index of excluded cams for this keypoint
    # print('Loop ended')
    
    if 'best_cams' in locals():
        # print(id_cams_off_tot)
        # print('len(id_cams_off_tot) ', len(id_cams_off_tot))
        # print('id_cams_off_tot ', id_cams_off_tot)
        id_excluded_cams = id_cams_off_tot[best_cams]
        # print('id_excluded_cams ', id_excluded_cams)
    else:
        id_excluded_cams = list(range(n_cams))
        nb_cams_excluded = n_cams
    # print('id_excluded_cams ', id_excluded_cams)
    
    # If triangulation not successful, error = nan,  and 3D coordinates as missing values
    if error_min > error_threshold_triangulation:
        error_min = np.nan
        Q = np.array([np.nan, np.nan, np.nan])
        
    return Q, error_min, nb_cams_excluded, id_excluded_cams


def extract_files_frame_f(json_tracked_files_f, keypoints_ids, nb_persons_to_detect):
    '''
    Extract data from json files for frame f, 
    in the order of the body model hierarchy.

    INPUTS:
    - json_tracked_files_f: list of str. Paths of json_files for frame f.
    - keypoints_ids: list of int. Keypoints IDs in the order of the hierarchy.
    - nb_persons_to_detect: int

    OUTPUTS:
    - x_files, y_files, likelihood_files: [[[list of coordinates] * n_cams ] * nb_persons_to_detect]
    '''

    n_cams = len(json_tracked_files_f)
    
    x_files = [[] for n in range(nb_persons_to_detect)]
    y_files = [[] for n in range(nb_persons_to_detect)]
    likelihood_files = [[] for n in range(nb_persons_to_detect)]
    for n in range(nb_persons_to_detect):
        for cam_nb in range(n_cams):
            x_files_cam, y_files_cam, likelihood_files_cam = [], [], []
            try:
                with open(json_tracked_files_f[cam_nb], 'r') as json_f:
                    js = json.load(json_f)
                    for keypoint_id in keypoints_ids:
                        try:
                            x_files_cam.append( js['people'][n]['pose_keypoints_2d'][keypoint_id*3] )
                            y_files_cam.append( js['people'][n]['pose_keypoints_2d'][keypoint_id*3+1] )
                            likelihood_files_cam.append( js['people'][n]['pose_keypoints_2d'][keypoint_id*3+2] )
                        except:
                            x_files_cam.append( np.nan )
                            y_files_cam.append( np.nan )
                            likelihood_files_cam.append( np.nan )
            except:
                x_files_cam = [np.nan] * len(keypoints_ids)
                y_files_cam = [np.nan] * len(keypoints_ids)
                likelihood_files_cam = [np.nan] * len(keypoints_ids)
            x_files[n].append(x_files_cam)
            y_files[n].append(y_files_cam)
            likelihood_files[n].append(likelihood_files_cam)
        
    x_files = np.array(x_files)
    y_files = np.array(y_files)
    likelihood_files = np.array(likelihood_files)

    return x_files, y_files, likelihood_files


def triangulate_all(config_dict):
    '''
    For each frame
    For each keypoint
    - Triangulate keypoint
    - Reproject it on all cameras
    - Take off cameras until requirements are met
    Interpolate missing values
    Create trc file
    Print recap message
    
     INPUTS: 
    - a calibration file (.toml extension)
    - json files for each camera with indices matching the detected persons
    - a Config.toml file
    - a skeleton model
    
    OUTPUTS: 
    - a .trc file with 3D coordinates in Y-up system coordinates 
    '''
    
    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    session_dir = resolve_session_dir(project_dir)
    multi_person = config_dict.get('project').get('multi_person')
    pose_model = config_dict.get('pose').get('pose_model')
    frame_range = config_dict.get('project').get('frame_range')
    likelihood_threshold = config_dict.get('triangulation').get('likelihood_threshold_triangulation')
    interpolation_kind = config_dict.get('triangulation').get('interpolation')
    interp_gap_smaller_than = config_dict.get('triangulation').get('interp_if_gap_smaller_than')
    max_distance_m = config_dict.get('triangulation').get('max_distance_m', None)
    remove_incomplete_frames = config_dict.get('triangulation').get('remove_incomplete_frames', False)
    sections_to_keep = config_dict.get('triangulation').get('sections_to_keep')
    min_chunk_size = config_dict.get('triangulation').get('min_chunk_size', 10)
    fill_large_gaps_with = config_dict.get('triangulation').get('fill_large_gaps_with')
    show_interp_indices = config_dict.get('triangulation').get('show_interp_indices')
    undistort_points = config_dict.get('triangulation').get('undistort_points')
    make_c3d = config_dict.get('triangulation').get('make_c3d')
    
    try:
        calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, c)) and  'calib' in c.lower()][0]
    except:
        raise Exception(f'No .toml calibration direcctory found.')
    try:
        calib_files = glob.glob(os.path.join(calib_dir, '*.toml'))
        calib_file = max(calib_files, key=os.path.getctime) # lastly created calibration file
    except:
        raise Exception(f'No .toml calibration file found in the {calib_dir}.')
    pose_dir = os.path.join(project_dir, 'pose')
    poseSync_dir = os.path.join(project_dir, 'pose-sync')
    poseTracked_dir = os.path.join(project_dir, 'pose-associated')
    
    # Projection matrix from toml calibration file
    P = computeP(calib_file, undistort=undistort_points)
    calib_params = retrieve_calib_params(calib_file)
        
    # Retrieve keypoints from model
    try: # from skeletons.py
        if pose_model.upper() == 'BODY_WITH_FEET': pose_model = 'HALPE_26'
        elif pose_model.upper() == 'WHOLE_BODY_WRIST': pose_model = 'COCO_133_WRIST'
        elif pose_model.upper() == 'WHOLE_BODY': pose_model = 'COCO_133'
        elif pose_model.upper() == 'BODY': pose_model = 'COCO_17'
        elif pose_model.upper() == 'HAND': pose_model = 'HAND_21'
        elif pose_model.upper() == 'FACE': pose_model = 'FACE_106'
        elif pose_model.upper() == 'ANIMAL': pose_model = 'ANIMAL2D_17'
        else: pass
        model = eval(pose_model)
    except:
        try: # from Config.toml
            model = import_pose_model_from_config(config_dict, pose_model)
        except:
            raise NameError('{pose_model} not found in skeletons.py nor in Config.toml')
            
    keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_idx = list(range(len(keypoints_ids)))
    keypoints_nb = len(keypoints_ids)
    keypoint_neighbors = build_keypoint_neighbors(keypoints_names)
    # for pre, _, node in RenderTree(model): 
    #     print(f'{pre}{node.name} id={node.id}')
    
    # left/right swapped keypoints
    try:
        keypoints_names_swapped = ['L'+keypoint_name[1:] if keypoint_name.startswith('R') else 'R'+keypoint_name[1:] if keypoint_name.startswith('L') else keypoint_name for keypoint_name in keypoints_names]
        keypoints_names_swapped = [keypoint_name_swapped.replace('right', 'left') if keypoint_name_swapped.startswith('right') else keypoint_name_swapped.replace('left', 'right') if keypoint_name_swapped.startswith('left') else keypoint_name_swapped for keypoint_name_swapped in keypoints_names_swapped]
        keypoints_idx_swapped = [keypoints_names.index(keypoint_name_swapped) for keypoint_name_swapped in keypoints_names_swapped] # find index of new keypoint_name
    except:
        keypoints_names_swapped = keypoints_names
        keypoints_idx_swapped = keypoints_idx
        logging.warning('No left/right swap was performed.')
    
    # 2d-pose files selection
    try:
        pose_listdirs_names = next(os.walk(pose_dir))[1]
        os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
    except:
        raise ValueError(f'No json files found in {pose_dir} subdirectories. Make sure you run Pose2Sim.poseEstimation() first.')
    pose_listdirs_names = sort_stringlist_by_last_number(pose_listdirs_names)
    json_dirs_names = [k for k in pose_listdirs_names if 'json' in k]
    n_cams = len(json_dirs_names)
    try: 
        json_files_names = [fnmatch.filter(os.listdir(os.path.join(poseTracked_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
        pose_dir = poseTracked_dir
    except:
        try: 
            json_files_names = [fnmatch.filter(os.listdir(os.path.join(poseSync_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
            pose_dir = poseSync_dir
        except:
            try:
                json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
            except:
                raise Exception(f'No json files found in {pose_dir}, {poseSync_dir}, nor {poseTracked_dir} subdirectories. Make sure you run Pose2Sim.poseEstimation() first.')
    json_files_names = [sort_stringlist_by_last_number(js) for js in json_files_names]    

    # frame range selection
    f_range = [[0,min([len(j) for j in json_files_names])] if frame_range in ('all', 'auto', []) else frame_range][0]
    frame_nb = f_range[1] - f_range[0]
    
    # Check that camera number is consistent between calibration file and pose folders
    if n_cams != len(P):
        raise Exception(f'Error: The number of cameras is not consistent: Found {len(P)} cameras in the calibration file, and {n_cams} cameras based on the number of pose folders.')
    
    # Triangulation
    if multi_person:
        nb_persons_to_detect = max(max(count_persons_in_json(os.path.join(pose_dir, json_dirs_names[c], json_fname)) for json_fname in json_files_names[c]) for c in range(n_cams))
    else:
        nb_persons_to_detect = 1

    Q = [[[np.nan]*3]*keypoints_nb for n in range(nb_persons_to_detect)]
    Q_old = [[[np.nan]*3]*keypoints_nb for n in range(nb_persons_to_detect)]
    prev_coords_2d = [None for _ in range(nb_persons_to_detect)]
    bone_histories = [dict() for _ in range(nb_persons_to_detect)]
    error = [[] for n in range(nb_persons_to_detect)]
    nb_cams_excluded = [[] for n in range(nb_persons_to_detect)]
    id_excluded_cams = [[] for n in range(nb_persons_to_detect)]
    Q_tot, error_tot, nb_cams_excluded_tot, cam_excluded_count, id_excluded_cams_tot = [], [], [], [], []
    interp_frames, non_interp_frames, f_range_trimmed = [], [], []
    trc_paths, c3d_paths = [], []
    for f in tqdm(range(*f_range)):
        # print(f'\nFrame {f}:')        
        # Get x,y,likelihood values from files
        json_files_names_f = [[j for j in json_files_names[c] if int(re.split(r'(\d+)',j)[-2])==f] for c in range(n_cams)]
        json_files_names_f = [j for j_list in json_files_names_f for j in (j_list or ['none'])]
        json_files_f = [os.path.join(pose_dir, json_dirs_names[c], json_files_names_f[c]) for c in range(n_cams)]

        x_files, y_files, likelihood_files = extract_files_frame_f(json_files_f, keypoints_ids, nb_persons_to_detect)
        # [[[list of coordinates] * n_cams ] * nb_persons_to_detect]
        # vs. [[list of coordinates] * n_cams ] 
        
        # undistort points
        if undistort_points:
            for n in range(nb_persons_to_detect):
                points = [np.array(tuple(zip(x_files[n][i],y_files[n][i]))).reshape(-1, 1, 2).astype('float32') for i in range(n_cams)]
                undistorted_points = [cv2.undistortPoints(points[i], calib_params['K'][i], calib_params['dist'][i], None, calib_params['optim_K'][i]) for i in range(n_cams)]
                x_files[n] =  np.array([[u[i][0][0] for i in range(len(u))] for u in undistorted_points])
                y_files[n] =  np.array([[u[i][0][1] for i in range(len(u))] for u in undistorted_points])
                # This is good for slight distortion. For fisheye camera, the model does not work anymore. See there for an example https://github.com/lambdaloop/aniposelib/blob/d03b485c4e178d7cff076e9fe1ac36837db49158/aniposelib/cameras.py#L301

        # Replace likelihood by 0 if under likelihood_threshold
        with np.errstate(invalid='ignore'):
            for n in range(nb_persons_to_detect):
                x_files[n][likelihood_files[n] < likelihood_threshold] = np.nan
                y_files[n][likelihood_files[n] < likelihood_threshold] = np.nan
                likelihood_files[n][likelihood_files[n] < likelihood_threshold] = np.nan

        for n in range(nb_persons_to_detect):
            prev_coords = None if multi_person else prev_coords_2d[n]
            x_files[n], y_files[n], likelihood_files[n] = clean_2d_keypoints(
                config_dict,
                keypoints_names,
                x_files[n],
                y_files[n],
                likelihood_files[n],
                prev_coords=prev_coords
            )
        
        # Q_old = Q except when it has nan, otherwise it takes the Q_old value
        nan_mask = np.isnan(Q)
        Q_old = np.where(nan_mask, Q_old, Q)
        Q = [[] for n in range(nb_persons_to_detect)]
        error = [[] for n in range(nb_persons_to_detect)]
        nb_cams_excluded = [[] for n in range(nb_persons_to_detect)]
        id_excluded_cams = [[] for n in range(nb_persons_to_detect)]
        
        for n in range(nb_persons_to_detect):
            for keypoint_idx in keypoints_idx:
            # keypoints_nb = 2
            # for keypoint_idx in range(2):
            # Triangulate cameras with min reprojection error
                # print('\n', keypoints_names[keypoint_idx])
                coords_2D_kpt = np.array( (x_files[n][:, keypoint_idx], y_files[n][:, keypoint_idx], likelihood_files[n][:, keypoint_idx]) )
                coords_2D_kpt_swapped = np.array(( x_files[n][:, keypoints_idx_swapped[keypoint_idx]], y_files[n][:, keypoints_idx_swapped[keypoint_idx]], likelihood_files[n][:, keypoints_idx_swapped[keypoint_idx]] ))
                previous_point_3d = np.asarray(Q_old[n][keypoint_idx], dtype=float)
                if not np.all(np.isfinite(previous_point_3d)):
                    previous_point_3d = None

                Q_kpt, error_kpt, nb_cams_excluded_kpt, id_excluded_cams_kpt = triangulation_from_best_cameras(
                    config_dict,
                    coords_2D_kpt,
                    coords_2D_kpt_swapped,
                    P,
                    calib_params,
                    keypoint_name=keypoints_names[keypoint_idx],
                    previous_point_3d=previous_point_3d,
                    current_points_3d=Q[n],
                    current_point_names=keypoints_names,
                    bone_history=bone_histories[n],
                    keypoint_neighbors=keypoint_neighbors
                ) # P has been modified if undistort_points=True

                Q[n].append(Q_kpt)
                error[n].append(error_kpt)
                nb_cams_excluded[n].append(nb_cams_excluded_kpt)
                id_excluded_cams[n].append(id_excluded_cams_kpt)

            prev_coords_2d[n] = (
                np.array(x_files[n], copy=True),
                np.array(y_files[n], copy=True),
                np.array(likelihood_files[n], copy=True)
            )
            bone_histories[n] = update_bone_history(bone_histories[n], keypoints_names, Q[n])
        
        if multi_person:
            # reID persons across frames by checking the distance from one frame to another
            # print('Q before ordering ', np.array(Q)[:,:2])
            if f !=0:
                Q = np.array(Q)
                Q_old, Q, sorted_ids = sort_people_sports2d(Q_old, np.array(Q), max_dist=max_distance_m)
                
                error_sorted, nb_cams_excluded_sorted, id_excluded_cams_sorted = [], [], []
                for n in range(nb_persons_to_detect):
                    detection_idx = sorted_ids[n]
                    if detection_idx >= 0:  # Person is detected in current frame
                        error_sorted.append(error[detection_idx])
                        nb_cams_excluded_sorted.append(nb_cams_excluded[detection_idx])
                        id_excluded_cams_sorted.append(id_excluded_cams[detection_idx])
                    else:  # Person is not detected in current frame
                        error_sorted.append([np.nan] * keypoints_nb)
                        nb_cams_excluded_sorted.append([n_cams] * keypoints_nb)
                        id_excluded_cams_sorted.append([list(range(n_cams))] * keypoints_nb)
                error, nb_cams_excluded, id_excluded_cams = error_sorted, nb_cams_excluded_sorted, id_excluded_cams_sorted
        
        # TODO: if distance > threshold, new person
        
        # Add triangulated points, errors and excluded cameras to pandas dataframes
        Q_tot.append([np.concatenate(Q[n]) for n in range(nb_persons_to_detect)])
        error_tot.append([error[n] for n in range(nb_persons_to_detect)])
        nb_cams_excluded_tot.append([nb_cams_excluded[n] for n in range(nb_persons_to_detect)])
        id_excluded_cams = [[id_excluded_cams[n][k] for k in range(keypoints_nb)] for n in range(nb_persons_to_detect)]
        id_excluded_cams_tot.append(id_excluded_cams)
            
    # fill values for if a person that was not initially detected has entered the frame 
    Q_tot = [list(tpl) for tpl in zip(*it.zip_longest(*Q_tot, fillvalue=[np.nan]*keypoints_nb*3))]
    error_tot = [list(tpl) for tpl in zip(*it.zip_longest(*error_tot, fillvalue=[np.nan]*keypoints_nb*3))]
    nb_cams_excluded_tot = [list(tpl) for tpl in zip(*it.zip_longest(*nb_cams_excluded_tot, fillvalue=[np.nan]*keypoints_nb*3))]
    id_excluded_cams_tot = [list(tpl) for tpl in zip(*it.zip_longest(*id_excluded_cams_tot, fillvalue=[np.nan]*keypoints_nb*3))]

    # dataframes for each person
    Q_tot = [pd.DataFrame([Q_tot_f[n] for Q_tot_f in Q_tot], index=range(*f_range)) for n in range(nb_persons_to_detect)]
    error_tot = [pd.DataFrame([error_tot_f[n] for error_tot_f in error_tot], index=range(*f_range)) for n in range(nb_persons_to_detect)]
    nb_cams_excluded_tot = [pd.DataFrame([nb_cams_excluded_tot_f[n] for nb_cams_excluded_tot_f in nb_cams_excluded_tot], index=range(*f_range)) for n in range(nb_persons_to_detect)]
    id_excluded_cams_tot = [pd.DataFrame([id_excluded_cams_tot_f[n] for id_excluded_cams_tot_f in id_excluded_cams_tot], index=range(*f_range)) for n in range(nb_persons_to_detect)]

    # Interpolate small missing sections
    for n in range(nb_persons_to_detect):
        if interpolation_kind != 'none':
            try:
                Q_tot[n] = Q_tot[n].apply(interpolate_zeros_nans, axis=0, args=[interp_gap_smaller_than, interpolation_kind])
            except:
                logging.warning(f'Interpolation was not possible for person {n}. This means that not enough points are available, which is often due to a bad calibration.')

        # Determine frames where the person is out of the frame
        error_tot[n]['mean'] = error_tot[n].mean(axis = 1, skipna = not remove_incomplete_frames)
        nb_cams_excluded_tot[n]['mean'] = nb_cams_excluded_tot[n].mean(axis=1)
        first_run_start_min, last_run_end_max = indices_of_first_last_non_nan_chunks(error_tot[n]['mean'], min_chunk_size=min_chunk_size, chunk_choice_method=sections_to_keep)
        f_range_trimmed.append([first_run_start_min, last_run_end_max])

        # Skip person if not correctly triangulated
        if last_run_end_max - first_run_start_min <= min_chunk_size:
            nb_cams_excluded_tot[n] = pd.DataFrame(columns=nb_cams_excluded_tot[n].columns)
            cam_excluded_count.append({})
            interp_frames.append([])
            non_interp_frames.append([])
            trc_paths.append ('')
            logging.info(f'\nPerson {n}: Less than {min_chunk_size} valid frames in a row. Deleting person.')
            continue

        # Trim around good frames
        Q_tot[n] = Q_tot[n].iloc[first_run_start_min:last_run_end_max]
        error_tot[n] = error_tot[n].iloc[first_run_start_min:last_run_end_max]
        nb_cams_excluded_tot[n] = nb_cams_excluded_tot[n].iloc[first_run_start_min:last_run_end_max]
        id_excluded_cams_tot[n] = id_excluded_cams_tot[n].iloc[first_run_start_min:last_run_end_max]
        zero_nan_frames = np.where( Q_tot[n].iloc[:,::3].T.eq(0) | ~np.isfinite(Q_tot[n].iloc[:,::3].T) )
        zero_nan_frames_per_kpt = [zero_nan_frames[1][np.where(zero_nan_frames[0]==k)[0]] for k in range(keypoints_nb)]
        zero_nan_frames_per_kpt = [z[(first_run_start_min < z) & (last_run_end_max > z)] for z in zero_nan_frames_per_kpt]

        # Fill non-interpolated values with last valid one
        if fill_large_gaps_with == 'last_value':
            Q_tot[n] = Q_tot[n].ffill(axis=0).bfill(axis=0)
            Q_tot[n].replace([np.nan, np.inf], 0, inplace=True)
        elif fill_large_gaps_with == 'zeros':
            Q_tot[n].replace([np.nan, np.inf], 0, inplace=True)

        # Create TRC file
        trc_paths.append(make_trc(config_dict, Q_tot[n], keypoints_names, id_person=n))
        if make_c3d:
            c3d_paths.append(convert_to_c3d(t) for t in trc_paths)

        # IDs of excluded cameras
        frame_count = len(Q_tot[n])
        cam_exclusion_counts = {cam_id: 0 for cam_id in range(n_cams)} # initialize at zero
        for excluded_at_frame in id_excluded_cams_tot[n].values:
            for keypoint_cams in excluded_at_frame:
                if isinstance(keypoint_cams, (list, np.ndarray)):
                    for cam_id in keypoint_cams:
                        if isinstance(cam_id, (int, np.integer)):
                            cam_exclusion_counts[cam_id] += 1
        total_opportunities = frame_count * keypoints_nb
        cam_excluded_count.append({k: v/total_opportunities for k, v in cam_exclusion_counts.items()})

        # Optionally, for each person, for each keypoint, show indices of frames that should be interpolated
        if show_interp_indices:
            gaps = [np.where(np.diff(zero_nan_frames_per_kpt[k]) > 1)[0] + 1 for k in range(keypoints_nb)]
            sequences = [np.split(zero_nan_frames_per_kpt[k], gaps[k]) for k in range(keypoints_nb)]
            interp_frames.append([[f'{seq[0]}:{seq[-1]}' for seq in seq_kpt if len(seq)<=interp_gap_smaller_than and len(seq)>0] for seq_kpt in sequences])
            non_interp_frames.append([[f'{seq[0]}:{seq[-1]}' for seq in seq_kpt if len(seq)>interp_gap_smaller_than] for seq_kpt in sequences])
        else:
            interp_frames.append(None)
            non_interp_frames.append([])

    if np.all(np.diff(np.array(f_range_trimmed))==0):
        raise Exception('No persons have been triangulated. Please check your calibration and your synchronization, or the triangulation parameters in Config.toml.')

    # Recap message
    recap_triangulate(config_dict, error_tot, nb_cams_excluded_tot, keypoints_names, cam_excluded_count, interp_frames, non_interp_frames, f_range_trimmed, f_range, trc_paths)
