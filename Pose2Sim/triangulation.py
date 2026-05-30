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
import time
import pandas as pd
import cv2
import toml
from tqdm import tqdm
from collections import Counter
from collections.abc import Mapping
from anytree import RenderTree
from anytree.importer import DictImporter
from scipy.optimize import least_squares
import logging

from Pose2Sim.common import retrieve_calib_params, computeP, weighted_triangulation, \
    reprojection, euclidean_distance, sort_people_sports2d, interpolate_zeros_nans, \
    sort_stringlist_by_last_number, zup2yup, convert_to_c3d, is_video_file
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


## FUNCTIONS
def _to_picklable_builtin(value):
    '''
    Convert toml parser container subclasses to built-in containers.

    Windows multiprocessing starts fresh Python processes and pickles every
    worker argument. The toml package represents inline tables with a local
    DynamicInlineTableDict class, which cannot be pickled.
    '''

    if isinstance(value, Mapping):
        return {key: _to_picklable_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_picklable_builtin(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_picklable_builtin(item) for item in value)
    return value


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


def _normalize_large_gap_fill_mode(mode, default='nan'):
    mode = str(mode or default).lower()
    if mode in ('last_value', 'nan', 'zeros'):
        return mode
    logging.warning(
        'Invalid large-gap fill mode "%s". Falling back to "%s".',
        mode,
        default,
    )
    return default


def _large_gap_marker_fill_overrides(config_dict, keypoints_names):
    raw_overrides = config_dict.get('triangulation', {}).get(
        'fill_large_gaps_marker_overrides',
        {},
    )
    if raw_overrides in (None, ''):
        return {}
    if not isinstance(raw_overrides, dict):
        logging.warning(
            'triangulation.fill_large_gaps_marker_overrides must be a marker-to-mode table; ignoring it.'
        )
        return {}

    overrides = {}
    for marker_name, fill_mode in raw_overrides.items():
        if marker_name not in keypoints_names:
            logging.warning(
                'Ignoring large-gap fill override for unknown marker "%s".',
                marker_name,
            )
            continue
        overrides[keypoints_names.index(marker_name)] = _normalize_large_gap_fill_mode(fill_mode)
    return overrides


def _apply_large_gap_fill(Q, zero_nan_frames_per_kpt, keypoints_names, fill_large_gaps_with, marker_fill_overrides):
    fill_large_gaps_with = _normalize_large_gap_fill_mode(fill_large_gaps_with)

    if fill_large_gaps_with == 'last_value':
        Q_filled = Q.ffill(axis=0)
        Q_filled.replace([np.inf, -np.inf], np.nan, inplace=True)
    elif fill_large_gaps_with == 'zeros':
        Q_filled = Q.copy()
        Q_filled.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
    else:
        Q_filled = Q.copy()
        Q_filled.replace([np.inf, -np.inf], np.nan, inplace=True)

    for keypoint_id, fill_mode in marker_fill_overrides.items():
        if keypoint_id >= len(keypoints_names) or fill_mode == fill_large_gaps_with:
            continue

        missing_positions = zero_nan_frames_per_kpt[keypoint_id]
        if len(missing_positions) == 0:
            continue

        column_positions = slice(keypoint_id * 3, keypoint_id * 3 + 3)
        if fill_mode == 'last_value':
            Q_filled.iloc[:, column_positions] = Q.iloc[:, column_positions].ffill(axis=0)
        elif fill_mode == 'zeros':
            Q_filled.iloc[missing_positions, column_positions] = 0
        else:
            Q_filled.iloc[missing_positions, column_positions] = np.nan

    Q_filled.replace([np.inf, -np.inf], np.nan, inplace=True)
    return Q_filled


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
    project_dir = config_dict.get('project', {}).get('project_dir', '.')
    multi_person = config_dict.get('project', {}).get('multi_person', False)
    if multi_person:
        seq_name = f'{os.path.basename(os.path.realpath(project_dir))}_P{id_person}'
    else:
        seq_name = f'{os.path.basename(os.path.realpath(project_dir))}'
    pose3d_dir = os.path.join(project_dir, 'pose-3d')

    # Get frame_rate
    video_dir = os.path.join(project_dir, 'videos')
    video_files = sorted([f for f in glob.glob(os.path.join(video_dir, '*')) if is_video_file(f)])
    frame_rate = config_dict.get('project', {}).get('frame_rate', 'auto')
    if frame_rate == 'auto': 
        try:
            cap = cv2.VideoCapture(video_files[0])
            cap.read()
            if cap.read()[0] == False:
                raise
            frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
        except:
            logging.warning(f'Cannot read video. Frame rate will be set to 30 fps.')
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
    # if batch
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
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


def triangulation_from_best_cameras(config_dict, coords_2D_kpt, coords_2D_kpt_swapped, projection_matrices, calib_params):
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
    error_threshold_triangulation = config_dict.get('triangulation', {}).get('reproj_error_threshold_triangulation', 15)
    min_cameras_for_triangulation = config_dict.get('triangulation', {}).get('min_cameras_for_triangulation', 2)
    handle_LR_swap = config_dict.get('triangulation', {}).get('handle_LR_swap', False)

    undistort_points = config_dict.get('triangulation', {}).get('undistort_points', False)
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
    
    nb_cams_off = 0 # cameras will be taken-off until reprojection error is under threshold
    # print('\n')
    while error_min > error_threshold_triangulation and n_cams - nb_cams_off >= min_cameras_for_triangulation:
        # print("error min ", error_min, "thresh ", error_threshold_triangulation, 'nb_cams_off ', nb_cams_off)
        # Create subsets with "nb_cams_off" cameras excluded
        id_cams_off = np.array(list(it.combinations(range(n_cams), nb_cams_off)))
        
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
                x_files_filt[i][id_cams_off[i]] = np.nan
                y_files_filt[i][id_cams_off[i]] = np.nan
                x_files_swapped_filt[i][id_cams_off[i]] = np.nan
                y_files_swapped_filt[i][id_cams_off[i]] = np.nan
                likelihood_files_filt[i][id_cams_off[i]] = np.nan
        
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
            q_file = [(x_files_filt[config_off_id][i], y_files_filt[config_off_id][i]) for i in range(len(x_files_filt[config_off_id]))]
            q_calc = [(x_calc_filt[config_off_id][i], y_calc_filt[config_off_id][i]) for i in range(len(x_calc_filt[config_off_id]))]
            error.append( np.mean( [euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))] ) )
        # print('error ', error)
            
        # Choosing best triangulation (with min reprojection error)
        # print('\n', error)
        # print('len(error) ', len(error))
        # print('len(x_calc_filt) ', len(x_calc_filt))
        # print('len(likelihood_files_filt) ', len(likelihood_files_filt))
        # print('len(id_cams_off_tot) ', len(id_cams_off_tot))
        # print('min error ', np.nanmin(error))
        # print('argmin error ', np.nanargmin(error))
        error_min = np.nanmin(error)
        # print(error_min)
        best_cams = np.nanargmin(error)
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
                id_cams_swapped = np.array(list(it.combinations(range(n_cams-nb_cams_off_tot), n_cams_swapped)))
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
                        # print(id_off,id_swapped,n_cams,nb_cams_off)
                        # print(repr(x_files_filt_off_swap))
                        q_file_off_swap = [(x_files_filt_off_swap[id_off][id_swapped][i], y_files_filt_off_swap[id_off][id_swapped][i]) for i in range(n_cams - nb_cams_off_tot)]
                        q_calc_off_swap = [(x_calc_off_swap[id_off][id_swapped][i], y_calc_off_swap[id_off][id_swapped][i]) for i in range(n_cams - nb_cams_off_tot)]
                        error_percam.append( np.mean( [euclidean_distance(q_file_off_swap[i], q_calc_off_swap[i]) for i in range(len(q_file_off_swap))] ) )
                    error_off_swap.append(error_percam)
                error_off_swap = np.array(error_off_swap)
                # print('error_off_swap ', error_off_swap)
                
                # Choosing best triangulation (with min reprojection error)
                error_off_swap_min = np.min(error_off_swap)
                best_off_swap_config = np.unravel_index(error_off_swap.argmin(), error_off_swap.shape)
                
                id_off_cams = best_off_swap_config[0]
                id_swapped_cams = id_cams_swapped[best_off_swap_config[1]]
                Q_best = Q_filt_off_swap[best_off_swap_config][:-1]

                n_cams_swapped += 1

            if error_off_swap_min < error_min:
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


def parse_rigid_marker_groups(config_dict, keypoints_names):
    '''
    Read optional rigid-marker groups from Config.toml and map marker names to
    triangulation column indices.
    '''

    triangulation_config = config_dict.get('triangulation', {})
    raw_groups = triangulation_config.get('rigid_marker_groups', [])
    if raw_groups in (None, False, []):
        return []
    if not isinstance(raw_groups, list):
        logging.warning('rigid_marker_groups must be a list of marker-name lists. Ignoring rigid triangulation groups.')
        return []

    min_markers = triangulation_config.get('rigid_group_min_markers', 3)
    groups = []
    seen_groups = set()
    for group_id, raw_group in enumerate(raw_groups):
        group_name = None
        markers = raw_group
        if isinstance(raw_group, dict):
            group_name = raw_group.get('name')
            markers = raw_group.get('markers', raw_group.get('names', raw_group.get('keypoints')))
        if isinstance(markers, str):
            markers = [marker.strip() for marker in markers.split(',') if marker.strip()]
        if not markers:
            logging.warning(f'Rigid marker group {group_id} has no markers. Skipping it.')
            continue

        present_markers = []
        missing_markers = []
        for marker in markers:
            if marker in keypoints_names and marker not in present_markers:
                present_markers.append(marker)
            else:
                missing_markers.append(marker)

        if missing_markers:
            logging.warning(
                f"Rigid marker group {group_name or group_id} skipped missing markers: {missing_markers}."
            )
        if len(present_markers) < min_markers:
            logging.warning(
                f"Rigid marker group {group_name or group_id} needs at least {min_markers} available markers. "
                f"Only found {present_markers}. Skipping it."
            )
            continue

        indices = [keypoints_names.index(marker) for marker in present_markers]
        group_key = tuple(indices)
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)

        # Collect any per-group overrides (everything that is not the name or
        # marker list). Short keys such as ``fill_missing`` are normalized to the
        # ``rigid_group_*`` names the helpers look up, so a group can locally tune
        # behaviour without touching the global defaults used by other groups.
        options = {}
        if isinstance(raw_group, dict):
            reserved = {'name', 'markers', 'names', 'keypoints'}
            for key, value in raw_group.items():
                if key in reserved:
                    continue
                opt_key = key if key.startswith('rigid_group_') else f'rigid_group_{key}'
                options[opt_key] = value

        groups.append({
            'name': group_name or '+'.join(present_markers),
            'markers': present_markers,
            'indices': indices,
            'options': options,
        })

    return groups


def _rigid_param(group_options, triangulation_config, key, default):
    '''Resolve a rigid-group parameter: per-group override first, then the
    global triangulation config, then the supplied default.'''
    if group_options is not None and key in group_options:
        return group_options[key]
    return triangulation_config.get(key, default)


def _rigid_group_columns(keypoint_indices):
    return np.array([keypoint_idx * 3 + axis for keypoint_idx in keypoint_indices for axis in range(3)], dtype=int)


def _extract_group_points(Q_df, keypoint_indices):
    columns = _rigid_group_columns(keypoint_indices)
    return Q_df.iloc[:, columns].to_numpy(dtype=float).reshape(len(Q_df), len(keypoint_indices), 3)


def _build_rigid_template(group_points, group_name, config_dict, group_options=None):
    triangulation_config = config_dict.get('triangulation', {})
    min_template_frames = _rigid_param(group_options, triangulation_config, 'rigid_group_template_min_frames', 20)
    mad_factor = _rigid_param(group_options, triangulation_config, 'rigid_group_template_mad_factor', 5.0)
    distance_floor_m = _rigid_param(group_options, triangulation_config, 'rigid_group_template_distance_floor_m', 0.02)

    complete_frames = np.all(np.isfinite(group_points), axis=(1, 2))
    if np.count_nonzero(complete_frames) < 3:
        logging.warning(
            f'Rigid marker group {group_name} has fewer than 3 complete baseline frames. '
            'Keeping independent triangulation for this group.'
        )
        return None

    candidate_points = group_points[complete_frames]
    if len(candidate_points) < min_template_frames:
        logging.warning(
            f'Rigid marker group {group_name} has only {len(candidate_points)} complete template frames '
            f'(requested {min_template_frames}). Using the available frames.'
        )

    if candidate_points.shape[1] >= 2 and len(candidate_points) >= 5:
        pairwise_distances = []
        for marker_i, marker_j in it.combinations(range(candidate_points.shape[1]), 2):
            pairwise_distances.append(np.linalg.norm(candidate_points[:, marker_i] - candidate_points[:, marker_j], axis=1))
        pairwise_distances = np.array(pairwise_distances).T
        median_distances = np.nanmedian(pairwise_distances, axis=0)
        mad_distances = np.nanmedian(np.abs(pairwise_distances - median_distances), axis=0)
        thresholds = np.maximum(distance_floor_m, mad_factor * 1.4826 * mad_distances)
        stable_frames = np.all(np.abs(pairwise_distances - median_distances) <= thresholds, axis=1)
        if np.count_nonzero(stable_frames) >= 3:
            candidate_points = candidate_points[stable_frames]

    centered_points = candidate_points - np.mean(candidate_points, axis=1, keepdims=True)
    template = np.nanmedian(centered_points, axis=0)
    template = template - np.mean(template, axis=0, keepdims=True)
    if not np.all(np.isfinite(template)):
        logging.warning(f'Rigid marker group {group_name} produced a non-finite template. Skipping it.')
        return None
    return template


def _initial_rigid_params(template, baseline_points):
    valid_markers = np.all(np.isfinite(baseline_points), axis=1)
    if np.count_nonzero(valid_markers) < 1:
        return np.zeros(6)

    source = template[valid_markers]
    target = baseline_points[valid_markers]
    if len(source) >= 3:
        source_centroid = source.mean(axis=0)
        target_centroid = target.mean(axis=0)
        source_centered = source - source_centroid
        target_centered = target - target_centroid
        U, _, Vt = np.linalg.svd(source_centered.T @ target_centered)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = target_centroid - R @ source_centroid
        rvec = cv2.Rodrigues(R)[0].ravel()
        return np.concatenate([rvec, t])

    t = np.mean(target - source, axis=0)
    return np.concatenate([np.zeros(3), t])


def _rigid_points_from_params(params, template):
    R = cv2.Rodrigues(params[:3])[0]
    return (R @ template.T).T + params[3:6]


def _project_rigid_point(projection_matrix, point):
    point_h = np.append(point, 1.0)
    projected = projection_matrix @ point_h
    if not np.isfinite(projected[2]) or projected[2] == 0:
        return np.array([np.nan, np.nan])
    return projected[:2] / projected[2]


def _rigid_group_residuals(params, template, x_obs, y_obs, likelihood_obs, projection_matrices, camera_ids):
    points = _rigid_points_from_params(params, template)
    residuals = []
    for cam_id in camera_ids:
        for marker_id, point in enumerate(points):
            if not (
                np.isfinite(x_obs[cam_id, marker_id])
                and np.isfinite(y_obs[cam_id, marker_id])
                and np.isfinite(likelihood_obs[cam_id, marker_id])
                and likelihood_obs[cam_id, marker_id] > 0
            ):
                continue
            projected = _project_rigid_point(projection_matrices[cam_id], point)
            if not np.all(np.isfinite(projected)):
                continue
            weight = np.sqrt(max(float(likelihood_obs[cam_id, marker_id]), 0.0))
            residuals.extend((projected - np.array([x_obs[cam_id, marker_id], y_obs[cam_id, marker_id]])) * weight)
    if not residuals:
        return np.array([1e6])
    return np.array(residuals)


def _rigid_reprojection_error(points, x_obs, y_obs, likelihood_obs, projection_matrices, camera_ids):
    marker_errors = [[] for _ in range(points.shape[0])]
    all_errors = []
    for cam_id in camera_ids:
        for marker_id, point in enumerate(points):
            if not (
                np.isfinite(x_obs[cam_id, marker_id])
                and np.isfinite(y_obs[cam_id, marker_id])
                and np.isfinite(likelihood_obs[cam_id, marker_id])
                and likelihood_obs[cam_id, marker_id] > 0
            ):
                continue
            projected = _project_rigid_point(projection_matrices[cam_id], point)
            if not np.all(np.isfinite(projected)):
                continue
            observed = np.array([x_obs[cam_id, marker_id], y_obs[cam_id, marker_id]])
            error = euclidean_distance(projected, observed)
            marker_errors[marker_id].append(error)
            all_errors.append(error)

    if not all_errors:
        return np.nan, np.full(points.shape[0], np.nan)
    return float(np.mean(all_errors)), np.array([
        float(np.mean(errors)) if errors else np.nan for errors in marker_errors
    ])


def _fit_rigid_group_frame(config_dict, template, baseline_points, x_obs, y_obs, likelihood_obs, projection_matrices, group_options=None):
    triangulation_config = config_dict.get('triangulation', {})
    error_threshold = _rigid_param(
        group_options, triangulation_config, 'rigid_group_reproj_error_threshold',
        triangulation_config.get('reproj_error_threshold_triangulation', 15),
    )
    min_cameras = triangulation_config.get('min_cameras_for_triangulation', 2)
    min_markers = _rigid_param(group_options, triangulation_config, 'rigid_group_min_markers', 3)
    max_nfev = _rigid_param(group_options, triangulation_config, 'rigid_group_max_nfev', 30)
    loss = _rigid_param(group_options, triangulation_config, 'rigid_group_loss', 'soft_l1')
    loss_scale = _rigid_param(group_options, triangulation_config, 'rigid_group_loss_scale_px', 5.0)
    max_cams_to_exclude = _rigid_param(group_options, triangulation_config, 'rigid_group_max_cams_to_exclude', 1)
    if max_nfev in (None, False):
        max_nfev = None
    else:
        max_nfev = int(max_nfev)
    if max_cams_to_exclude in (None, False):
        max_cams_to_exclude = 0
    else:
        max_cams_to_exclude = int(max_cams_to_exclude)

    valid_observations = (
        np.isfinite(x_obs)
        & np.isfinite(y_obs)
        & np.isfinite(likelihood_obs)
        & (likelihood_obs > 0)
    )
    n_cams = len(projection_matrices)
    if np.count_nonzero(valid_observations.any(axis=1)) < min_cameras:
        return None
    if np.count_nonzero(valid_observations.any(axis=0)) < min_markers:
        return None

    initial_params = _initial_rigid_params(template, baseline_points)
    best_fit = None
    valid_camera_ids = [cam_id for cam_id in range(n_cams) if valid_observations[cam_id].any()]
    max_cams_to_exclude = min(max(0, max_cams_to_exclude), len(valid_camera_ids) - min_cameras)
    for nb_cams_off in range(max_cams_to_exclude + 1):
        for excluded_cams in it.combinations(valid_camera_ids, nb_cams_off):
            used_camera_ids = [cam_id for cam_id in valid_camera_ids if cam_id not in excluded_cams]
            if len(used_camera_ids) < min_cameras:
                continue
            if np.count_nonzero(valid_observations[used_camera_ids].any(axis=0)) < min_markers:
                continue
            if np.count_nonzero(valid_observations[used_camera_ids]) * 2 < 6:
                continue

            initial_points = _rigid_points_from_params(initial_params, template)
            initial_error, initial_marker_errors = _rigid_reprojection_error(
                initial_points, x_obs, y_obs, likelihood_obs, projection_matrices, used_camera_ids
            )
            if np.isfinite(initial_error):
                deliberately_excluded = set(range(n_cams)) - set(used_camera_ids)
                candidate = {
                    'points': initial_points,
                    'params': initial_params,
                    'error': initial_error,
                    'marker_errors': initial_marker_errors,
                    'excluded_cams': sorted(deliberately_excluded),
                    'used_cams': used_camera_ids,
                }
                if best_fit is None or candidate['error'] < best_fit['error']:
                    best_fit = candidate
                if initial_error <= error_threshold:
                    return candidate

            result = least_squares(
                _rigid_group_residuals,
                initial_params,
                args=(template, x_obs, y_obs, likelihood_obs, projection_matrices, used_camera_ids),
                loss=loss,
                f_scale=loss_scale,
                max_nfev=max_nfev,
            )
            if not np.all(np.isfinite(result.x)):
                continue
            fitted_points = _rigid_points_from_params(result.x, template)
            error, marker_errors = _rigid_reprojection_error(
                fitted_points, x_obs, y_obs, likelihood_obs, projection_matrices, used_camera_ids
            )
            if not np.isfinite(error):
                continue
            deliberately_excluded = set(range(n_cams)) - set(used_camera_ids)
            candidate = {
                'points': fitted_points,
                'params': result.x,
                'error': error,
                'marker_errors': marker_errors,
                'excluded_cams': sorted(deliberately_excluded),
                'used_cams': used_camera_ids,
            }
            if best_fit is None or candidate['error'] < best_fit['error']:
                best_fit = candidate

        if best_fit is not None and best_fit['error'] <= error_threshold:
            break

    if best_fit is None or best_fit['error'] > error_threshold:
        return None
    return best_fit


def _smooth_rigid_params(fits, config_dict, group_options=None):
    triangulation_config = config_dict.get('triangulation', {})
    window = _rigid_param(group_options, triangulation_config, 'rigid_group_smoothing_window', 5)
    if window in (None, False) or window <= 1:
        return [fit.get('params') if fit is not None else None for fit in fits]
    method = str(_rigid_param(group_options, triangulation_config, 'rigid_group_smoothing_method', 'median')).lower()

    params = np.full((len(fits), 6), np.nan)
    for frame_id, fit in enumerate(fits):
        if fit is not None:
            params[frame_id] = fit['params']
    if np.count_nonzero(np.all(np.isfinite(params), axis=1)) < 2:
        return [fit.get('params') if fit is not None else None for fit in fits]

    params_df = pd.DataFrame(params)
    roll = params_df.rolling(window=int(window), center=True, min_periods=1)
    # 'median' (default) rejects outlier single-frame fits but is edge-preserving,
    # so it leaves depth-ambiguity steps at gap boundaries untouched. 'mean' (and
    # 'median_then_mean') distribute such steps into a gentle ramp — appropriate for
    # slow-moving groups like the pelvis where genuine motion has no sharp edges.
    if method == 'mean':
        smoothed = roll.mean().to_numpy()
    elif method in ('median_then_mean', 'median_mean'):
        median_df = roll.median()
        smoothed = median_df.rolling(window=int(window), center=True, min_periods=1).mean().to_numpy()
    else:
        smoothed = roll.median().to_numpy()
    return [
        smoothed[frame_id] if fit is not None and np.all(np.isfinite(smoothed[frame_id])) else (
            fit.get('params') if fit is not None else None
        )
        for frame_id, fit in enumerate(fits)
    ]


def _pairwise_distance_change_ratio(reference_points, candidate_points, valid_markers):
    ratios = []
    for marker_i, marker_j in it.combinations(range(len(reference_points)), 2):
        if not (valid_markers[marker_i] and valid_markers[marker_j]):
            continue
        reference_distance = np.linalg.norm(reference_points[marker_i] - reference_points[marker_j])
        candidate_distance = np.linalg.norm(candidate_points[marker_i] - candidate_points[marker_j])
        if not np.isfinite(reference_distance) or not np.isfinite(candidate_distance):
            continue
        if reference_distance <= 1e-9:
            if candidate_distance > 1e-9:
                ratios.append(np.inf)
            continue
        ratios.append(abs(candidate_distance - reference_distance) / reference_distance)

    return max(ratios) if ratios else 0.0


def _guarded_rigid_points(config_dict, baseline_points, rigid_points, group_options=None):
    triangulation_config = config_dict.get('triangulation', {})
    blend = _rigid_param(group_options, triangulation_config, 'rigid_group_blend', 0.7)
    max_correction_m = _rigid_param(group_options, triangulation_config, 'rigid_group_max_correction_m', 0.05)
    max_pairwise_change_ratio = _rigid_param(group_options, triangulation_config, 'rigid_group_max_pairwise_change_ratio', 0.15)
    fill_missing = bool(_rigid_param(group_options, triangulation_config, 'rigid_group_fill_missing', False))

    blend = 0.7 if blend is None else float(blend)
    blend = float(np.clip(blend, 0.0, 1.0))
    if max_correction_m in (None, False):
        max_correction_m = None
    else:
        max_correction_m = float(max_correction_m)
    if max_pairwise_change_ratio in (None, False):
        max_pairwise_change_ratio = None
    else:
        max_pairwise_change_ratio = float(max_pairwise_change_ratio)

    finite_baseline = np.all(np.isfinite(baseline_points), axis=1)
    finite_rigid = np.all(np.isfinite(rigid_points), axis=1)
    blended_markers = finite_baseline & finite_rigid
    filled_markers = (~finite_baseline) & finite_rigid if fill_missing else np.zeros_like(finite_rigid, dtype=bool)

    stats = {
        'applied': False,
        'effective_blend': blend,
        'correction_limited': False,
        'pairwise_guarded': False,
        'pairwise_rejected': False,
        'delta_mm': [],
    }
    if not np.any(blended_markers) and not np.any(filled_markers):
        return baseline_points, stats

    if max_correction_m is not None and max_correction_m >= 0 and np.any(blended_markers):
        correction_norms = np.linalg.norm(rigid_points[blended_markers] - baseline_points[blended_markers], axis=1)
        max_full_correction = np.nanmax(correction_norms) if len(correction_norms) else 0.0
        if np.isfinite(max_full_correction) and max_full_correction > 0 and blend * max_full_correction > max_correction_m:
            blend = max_correction_m / max_full_correction
            stats['correction_limited'] = True

    def build_candidate(candidate_blend):
        candidate_points = baseline_points.copy()
        if np.any(blended_markers):
            candidate_points[blended_markers] = (
                (1.0 - candidate_blend) * baseline_points[blended_markers]
                + candidate_blend * rigid_points[blended_markers]
            )
        if np.any(filled_markers):
            candidate_points[filled_markers] = rigid_points[filled_markers]
        return candidate_points

    corrected_points = build_candidate(blend)
    if max_pairwise_change_ratio is not None and max_pairwise_change_ratio >= 0:
        ratio = _pairwise_distance_change_ratio(baseline_points, corrected_points, blended_markers)
        if ratio > max_pairwise_change_ratio:
            stats['pairwise_guarded'] = True
            low_blend, high_blend = 0.0, blend
            for _ in range(24):
                mid_blend = (low_blend + high_blend) / 2.0
                mid_points = build_candidate(mid_blend)
                mid_ratio = _pairwise_distance_change_ratio(baseline_points, mid_points, blended_markers)
                if mid_ratio <= max_pairwise_change_ratio:
                    low_blend = mid_blend
                else:
                    high_blend = mid_blend
            blend = low_blend
            corrected_points = build_candidate(blend)
            if blend <= 1e-6 and not np.any(filled_markers):
                stats['pairwise_rejected'] = True

    stats['effective_blend'] = blend
    finite_corrected = np.all(np.isfinite(corrected_points), axis=1)
    changed_markers = finite_baseline & finite_corrected
    if np.any(changed_markers):
        deltas_mm = np.linalg.norm(corrected_points[changed_markers] - baseline_points[changed_markers], axis=1) * 1000.0
        stats['delta_mm'] = [float(delta) for delta in deltas_mm if np.isfinite(delta)]
    stats['applied'] = (
        (np.any(changed_markers) and np.nanmax(np.linalg.norm(corrected_points[changed_markers] - baseline_points[changed_markers], axis=1)) > 0)
        or np.any(filled_markers)
    )
    return corrected_points, stats


def refine_rigid_marker_groups(config_dict, Q_df, error_df, nb_cams_excluded_df, id_excluded_cams_df,
                               observations, projection_matrices, rigid_groups, id_person=0):
    '''
    Stabilize configured marker groups by fitting a rigid transform to 2D
    observations, then blend the rigid result back into the independent 3D
    triangulation with proportion guards.
    '''

    if not rigid_groups or observations is None:
        return []

    x_obs = observations.get('x')
    y_obs = observations.get('y')
    likelihood_obs = observations.get('likelihood')
    if x_obs is None or y_obs is None or likelihood_obs is None:
        return []
    if len(x_obs) != len(Q_df):
        logging.warning(
            f'Rigid triangulation observations ({len(x_obs)} frames) do not match '
            f'3D results ({len(Q_df)} frames). Skipping rigid refinement for person {id_person}.'
        )
        return []

    stats = []
    for group in rigid_groups:
        start_time = time.perf_counter()
        group_name = group['name']
        keypoint_indices = group['indices']
        group_options = group.get('options')
        group_points = _extract_group_points(Q_df, keypoint_indices)
        template = _build_rigid_template(group_points, group_name, config_dict, group_options)
        if template is None:
            stats.append({
                'name': group_name,
                'accepted': 0,
                'applied': 0,
                'total': len(Q_df),
                'mean_error': np.nan,
                'mean_delta_mm': np.nan,
                'max_delta_mm': np.nan,
                'elapsed_seconds': time.perf_counter() - start_time,
            })
            continue

        fits = []
        for row_id in range(len(Q_df)):
            fits.append(_fit_rigid_group_frame(
                config_dict,
                template,
                group_points[row_id],
                x_obs[row_id][:, keypoint_indices],
                y_obs[row_id][:, keypoint_indices],
                likelihood_obs[row_id][:, keypoint_indices],
                projection_matrices,
                group_options,
            ))

        accepted_frames = 0
        applied_frames = 0
        accepted_errors = []
        correction_limited_frames = 0
        pairwise_guarded_frames = 0
        pairwise_rejected_frames = 0
        correction_deltas_mm = []
        columns = _rigid_group_columns(keypoint_indices)
        smoothed_params = _smooth_rigid_params(fits, config_dict, group_options)
        for row_id, fit in enumerate(fits):
            if fit is None:
                continue

            rigid_points = _rigid_points_from_params(smoothed_params[row_id], template)
            error, marker_errors = _rigid_reprojection_error(
                rigid_points,
                x_obs[row_id][:, keypoint_indices],
                y_obs[row_id][:, keypoint_indices],
                likelihood_obs[row_id][:, keypoint_indices],
                projection_matrices,
                fit['used_cams'],
            )
            if not np.isfinite(error):
                rigid_points = fit['points']
                error = fit['error']
                marker_errors = fit['marker_errors']

            accepted_frames += 1
            corrected_points, guard_stats = _guarded_rigid_points(
                config_dict,
                group_points[row_id],
                rigid_points,
                group_options,
            )
            if guard_stats['correction_limited']:
                correction_limited_frames += 1
            if guard_stats['pairwise_guarded']:
                pairwise_guarded_frames += 1
            if guard_stats['pairwise_rejected']:
                pairwise_rejected_frames += 1
            correction_deltas_mm.extend(guard_stats['delta_mm'])

            if guard_stats['applied']:
                error, marker_errors = _rigid_reprojection_error(
                    corrected_points,
                    x_obs[row_id][:, keypoint_indices],
                    y_obs[row_id][:, keypoint_indices],
                    likelihood_obs[row_id][:, keypoint_indices],
                    projection_matrices,
                    fit['used_cams'],
                )
                if not np.isfinite(error):
                    error = fit['error']
                    marker_errors = fit['marker_errors']

                Q_df.iloc[row_id, columns] = corrected_points.reshape(-1)
                for local_marker_id, keypoint_idx in enumerate(keypoint_indices):
                    marker_error = marker_errors[local_marker_id]
                    error_df.iat[row_id, keypoint_idx] = marker_error if np.isfinite(marker_error) else error
                    nb_cams_excluded_df.iat[row_id, keypoint_idx] = len(fit['excluded_cams'])
                    id_excluded_cams_df.iat[row_id, keypoint_idx] = fit['excluded_cams']
                applied_frames += 1

            accepted_errors.append(error)

        mean_error = float(np.mean(accepted_errors)) if accepted_errors else np.nan
        mean_delta_mm = float(np.mean(correction_deltas_mm)) if correction_deltas_mm else np.nan
        max_delta_mm = float(np.max(correction_deltas_mm)) if correction_deltas_mm else np.nan
        elapsed_seconds = time.perf_counter() - start_time
        stats.append({
            'name': group_name,
            'accepted': accepted_frames,
            'applied': applied_frames,
            'total': len(Q_df),
            'mean_error': mean_error,
            'mean_delta_mm': mean_delta_mm,
            'max_delta_mm': max_delta_mm,
            'correction_limited_frames': correction_limited_frames,
            'pairwise_guarded_frames': pairwise_guarded_frames,
            'pairwise_rejected_frames': pairwise_rejected_frames,
            'elapsed_seconds': elapsed_seconds,
        })
        logging.info(
            f"Rigid marker group {group_name} for person {id_person}: accepted "
            f"{accepted_frames}/{len(Q_df)} frames"
            + (f" with mean joint reprojection error {mean_error:.1f} px" if np.isfinite(mean_error) else "")
            + f"; applied {applied_frames}/{len(Q_df)} frames"
            + (f"; mean correction {mean_delta_mm:.1f} mm" if np.isfinite(mean_delta_mm) else "")
            + (f", max correction {max_delta_mm:.1f} mm" if np.isfinite(max_delta_mm) else "")
            + f"; correction limit adjusted {correction_limited_frames} frames"
            + f"; pairwise guard adjusted {pairwise_guarded_frames} frames"
            + f", rejected {pairwise_rejected_frames}"
            + f"; elapsed {elapsed_seconds:.2f} s."
        )

    return stats


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


def triangulate_single_frame(f, json_dirs_names, json_files_names, pose_dir,
                             keypoints_ids, keypoints_idx, keypoints_idx_swapped,
                             nb_persons_to_detect, n_cams, P, calib_params,
                             config_dict, undistort_points, return_observations=False):
    '''
    Pure per-frame triangulation function for process-based parallelism.
    '''

    json_files_names_f = [[j for j in json_files_names[c] if int(re.split(r'(\d+)', j)[-2]) == f] for c in range(n_cams)]
    json_files_names_f = [j for j_list in json_files_names_f for j in (j_list or ['none'])]
    json_files_f = [os.path.join(pose_dir, json_dirs_names[c], json_files_names_f[c]) for c in range(n_cams)]

    x_files, y_files, likelihood_files = extract_files_frame_f(json_files_f, keypoints_ids, nb_persons_to_detect)

    if undistort_points:
        for n in range(nb_persons_to_detect):
            points = [np.array(tuple(zip(x_files[n][i], y_files[n][i]))).reshape(-1, 1, 2).astype('float32') for i in range(n_cams)]
            undistorted_points = [cv2.undistortPoints(points[i], calib_params['K'][i], calib_params['dist'][i], None, calib_params['optim_K'][i]) for i in range(n_cams)]
            x_files[n] = np.array([[u[i][0][0] for i in range(len(u))] for u in undistorted_points])
            y_files[n] = np.array([[u[i][0][1] for i in range(len(u))] for u in undistorted_points])

    likelihood_threshold = config_dict.get('triangulation', {}).get('likelihood_threshold_triangulation', 0.3)
    with np.errstate(invalid='ignore'):
        for n in range(nb_persons_to_detect):
            x_files[n][likelihood_files[n] < likelihood_threshold] = np.nan
            y_files[n][likelihood_files[n] < likelihood_threshold] = np.nan
            likelihood_files[n][likelihood_files[n] < likelihood_threshold] = np.nan

    Q = [[] for _ in range(nb_persons_to_detect)]
    error = [[] for _ in range(nb_persons_to_detect)]
    nb_cams_excluded = [[] for _ in range(nb_persons_to_detect)]
    id_excluded_cams = [[] for _ in range(nb_persons_to_detect)]

    for n in range(nb_persons_to_detect):
        for keypoint_idx in keypoints_idx:
            coords_2D_kpt = np.array((x_files[n][:, keypoint_idx], y_files[n][:, keypoint_idx], likelihood_files[n][:, keypoint_idx]))
            coords_2D_kpt_swapped = np.array((
                x_files[n][:, keypoints_idx_swapped[keypoint_idx]],
                y_files[n][:, keypoints_idx_swapped[keypoint_idx]],
                likelihood_files[n][:, keypoints_idx_swapped[keypoint_idx]],
            ))

            Q_kpt, error_kpt, nb_cams_excluded_kpt, id_excluded_cams_kpt = triangulation_from_best_cameras(
                config_dict, coords_2D_kpt, coords_2D_kpt_swapped, P, calib_params,
            )

            Q[n].append(Q_kpt)
            error[n].append(error_kpt)
            nb_cams_excluded[n].append(nb_cams_excluded_kpt)
            id_excluded_cams[n].append(id_excluded_cams_kpt)

    if return_observations:
        observations = {
            'x': x_files,
            'y': y_files,
            'likelihood': likelihood_files,
        }
        return Q, error, nb_cams_excluded, id_excluded_cams, observations
    return Q, error, nb_cams_excluded, id_excluded_cams


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
    project_dir = config_dict.get('project', {}).get('project_dir', '.')
    # if batch
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    multi_person = config_dict.get('project', {}).get('multi_person', False)
    pose_model = config_dict.get('pose', {}).get('pose_model', 'Body_with_feet')
    frame_range = config_dict.get('project', {}).get('frame_range', 'auto')
    interpolation_kind = config_dict.get('triangulation', {}).get('interpolation', 'linear')
    interp_gap_smaller_than = config_dict.get('triangulation', {}).get('interp_if_gap_smaller_than', 20)
    parallel_triangulation = config_dict.get('triangulation', {}).get('parallel_triangulation', 'auto')
    max_distance_m = config_dict.get('triangulation', {}).get('max_distance_m', 1.0)
    remove_incomplete_frames = config_dict.get('triangulation', {}).get('remove_incomplete_frames', False)
    sections_to_keep = config_dict.get('triangulation', {}).get('sections_to_keep', 'all')
    min_chunk_size = config_dict.get('triangulation', {}).get('min_chunk_size', 10)
    fill_large_gaps_with = config_dict.get('triangulation', {}).get('fill_large_gaps_with', 'last_value')
    show_interp_indices = config_dict.get('triangulation', {}).get('show_interp_indices', True)
    undistort_points = config_dict.get('triangulation', {}).get('undistort_points', False)
    make_c3d = config_dict.get('triangulation', {}).get('make_c3d', True)
    
    try:
        calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, c)) and  'calib' in c.lower()][0]
    except:
        raise Exception(f'No .toml calibration directory found.')
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
            model = DictImporter().import_(config_dict.get('pose').get(pose_model))
            if model.id == 'None':
                model.id = None
        except:
            raise NameError('{pose_model} not found in skeletons.py nor in Config.toml')
            
    keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_idx = list(range(len(keypoints_ids)))
    keypoints_nb = len(keypoints_ids)
    marker_fill_overrides = _large_gap_marker_fill_overrides(config_dict, keypoints_names)
    if marker_fill_overrides:
        logging.info(
            'Large-gap fill marker overrides: %s.',
            ', '.join(
                f'{keypoints_names[keypoint_id]}={fill_mode}'
                for keypoint_id, fill_mode in marker_fill_overrides.items()
            ),
        )
    rigid_groups = parse_rigid_marker_groups(config_dict, keypoints_names)
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
        logging.info('Counting the number of persons... This can take a while if you use numerous cameras or recorded for a long time.')
        nb_persons_to_detect = max(max(count_persons_in_json(os.path.join(pose_dir, json_dirs_names[c], json_fname)) for json_fname in json_files_names[c]) for c in range(n_cams))
        logging.info(f'{nb_persons_to_detect} person(s) detected. Triangulating now...')
    else:
        nb_persons_to_detect = 1

    Q = np.full((nb_persons_to_detect, keypoints_nb, 3), np.nan)
    Q_old = np.full((nb_persons_to_detect, keypoints_nb, 3), np.nan)
    error = [[] for n in range(nb_persons_to_detect)]
    nb_cams_excluded = [[] for n in range(nb_persons_to_detect)]
    id_excluded_cams = [[] for n in range(nb_persons_to_detect)]
    Q_tot, error_tot, nb_cams_excluded_tot, cam_excluded_count, id_excluded_cams_tot = [], [], [], [], []
    observations_tot = [{'x': [], 'y': [], 'likelihood': []} for _ in range(nb_persons_to_detect)] if rigid_groups else None
    interp_frames, non_interp_frames, f_range_trimmed = [], [], []
    trc_paths, c3d_paths = [], []
    if parallel_triangulation not in ('auto', False) and not isinstance(parallel_triangulation, int):
        raise ValueError("parallel_triangulation must be 'auto', an integer, or false.")
    if parallel_triangulation in (False, 1) or frame_nb <= 1:
        triangulation_workers = 1
    elif parallel_triangulation == 'auto':
        triangulation_workers = min(os.cpu_count() or 1, frame_nb)
    elif parallel_triangulation < 1:
        raise ValueError('parallel_triangulation must be greater or equal to 1 when set to an integer.')
    else:
        triangulation_workers = min(int(parallel_triangulation), frame_nb)

    frames_to_process = range(*f_range)
    if triangulation_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        logging.info(f'Triangulating frames in parallel with {triangulation_workers} worker processes.')
        worker_config_dict = _to_picklable_builtin(config_dict)
        chunksize = max(1, frame_nb // max(1, triangulation_workers * 4))
        with ProcessPoolExecutor(max_workers=triangulation_workers) as executor:
            frame_results = list(tqdm(
                executor.map(
                    triangulate_single_frame,
                    frames_to_process,
                    it.repeat(json_dirs_names),
                    it.repeat(json_files_names),
                    it.repeat(pose_dir),
                    it.repeat(keypoints_ids),
                    it.repeat(keypoints_idx),
                    it.repeat(keypoints_idx_swapped),
                    it.repeat(nb_persons_to_detect),
                    it.repeat(n_cams),
                    it.repeat(P),
                    it.repeat(calib_params),
                    it.repeat(worker_config_dict),
                    it.repeat(undistort_points),
                    it.repeat(bool(rigid_groups)),
                    chunksize=chunksize,
                ),
                total=frame_nb,
            ))
    else:
        frame_results = [
            triangulate_single_frame(
                f, json_dirs_names, json_files_names, pose_dir, keypoints_ids, keypoints_idx,
                keypoints_idx_swapped, nb_persons_to_detect, n_cams, P, calib_params,
                config_dict, undistort_points, bool(rigid_groups),
            )
            for f in tqdm(frames_to_process)
        ]

    for f, frame_result in zip(range(*f_range), frame_results):
        if rigid_groups:
            raw_Q, error, nb_cams_excluded, id_excluded_cams, observations = frame_result
            x_obs = np.array(observations['x'], dtype=np.float64)
            y_obs = np.array(observations['y'], dtype=np.float64)
            likelihood_obs = np.array(observations['likelihood'], dtype=np.float64)
        else:
            raw_Q, error, nb_cams_excluded, id_excluded_cams = frame_result

        nan_mask = np.isnan(Q)
        Q_old = np.where(nan_mask, Q_old, Q)
        Q = np.array(raw_Q, dtype=np.float64)

        if multi_person:
            if f != 0:
                Q_old, Q, sorted_ids = sort_people_sports2d(Q_old, Q, max_dist=max_distance_m)

                error_sorted, nb_cams_excluded_sorted, id_excluded_cams_sorted = [], [], []
                if rigid_groups:
                    x_obs_sorted = np.full_like(x_obs, np.nan)
                    y_obs_sorted = np.full_like(y_obs, np.nan)
                    likelihood_obs_sorted = np.full_like(likelihood_obs, np.nan)
                for n in range(nb_persons_to_detect):
                    detection_idx = sorted_ids[n]
                    if detection_idx >= 0:
                        error_sorted.append(error[detection_idx])
                        nb_cams_excluded_sorted.append(nb_cams_excluded[detection_idx])
                        id_excluded_cams_sorted.append(id_excluded_cams[detection_idx])
                        if rigid_groups:
                            x_obs_sorted[n] = x_obs[detection_idx]
                            y_obs_sorted[n] = y_obs[detection_idx]
                            likelihood_obs_sorted[n] = likelihood_obs[detection_idx]
                    else:
                        error_sorted.append([np.nan] * keypoints_nb)
                        nb_cams_excluded_sorted.append([n_cams] * keypoints_nb)
                        id_excluded_cams_sorted.append([list(range(n_cams))] * keypoints_nb)
                error, nb_cams_excluded, id_excluded_cams = error_sorted, nb_cams_excluded_sorted, id_excluded_cams_sorted
                if rigid_groups:
                    x_obs, y_obs, likelihood_obs = x_obs_sorted, y_obs_sorted, likelihood_obs_sorted

        Q_tot.append([np.concatenate(Q[n]) for n in range(nb_persons_to_detect)])
        error_tot.append([error[n] for n in range(nb_persons_to_detect)])
        nb_cams_excluded_tot.append([nb_cams_excluded[n] for n in range(nb_persons_to_detect)])
        id_excluded_cams = [[id_excluded_cams[n][k] for k in range(keypoints_nb)] for n in range(nb_persons_to_detect)]
        id_excluded_cams_tot.append(id_excluded_cams)
        if rigid_groups:
            for n in range(nb_persons_to_detect):
                observations_tot[n]['x'].append(x_obs[n])
                observations_tot[n]['y'].append(y_obs[n])
                observations_tot[n]['likelihood'].append(likelihood_obs[n])
            
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
    if rigid_groups:
        observations_tot = [
            {key: np.array(value, dtype=np.float64) for key, value in observations.items()}
            for observations in observations_tot
        ]
        logging.info(
            '\nApplying rigid marker group triangulation to: '
            + ', '.join(group['name'] for group in rigid_groups)
            + '.'
        )
        for n in range(nb_persons_to_detect):
            refine_rigid_marker_groups(
                config_dict,
                Q_tot[n],
                error_tot[n],
                nb_cams_excluded_tot[n],
                id_excluded_cams_tot[n],
                observations_tot[n],
                P,
                rigid_groups,
                id_person=n,
            )

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

        Q_tot[n] = _apply_large_gap_fill(
            Q_tot[n],
            zero_nan_frames_per_kpt,
            keypoints_names,
            fill_large_gaps_with,
            marker_fill_overrides,
        )

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
