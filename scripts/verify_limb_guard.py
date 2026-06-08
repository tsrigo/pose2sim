#!/usr/bin/env python3
"""Verify the non-physical limb-length guard:
  (1) lhl_action02 155-176: the exploded shoulders/arms (1-2 m bones) are repaired.
  (2) clean trials (zjg/zf static): the guard is a NO-OP (0 rejections), so generality holds.
Runs each trial with the guard OFF then ON from the SAME cached 2D and compares the max
shoulder/elbow/wrist bone length in the window.
"""
import os, sys, glob, shutil, logging
from pathlib import Path
import numpy as np
import toml
sys.path.insert(0, '.')
from Pose2Sim.Utilities.avi2trc import triangulate_trial

logging.basicConfig(level=logging.INFO, format='%(message)s')

PELVIS = {'name': 'pelvis', 'markers': ['Hip', 'RHip', 'LHip'], 'fill_missing': True,
          'blend': 1.0, 'max_correction_m': False, 'max_pairwise_change_ratio': False,
          'reproj_error_threshold': 45, 'smoothing_window': 9, 'smoothing_method': 'robust',
          'orientation_window': 15, 'orientation_tol_deg': 35, 'twist_guard': True,
          'max_twist_deg': 50, 'max_cams_to_exclude': 1}
HEAD = {'name': 'head', 'markers': ['Head', 'Nose', 'REye', 'LEye', 'REar', 'LEar'],
        'fill_missing': True, 'blend': 1.0, 'max_correction_m': False,
        'max_pairwise_change_ratio': False, 'reproj_error_threshold': 25,
        'smoothing_window': 9, 'max_cams_to_exclude': 1}

IDX = {n: i for i, n in enumerate(
    ['Hip','RHip','RKnee','RAnkle','RBigToe','RSmallToe','RHeel','LHip','LKnee','LAnkle',
     'LBigToe','LSmallToe','LHeel','Neck','Head','Nose','REye','LEye','REar','LEar',
     'RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist'], start=1)}
ARM_BONES = [('Neck','RShoulder'),('RShoulder','RElbow'),('RElbow','RWrist'),
             ('Neck','LShoulder'),('LShoulder','LElbow'),('LElbow','LWrist')]


def load(trc):
    rows = []
    for ln in open(trc).readlines()[5:]:
        p = ln.split('\t')
        if len(p) < 5: continue
        rows.append([float(x) if x.strip() not in ('', '\n') else np.nan for x in p])
    return np.array(rows)


def xyz(arr, i, name):
    c = 2 + (IDX[name]-1)*3
    return arr[i, c:c+3]


def max_arm_bone(trc, lo, hi):
    arr = load(trc); fr = arr[:, 0].astype(int)
    worst = 0.0; worst_info = None
    for f in range(lo, hi+1):
        w = np.where(fr == f)[0]
        if not len(w): continue
        i = w[0]
        for m1, m2 in ARM_BONES:
            d = np.linalg.norm(xyz(arr, i, m1) - xyz(arr, i, m2))
            if np.isfinite(d) and d > worst:
                worst = d; worst_info = (f, m1, m2)
    return worst, worst_info


def run(src, guard, out):
    src = Path(src)
    cfgs = [c for c in (list(src.glob('Config*.toml')) + list(src.glob('*.toml'))) if 'alib' not in c.name]
    cfg = toml.load(str(cfgs[0])) if cfgs else toml.load('deliverables/pnvision_20260530_video_filter/Config.toml')
    tri = cfg.setdefault('triangulation', {})
    groups = [g for g in (tri.get('rigid_marker_groups') or [])
              if not (isinstance(g, dict) and g.get('name') in ('pelvis', 'head'))]
    tri['rigid_marker_groups'] = [PELVIS, HEAD] + groups
    tri['reject_nonphysical_limbs'] = guard
    scratch = Path('/tmp/sc_lg')
    if scratch.exists(): shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    os.symlink((src / 'pose').resolve(), scratch / 'pose')
    os.symlink((src / 'calibration').resolve(), scratch / 'calibration')
    (scratch / 'pose-3d').mkdir()
    cfg.setdefault('project', {}).update(project_dir=str(scratch), frame_range='all', multi_person=False)
    cfg.setdefault('pose', {})['save_video'] = 'none'
    cfg.setdefault('filtering', {})['filter'] = False
    triangulate_trial(cfg)
    trc = glob.glob(str(scratch / 'pose-3d' / '*.trc'))[0]
    shutil.copy(trc, out)
    return out


if __name__ == '__main__':
    cases = [
        ('lhl', 'data/pnvision_action02_rtmpose_smoothing_20260519/rec_20260507_134131_lhl_action02', 155, 176),
        ('zjg', 'data/exp_lr_swap/prod_20260602/zjg_20260602_145557_静态站立', 850, 980),
        ('zf',  'data/exp_lr_swap/prod_20260602/zf_20260602_144726_静态站立', 1150, 1265),
    ]
    for tag, src, lo, hi in cases:
        print(f'\n===== {tag}  window {lo}-{hi} =====')
        print('--- GUARD OFF ---')
        off = run(src, False, f'/tmp/{tag}_off.trc')
        w_off, i_off = max_arm_bone(off, lo, hi)
        print('--- GUARD ON ---')
        on = run(src, True, f'/tmp/{tag}_on.trc')
        w_on, i_on = max_arm_bone(on, lo, hi)
        print(f'[{tag}] max arm bone in window:  OFF={w_off*100:.1f}cm {i_off}   ON={w_on*100:.1f}cm {i_on}')
