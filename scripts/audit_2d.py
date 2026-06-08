#!/usr/bin/env python3
"""Audit cached 2D pose JSON (HALPE_26) for a trial, per camera, over a frame window.

Flags the things that drive bad pelvis 3D:
  - low confidence on Hip/RHip/LHip
  - regressed-Hip inconsistency: |Hip - midpoint(RHip,LHip)| in px (kp19 is virtual)
  - L/R lateral-order sign flips of (RHip.x - LHip.x): a mid-window flip is either a
    genuine turn through profile or a 2D left/right swap
  - 2D jitter: median frame-to-frame displacement of RHip/LHip (px)
Usage: audit_2d.py <pose_dir> <lo> <hi>   (pose_dir holds cam*_json/)
"""
import sys, glob, json
from pathlib import Path
import numpy as np

H = {'Nose':0,'LShoulder':5,'RShoulder':6,'LHip':11,'RHip':12,'Hip':19}


def load_cam(jdir, lo, hi):
    files = sorted(glob.glob(str(Path(jdir) / '*.json')))
    rows = {}
    for f in files:
        idx = int(Path(f).stem.split('_')[-1])
        if idx < lo or idx > hi:
            continue
        d = json.load(open(f))
        if not d['people']:
            rows[idx] = None
            continue
        a = np.array(d['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        rows[idx] = a
    return rows


def audit(pose_dir, lo, hi):
    cams = sorted(glob.glob(str(Path(pose_dir) / 'cam*_json')))
    print(f'  window [{lo},{hi}]  {len(cams)} cams')
    for jdir in cams:
        name = Path(jdir).name.replace('_json', '')
        rows = load_cam(jdir, lo, hi)
        idxs = sorted(rows)
        miss = sum(1 for i in idxs if rows[i] is None)
        valid = [(i, rows[i]) for i in idxs if rows[i] is not None]
        if not valid:
            print(f'  {name}: NO DETECTIONS ({miss} missing)'); continue
        cR = np.array([v[H['RHip'], 2] for _, v in valid])
        cL = np.array([v[H['LHip'], 2] for _, v in valid])
        cH = np.array([v[H['Hip'], 2] for _, v in valid])
        # regressed-hip inconsistency
        mid = np.array([(v[H['RHip'], :2] + v[H['LHip'], :2]) / 2 for _, v in valid])
        hip = np.array([v[H['Hip'], :2] for _, v in valid])
        hip_err = np.linalg.norm(hip - mid, axis=1)
        # L/R lateral order
        dx = np.array([v[H['RHip'], 0] - v[H['LHip'], 0] for _, v in valid])
        flips = int(np.sum(np.diff(np.sign(dx)) != 0))
        # jitter
        R = np.array([v[H['RHip'], :2] for _, v in valid])
        L = np.array([v[H['LHip'], :2] for _, v in valid])
        jit = np.median(np.r_[np.linalg.norm(np.diff(R, axis=0), axis=1),
                              np.linalg.norm(np.diff(L, axis=0), axis=1)])
        lowconf = int(np.sum((cR < 0.3) | (cL < 0.3)))
        print(f'  {name}: miss={miss} lowconf(RHip|LHip<.3)={lowconf}/{len(valid)} '
              f'| conf R/L/Hip={cR.mean():.2f}/{cL.mean():.2f}/{cH.mean():.2f} '
              f'| HipRegErr px med={np.median(hip_err):.1f} max={hip_err.max():.1f} '
              f'| LR-sign-flips={flips} | jitter px/frm={jit:.1f}')


if __name__ == '__main__':
    audit(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
