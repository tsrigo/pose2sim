#!/usr/bin/env python3
"""Triangulate many trials with the production pelvis recipe injected into each trial's
OWN config (so skeleton/calibration match), then run the artifact sweep on all of them.

Recipe: pelvis rigid group with robust orientation smoothing + chirality hold + twist
clamp. Writes raw TRC to /tmp/batch/<tag>.trc and prints a one-line sweep per trial.
"""
import os, sys, glob, shutil
from pathlib import Path
import toml
sys.path.insert(0, '.')
from Pose2Sim.Utilities.avi2trc import triangulate_trial

PELVIS = {'name': 'pelvis', 'markers': ['Hip', 'RHip', 'LHip'], 'fill_missing': True,
          'blend': 1.0, 'max_correction_m': False, 'max_pairwise_change_ratio': False,
          'reproj_error_threshold': 45, 'smoothing_window': 9, 'smoothing_method': 'robust',
          'orientation_window': 15, 'orientation_tol_deg': 35, 'twist_guard': True,
          'max_twist_deg': 50, 'max_cams_to_exclude': 1}
HEAD = {'name': 'head', 'markers': ['Head', 'Nose', 'REye', 'LEye', 'REar', 'LEar'],
        'fill_missing': True, 'blend': 1.0, 'max_correction_m': False,
        'max_pairwise_change_ratio': False, 'reproj_error_threshold': 25,
        'smoothing_window': 9, 'max_cams_to_exclude': 1}

OUT = Path('/tmp/batch'); OUT.mkdir(parents=True, exist_ok=True)


def run(tag, src):
    src = Path(src)
    cfgs = list(src.glob('Config*.toml')) + list(src.glob('*.toml'))
    cfgs = [c for c in cfgs if 'alib' not in c.name]
    if not cfgs:
        # fall back to production config
        cfg = toml.load('deliverables/pnvision_20260530_video_filter/Config.toml')
    else:
        cfg = toml.load(str(cfgs[0]))
    tri = cfg.setdefault('triangulation', {})
    groups = tri.get('rigid_marker_groups') or []
    groups = [g for g in groups if not (isinstance(g, dict) and g.get('name') in ('pelvis', 'head'))]
    tri['rigid_marker_groups'] = [PELVIS, HEAD] + groups
    scratch = Path('/tmp/sc_batch');
    if scratch.exists(): shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    os.symlink((src / 'pose').resolve(), scratch / 'pose')
    os.symlink((src / 'calibration').resolve(), scratch / 'calibration')
    (scratch / 'pose-3d').mkdir()
    cfg.setdefault('project', {}).update(project_dir=str(scratch), frame_range='all', multi_person=False)
    cfg.setdefault('pose', {})['save_video'] = 'none'
    cfg.setdefault('filtering', {})['filter'] = False
    try:
        triangulate_trial(cfg)
    except Exception as e:
        print(f'{tag}: TRIANGULATION FAILED: {e}')
        return None
    trcs = glob.glob(str(scratch / 'pose-3d' / '*.trc'))
    if not trcs:
        print(f'{tag}: no TRC produced'); return None
    out = OUT / f'{tag}.trc'
    shutil.copy(trcs[0], out)
    return str(out)


if __name__ == '__main__':
    trials = [ln.split(None, 1) for ln in sys.stdin.read().strip().splitlines() if ln.strip()]
    done = []
    for tag, src in trials:
        r = run(tag.strip(), src.strip())
        if r:
            done.append((tag.strip(), r))
            print(f'OK {tag.strip()} -> {r}', flush=True)
    print('\nBATCH DONE:', len(done), 'trials')
