"""Reproduce the saved V9 descending-seed failure and audit deterministic reseeding.

Only local algebra/root solves are evaluated. No spindle integration is run.
The frozen pre-change solver is required so originally successful outputs can be
compared bit for bit, including diagnostic and iteration-count fields.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

from spindle import tai_bearing as b, tai_system
from utils.clearance_v9_contact_boundary import (
    loaded_loss_limit, prescribed_loss_limit, trace_dual_branch,
)


ROOT = Path('experiments/clearance_v9_tai_bearing')


def load_previous(path):
    name = 'v9_solver_before_retry'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False,
                               indent=2, allow_nan=False), encoding='utf-8')


def replay_failure(directory, previous):
    config = json.loads((directory/'actual_config.json').read_text(encoding='utf-8'))
    args = tai_system.prepare(config['parameters'])
    with np.load(directory/'failure_state.npz') as data:
        t, q, v = float(data['t_s']), data['q'].copy(), data['v'].copy()
    with np.load(directory/'initial_seed.npz') as data:
        start = float(data['absolute_time_s'])
        seed = data['state'].copy()
    failed = []
    unchanged = 0
    new_failed = 0
    for name, row, p in zip(tai_system.NAMES, args[4], args[5]):
        z, endpoint, sign, _, _, phase0, factor = row
        endpoint = int(endpoint)
        dx, dy = q[0]+z*q[9]-q[endpoint], q[1]-z*q[8]-q[endpoint+1]
        b0 = (p[b.FI]+p[b.FO]-1)*p[b.D]
        ric = p[b.DM]/2+(p[b.FI]-.5)*p[b.D]*math.cos(p[b.ALPHA0])
        for k in range(int(p[b.Z])):
            phase = sign*(phase0+args[7]*p[b.CAGE_RATIO]*factor*t+2*math.pi*k/p[b.Z])
            sn, cs = math.sin(phase), math.cos(phase)
            A = b0*math.sin(p[b.ALPHA0])+p[b.A_PRE]-ric*sign*q[9]*cs+ric*q[8]*sn
            B = b0*math.cos(p[b.ALPHA0])+dx*cs+sign*dy*sn
            omega = sign*args[7]
            call = (A, B, omega, p, args[8], args[9], args[10], True, factor)
            old, new = previous.solve_ball_kernel(*call), b.solve_ball_kernel(*call)
            new_failed += int(new[b.STATUS] < 0)
            if old[b.STATUS] >= 0:
                unchanged += int(np.array_equal(old.view(np.uint64), new.view(np.uint64)))
                continue
            reference = previous.solve_ball_reference(A, B, omega, p)
            qi = brentq(lambda load: trace_dual_branch(A, load, omega, p)['B_m']-B,
                        1e-8, 100., xtol=1e-12)
            traced = trace_dual_branch(A, qi, omega, p)
            fixed_limit = prescribed_loss_limit(A, omega, p)
            loaded_limit = loaded_loss_limit(A, omega, p)
            angle = math.atan2(A, B)
            ki, ko = b.hertz_kernel(angle, p, True)[0], b.hertz_kernel(angle, p, False)[0]
            static_overlap = math.hypot(A, B)-b0
            static_load = (ki**(-2/3)+ko**(-2/3))**(-1.5)*max(static_overlap, 0.)**1.5
            physical = list(range(b.U, b.KO+1))
            reference_array = np.array([reference[b.BALL_NAMES[i]] for i in physical])
            failed.append(dict(bearing=name, ball_index_zero_based=k, A_m=A, B_m=B,
                omega_local_rad_s=omega, parameters=p, original=previous._ball_dict(old),
                repaired=b._ball_dict(new), independent_four_unknown_reference=reference,
                independent_prescribed_Qi_trace=traced,
                new_vs_reference_max_scaled_difference=float(np.max(
                    np.abs(new[physical]-reference_array)/np.maximum(np.abs(reference_array), 1e-12))),
                F10_prescribed_limit=fixed_limit, F07_loaded_limit=loaded_limit,
                B_above_F07_boundary_um=(B-loaded_limit['B_m'])*1e6,
                inside_F07_F10_boundary_interval=bool(min(fixed_limit['B_m'],loaded_limit['B_m']) <= B
                    <= max(fixed_limit['B_m'],loaded_limit['B_m'])),
                static_series_overlap_um=static_overlap*1e6,
                original_static_series_seed_load_N=static_load))
    return dict(scope='Local replay only, no integration and no physical parameter change.',
        failed_directory=str(directory), source_error=json.loads((directory/'failed.json').read_text(encoding='utf-8')),
        seed_time_s=start, failure_time_s=t, elapsed_after_seed_us=(t-start)*1e6,
        seed_state=seed, failure_q=q, failure_v=v,
        all_balls_count=sum(int(p[b.Z]) for p in args[5]),
        old_failed_ball_count=len(failed), old_successful_balls_bitwise_unchanged=unchanged,
        new_failed_ball_count=new_failed, failed_balls=failed,
        accepted_state_status='unknown: saved during an RHS evaluation; no accepted-step mesh or DOP853 stage identity was saved',
        accepted_state_evidence=[
            'dynamics_model.simulate calls solve_ivp with DOP853 and dense_output=True.',
            'The failure propagates inside RHS; accepted_steps are copied only after solve_ivp returns.',
            'This failed directory contains no accepted_steps, checkpoint, or native trajectory.',
            'The saved evaluation may be a step stage, rejected trial, or dense-output auxiliary stage; it is not certified as an accepted state.'
        ],
        conclusion='Valid outer-control dual root exists. Original deterministic Newton path failed; the F07/F10 contact-loss closure gap is not the cause of this saved failure.')


def random_regression(config, previous):
    cfg = json.loads(config.read_text(encoding='utf-8'))['parameters']['tai_bearing']
    rng = np.random.default_rng(902)
    families = {}
    for name in ('front', 'rear'):
        p = np.array(cfg['families'][name]['initialized_parameter_vector'])
        b0 = (p[b.FI]+p[b.FO]-1)*p[b.D]
        A0, B0 = b0*math.sin(p[b.ALPHA0])+p[b.A_PRE], b0*math.cos(p[b.ALPHA0])
        changed, old_failed, recovered = [], [], []
        maximum = np.zeros(32)
        for i in range(2500):
            A, B = A0+rng.uniform(-5e-6,5e-6), B0+rng.uniform(-1e-5,1e-5)
            omega = (3000,7000,10000)[i%3]*2*math.pi/60
            for diagnostics in (True, False):
                call = (A,B,omega,p,1+i%2,i%2,1.,diagnostics)
                old, new = previous.solve_ball_kernel(*call), b.solve_ball_kernel(*call)
                if old[b.STATUS] < 0:
                    old_failed.append([i,diagnostics])
                    if new[b.STATUS] >= 0:
                        recovered.append([i,diagnostics])
                else:
                    maximum = np.maximum(maximum, np.nan_to_num(np.abs(new-old)))
                    if not np.array_equal(old.view(np.uint64),new.view(np.uint64)):
                        changed.append([i,diagnostics])
        families[name] = dict(state_count=2500, diagnostics_modes=[True,False],
            old_failures=old_failed, recovered=recovered, changed_original_successes=changed,
            max_absolute_change_by_field=dict(zip(b.BALL_NAMES, maximum)),
            bitwise_unchanged=not changed)
    return dict(generator='numpy.default_rng(902), continuous stream front then rear',
        A_offset_uniform_m=[-5e-6,5e-6], B_offset_uniform_m=[-1e-5,1e-5],
        rpm_cycle=[3000,7000,10000], mode='1+i%2',control='i%2',families=families,
        passed=all(item['bitwise_unchanged'] and not item['old_failures'] for item in families.values()))


def boundary_regression(path, config, previous):
    saved = json.loads(path.read_text(encoding='utf-8'))
    families = json.loads(config.read_text(encoding='utf-8'))['parameters']['tai_bearing']['families']
    old_failed = new_failed = recovered = successful_changed = count = 0
    for case in saved['cases']:
        p = np.array(families[case['family']]['initialized_parameter_vector'])
        omega = case['rpm_magnitude']*2*math.pi/60
        for sample in case['production_samples']:
            count += 1
            call = (sample['A_m'],sample['B_m'],omega,p,2,0,1.)
            old,new = previous.solve_ball_kernel(*call),b.solve_ball_kernel(*call)
            old_failed += int(old[b.STATUS] < 0)
            new_failed += int(new[b.STATUS] < 0)
            recovered += int(old[b.STATUS] < 0 and new[b.STATUS] >= 0)
            successful_changed += int(old[b.STATUS] >= 0 and not np.array_equal(
                old.view(np.uint64),new.view(np.uint64)))
    return dict(sample_count=count,original_failures=old_failed,new_failures=new_failed,
        recovered_samples=recovered,original_successes_changed=successful_changed,
        note='Local sampling only. Deterministic reseeding does not alter the inconsistent F07/F10 limiting closures or claim general boundary continuity.')


def assembled_rhs_replay(directory):
    from spindle.fast_rhs import prepare_fast_rhs
    config = json.loads((directory/'actual_config.json').read_text(encoding='utf-8'))
    drive = config['excitation']
    rhs = prepare_fast_rhs(config['parameters'], (drive['U_kg_m'],drive['phase_rad']))
    with np.load(directory/'failure_state.npz') as data:
        t = float(data['t_s'])
        state = np.concatenate((data['q'],data['v']))
    result = rhs(t,state)
    return dict(scope='One complete assembled RHS evaluation at the saved failed state; no integration.',
        failure_directory=str(directory), numba_cache_dir=os.environ.get('NUMBA_CACHE_DIR'),
        t_s=t, derivative=result, passed=bool(np.isfinite(result).all()),
        exception=None, source='spindle.fast_rhs.prepare_fast_rhs -> spindle.tai_system.rhs_kernel')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/clearance_v9_config.json'))
    parser.add_argument('--failure-dir',type=Path,
        default=ROOT/'sensitivity/initial_state_down_10_to_5/N3/gap_5um')
    parser.add_argument('--old-source',type=Path,default=ROOT/'validation/solver_retry_before/tai_bearing.py')
    parser.add_argument('--boundary',type=Path,default=ROOT/'validation/contact_boundary.json',
        help='Saved contact-boundary report used for the local old/new solver comparison.')
    parser.add_argument('--output',type=Path,default=ROOT/'validation')
    parser.add_argument('--rhs-only',action='store_true',
        help='Evaluate only the complete RHS; set an isolated NUMBA_CACHE_DIR before running.')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.rhs_only:
        result = assembled_rhs_replay(args.failure_dir)
        write_json(args.output/'solver_retry_assembled_rhs.json',result)
        print(json.dumps(jsonable(result),ensure_ascii=False,indent=2))
        return 0 if result['passed'] else 1
    previous = load_previous(args.old_source)
    replay = replay_failure(args.failure_dir,previous)
    write_json(args.output/'descending_seed_failure_replay.json',replay)
    random = random_regression(args.config,previous)
    boundary = boundary_regression(args.boundary,args.config,previous)
    report = dict(old_source=str(args.old_source),new_source='src/spindle/tai_bearing.py',
        scope='Local-only numerical solver audit; no spindle trajectory.',random_regression=random,
        saved_boundary_samples=boundary,
        passed=bool(random['passed'] and boundary['original_successes_changed']==0
                    and replay['new_failed_ball_count']==0))
    write_json(args.output/'solver_retry_regression.json',report)
    print(json.dumps(dict(passed=report['passed'],failure_recovered=replay['new_failed_ball_count']==0,
        old_successful_balls_bitwise_unchanged=replay['old_successful_balls_bitwise_unchanged'],
        random_5000_bitwise_unchanged=random['passed'],boundary=boundary),ensure_ascii=False,indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
