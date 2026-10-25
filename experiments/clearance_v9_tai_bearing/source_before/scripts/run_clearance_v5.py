"""V5 force sensitivity with same-force healthy controls and frozen mechanics."""
import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from spindle.dynamics_model import front_group_parameters, section_matrix
from scripts.run_clearance_v3 import write_json
from scripts.run_clearance_v4 import run_case
from utils.clearance_v5_analysis import analyze, compare_metrics


def force_config(baseline, force, gap, directions=256, step_factor=1, backend='numba'):
    config = copy.deepcopy(baseline)
    p = front_group_parameters(config['parameters'], gap)
    p.update(N=directions, rhs_backend=backend, diagnostic_single_front=False)
    config['parameters'] = p
    omega = p['rpm'] * 2*np.pi/60
    config['excitation'].update(U_kg_m=force/omega**2, peak_force_N=force,
        source='V5 authorized sensitivity input; replaces total rotating force; assumed_demo')
    config['simulation']['max_step_s'] *= step_factor
    config['simulation']['maximum_warmup_s'] = 10.0
    config['parameter_provenance']['excitation_status'] = 'assumed_demo_sensitivity'
    return config


def execute(config, folder):
    folder = Path(folder)
    if (folder/'metrics.json').exists():
        saved=json.loads((folder/'actual_config.json').read_text(encoding='utf-8'))
        if saved['parameters']!=config['parameters'] or saved['excitation']!=config['excitation'] or any(
            saved['simulation'][key]!=value for key,value in config['simulation'].items() if key!='warmup_s'):
            raise ValueError(f'Existing result has a different actual configuration: {folder}')
        return json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    began = time.time()
    metrics = run_case(config, folder, analyze, compare_metrics)
    p, excitation = config['parameters'], config['excitation']
    with np.load(folder/'native.npz') as data:
        times = data['t_raw']
    angle = p['rpm']*2*np.pi/60*times + excitation['phase_rad']
    force = excitation['U_kg_m']*(p['rpm']*2*np.pi/60)**2
    xy = force*np.column_stack((np.cos(angle), np.sin(angle)))
    generalized = xy @ section_matrix(p['z_u'])
    np.savez_compressed(folder/'actual_input.npz', t_raw=times, angle_rad=angle,
        force_xy_N=xy, generalized_rotor_force=generalized)
    metrics.update(force_N=force, U_kg_m=excitation['U_kg_m'])
    write_json(folder/'metrics.json', metrics)
    write_json(folder/'input_check.json', dict(
        force_N=force, norm_max_error_N=float(np.max(abs(np.linalg.norm(xy,axis=1)-force))),
        generalized_rotor_indices=[0,1,8,9], generalized_rotor_units=['N','N','N m','N m'],
        torque_max_error_Nm=float(np.max(abs(generalized[:,2:]-
            np.column_stack((-p['z_u']*xy[:,1],p['z_u']*xy[:,0]))))),
        start_epoch_s=began, end_epoch_s=time.time(), status='completed',
        common_clearance_um=gap if (gap:=p['fit_clearance_um'])==p['fit2_clearance_um'] else None,
        parameter_identity='Exact configuration JSON and explicit differences; no hashes.'))
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/clearance_v5_config.json'))
    parser.add_argument('--output',type=Path,default=Path('experiments/clearance_v5_excitation'))
    parser.add_argument('--group',default='data')
    parser.add_argument('--forces',type=float,nargs='+',required=True)
    parser.add_argument('--gaps',type=float,nargs='+',default=[0,10])
    parser.add_argument('--directions',type=int,default=256)
    parser.add_argument('--step-factor',type=float,default=1)
    parser.add_argument('--warmup',type=float)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--backend',choices=['python','numba'],default='numba')
    args=parser.parse_args()
    baseline=json.loads(args.config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for force in args.forces:
        for gap in args.gaps:
            if not (0<=gap<=10) or not (0<force<=30):
                raise ValueError('V5 input domain is 0<c<=10 um (including zero), 0<F<=30 N')
            config=force_config(baseline,force,gap,args.directions,args.step_factor,args.backend)
            if args.warmup is not None:config['simulation']['warmup_s']=args.warmup
            folder=args.output/args.group/f'F_{force:g}N'/f'gap_{gap:g}um'
            jobs.append((config,folder))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(execute,*job):job[1] for job in jobs}
        for future in as_completed(futures):
            try:
                metrics=future.result()
                print(json.dumps(dict(folder=str(futures[future]),E_um=metrics['E_dir_um'],
                    steady=metrics['steady_pass'],domain=metrics['domain_pass'])),flush=True)
            except Exception as error:
                write_json(futures[future]/'failed.json',dict(status='failed',error=str(error)))
                raise


if __name__=='__main__':main()
