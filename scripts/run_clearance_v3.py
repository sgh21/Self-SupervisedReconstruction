"""Reproducible fixed-configuration 10-DOF clearance runs; preserves V1 results."""
import argparse
import copy
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from spindle.dynamics_model import (equilibrium_continuation, simulate,
                                    rotor_section, bearing_force)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_window(config):
    p, s = config['parameters'], config['simulation']
    rate = abs(p['rpm'])/60
    cycles = np.sign(p['rpm'])*config['excitation']['phase_rad']/(2*np.pi)
    start = (np.ceil(s['warmup_s']*rate+cycles)-cycles)/rate
    return float(start), float(start+s['retained_turns']/rate)


def run_case(config, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/'actual_config.json', config)
    p, s = config['parameters'], config['simulation']
    start, end = export_window(config)
    fs = s['fs_raw_hz']
    keep_from = np.floor((start-s['guard_s'])*fs)/fs
    finish = np.ceil((end+s['guard_s'])*fs)/fs
    atol = np.r_[np.full(8, s['atol_translation_m']), np.full(2, s['atol_angle_rad']),
                  np.full(8, s['atol_velocity_m_s']), np.full(2, s['atol_angle_rate_rad_s'])]
    q0, _ = equilibrium_continuation(p)
    began = time.perf_counter()
    result = simulate(p, 0, [], [], warmup=keep_from, duration=finish-keep_from,
                      fs=fs, max_step=s['max_step_s'], rtol=s['rtol'], atol=atol,
                      method=s['method'], q_initial=q0,
                      unbalance=(config['excitation']['U_kg_m'], config['excitation']['phase_rad']))
    elapsed = time.perf_counter()-began
    from utils.clearance_v3_analysis import analyze_v3, compare_v3
    metrics, turns, output, steady = analyze_v3(result, config, start, end)
    coarse, _, _, _ = analyze_v3(result, config, start, end, stride=2)
    sampling = compare_v3(coarse, metrics, config)
    metrics.update(gap_um=p['fit_clearance_um'], steady_pass=steady['passed'],
                   wall_seconds=elapsed, nfev=result['nfev'])
    t, q, v = result['t'], result['q'], result['v']
    valid = (t >= start) & (t < end)
    relative_f = rotor_section(q, p['z_f'])-q[:, 6:8]
    velocity_f = rotor_section(v, p['z_f'])-v[:, 6:8]
    bearing = np.asarray([bearing_force(ti, ri, vi, p)
                          for ti, ri, vi in zip(t, relative_f, velocity_f)])
    healthy_relative = np.stack([rotor_section(q,h['z_m'])-q[:,4:6]
                                for h in p['healthy_supports']], axis=1)
    healthy = np.stack([(rotor_section(q,h['z_m'])-q[:,4:6])@np.asarray(h['K']).T
                       +(rotor_section(v,h['z_m'])-v[:,4:6])@np.asarray(h['C']).T
                       for h in p['healthy_supports']], axis=1)
    directions = np.column_stack((np.cos(2*np.pi*np.arange(p['N'])/p['N']),
                                  np.sin(2*np.pi*np.arange(p['N'])/p['N'])))
    compression = np.maximum((q[:,6:8]-q[:,4:6])@directions.T-p['fit_clearance_um']*1e-6,0)
    metrics['fit_max_compression_um'] = float(compression[valid].max()*1e6)
    metrics['rotor_max_tilt_rad'] = float(np.linalg.norm(q[valid,8:10],axis=1).max())
    metrics['bearing_peak_N'] = float(np.linalg.norm(bearing[valid],axis=1).max())
    metadata = dict(config=config, actual_solver_settings=result['actual_solver_settings'],
        export_start_s=start, export_end_s=end, metrics=metrics,
        state_order=['Sx','Sy','Tx','Ty','Hx','Hy','Ox','Oy','theta_x','theta_y'],
        state_units=['m']*8+['rad']*2,
        velocity_units=['m/s']*8+['rad/s']*2,
        acceleration_units=['m/s2']*8+['rad/s2']*2,
        processing='H state velocity and ODE acceleration: common resample_poly Kaiser beta5 AA lowpass to 8192Hz, real guards discarded; no bandpass, no normalization. Raw arrays include guards; exact validated export_start/end identify main window.',
        main_metric='max(mean(turn_pp_x_um),mean(turn_pp_y_um)); includes 1X',
        reference='rigid rotor section z_m relative fixed ground/table demo; no measured surface geometry',
        source_identity='source_final snapshot in experiment root; no hash',
        equilibrium_residual_equivalent_N=result['equilibrium_residual_N'])
    tout, hv, ha = output
    np.savez_compressed(folder/'native.npz', t_raw=t, q_raw_m=q, v_raw_m_s=v,
        a_raw_m_s2=result['acc'], r_m_raw_m=result['measured_relative_xy_m'],
        section_xy_m=result['measured_section_xy_m'], probe_reference_xy_m=result['probe_reference_xy_m'],
        rO_raw_m=q[:,6:8]-q[:,4:6], S_relative_H_m=q[:,:2]-q[:,4:6],
        M_relative_H_m=result['measured_section_xy_m']-q[:,4:6],
        angle_rad=p['rpm']*2*np.pi/60*t+config['excitation']['phase_rad'],
        fit_force_N=result['fit_force_N'], fit_active_count=result['fit_active_count'],
        target_bearing_force_N=bearing, healthy_support_forces_N=healthy,
        healthy_relative_m=healthy_relative, q_equilibrium=result['q_equilibrium'],
        t_out=tout, H_velocity_mm_s=hv, H_acceleration_m_s2=ha,
        turn_id=turns['turn_id'], turn_start_s=turns['start_time_s'],
        turn_end_s=turns['end_time_s'], turn_pp_um=turns['peak_to_peak_um'],
        turn_fit_impulse_N_s=turns['fit_impulse_N_s'], turn_fit_peak_N=turns['fit_peak_N'],
        turn_contact_fraction=turns['fit_contact_fraction'],
        metadata_json=json.dumps(metadata,ensure_ascii=False))
    write_json(folder/'metrics.json', metrics)
    write_json(folder/'stationarity.json', steady)
    write_json(folder/'sampling_convergence.json', sampling)
    write_json(folder/'actual_solver_settings.json', result['actual_solver_settings'])
    write_csv(folder/'turns.csv', [dict(turn_id=int(i),start_s=float(a),end_s=float(b),
              Ex_um=float(x),Ey_um=float(y),fit_impulse_N_s=float(impulse),
              fit_peak_N=float(peak),contact_fraction=float(contact))
              for i,a,b,(x,y),impulse,peak,contact in zip(turns['turn_id'],
              turns['start_time_s'],turns['end_time_s'],turns['peak_to_peak_um'],
              turns['fit_impulse_N_s'],turns['fit_peak_N'],turns['fit_contact_fraction'])])
    print(json.dumps(dict(folder=str(folder),**metrics)), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/clearance_v3_config.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gaps',type=float,nargs='+')
    parser.add_argument('--directions',type=int)
    parser.add_argument('--step-factor',type=float,default=1)
    parser.add_argument('--warmup',type=float)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.directions is not None: config['parameters']['N']=args.directions
    config['simulation']['max_step_s']*=args.step_factor
    if args.warmup is not None: config['simulation']['warmup_s']=args.warmup
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'run_settings.json',config)
    gaps=config['clearances_um'] if args.gaps is None else args.gaps
    jobs=[]
    for gap in gaps:
        local=copy.deepcopy(config)
        local['parameters']['fit_clearance_um']=gap
        jobs.append((local,args.output/f'gap_{gap:g}um'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_case,*job) for job in jobs]
        rows=[f.result() for f in as_completed(futures)]
    rows.sort(key=lambda x:x['gap_um'])
    write_csv(args.output/'summary.csv',rows)


if __name__=='__main__':
    main()
