"""V6 GMN runs: one actual U, two common front gaps, retained native evidence."""
import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from spindle import dynamics_model as model
from scripts.run_clearance_v3 import export_window, write_json, write_csv
from utils.clearance_v6_analysis import analyze, compare_metrics, domain_check


ARRAY_KEYS = ('t', 'q', 'v', 'acc', 'measured_relative_xy_m', 'fit_force_N',
              'fit_active_count', 'fit2_force_N', 'fit2_active_count')


def case_config(baseline, gap, force=None, directions=None, step_factor=1):
    config = copy.deepcopy(baseline)
    config['parameters'] = model.front_group_parameters(config['parameters'], gap)
    p, s, e = config['parameters'], config['simulation'], config['excitation']
    if force is not None:
        if not 0 <= force <= 30:
            raise ValueError('The authorized force range is 0 to 30 N.')
        e['U_kg_m'] = force/(p['rpm']*2*np.pi/60)**2
    if e['U_kg_m'] is None:
        raise ValueError('Calibrate U or explicitly supply a calibration force.')
    e['peak_force_N'] = e['U_kg_m']*(p['rpm']*2*np.pi/60)**2
    if e['peak_force_N'] > 30 + 1e-10:
        raise ValueError('Actual rotating force exceeds 30 N.')
    if directions is not None:
        p['N'] = directions
    s['max_step_s'] *= step_factor
    s['dense_sampling_multiplier'] = 2
    return config


def execute(config, folder, seed_folder=None):
    folder = Path(folder)
    if (folder/'metrics.json').exists():
        saved = json.loads((folder/'actual_config.json').read_text(encoding='utf-8'))
        if any(saved[key] != config[key] for key in ('parameters', 'excitation', 'processing')):
            raise ValueError(f'Existing case has different physics: {folder}')
        for key, value in config['simulation'].items():
            if key != 'warmup_s' and saved['simulation'][key] != value:
                raise ValueError(f'Existing case has different solver: {folder}, {key}')
        return json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    folder.mkdir(parents=True, exist_ok=False)
    # Mechanical caches belong to the calculation copy, never the saved contract.
    p, s = copy.deepcopy(config['parameters']), config['simulation']
    write_json(folder/'actual_config.json', config)
    q0, static_residual = model.equilibrium_continuation(p)
    aq = np.full(12, s['atol_translation_m']); aq[8:10] = s['atol_angle_rad']
    av = np.full(12, s['atol_velocity_m_s']); av[8:10] = s['atol_angle_rate_rad_s']
    fs = s['fs_raw_hz']*2
    result = None; history = []; solver_history = []
    seed = None; epoch = 0.
    if seed_folder is not None:
        with np.load(Path(seed_folder)/'native.npz') as data:
            seed = data['state_final']; epoch = float(data['integration_end_s'])
        np.savez_compressed(folder/'initial_seed.npz', state=seed, absolute_time_s=epoch)
        s['warmup_s'] += epoch
        s['maximum_warmup_s'] += epoch
    began = time.perf_counter()
    while True:
        start, end = export_window(config)
        keep = np.floor((start-s['guard_s'])*fs)/fs
        finish = np.ceil((end+s['guard_s'])*fs)/fs
        previous = epoch if result is None else float(result['t'][-1])
        new = model.simulate(p, 0, [], [], warmup=keep-epoch if result is None else 0,
            duration=finish-keep if result is None else finish-previous,
            fs=fs, max_step=s['max_step_s'], rtol=s['rtol'], atol=np.r_[aq, av],
            method=s['method'], q_initial=q0, t_start=previous,
            state_initial=seed if result is None else result['state_final'],
            unbalance=(config['excitation']['U_kg_m'], config['excitation']['phase_rad']))
        solver_history.append(new['actual_solver_settings'])
        if result is not None:
            for key in ARRAY_KEYS:
                new[key] = np.concatenate((result[key], new[key][1:]), axis=0)
            new['nfev'] += result['nfev']
        result = {key: new[key] for key in ARRAY_KEYS + ('state_final', 'nfev')}
        metrics, turns, output, steady = analyze(result, config, start, end, stride=2)
        history.append(dict(warmup_s=s['warmup_s'], start_s=start, end_s=end,
                            stationarity=steady))
        print(json.dumps(dict(folder=str(folder), warmup=s['warmup_s'],
            E_um=metrics['E_dir_um'], steady=steady['passed'])), flush=True)
        write_json(folder/'progress.json', dict(history=history, wall_seconds=time.perf_counter()-began))
        if steady['passed'] or s['warmup_s'] >= s['maximum_warmup_s']:
            break
        s['warmup_s'] = min(s['maximum_warmup_s'], s['warmup_s']+s['extension_s'])
    fine, _, _, _ = analyze(result, config, start, end)
    sampling = compare_metrics(metrics, fine, config)
    sampling.update(base_fs_hz=s['fs_raw_hz'], fine_fs_hz=fs,
                    same_dense_solution=True, independent_refinement_applied=True)
    cut = np.searchsorted(result['t'], np.floor((start-s['guard_s'])*fs)/fs)
    cut -= cut % int(fs/s['fs_out_hz'])
    dense_use = slice(cut, None)
    use = slice(cut, None, 2)
    t, q, v, acc = (result[key][use] for key in ('t', 'q', 'v', 'acc'))
    valid = (t >= start) & (t < end)
    # Full twice-rate center motion is retained to reproduce the peak test.
    dense_extra = {}
    if config.get('analysis_version') == 'v8':
        dense_extra = dict(q_raw_m=result['q'][dense_use], v_raw_m_s=result['v'][dense_use])
    np.savez_compressed(folder/'sampling_131072.npz', t_raw=result['t'][dense_use],
        r_m_raw_m=result['measured_relative_xy_m'][dense_use], **dense_extra)
    contacts = {}; diagnostics = {}; forces = {}; relative = {}
    wrench_arrays = {}
    rear = model.rear_bearing_parameters(p)
    specifications = [('F1', p, p['z_f'], 6), ('F2', p, p['z_f2'], 10)]
    specifications += [(h['name'], rear, h['z_m'], 4) for h in p['rear_supports']]
    for name, bp, z, outer in specifications:
        r = model.rotor_section(q, z)-q[:, outer:outer+2]
        vr = model.rotor_section(v, z)-v[:, outer:outer+2]
        if config.get('analysis_version') == 'v8':
            wrench = model.bearing_wrench_at(t, q, v, p, name)
            force, state = wrench['force_N'], wrench['diagnostics']
            wrench_arrays.update({name+'_radial_restoring_N': force,
                name+'_local_restoring_Nm': wrench['moment_Nm'],
                name+'_axial_restoring_N': state['axial_restoring_N']})
        else:
            force, state = model.bearing_contact_state(t, r, vr, bp)
        forces[name] = force; relative[name] = r
        delta, normal = state['delta_m'], state['normal_force_N']
        contacts[name+'_delta_m'] = delta
        contacts[name+'_normal_force_N'] = normal
        diagnostics[name] = dict(delta_min_m=float(delta[valid].min()),
            delta_max_m=float(delta[valid].max()), Q_min_N=float(normal[valid].min()),
            Q_max_N=float(normal[valid].max()),
            delta_over_diameter_max=float(delta[valid].max()/bp['ball_diameter']),
            delta_over_overlap_max=float(delta[valid].max()/bp['normal_overlap']),
            mean_reaction_N=np.mean(force[valid], axis=0).tolist(),
            peak_reaction_N=float(np.linalg.norm(force[valid], axis=1).max()))
        if config.get('analysis_version') == 'v8':
            spec = next(row for row in p['bearing_layout'] if row['name'] == name)
            eta = spec['rho_m']*np.sin(bp['contact_angle'])*np.linalg.norm(q[valid,8:10],axis=1)/bp['normal_overlap']
            diagnostics[name].update(eta_tilt_max=float(eta.max()),
                eta_tilt_above_half=bool(eta.max() > .5),
                maximum_actual_tilt_compression_over_overlap=float(abs(state['tilt_compression_m'][valid]).max()/bp['normal_overlap']),
                outer_tilt_constraint='fixed; not removed on radial fit opening')
    np.savez_compressed(folder/'bearing_contacts.npz', t_raw=t, **contacts)
    domain = domain_check(t[valid], q[valid], config, diagnostics)
    f1, d1 = model.fit_contact_summary(q[:, 6:8]-q[:, 4:6], v[:, 6:8]-v[:, 4:6], p)
    _, f2, d2 = model.front2_forces(q, v, p, summary=True, t=t)
    metrics.update(domain_pass=domain['passed'], sampling_pass=sampling['passed'],
        force_N=config['excitation']['peak_force_N'], U_kg_m=config['excitation']['U_kg_m'],
        wall_seconds=time.perf_counter()-began, nfev=result['nfev'])
    tout, hv, ha = output
    metadata = dict(config=config, export_start_s=start, export_end_s=end,
        actual_solver_history=solver_history,
        state_order=['Sx','Sy','Tx','Ty','Hx','Hy','O1x','O1y','theta_x','theta_y','O2x','O2y'],
        coordinate_units=['m']*8+['rad']*2+['m']*2,
        velocity_units=['m/s']*8+['rad/s']*2+['m/s']*2,
        acceleration_units=['m/s2']*8+['rad/s2']*2+['m/s2']*2,
        main_metric='max(mean(turn_PP_x),mean(turn_PP_y)); includes 1X',
        reference='rotor M at fixed z_m relative fixed probe; conditional model',
        processing='Native SI; common guarded FIR, main 10-900 Hz; no normalization')
    arrays = dict(t_raw=t, q_raw_m=q, v_raw_m_s=v, a_raw_m_s2=acc,
        r_m_raw_m=result['measured_relative_xy_m'][use],
        M_relative_H_m=model.rotor_section(q,p['z_m'])-q[:,4:6],
        S_relative_H_m=q[:,:2]-q[:,4:6], rO1_raw_m=q[:,6:8]-q[:,4:6],
        rO2_raw_m=q[:,10:12]-q[:,4:6], fit_force_N=f1, fit2_force_N=f2,
        fit_active_count=d1['active_count'], fit2_active_count=d2['active_count'],
        fit_max_compression_m=d1['max_compression_m'], fit2_max_compression_m=d2['max_compression_m'],
        bearing1_force_N=forces['F1'], bearing2_force_N=forces['F2'],
        rear_force_N=np.stack([forces[h['name']] for h in p['rear_supports']], axis=1),
        bearing1_relative_m=relative['F1'], bearing2_relative_m=relative['F2'],
        rear_relative_m=np.stack([relative[h['name']] for h in p['rear_supports']], axis=1),
        q_equilibrium=q0, equilibrium_residual_N=static_residual,
        t_out=tout, H_velocity_mm_s=hv, H_acceleration_m_s2=ha,
        state_final=result['state_final'], integration_end_s=result['t'][-1],
        metadata_json=json.dumps(metadata, ensure_ascii=False),
        **{'turn_'+key:value for key,value in turns.items()}, **wrench_arrays)
    np.savez_compressed(folder/'native.npz', **arrays)
    angle = p['rpm']*2*np.pi/60*t+config['excitation']['phase_rad']
    xy = config['excitation']['peak_force_N']*np.column_stack((np.cos(angle),np.sin(angle)))
    np.savez_compressed(folder/'actual_input.npz',t_raw=t,angle_rad=angle,force_xy_N=xy,
                        generalized_rotor_force=xy@model.section_matrix(p['z_u']))
    write_json(folder/'input_check.json', dict(status='completed',
        generalized_rotor_indices=[0, 1, 8, 9],
        generalized_rotor_units=['N', 'N', 'N m', 'N m'],
        force_N=config['excitation']['peak_force_N'], U_kg_m=config['excitation']['U_kg_m'],
        force_norm_error_N=float(np.max(abs(np.linalg.norm(xy,axis=1)-config['excitation']['peak_force_N'])))))
    for filename, value in [('metrics',metrics),('stationarity',steady),('warmup_history',history),
        ('domain',domain),('bearing_diagnostics',diagnostics),('sampling_convergence',sampling),
        ('actual_solver_settings',solver_history),('actual_config',config)]:
        write_json(folder/(filename+'.json'),value)
    rows=[]
    for i in range(len(turns['turn_id'])):
        row={key:float(value[i]) for key,value in turns.items() if value.ndim==1}
        row.update(Ex_um=float(turns['peak_to_peak_um'][i,0]),Ey_um=float(turns['peak_to_peak_um'][i,1]))
        rows.append(row)
    write_csv(folder/'turns.csv',rows)
    if config.get('analysis_version') == 'v8':
        from utils.clearance_v8_analysis import saved_wrench_report
        saved_wrench_report(folder)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/clearance_v6_config.json'))
    parser.add_argument('--output',type=Path,default=Path('experiments/clearance_v6_GMN'))
    parser.add_argument('--group',default='data')
    parser.add_argument('--gaps',type=float,nargs='+',default=[2,3,5,7.5,10,0])
    parser.add_argument('--forces',type=float,nargs='+')
    parser.add_argument('--directions',type=int)
    parser.add_argument('--step-factor',type=float,default=1.)
    parser.add_argument('--warmup',type=float)
    parser.add_argument('--turns',type=int)
    parser.add_argument('--seed',type=Path)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    baseline=json.loads(args.config.read_text(encoding='utf-8'))
    jobs=[]
    for force in args.forces or [None]:
        for gap in args.gaps:
            config=case_config(baseline,gap,force,args.directions,args.step_factor)
            if args.warmup is not None: config['simulation']['warmup_s']=args.warmup
            if args.turns is not None: config['simulation']['retained_turns']=args.turns
            parent=args.output/args.group
            if force is not None: parent=parent/f'F_{force:g}N'
            jobs.append((config,parent/f'gap_{gap:g}um',args.seed))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(execute,*job):job[1] for job in jobs}
        for future in as_completed(futures):
            try:
                metrics=future.result()
                print(json.dumps(dict(folder=str(futures[future]),E_um=metrics['E_dir_um'],
                    steady=metrics['steady_pass'],domain=metrics['domain_pass'])),flush=True)
            except Exception as error:
                write_json(futures[future]/'failed.json',dict(error=str(error)))
                raise


if __name__=='__main__': main()
