"""Run conditional V9 Tai bearing cases and retain solver-level evidence."""
import argparse
import copy
import json
import sys
import time
import zlib
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from spindle import dynamics_model as model
from scripts.run_clearance_v3 import export_window, write_csv, write_json
from utils import clearance_v9_analysis as analysis
from utils import clearance_v9_parameters as v9


ARRAY_KEYS = ('t', 'q', 'v', 'acc', 'measured_relative_xy_m', 'fit_force_N', 'fit_active_count',
              'fit2_force_N', 'fit2_active_count')


def _variant_config(config, variant):
    config = copy.deepcopy(config)
    if config.get('analysis_version') != 'v9':
        config = v9.build_config(variant or 'N3', base=config)
    elif variant is not None:
        if variant not in v9.MODES:
            raise ValueError(f'Unknown V9 mechanism variant: {variant}')
        config['mechanism_variant'] = variant
        p = config['parameters']
        p['bearing_contact_model'] = 'relative_tilt_v8' if variant.startswith('B0') else 'quasistatic_tai_v9'
        p['damping_policy'] = 'v8_clipped_contact' if variant == 'B0' else 'common_structural_ring'
        p.setdefault('tai_bearing', {})['ball_inertia_mode'] = v9.MODES[variant] or 'none'
    if not all(family.get('initialized_parameter_vector')
               for family in config['parameters'].get('tai_bearing', {}).get('families', {}).values()):
        config = v9.initialize_preload(config)
    return config


def _append_result(previous, current):
    if previous is None:
        return {key: current[key] for key in ARRAY_KEYS} | {
            'state_final': current['state_final'], 'nfev': current['nfev']}
    merged = {key: np.concatenate((previous[key], current[key][1:]), axis=0)
              for key in ARRAY_KEYS}
    merged.update(state_final=current['state_final'], nfev=previous['nfev']+current['nfev'])
    return merged


def _domain_check(result, steady, config, local_metrics):
    p = config['parameters']
    tilt = float(np.linalg.norm(result['q'][:, 8:10], axis=1).max())
    tilt_limit = float(config.get('constitutive_domains', {}).get('maximum_tilt_rad', .001))
    root_limit = float(config['v9_acceptance']['local_force_absolute_N'])
    geometry_limit = float(config['v9_acceptance']['local_geometry_normalized'])
    local_checks = {}
    for key, row in local_metrics.items():
        if key.endswith('_max_force_residual_tolerance_ratio'):
            local_checks[key] = dict(value=float(row), tolerance=1.0,
                                     passed=bool(row <= 1.0))
        elif key.endswith('_max_geometry_residual_normalized'):
            local_checks[key] = dict(value=float(row), tolerance=geometry_limit,
                                     passed=bool(row <= geometry_limit))
        elif key.endswith('_status_failed_count'):
            local_checks[key] = dict(value=float(row),tolerance=0.,passed=bool(row==0))
    control = steady.get('ring_control_self_consistency', {})
    phase = [float(row['value']) for row in steady.get('phase_speed_warnings', {}).values()]
    phase_limit = float(config['v9_acceptance']['phase_speed_warning_fraction'])
    phase_max = max(phase, default=0.0)
    checks = dict(finite_state=bool(np.isfinite(result['q']).all()
                                    and np.isfinite(result['v']).all()
                                    and np.isfinite(result['acc']).all()),
        small_angle=bool(tilt <= tilt_limit),
        local_root_residuals=bool(all(row['passed'] for row in local_checks.values())))
    return dict(passed=bool(all(checks.values())), checks=checks,
        values=dict(maximum_tilt_rad=tilt, maximum_tilt_limit_rad=tilt_limit,
            max_local_force_residual_N=max((float(row) for key,row in local_metrics.items()
                if key.endswith('_max_force_residual_N')), default=None),
            max_local_force_residual_tolerance_ratio=max((row['value'] for key,row in local_checks.items()
                if key.endswith('_max_force_residual_tolerance_ratio')), default=None),
            max_local_geometry_residual_normalized=max((row['value'] for key,row in local_checks.items()
                if key.endswith('_max_geometry_residual_normalized')), default=None),
            max_relative_phase_speed_difference=phase_max,
            phase_speed_warning_fraction=phase_limit),
        local_contact_checks=local_checks, ring_control_checks=control,
        phase_speed_warnings=steady.get('phase_speed_warnings', {}),
        physical_closure_status='conditional' if any(not row['passed'] for row in control.values()) or phase_max > phase_limit else 'within_declared_screens',
        branch_comparison_required=bool(any(not row['passed'] for row in control.values())),
        phase_speed_warning_exceeded=bool(phase_max > phase_limit),
        contact_pressure_strength='not_validated_missing_running_groove_curvature_limit',
        scope='Conditional Tai local-contact and fixed-ring screen; no unsupported Hertz pressure cutoff.')


def _failed(folder, error):
    record = dict(error=str(error), error_type=type(error).__name__)
    state = getattr(error, 'state', None)
    if state is not None:
        record['failure_state_saved'] = True
        np.savez_compressed(Path(folder)/'failure_state.npz',
            t_s=np.asarray(state.get('t_s', np.nan)),
            q=np.asarray(state.get('q', []), dtype=float),
            v=np.asarray(state.get('v', []), dtype=float))
        record['local_failure'] = {key:value for key,value in state.items()
                                   if key not in ('q','v')}
    write_json(Path(folder)/'failed.json', record)


def execute(config, folder, seed_path=None, resume_path=None):
    folder = Path(folder)
    if (folder/'metrics.json').exists():
        saved = json.loads((folder/'actual_config.json').read_text(encoding='utf-8'))
        for key in ('parameters', 'excitation', 'processing', 'mechanism_variant'):
            if saved.get(key) != config.get(key):
                raise ValueError(f'Existing V9 case has different physics: {folder} ({key})')
        for key, value in config['simulation'].items():
            if key not in ('warmup_s', 'maximum_warmup_s', 'extension_s') and \
                    saved['simulation'].get(key) != value:
                raise ValueError(f'Existing V9 case has different solver settings: {folder} ({key})')
        return json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/'actual_config.json', config)
    source_paths = [Path(__file__), Path(model.__file__), Path(analysis.__file__),
                    Path(analysis.tai_bearing.__file__), Path(analysis.tai_system.__file__)]
    write_json(folder/'implementation_at_start.json',dict(
        algorithm='CRC32', files=[dict(path=str(path),bytes=path.stat().st_size,
            crc32=f'{zlib.crc32(path.read_bytes()) & 0xffffffff:08x}') for path in source_paths],
        note='Source identities recorded at case start; no SHA256. Earlier campaign cases use source_at_A and the recorded later numerical-solver patch.'))
    p = copy.deepcopy(config['parameters'])
    sim = config['simulation']
    try:
        q0, static_residual = model.equilibrium_continuation(p)
    except Exception as error:
        _failed(folder,error)
        raise
    atol_q = np.full(12, sim['atol_translation_m'])
    atol_q[8:10] = sim['atol_angle_rad']
    atol_v = np.full(12, sim['atol_velocity_m_s'])
    atol_v[8:10] = sim['atol_angle_rate_rad_s']
    dense_fs = int(sim['fs_raw_hz']*2)
    result, history, solver_history, accepted, offsets = None, [], [], [], [0]
    epoch, seed = 0.0, None
    inherited_wall_seconds = 0.
    if resume_path is not None:
        if seed_path is not None:
            raise ValueError('Use either an independent seed or a same-case continuation')
        resume_path = Path(resume_path)
        previous_config = json.loads((resume_path/'actual_config.json').read_text(encoding='utf-8'))
        for key in ('parameters','excitation','processing'):
            if previous_config[key] != config[key]:
                raise ValueError(f'Same-case continuation changed {key}')
        for key in ('method','rtol','max_step_s','fs_raw_hz','atol_translation_m',
                    'atol_velocity_m_s','atol_angle_rad','atol_angle_rate_rad_s'):
            if previous_config['simulation'][key] != sim[key]:
                raise ValueError(f'Same-case continuation changed numerical setting {key}')
        with np.load(resume_path/'integration_checkpoint.npz') as data:
            result={key:data[key] for key in ARRAY_KEYS}
            result.update(state_final=data['state_final'],nfev=int(data['nfev']))
        with np.load(resume_path/'accepted_steps.npz') as data:
            accepted=[data['step_s']]
            offsets=data['segment_offsets'].tolist()
        solver_history=json.loads((resume_path/'actual_solver_settings.json').read_text(encoding='utf-8'))
        prior_metrics=json.loads((resume_path/'metrics.json').read_text(encoding='utf-8'))
        inherited_wall_seconds=prior_metrics['wall_seconds']
        write_json(folder/'continuation_provenance.json',dict(source=str(resume_path),
            original_export_start_s=prior_metrics['measurement_start_s'],
            original_export_end_s=prior_metrics['measurement_end_s'],
            inherited_nfev=result['nfev'],inherited_wall_seconds=inherited_wall_seconds,
            physical_and_numerical_parameters_identical=True,
            operation='Extend the original cold-start trajectory from its exact saved final state.'))
    if seed_path is not None:
        with np.load(Path(seed_path)/'native.npz') as data:
            seed = data['state_final']
            epoch = float(data['integration_end_s'])
        np.savez_compressed(folder/'initial_seed.npz', state=seed, absolute_time_s=epoch)
        sim['warmup_s'] += epoch
        sim['maximum_warmup_s'] += epoch
    started = time.perf_counter()
    final_start = final_end = None
    try:
        while True:
            start, end = export_window(config)
            keep = np.floor((start-sim['guard_s'])*dense_fs)/dense_fs
            finish = np.ceil((end+sim['guard_s'])*dense_fs)/dense_fs
            previous = epoch if result is None else float(result['t'][-1])
            new = model.simulate(p, 0, [], [], warmup=keep-epoch if result is None else 0,
                duration=finish-keep if result is None else finish-previous,
                fs=dense_fs, max_step=sim['max_step_s'], rtol=sim['rtol'],
                atol=np.r_[atol_q, atol_v], method=sim['method'], q_initial=q0,
                t_start=previous, state_initial=seed if result is None else result['state_final'],
                unbalance=(config['excitation']['U_kg_m'], config['excitation']['phase_rad']))
            new_steps = np.asarray(new['integration_accepted_steps_s'], dtype=float)
            accepted.append(new_steps)
            offsets.append(offsets[-1]+len(new_steps))
            solver_history.append(new['actual_solver_settings'])
            result = _append_result(result, new)
            # Preserve the integration before filtering or diagnostic export.
            np.savez_compressed(folder/'integration_checkpoint.npz', **result)
            write_json(folder/'actual_config.json', config)
            final_start, final_end = start, end
            metrics, _, _, steady, _ = analysis.analyze(
                result, config, start, end, stride=2,
                include_wrench_timeseries=False, include_ball_window=False)
            row = dict(warmup_s=float(sim['warmup_s']), start_s=start, end_s=end,
                       signal_steady_pass=steady['signal_stationarity_pass'],
                       bearing_steady_pass=steady['bearing_stationarity_pass'],
                       steady_pass=steady['passed'])
            history.append(row)
            write_json(folder/'progress.json', dict(history=history,
                wall_seconds=time.perf_counter()-started))
            print(json.dumps(dict(folder=str(folder), warmup_s=sim['warmup_s'],
                signal_steady=row['signal_steady_pass'], bearing_steady=row['bearing_steady_pass']),
                ensure_ascii=False), flush=True)
            if steady['passed'] or sim['warmup_s'] >= sim['maximum_warmup_s']:
                break
            sim['warmup_s'] = min(sim['maximum_warmup_s'], sim['warmup_s']+sim['extension_s'])
        # Keep the complete physical window with full per-ring evidence at base rate.
        metrics, turns, output, steady, evidence = analysis.analyze(
            result, config, final_start, final_end, stride=2,
            include_wrench_timeseries=True, include_ball_window=True)
        fine_metrics, _, _, _, _ = analysis.analyze(
            result, config, final_start, final_end, stride=1,
            include_wrench_timeseries=False, include_ball_window=False)
        sampling = analysis.compare_metrics(metrics, fine_metrics, config)
        sampling.update(base_fs_hz=sim['fs_raw_hz'], dense_fs_hz=dense_fs,
                        same_dense_solution=True, independent_refinement_applied=False)
        domain = _domain_check(result, steady, config, evidence['metrics'])
        metrics.update(domain_pass=domain['passed'], sampling_pass=sampling['passed'],
            force_N=config['excitation']['peak_force_N'], U_kg_m=config['excitation']['U_kg_m'],
            wall_seconds=time.perf_counter()-started+inherited_wall_seconds, nfev=result['nfev'],
            continuation_wall_seconds=time.perf_counter()-started,
            inherited_wall_seconds=inherited_wall_seconds,
            equilibrium_residual_N=float(static_residual),
            physical_closure_status=domain['physical_closure_status'],
            simulated_seconds=float(result['t'][-1]-epoch),
            seconds_per_simulated_second=(time.perf_counter()-started+inherited_wall_seconds)/(result['t'][-1]-epoch))
        # The native product is the base-rate path; preserve the dense q/v path separately.
        t_dense = np.asarray(result['t'])
        start_index = int(np.searchsorted(t_dense, np.floor((final_start-sim['guard_s'])*dense_fs)/dense_fs))
        start_index -= start_index % 2
        t_dense = t_dense[start_index:]
        q_dense, v_dense = result['q'][start_index:], result['v'][start_index:]
        base = slice(None, None, 2)
        t, q, vel, acc = (np.asarray(result[key])[start_index:][base]
                          for key in ('t','q','v','acc'))
        fit1, fit1_count, fit2, fit2_count = (
            np.asarray(result[key])[start_index:][base] for key in
            ('fit_force_N','fit_active_count','fit2_force_N','fit2_active_count'))
        relative = model.rotor_section(q, p['z_m']) - (
            q[:,4:6] if p.get('probe_reference') == 'housing' else np.zeros((len(q),2)))
        tout, h_velocity, h_acceleration = output
        metadata = dict(config=config, export_start_s=final_start, export_end_s=final_end,
            actual_solver_history=solver_history, state_order=['Sx','Sy','Tx','Ty','Hx','Hy',
                'O1x','O1y','theta_x','theta_y','O2x','O2y'],
            coordinate_units=['m']*8+['rad']*2+['m']*2,
            velocity_units=['m/s']*8+['rad/s']*2+['m/s']*2,
            main_metric='max(mean(turn_PP_x),mean(turn_PP_y)); retains 1X',
            measurement_reference=p.get('probe_reference'),
            analysis_scope='Conditional V9 mechanism; no real-machine validation implied.')
        np.savez_compressed(folder/'native.npz', t_raw=t, q_raw_m=q, v_raw_m_s=vel,
            a_raw_m_s2=acc, r_m_raw_m=relative, fit_force_N=fit1,
            fit_active_count=fit1_count, fit2_force_N=fit2, fit2_active_count=fit2_count,
            M_relative_H_m=model.rotor_section(q,p['z_m'])-q[:,4:6],
            S_relative_H_m=q[:,:2]-q[:,4:6],rO1_raw_m=q[:,6:8]-q[:,4:6],
            rO2_raw_m=q[:,10:12]-q[:,4:6],
            q_equilibrium=q0, equilibrium_residual_N=static_residual,
            state_final=result['state_final'], integration_end_s=float(result['t'][-1]),
            t_out=tout, H_velocity_mm_s=h_velocity, H_acceleration_m_s2=h_acceleration,
            metadata_json=json.dumps(metadata, ensure_ascii=False),
            **{'turn_'+key:value for key,value in turns.items()})
        np.savez_compressed(folder/f'sampling_{dense_fs}.npz', t_raw=t_dense,
                            q_raw_m=q_dense, v_raw_m_s=v_dense)
        accepted_steps = np.concatenate(accepted) if accepted else np.empty(0)
        np.savez_compressed(folder/'accepted_steps.npz', step_s=accepted_steps,
                            segment_offsets=np.asarray(offsets, dtype=np.int64))
        analysis.save_bearing_evidence(folder, evidence, config)
        angle = p['rpm']*2*np.pi/60*t+config['excitation']['phase_rad']
        xy = config['excitation']['peak_force_N']*np.column_stack((np.cos(angle),np.sin(angle)))
        np.savez_compressed(folder/'actual_input.npz',t_raw=t,angle_rad=angle,force_xy_N=xy,
            generalized_rotor_force=xy@model.section_matrix(p['z_u']))
        write_json(folder/'input_check.json', dict(status='completed',
            U_kg_m=config['excitation']['U_kg_m'], force_N=config['excitation']['peak_force_N'],
            rpm_signed=p['rpm'], phase_rad=config['excitation']['phase_rad'],
            force_norm_error_N=float(np.max(abs(np.linalg.norm(xy,axis=1)-config['excitation']['peak_force_N']))),
            generalized_rotor_indices=[0,1,8,9],generalized_rotor_units=['N','N','Nm','Nm'],
            generalized_force='analytic circular unbalance at z_u; same signed phase as RHS'))
        write_json(folder/'metrics.json', metrics)
        write_json(folder/'stationarity.json', steady)
        write_json(folder/'sampling_convergence.json', sampling)
        write_json(folder/'domain.json', domain)
        write_json(folder/'warmup_history.json', history)
        write_json(folder/'actual_solver_settings.json', solver_history)
        write_json(folder/'actual_config.json',config)
        write_csv(folder/'turns.csv', [dict(turn_id=int(turns['turn_id'][i]),
            start_s=float(turns['start_time_s'][i]), end_s=float(turns['end_time_s'][i]),
            Ex_um=float(turns['peak_to_peak_um'][i,0]), Ey_um=float(turns['peak_to_peak_um'][i,1]))
            for i in range(len(turns['turn_id']))])
    except Exception as error:
        _failed(folder, error)
        raise
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/clearance_v9_config.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', default='data_A')
    parser.add_argument('--gaps', type=float, nargs='+', default=[2,3,5,7.5,10])
    parser.add_argument('--U', '--u', dest='U', type=float)
    parser.add_argument('--variant', choices=tuple(v9.MODES))
    parser.add_argument('--root-factor', type=float, default=1.)
    parser.add_argument('--step-factor', type=float, default=1.)
    parser.add_argument('--N', '--directions', dest='directions', type=int)
    parser.add_argument('--turns', type=int)
    parser.add_argument('--warmup', type=float)
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--resume',type=Path,
        help='Extend one completed case from integration_checkpoint.npz; retain its original cold-start history')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    base = _variant_config(json.loads(args.config.read_text(encoding='utf-8')), args.variant)
    if args.U is not None:
        base['excitation']['U_kg_m'] = args.U
    variant = base.get('mechanism_variant', 'N3')
    jobs = []
    for gap in args.gaps:
        config = v9.case_config(base, gap, U=args.U, directions=args.directions,
            step_factor=args.step_factor, root_factor=args.root_factor,
            turns=args.turns, warmup=args.warmup)
        if config['simulation']['warmup_s'] > config['simulation']['maximum_warmup_s']:
            config['simulation']['maximum_warmup_s'] = config['simulation']['warmup_s']
        parent = args.output/args.group/variant
        if args.U is not None:
            parent = parent/f"U_{config['excitation']['U_kg_m']:.12g}"
        folder = parent/f"gap_{gap:g}um"
        if args.resume is not None and len(args.gaps)!=1:
            raise ValueError('--resume extends exactly one specified gap')
        jobs.append((config, folder, args.seed, args.resume))
    args.output.mkdir(parents=True, exist_ok=True)
    run_parent = args.output/args.group/variant
    run_parent.mkdir(parents=True,exist_ok=True)
    write_json(run_parent/'actual_command.json',dict(
        executable=sys.executable,argv=sys.argv,working_directory=str(Path.cwd()),
        started_utc=datetime.now(timezone.utc).isoformat()))
    write_json(run_parent/'run_settings.json', dict(config=base, group=args.group,
        gaps=args.gaps, workers=args.workers, root_factor=args.root_factor,
        step_factor=args.step_factor))
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, *job): job[1] for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
                print(json.dumps(dict(folder=str(futures[future]), E_dir_um=result.get('E_dir_um'),
                    steady_pass=result.get('steady_pass'), domain_pass=result.get('domain_pass')),
                    ensure_ascii=False), flush=True)
            except Exception as error:
                _failed(futures[future],error)
                failures.append(dict(folder=str(futures[future]), error=str(error)))
                print(json.dumps(failures[-1], ensure_ascii=False), flush=True)
    write_json(run_parent/'run_failures.json', failures)
    if failures:
        raise SystemExit(f'{len(failures)} V9 case(s) failed; inspect run_failures.json')


if __name__ == '__main__':
    main()
