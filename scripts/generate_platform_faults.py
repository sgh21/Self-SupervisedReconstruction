"""Generate matched seven-DOF platform/fault cases and check numerical resolution."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import firwin, freqz

from spindle.dynamics_model import (
    displacement_metrics,
    harmonic_response,
    inner_race_pit_geometry,
    outer_race_pit_geometry,
    simulate,
)
from utils.data_io import save_json
from utils.pitting_analysis import filter_vibration, resample_to


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT/'config/platform_faults_20261022.json'
DEFAULT_OUTPUT = ROOT/'experiments/platform_faults_20261022'
SIMULATION = dict(native_fs_hz=65536, warmup_s=1., duration_s=3.,
                  max_step_s=1/8192, pit_steps_per_passage=4)
KINDS = ('acceleration', 'velocity', 'displacement')
HOLDERS = ('holder_absolute_um', 'holder_relative_platform_um')
RMS_KEYS = dict(acceleration='acceleration_rms_m_s2', velocity='velocity_rms_mm_s',
                displacement='displacement_rms_um')


def case_parameters(config, case):
    return dict(config['parameters'], **case.get('parameter_overrides', {}))


def drive(config):
    excitation = config['excitation']
    coefficients = (excitation['peak_force_N']*np.exp(1j*excitation['phase_rad'])
                    * np.array([[1., 1j*excitation.get('y_phase_sign', 1)]]))
    return np.array([excitation['frequency_hz']]), coefficients


def run_case(config, case, native_fs=None, refinement=1, duration=None):
    setting = dict(SIMULATION, **config.get('simulation', {}))
    parameters = case_parameters(config, case)
    frequency, coefficient = drive(config)
    native_fs = setting['native_fs_hz'] if native_fs is None else native_fs
    duration = setting['duration_s'] if duration is None else duration
    return simulate(parameters, case.get('lost_area', 0.), frequency, coefficient,
        duration=duration, warmup=setting['warmup_s'], fs=native_fs,
        max_step=setting['max_step_s']/refinement,
        pit_steps_per_passage=setting['pit_steps_per_passage']*refinement)


def process(raw, config, source_fs):
    """Bandlimit measured-node outputs; preserve unbandpassed holder positions."""
    processing = config['processing']
    fs, guard = processing['fs_hz'], processing['guard_s']
    start = round(guard*fs)
    stop = round((raw['t'][-1]-raw['t'][0]-guard)*fs)
    payload = {}
    for kind, key, multiplier in [('acceleration', 'acc', 1.),
                                  ('velocity', 'v', 1000.),
                                  ('displacement', 'q', 1e6)]:
        signal = raw[key][:-1, [4, 5, 6]].T*multiplier
        if kind == 'displacement':
            signal = signal-signal.mean(axis=1, keepdims=True)
        payload[kind] = filter_vibration(signal, source_fs, processing)[:, start:stop]
    absolute = raw['q'][:-1, 2:4].T
    relative = absolute.copy()
    relative[1] -= raw['q'][:-1, 6]
    for name, signal in zip(HOLDERS, (absolute, relative)):
        payload[name] = resample_to(signal*1e6, source_fs, fs)[:, start:stop]
    payload['time_s'] = np.arange(stop-start)/fs
    payload['t'] = payload['time_s']
    payload['absolute_start_time_s'] = float(raw['t'][0]+guard)
    return payload


def metrics(payload):
    result = {RMS_KEYS[kind]: np.sqrt(np.mean(payload[kind]**2, axis=1)).tolist()
              for kind in KINDS}
    for name in HOLDERS:
        result[name.removesuffix('_um')] = displacement_metrics(payload[name].T*1e-6)
    return result


def healthy_response_check(config, case, payload):
    """Use the generated record to compare early/late motion with linear steady response."""
    frequency, coefficient = drive(config)
    omega = 2*np.pi*frequency[0]
    response = harmonic_response(case_parameters(config, case), 0., frequency, coefficient)[0]
    processing = config['processing']
    taps = firwin(processing['fir_taps'], processing['band_hz'], pass_zero=False,
                  fs=processing['fs_hz'], window=('kaiser', processing['kaiser_beta']))
    gain = abs(freqz(taps, worN=frequency, fs=processing['fs_hz'])[1][0])
    time_grid = payload['absolute_start_time_s']+payload['time_s']
    wave = np.exp(1j*omega*time_grid)
    predicted = {}
    for kind, scale in [('acceleration', -omega**2), ('velocity', 1j*omega*1000),
                         ('displacement', 1e6)]:
        predicted[kind] = np.real((response[[4, 5, 6]]*scale*gain)[:, None]*wave)
    absolute = response[2:4].copy()
    relative = absolute.copy()
    relative[1] -= response[6]
    for key, amplitude in zip(HOLDERS, (absolute, relative)):
        predicted[key] = np.real(amplitude[:, None]*1e6*wave)
    count = round(.5*processing['fs_hz'])
    errors = {}
    for label, selection in [('first_0.5s', slice(0, count)), ('last_0.5s', slice(-count, None))]:
        actual_slice = {key: payload[key][:, selection] for key in (*KINDS, *HOLDERS)}
        predicted_slice = {key: predicted[key][:, selection] for key in (*KINDS, *HOLDERS)}
        errors[label] = compare(actual_slice, predicted_slice)
    threshold = .03
    return dict(linear_response_phasor_real_m=response.real.tolist(),
        linear_response_phasor_imag_m=response.imag.tolist(),
        relative_l2_vs_linear_steady_state=errors,
        last_interval_within_3percent=all(max(value) <= threshold for value in errors['last_0.5s'].values()),
        note='Independent steady-response diagnostic; integration convergence alone does not establish settled motion. Holder comparisons remove interval means.')


def save_case(config, case, output, raw, elapsed):
    native_fs = int(raw['output_fs_hz'])
    payload = process(raw, config, native_fs)
    parameters = case_parameters(config, case)
    metadata = dict(
        channel_order=['housing_X', 'housing_Y', 'platform_Y'],
        units=dict(acceleration='m/s^2', velocity='mm/s', displacement='um',
                   holder_absolute_um='um', holder_relative_platform_um='um',
                   raw_q='m', raw_v='m/s', raw_acc='m/s^2', time_s='s', raw_t='s'),
        raw_coordinate_order=['shaft_X', 'shaft_Y', 'holder_X', 'holder_Y',
                              'housing_X', 'housing_Y', 'platform_Y'],
        native_fs_hz=native_fs, processing=config['processing'],
        absolute_start_time_s=payload['absolute_start_time_s'],
        holder_processing='Relative Y formed before anti-alias resampling; no bandpass or mean removal.',
        displacement_processing='Native full-record mean removed before 10–2048 Hz FIR; no amplitude normalization. Raw and holder positions retain static values.',
        excitation=config['excitation'], parameters=parameters,
        lost_area_m2=case.get('lost_area', 0.),
        note='Platform is a one-direction model reference, not an extra spindle radial axis.',
    )
    path = Path(output)/'cases'/f"{case['case_id']}.npz"
    np.savez_compressed(path, **payload, raw_t=raw['t'], raw_q=raw['q'], raw_v=raw['v'],
        raw_acc=raw['acc'], raw_fs_hz=native_fs,
        processed_fs_hz=config['processing']['fs_hz'], metadata_json=json.dumps(metadata))
    finite = all(np.isfinite(payload[key]).all() for key in (*KINDS, *HOLDERS))
    if not finite:
        raise ValueError(f"Nonfinite processed case: {case['case_id']}")
    row = dict(case_id=case['case_id'], fault_type=case['fault_type'],
        severity=case['severity'], label=case['label'], path=str(path),
        parameters=parameters, lost_area_m2=case.get('lost_area', 0.),
        metrics=metrics(payload), finite=True, native_fs_hz=native_fs,
        geometry=dict(outer=outer_race_pit_geometry(parameters), inner=inner_race_pit_geometry(parameters)),
        nfev=int(raw['nfev']), integration_segments=int(raw['integration_segments']),
        integration_max_step_s=float(raw['integration_max_step_s']),
        pit_steps_per_passage=int(raw['pit_steps_per_passage']),
        equilibrium_residual_N=float(raw['equilibrium_residual_N']),
        elapsed_s=elapsed)
    if case['fault_type'] == 'healthy':
        row['steady_state_check'] = healthy_response_check(config, case, payload)
    return row


def generate_one(config, case, output, native_fs=None, refinement=1):
    started = time.perf_counter()
    print(f"Integrating {case['case_id']}", flush=True)
    raw = run_case(config, case, native_fs=native_fs, refinement=refinement)
    result = save_case(config, case, output, raw, time.perf_counter()-started)
    print(f"Saved {case['case_id']} ({result['elapsed_s']:.1f} s)", flush=True)
    return result


def load_payload(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in (*KINDS, *HOLDERS)}


def add_healthy_deltas(rows):
    healthy = load_payload(next(row['path'] for row in rows if row['fault_type'] == 'healthy'))
    for row in rows:
        payload = load_payload(row['path'])
        delta = {key: payload[key]-healthy[key] for key in healthy}
        row['metrics']['healthy_delta'] = metrics(delta)
        row['metrics']['healthy_delta']['definition'] = 'Pointwise matched-drive difference; holder metrics subtract the difference mean.'


def reprocess_cases(config, rows, output):
    updated = []
    cases = {case['case_id']: case for case in config['cases']}
    for row in rows:
        with np.load(row['path']) as saved:
            raw = {key: saved['raw_'+key] for key in ('t', 'q', 'v', 'acc')}
        raw.update(output_fs_hz=row['native_fs_hz'], nfev=row['nfev'],
            integration_segments=row['integration_segments'],
            integration_max_step_s=row['integration_max_step_s'],
            pit_steps_per_passage=row['pit_steps_per_passage'],
            equilibrium_residual_N=row['equilibrium_residual_N'])
        updated.append(save_case(config, cases[row['case_id']], output, raw, row['elapsed_s']))
    add_healthy_deltas(updated)
    save_json(Path(output)/'generation_summary.json', generation_summary(config, updated, True))
    return updated


def generation_summary(config, rows, completed):
    order = {case['case_id']: i for i, case in enumerate(config['cases'])}
    return dict(config=config, completed=completed,
                cases=sorted(rows, key=lambda row: order[row['case_id']]),
                output_channels=['housing_X', 'housing_Y', 'platform_Y'],
                processing=config['processing'],
                holder_metrics_note='Dynamic displacement metrics, not validated standard error-motion values.')


def compare(a, b):
    """Per-channel relative L2 on a shared interval; position metrics remove DC."""
    result = {}
    for key in (*KINDS, *HOLDERS):
        count = min(a[key].shape[-1], b[key].shape[-1])
        x, y = a[key][..., :count], b[key][..., :count]
        if key in HOLDERS or key == 'displacement':
            x, y = x-x.mean(axis=1, keepdims=True), y-y.mean(axis=1, keepdims=True)
        result[key] = (np.linalg.norm(x-y, axis=1)/np.linalg.norm(y, axis=1)).tolist()
    return result


def passed(errors, tolerances):
    return all(max(error) <= tolerances.get(key, .005) for key, error in errors.items())


def verify_one(config, case, row, output):
    """Separate integration error from output-sampling error with one finer solution."""
    verification = config.get('verification', {})
    seconds = verification.get('duration_s', .5)
    duration = seconds+2*config['processing']['guard_s']
    tolerances = dict(dict(acceleration=.02, velocity=.005, displacement=.005,
                          holder_absolute_um=.005, holder_relative_platform_um=.005),
                      **verification.get('relative_l2_tolerances', {}))
    native_fs = row['native_fs_hz']
    print(f"Refining {case['case_id']} at {2*native_fs} Hz", flush=True)
    fine = run_case(config, case, native_fs=2*native_fs, refinement=2, duration=duration)
    fine_payload = process(fine, config, 2*native_fs)
    decimated = dict(fine)
    for key in ('t', 'q', 'v', 'acc'):
        decimated[key] = fine[key][::2]
    # Deliberate point decimation emulates native coarse sampling of exactly
    # the same continuous solution; the difference reveals aliasing before FIR.
    sampled_payload = process(decimated, config, native_fs)
    sampling_error = compare(sampled_payload, fine_payload)
    coarse = load_payload(row['path'])
    integration_error = compare(coarse, sampled_payload)
    item = dict(case_id=case['case_id'], compared_duration_s=seconds,
        absolute_start_time_s=fine_payload['absolute_start_time_s'],
        coarse_native_fs_hz=native_fs, fine_native_fs_hz=2*native_fs,
        fine_max_step_s=float(fine['integration_max_step_s']),
        fine_pit_steps_per_passage=int(fine['pit_steps_per_passage']),
        sampling_relative_l2=sampling_error, integration_relative_l2=integration_error,
        sampling_passed=passed(sampling_error, tolerances),
        integration_passed=passed(integration_error, tolerances), tolerances=tolerances)
    # Preserve compact evidence, rather than another full-rate multisecond file.
    evidence = Path(output)/'verification'
    evidence.mkdir(exist_ok=True)
    np.savez_compressed(evidence/f"{case['case_id']}.npz",
        **{f'fine_{key}': fine_payload[key] for key in (*KINDS, *HOLDERS)},
        **{f'sampled_{key}': sampled_payload[key] for key in (*KINDS, *HOLDERS)},
        **{f'coarse_{key}': coarse[key][..., :fine_payload['time_s'].size]
           for key in (*KINDS, *HOLDERS)}, time_s=fine_payload['time_s'])
    item['passed'] = item['sampling_passed'] and item['integration_passed']
    print(f"Checked {case['case_id']}: sampling={item['sampling_passed']} integration={item['integration_passed']}", flush=True)
    return item


def settings_for_row(config, row):
    revised = dict(config)
    revised['simulation'] = dict(SIMULATION, **config.get('simulation', {}))
    revised['simulation'].update(max_step_s=row['integration_max_step_s'],
                                 pit_steps_per_passage=row['pit_steps_per_passage'])
    return revised


def refine_failed(config, case, row, check, output):
    rate = row['native_fs_hz']*(2 if not check['sampling_passed'] else 1)
    refinement = 2 if not check['integration_passed'] else 1
    revised = settings_for_row(config, row)
    replacement = generate_one(revised, case, output, native_fs=rate, refinement=refinement)
    updated = verify_one(settings_for_row(config, replacement), case, replacement, output)
    updated['previous_check'] = check
    return replacement, updated


def verify(config, rows, output, workers, resume=False):
    groups = {}
    for case in config['cases']:
        if case['fault_type'] != 'healthy':
            groups.setdefault(case['fault_type'], []).append(case)
    selected = {max(group, key=lambda c: c['severity'])['case_id']
                for group in groups.values()}
    # Output aliasing is not monotone with pit width, so check every pit size.
    for group in groups.values():
        if any('pit_area_m2' in key for key in group[0].get('parameter_overrides', {})):
            selected.update(case['case_id'] for case in group)
    by_id = {row['case_id']: row for row in rows}
    cases = {case['case_id']: case for case in config['cases']}
    checks = []
    verification_path = Path(output)/'verification.json'
    if resume:
        previous = json.loads(verification_path.read_text(encoding='utf-8'))['checks']
        # Explicit resume is for the same already-generated cases; never used
        # automatically when a fresh generation could have changed mechanics.
        checks = [check for check in previous if check['case_id'] in selected
                  and check['coarse_native_fs_hz'] == by_id[check['case_id']]['native_fs_hz']
                  and check['fine_max_step_s']*2 == by_id[check['case_id']]['integration_max_step_s']]
    pending = selected-{check['case_id'] for check in checks}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(verify_one, settings_for_row(config,by_id[name]),
                               cases[name], by_id[name], output): name for name in pending}
        for future in as_completed(futures):
            checks.append(future.result())
            save_json(verification_path, dict(completed=False, checks=checks))
    for _ in range(3):
        failed = [check for check in checks if not check['passed']]
        if not failed:
            break
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(refine_failed,config,cases[check['case_id']],
                                   by_id[check['case_id']],check,output) for check in failed]
            for future in as_completed(futures):
                replacement, updated = future.result()
                name = replacement['case_id']
                rows[rows.index(by_id[name])] = replacement
                by_id[name] = replacement
                index = next(i for i,check in enumerate(checks) if check['case_id']==name)
                checks[index] = updated
                save_json(verification_path, dict(completed=False, checks=checks))
                save_json(Path(output)/'generation_summary.json',generation_summary(config,rows,True))
    result = dict(completed=True, passed=all(check['passed'] for check in checks), checks=checks,
        note='Separate output-rate and integration refinements. Acceptance applies to retained 10–2048 Hz outputs and anti-aliased holder positions, not unresolved raw high-frequency carriers.')
    save_json(Path(output)/'verification.json', result)
    add_healthy_deltas(rows)
    save_json(Path(output)/'generation_summary.json', generation_summary(config, rows, True))
    if not result['passed']:
        raise RuntimeError('Numerical refinement still exceeds tolerance; inspect verification.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--resume-verification', action='store_true',
                        help='Reuse previous checks for the same existing cases; implies --verify-only.')
    parser.add_argument('--reprocess-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    (args.output/'cases').mkdir(parents=True, exist_ok=True)
    save_json(args.output/'frozen_config.json', config)
    summary_path = args.output/'generation_summary.json'
    if args.reprocess_only:
        reprocess_cases(config, json.loads(summary_path.read_text(encoding='utf-8'))['cases'], args.output)
        return
    if args.verify_only or args.resume_verification:
        rows = json.loads(summary_path.read_text(encoding='utf-8'))['cases']
    else:
        healthy = next(case for case in config['cases'] if case['fault_type'] == 'healthy')
        rows = [generate_one(config, healthy, args.output)]
        save_json(summary_path, generation_summary(config, rows, False))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(generate_one, config, case, args.output)
                       for case in config['cases'] if case['case_id'] != healthy['case_id']]
            for future in as_completed(futures):
                rows.append(future.result())
                save_json(summary_path, generation_summary(config, rows, False))
        add_healthy_deltas(rows)
        save_json(summary_path, generation_summary(config, rows, True))
    verify(config, rows, args.output, args.workers,resume=args.resume_verification)


if __name__ == '__main__':
    main()
