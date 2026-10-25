"""Generate group-isolated spindle velocity records and native-state turn labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

from spindle.dynamics_model import ball_spin_speed, cage_speed, simulate
from scripts.generate_platform_faults import case_parameters
from utils.data_io import save_json
from utils.pitting_analysis import filter_vibration, resample_to
from utils.runout_labels import complete_turn_runout, turns_in_window


ROOT = Path(__file__).resolve().parents[1]


def group_definition(config, group):
    """Engineering variability, fixed before outcomes; shared by all 13 cases."""
    rng = np.random.default_rng(group['seed'])
    variability = config['variability']
    support = {name: float(rng.uniform(*variability['support_multiplier_range']))
               for name in variability['support_parameters']}
    base = config['base']['parameters']
    overrides = {name: base[name]*factor for name, factor in support.items()}
    overrides['cage_phase'] = float(rng.uniform(0, 2*np.pi))
    base_drive = config['base']['excitation']
    amplitude = base_drive['peak_force_N']*rng.uniform(*variability['fundamental_multiplier_range'])
    frequency = np.r_[base_drive['frequency_hz'],
                      [rng.uniform(*band) for band in variability['additional_frequency_ranges_hz']]]
    amplitudes = np.r_[amplitude, amplitude*rng.uniform(
        *variability['additional_peak_fraction_range'], len(frequency)-1)]
    phases = rng.uniform(0, 2*np.pi, len(frequency))
    ellipticity = np.r_[1., rng.uniform(.6, 1., len(frequency)-1)]
    coefficients = amplitudes[:, None]*np.exp(1j*phases[:, None])*np.column_stack(
        (np.ones(len(frequency)), 1j*ellipticity))
    return dict(**group, parameter_overrides=overrides, support_multipliers=support,
        frequencies_hz=frequency.tolist(), peak_amplitudes_N=amplitudes.tolist(),
        phase_rad=phases.tolist(), y_to_x_amplitude=ellipticity.tolist(),
        coefficients_real=coefficients.real.tolist(), coefficients_imag=coefficients.imag.tolist())


def frequencies(parameters):
    rotation = abs(parameters['rpm'])/60
    cage = abs(cage_speed(parameters))/(2*np.pi)
    spin = abs(ball_spin_speed(parameters))/(2*np.pi)
    return dict(rotation=rotation, cage=cage, outer_pass=parameters['balls']*cage,
                inner_pass=parameters['balls']*(rotation-cage), ball_spin=spin,
                ball_waviness_2x=2*spin)


def severity_units(fault_type):
    return {'healthy': 'none', 'taper': 'fraction_of_nominal_contact_area',
            'outer': 'mm^2', 'inner': 'mm^2', 'ball': 'um'}[fault_type]


def generate_record(config, group, case, output, refinement=1, native_multiplier=1):
    started = time.perf_counter()
    sim_id = group['id']+'__'+case['case_id']
    print('Integrating '+sim_id, flush=True)
    settings = config['simulation']
    processing = config['base']['processing']
    fs = processing['fs_hz']
    native_fs = settings['native_fs_hz']*native_multiplier
    parameters = dict(case_parameters(config['base'], case), **group['parameter_overrides'])
    coefficients = np.array(group['coefficients_real'])+1j*np.array(group['coefficients_imag'])
    raw = simulate(parameters, case.get('lost_area', 0), group['frequencies_hz'], coefficients,
        duration=settings['duration_s']+2*processing['guard_s'], warmup=settings['warmup_s'],
        fs=native_fs, max_step=settings['max_step_s']/refinement,
        pit_steps_per_passage=settings['pit_steps_per_passage']*refinement)
    begin = round(processing['guard_s']*fs)
    count = round(settings['duration_s']*fs)
    start_s = float(raw['t'][0]+processing['guard_s'])
    end_s = start_s+settings['duration_s']
    velocity = filter_vibration(raw['v'][:-1, [4, 5, 6]].T, native_fs, processing)[:, begin:begin+count]*1000
    position_um = raw['q'][:, :4]*1e6
    positions = resample_to(position_um[:-1].T, native_fs, fs)[:, begin:begin+count]
    turns = complete_turn_runout(raw['t'], position_um, parameters['rpm'], start_s, end_s)
    path = Path(output)/'records'/f'{sim_id}.npz'
    np.savez_compressed(path, velocity_mm_s=velocity.astype(np.float32),
        time_s=start_s+np.arange(count)/fs, shaft_position_um=positions[:2], tool_position_um=positions[2:4],
        native_time_s=raw['t'], native_position_um=position_um,
        native_velocity_mm_s=raw['v'][:, [4, 5, 6]].T*1000,
        native_fs_hz=native_fs, sample_rate_hz=fs, rpm=parameters['rpm'],
        turn_id=turns['turn_id'], turn_start_time_s=turns['start_time_s'],
        turn_end_time_s=turns['end_time_s'], turn_min_um=turns['min_um'],
        turn_max_um=turns['max_um'], turn_shaft_peak_to_peak_um=turns['peak_to_peak_um'][:, :2],
        turn_tool_peak_to_peak_um=turns['peak_to_peak_um'][:, 2:4])
    row = dict(sim_id=sim_id, group_id=group['id'], split=group['split'],
        case_id=case['case_id'], class_index=case['class_index'], fault_type=case['fault_type'],
        severity=case['severity'], severity_unit=severity_units(case['fault_type']),
        lost_area_m2=case.get('lost_area', 0), parameters=parameters, excitation=group,
        characteristic_frequencies_hz=frequencies(parameters),
        path=str(path.resolve()), start_time_s=start_s, end_time_s=end_s,
        sample_rate_hz=fs, native_fs_hz=native_fs, channel_order=['ACC2', 'ACC1', 'ACC3'],
        channel_meanings=['housing X', 'housing Y', 'platform Y'],
        velocity_unit='mm/s', displacement_unit='um',
        native_position_order=['shaft_X', 'shaft_Y', 'tool_X', 'tool_Y'],
        primary_label_coordinate='tool absolute lumped-node XY (no 15 mm section mapping)',
        turn_definition='phase=2*pi*rpm*t/60; complete rotations; native-state max minus min including linearly interpolated boundaries',
        position_processing='Native position labels have no bandpass; 8192 Hz position copies only anti-alias resampled',
        processing=processing, full_turn_count=len(turns['turn_id']),
        velocity_rms_mm_s=np.sqrt(np.mean(velocity**2, axis=1)).tolist(),
        mean_turn_tool_peak_to_peak_um=turns['peak_to_peak_um'][:, 2:4].mean(axis=0).tolist(),
        mean_turn_shaft_peak_to_peak_um=turns['peak_to_peak_um'][:, :2].mean(axis=0).tolist(),
        finite=bool(np.isfinite(velocity).all() and np.isfinite(position_um).all()),
        equilibrium_residual_N=float(raw['equilibrium_residual_N']), nfev=int(raw['nfev']),
        integration_max_step_s=float(raw['integration_max_step_s']),
        pit_steps_per_passage=int(raw['pit_steps_per_passage']), elapsed_s=time.perf_counter()-started)
    if not row['finite'] or velocity.shape != (3, count) or not len(turns['turn_id']):
        raise ValueError('Invalid source record: '+sim_id)
    save_json(path.with_suffix('.json'), row)
    print(f"Saved {sim_id}: {row['elapsed_s']:.1f} s", flush=True)
    return row


def export_dataset(config, output):
    """Build a compact dataset; full-rate displacement remains in each record."""
    output = Path(output)
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((output/'records').glob('*.json'))]
    windows, turn_rows, arrays = [], [], {}
    fs = config['base']['processing']['fs_hz']
    # All cases share the recorded constant-rpm time grid. Keep its explicit
    # unwrapped shaft angle once and reference it instead of duplicating it 156 times.
    if rows:
        phase_reference = next((row for row in rows if row['split'] == 'train'), rows[0])
        with np.load(phase_reference['path']) as record:
            np.savez_compressed(output/'shared_shaft_phase.npz',
                native_time_s=record['native_time_s'],
                native_shaft_angle_rad=2*np.pi*float(record['rpm'])/60*record['native_time_s'],
                phase_at_zero_rad=0., rpm=float(record['rpm']))
    for split in ('train', 'validation', 'test'):
        signals, labels, tool_labels, shaft_labels, ids = [], [], [], [], []
        for row in (r for r in rows if r['split'] == split):
            with np.load(row['path']) as record:
                turns = dict(start_time_s=record['turn_start_time_s'], end_time_s=record['turn_end_time_s'])
                for j, turn_id in enumerate(record['turn_id']):
                    turn_rows.append(dict(sim_id=row['sim_id'], group_id=row['group_id'], split=split,
                        case_id=row['case_id'], turn_id=int(turn_id), start_time_s=float(turns['start_time_s'][j]),
                        end_time_s=float(turns['end_time_s'][j]),
                        shaft_X_um=float(record['turn_shaft_peak_to_peak_um'][j, 0]),
                        shaft_Y_um=float(record['turn_shaft_peak_to_peak_um'][j, 1]),
                        tool_X_um=float(record['turn_tool_peak_to_peak_um'][j, 0]),
                        tool_Y_um=float(record['turn_tool_peak_to_peak_um'][j, 1])))
                for index in range(record['velocity_mm_s'].shape[-1]//fs):
                    start = row['start_time_s']+index
                    selected = turns_in_window(turns, start, start+1)
                    window_id = row['sim_id']+f'__w{index:03d}'
                    signal = record['velocity_mm_s'][:, index*fs:(index+1)*fs]
                    tool = record['turn_tool_peak_to_peak_um'][selected].mean(axis=0)
                    shaft = record['turn_shaft_peak_to_peak_um'][selected].mean(axis=0)
                    signals.append(signal); labels.append(row['class_index'])
                    tool_labels.append(tool); shaft_labels.append(shaft); ids.append(window_id)
                    windows.append(dict(window_id=window_id, sim_id=row['sim_id'], group_id=row['group_id'],
                        split=split, dataset_index=len(signals)-1, case_id=row['case_id'],
                        class_index=row['class_index'], fault_type=row['fault_type'],
                        severity=row['severity'], severity_unit=row['severity_unit'],
                        start_time_s=start, end_time_s=start+1,
                        matched_healthy_window_id=row['group_id']+f'__healthy__w{index:03d}',
                        complete_turn_ids=record['turn_id'][selected].tolist(),
                        overlapping_turn_ids=record['turn_id'][
                            (turns['end_time_s'] > start) & (turns['start_time_s'] < start+1)].tolist(),
                        tool_mean_peak_to_peak_um=tool.tolist(), shaft_mean_peak_to_peak_um=shaft.tolist(),
                        tool_p95_peak_to_peak_um=np.percentile(record['turn_tool_peak_to_peak_um'][selected], 95, axis=0).tolist(),
                        tool_max_peak_to_peak_um=record['turn_tool_peak_to_peak_um'][selected].max(axis=0).tolist(),
                        shaft_p95_peak_to_peak_um=np.percentile(record['turn_shaft_peak_to_peak_um'][selected], 95, axis=0).tolist(),
                        shaft_max_peak_to_peak_um=record['turn_shaft_peak_to_peak_um'][selected].max(axis=0).tolist(),
                        complete_turn_count=len(selected), label_scope='complete revolutions wholly inside this velocity window',
                        label_provenance='simulation_state', runout_reference='absolute inertial XY coordinates',
                        runout_section='tool and shaft concentrated model nodes; not mapped to the 15 mm physical section',
                        angle_source=str((output/'shared_shaft_phase.npz').resolve()),
                        label_aggregation='mean of complete-turn X/Y peak-to-peak within window; individual turns retained',
                        source_path=row['path']))
        arrays['X_'+split] = np.asarray(signals, dtype=np.float32).reshape(-1, 3, fs)
        arrays['y_'+split] = np.asarray(labels, dtype=np.int64)
        arrays['E_'+split] = np.asarray(tool_labels, dtype=np.float64).reshape(-1, 2)
        arrays['E_shaft_'+split] = np.asarray(shaft_labels, dtype=np.float64).reshape(-1, 2)
        arrays['window_id_'+split] = np.asarray(ids, dtype='U80')
    dataset_path = output/'dataset.npz'
    temporary_path = output/'dataset.next.npz'
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(dataset_path)
    save_json(output/'windows.json', windows)
    save_json(output/'records.json', rows)
    if turn_rows:
        with (output/'turn_labels.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=list(turn_rows[0]))
            writer.writeheader(); writer.writerows(turn_rows)
    expected = len(config['groups'])*len(config['base']['cases'])
    summary = dict(completed=len(rows) == expected, expected_records=expected, completed_records=len(rows),
        independent_groups=len({row['group_id'] for row in rows}),
        windows_by_split={split: len(arrays['X_'+split]) for split in ('train', 'validation', 'test')},
        complete_turns=len(turn_rows), class_definitions=config['base']['cases'],
        sample_rate_hz=fs, velocity_unit='mm/s', label_unit='um',
        phase_reference='arbitrary simulation shaft phase zero at t=0; negative rpm rotates clockwise in model',
        primary_label='tool concentrated-node XY absolute displacement peak-to-peak per complete turn',
        limitations=config['limitations'])
    save_json(output/'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ('completed_records', 'windows_by_split')}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/spindle_sources_20261022.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--groups', nargs='*')
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    base_path = ROOT/config['base_config']
    config['base'] = json.loads(base_path.read_text(encoding='utf-8'))
    for index, case in enumerate(config['base']['cases']):
        case['class_index'] = index
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'records').mkdir(exist_ok=True)
    frozen_path = args.output/'frozen_config.json'
    if frozen_path.exists():
        if json.loads(frozen_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Output already contains a different frozen config; choose a new output directory')
        if not (args.resume or args.export_only):
            raise ValueError('Existing output requires --resume or --export-only')
    else:
        save_json(frozen_path, config)
    snapshot = args.output/'source_code'
    snapshot.mkdir(exist_ok=True)
    for source in (ROOT/'src/spindle/dynamics_model.py', Path(__file__), ROOT/'utils/runout_labels.py',
                   ROOT/'scripts/check_runout_labels.py', ROOT/'scripts/check_spindle_sources.py'):
        shutil.copyfile(source, snapshot/source.name)
    if args.export_only:
        export_dataset(config, args.output)
        return
    groups = [group_definition(config, group) for group in config['groups']]
    save_json(args.output/'groups.json', groups)
    pending = []
    for group in groups:
        if args.groups and group['id'] not in args.groups:
            continue
        for case in config['base']['cases']:
            if args.cases and case['case_id'] not in args.cases:
                continue
            path = args.output/'records'/(group['id']+'__'+case['case_id']+'.json')
            if not (args.resume and path.exists()):
                pending.append((group, case))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate_record, config, group, case, args.output) for group, case in pending]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 13 == 0:
                export_dataset(config, args.output)
    export_dataset(config, args.output)


if __name__ == '__main__':
    main()
