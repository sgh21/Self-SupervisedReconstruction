"""Audit spindle simulation records and refine selected training simulations."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.generate_spindle_sources import generate_record
from spindle.dynamics_model import harmonic_response
from utils.data_io import save_json
from utils.pitting_analysis import filter_vibration
from utils.runout_labels import complete_turn_runout


def plot_training_sources(rows, output):
    from scripts.analyze_platform_faults import (
        CHANNELS, COLORS, band_psd, configure_style, plt, save_figure,
    )
    configure_style()
    names = ('healthy', 'taper_50', 'outer_0p8', 'inner_0p8', 'ball_0p4')
    selected = {row['case_id']: row for row in rows if row['group_id'] == 'tr00'}
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), constrained_layout=True)
    label_fig, label_axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for name, color in zip(names, COLORS):
        if name not in selected:
            continue
        row = selected[name]
        with np.load(row['path']) as record:
            values = record['velocity_mm_s']
            frequency, power = band_psd(values, 8192)
            time_ms = (record['time_s'][:410]-record['time_s'][0])*1000
            for channel in range(3):
                axes[channel, 0].plot(time_ms, values[channel, :410], color=color, lw=.8, label=name)
                axes[channel, 1].semilogy(frequency, power[channel], color=color, lw=.8, label=name)
                axes[channel, 0].set(ylabel=CHANNELS[channel]+' (mm/s)', xlabel='Time from valid start (ms)')
                axes[channel, 1].set(xlim=(10, 2048), xlabel='Frequency (Hz)', ylabel='PSD ((mm/s)^2/Hz)')
            for axis, key, title in ((0, 'shaft_position_um', 'Shaft'), (1, 'tool_position_um', 'Tool T')):
                orbit = record[key][:, :819]-record[key][:, :819].mean(axis=1, keepdims=True)
                label_axes[axis, 0].plot(*orbit, color=color, lw=.8, label=name)
                label_axes[axis, 0].set(xlabel='X relative to interval mean (um)',
                                        ylabel='Y relative to interval mean (um)', title=title+' first 0.1 s orbit')
                label_axes[axis, 0].set_aspect('equal', adjustable='datalim')
                turns = record['turn_shaft_peak_to_peak_um' if axis == 0 else 'turn_tool_peak_to_peak_um']
                turn_number = record['turn_id'][:30]-record['turn_id'][0]
                label_axes[axis, 1].plot(turn_number, turns[:30, 0], color=color, lw=1, label=name+' X')
                label_axes[axis, 1].plot(turn_number, turns[:30, 1], color=color, lw=1, ls='--', label=name+' Y')
                label_axes[axis, 1].set(xlabel='Complete turn from valid start', ylabel='Peak-to-peak (um)', title=title+' native-state labels')
    axes[0, 0].legend(ncol=3, fontsize=8)
    label_axes[0, 0].legend(fontsize=8)
    label_axes[0, 1].text(.02, .98, 'Solid: X; dashed: Y', va='top', transform=label_axes[0, 1].transAxes)
    fig.suptitle('Training group tr00: shared forcing, physical amplitudes, 10-2048 Hz')
    label_fig.suptitle('Same simulations: positions from state; no housing integration')
    save_figure(fig, output/'figures', 'source_velocity')
    save_figure(label_fig, output/'figures', 'source_positions_and_turns')


def relative_l2(actual, reference):
    return np.linalg.norm(actual-reference, axis=-1)/np.maximum(np.linalg.norm(reference, axis=-1), 1e-30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--refine-cases', nargs='+', default=['outer_0p2', 'inner_0p8'])
    parser.add_argument('--refine-group',default='tr00')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'records').mkdir(exist_ok=True)
    config = json.loads((args.sources/'frozen_config.json').read_text(encoding='utf-8'))
    rows = json.loads((args.sources/'records.json').read_text(encoding='utf-8'))
    windows = json.loads((args.sources/'windows.json').read_text(encoding='utf-8'))
    groups = {row['group_id']: row['split'] for row in rows}
    assert all(groups[row['group_id']] == row['split'] for row in rows)
    known_windows = {row['window_id'] for row in windows}
    missing_matches = [row['window_id'] for row in windows if row['matched_healthy_window_id'] not in known_windows]
    assert not missing_matches
    shape_checks, reproducibility = [], []
    with np.load(args.sources/'dataset.npz') as dataset:
        for split in ('train', 'validation', 'test'):
            x = dataset['X_'+split]
            assert x.shape[1:] == (3, 8192) and np.isfinite(x).all()
            assert len(x) == len(dataset['y_'+split]) == len(dataset['E_'+split])
            shape_checks.append(dict(split=split, shape=list(x.shape)))
    # Numerical tests stay inside training sources, independently of held-out outcomes.
    for row in rows:
        if row['split'] != 'train':
            continue
        with np.load(row['path']) as record:
            recomputed = complete_turn_runout(record['native_time_s'], record['native_position_um'],
                float(record['rpm']), row['start_time_s'], row['end_time_s'])
            assert np.array_equal(recomputed['peak_to_peak_um'][:, 2:4], record['turn_tool_peak_to_peak_um'])
            assert np.array_equal(recomputed['peak_to_peak_um'][:, :2], record['turn_shaft_peak_to_peak_um'])
            assert len(record['turn_id']) == row['full_turn_count']
        reproducibility.append(row['sim_id'])
    diagnostics = []
    healthy = next(row for row in rows if row['split'] == 'train' and row['fault_type'] == 'healthy')
    with np.load(healthy['path']) as record:
        frequency = np.array(healthy['excitation']['frequencies_hz'])
        coefficient = (np.array(healthy['excitation']['coefficients_real'])
                       +1j*np.array(healthy['excitation']['coefficients_imag']))
        response = harmonic_response(healthy['parameters'], 0, frequency, coefficient)
        raw_time = record['native_time_s'][:-1]
        predicted = np.zeros((3, len(raw_time)))
        for f, mode in zip(frequency, response):
            predicted += np.real((mode[[4, 5, 6]]*2j*np.pi*f*1000)[:, None]
                                 *np.exp(2j*np.pi*f*raw_time))
        predicted = filter_vibration(predicted, healthy['native_fs_hz'], healthy['processing'])
        begin = round(healthy['processing']['guard_s']*8192)
        predicted = predicted[:, begin:begin+record['velocity_mm_s'].shape[-1]]
        error = relative_l2(record['velocity_mm_s'][:, -8192:], predicted[:, -8192:])
        diagnostics.append(dict(sim_id=healthy['sim_id'], check='last_second_vs_linear_steady_response',
                                relative_l2_per_channel=error.tolist(), threshold=.03,
                                passed=bool((error < .03).all())))
    refinements = []
    if args.refine:
        selected = [row for row in rows if row['group_id'] == args.refine_group and row['case_id'] in args.refine_cases]
        if not selected:raise ValueError('No requested training cases found for refinement')
        cases = {case['case_id']: case for case in config['base']['cases']}
        for row in selected:
            fine = generate_record(config, row['excitation'], cases[row['case_id']], args.output,
                                   refinement=2, native_multiplier=2)
            with np.load(row['path']) as coarse_record, np.load(fine['path']) as fine_record:
                velocity_error = relative_l2(coarse_record['velocity_mm_s'], fine_record['velocity_mm_s'])
                shaft_error = relative_l2(coarse_record['turn_shaft_peak_to_peak_um'].T,
                                          fine_record['turn_shaft_peak_to_peak_um'].T)
                tool_error = relative_l2(coarse_record['turn_tool_peak_to_peak_um'].T,
                                         fine_record['turn_tool_peak_to_peak_um'].T)
            refinements.append(dict(sim_id=row['sim_id'],
                coarse_native_fs_hz=row['native_fs_hz'], fine_native_fs_hz=fine['native_fs_hz'],
                coarse_step_s=row['integration_max_step_s'], fine_step_s=fine['integration_max_step_s'],
                velocity_relative_l2_per_channel=velocity_error.tolist(),
                shaft_turn_label_relative_l2_xy=shaft_error.tolist(),
                tool_turn_label_relative_l2_xy=tool_error.tolist(), threshold=.005,
                passed=bool(np.max(np.r_[velocity_error, shaft_error, tool_error]) < .005),
                note='Combined native sampling and integration refinement; not separated error estimates.'))
    result = dict(passed=all(item['passed'] for item in diagnostics+refinements),
        record_count=len(rows), independent_groups=len(groups), window_count=len(windows),
        shape_checks=shape_checks, matched_healthy_window_check=True, group_split_leakage_check=True,
        native_turn_label_exact_reproduction_count=len(reproducibility), diagnostics=diagnostics,
        refinements=refinements,
        scope='Training-label and numerical checks only; no held-out test similarity or fault scores inspected.')
    save_json(args.output/'source_checks.json', result)
    if args.plots:
        plot_training_sources(rows, args.output)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise RuntimeError('Source verification did not pass')


if __name__ == '__main__':
    main()
