"""V6 fixed-U physical-scale plots, including excluded results and calibration history."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram


def plot(root, base_group='data'):
    root = Path(root)
    output = root/'figures'; output.mkdir(exist_ok=True)
    acceptance = json.loads((root/'recomputed/acceptance.json').read_text(encoding='utf-8'))
    cases = sorted(acceptance['cases'], key=lambda c: c['metrics']['gap_um'])
    health = next(c for c in cases if c['metrics']['gap_um'] == 2)
    healthy_u = health['metrics']['U_kg_m']
    if any(c['metrics']['U_kg_m'] != healthy_u for c in cases):
        raise ValueError('Different U values cannot be plotted as one degradation curve')
    valid = [c for c in cases if c['main_valid']]
    candidates = valid or [c for c in cases if c['metrics'].get('steady_pass') and c['domain'].get('passed')]
    largest = max(candidates or [health], key=lambda c: c['metrics']['E_dir_um'])
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 140, 'axes.grid': True, 'grid.alpha': .25})
    generated = []

    def save(fig, name):
        fig.text(.5, .009, 'Fixed U; physical amplitudes | Red x: unsteady/outside screen | Black ring: independently certified | Dashed lines: guides',
                 ha='center', fontsize=7)
        fig.tight_layout(rect=(0, .045, 1, 1))
        fig.savefig(output/name); plt.close(fig); generated.append(name)

    def curve(ax, key, label, horizontal='gap_um'):
        x = np.array([c['metrics'][horizontal] for c in cases])
        y = np.array([c['metrics'][key] for c in cases])
        line, = ax.plot(x, y, '--', alpha=.45, label=label)
        stable = np.array([bool(c['metrics'].get('steady_pass') and c['domain'].get('passed')) for c in cases])
        certified = np.array([c['main_valid'] for c in cases])
        ax.scatter(x[stable], y[stable], color=line.get_color(), s=25)
        ax.scatter(x[~stable], y[~stable], color='crimson', marker='x', s=45)
        ax.scatter(x[certified], y[certified], facecolors='none', edgecolors='black', s=70)

    fig, axes = plt.subplots(1, 3, figsize=(13.3, 4))
    for key, label in [('E_x_um', 'X'), ('E_y_um', 'Y'), ('E_dir_um', 'E_dir')]:
        curve(axes[0], key, label)
    axes[0].axhspan(1.9, 2.1, color='green', alpha=.1, label='Healthy target')
    axes[0].axhline(20, color='.6', ls=':', label='20 um reference')
    for prefix, label in [('fit_', 'Front fit 1'), ('fit2_', 'Front fit 2')]:
        curve(axes[1], prefix+'contact_fraction', label)
    for axis in 'xy':
        curve(axes[2], f'mean_{axis}_um', axis.upper())
    for ax, title, unit in zip(axes, ('Absolute directional PP, includes 1X', 'Outer-ring contact occupancy', 'Static mean, excluded from PP'),
                               ('Mean per-turn PP (um)', 'Contact fraction', 'Axis-center mean (um)')):
        ax.set(xlabel='Common single-side radial gap (um)', ylabel=unit, title=title)
        ax.axvline(2, color='.5', ls=':'); ax.legend(fontsize=8)
    save(fig, 'gap_motion_contact.png')

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis in 'xy':
        for ax, key in zip(axes[0], (f'H_v_rms_{axis}_mm_s', f'H_v_1x_{axis}_mm_s', f'H_v_residual_{axis}_mm_s')):
            curve(ax, key, axis.upper())
        for ax, key in zip(axes[1], (f'H_v_rms_{axis}_mm_s', f'H_v_1x_{axis}_mm_s', f'H_v_residual_{axis}_mm_s')):
            curve(ax, key, axis.upper(), 'E_dir_um')
    for column, title in enumerate(('Total RMS', '1X peak amplitude', 'Non-1X RMS')):
        for row in range(2):
            axes[row, column].set(title=title+' (10-900 Hz)', ylabel='H velocity (mm/s)',
                xlabel='Gap (um)' if row == 0 else 'E_dir (um)')
            axes[row, column].legend()
    save(fig, 'gap_vibration_and_motion_scatter.png')

    response_file = root/'validation/response_classification.json'
    if response_file.exists():
        response = json.loads(response_file.read_text(encoding='utf-8'))
        repeats = {r['case']:r for r in response['cases'] if r['group']==base_group}
        fig, axes = plt.subplots(len(cases), 2, figsize=(11.7, 2.25*len(cases)), squeeze=False)
        for row, case in enumerate(cases):
            with (root/base_group/case['case']/'turns.csv').open(encoding='utf-8-sig') as stream:
                turns = list(csv.DictReader(stream))
            xy = np.array([[float(r['Ex_um']),float(r['Ey_um'])] for r in turns])
            info = repeats.get(case['case'], {})
            repeat = info.get('approximate_PP_repeat_turns')
            for axis, color in enumerate(('C0','C1')):
                axes[row,0].plot(np.arange(1,13),xy[:12,axis],'o-',ms=3,color=color,label='XY'[axis]+' per-turn PP')
            high=float(xy.max())
            bins=np.linspace(0,max(high*1.08,.005),49)
            axes[row,1].hist([xy[:,0],xy[:,1]],bins=bins,label=['X (all turns)','Y (all turns)'],
                             color=['C0','C1'],alpha=.65)
            formal=case['metrics']['E_dir_um']
            axes[row,0].axhline(formal,color='black',ls='--',lw=1,label='Formal max(mean X, mean Y)')
            axes[row,1].axvline(formal,color='black',ls='--',lw=1,label='Formal E_dir')
            axes[row,1].axvline(float(xy.max()),color='crimson',ls=':',lw=1.2,label='Largest single-turn PP')
            axes[row,0].set(title=f'c={case["metrics"]["gap_um"]:g} um; near {repeat}-turn PP repeat',
                            xlabel='Turn index in export window (first 12)',ylabel='Directional PP (um)',
                            ylim=(0,max(high*1.08,.005)))
            axes[row,1].set(title=f'All {len(xy)} turns: formal {formal:.6g} vs single max {xy.max():.6g} um',
                            xlabel='Directional PP (um)',ylabel='Turn count')
            if row==0:
                axes[row,0].legend(fontsize=7);axes[row,1].legend(fontsize=7)
        fig.suptitle('Per-turn PP recurrence is diagnostic, not proof of a full-system period',fontsize=11)
        save(fig,'per_turn_PP.png')

    signals = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for case in [health] if health is largest else [health, largest]:
        native = root/base_group/case['case']/'native.npz'
        with np.load(native) as data:
            metadata = json.loads(str(data['metadata_json']))
            period = 60/abs(metadata['config']['parameters']['rpm'])
            end = metadata['export_end_s']
            use = (data['t_raw'] >= end-3*period) & (data['t_raw'] < end)
            xy = data['r_m_raw_m'][use]*1e6
            label = f'c={case["metrics"]["gap_um"]:g} um'
            axes[0].plot(*xy.T, label=label)
            axes[1].plot(*(xy-xy.mean(axis=0)).T, label=label)
            signals.append(dict(case=case, time=data['t_out'].copy(), velocity=data['H_velocity_mm_s'].copy(),
                acceleration=data['H_acceleration_m_s2'].copy(), period=period, end=end,
                fs=metadata['config']['simulation']['fs_out_hz'],
                outer1=data['rO1_raw_m'][use]*1e6, outer2=data['rO2_raw_m'][use]*1e6))
    for ax, title in zip(axes, ('M relative fixed probe', 'Mean removed; 1X retained')):
        ax.set(xlabel='X (um)', ylabel='Y (um)', title=title, aspect='equal'); ax.legend()
    fig.suptitle(f'Healthy c=2 vs largest reported representative; U={healthy_u*1e6:.5g} g mm')
    save(fig, 'same_U_axis_trajectories.png')

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4))
    for bearing in ('F1', 'F2', 'R1', 'R2'):
        available = [c for c in cases if 'mean_reaction_N' in c['domain'].get('bearings', {}).get(bearing, {})]
        axes[0].plot([c['metrics']['gap_um'] for c in available],
            [c['domain']['bearings'][bearing]['mean_reaction_N'][1] for c in available],
            'o--', label=bearing)
    axes[0].set(xlabel='Common gap (um)', ylabel='Mean Y bearing force on outer support (N)',
                title='Static/dynamic-average load transfer')
    axes[0].legend()
    for signal in signals:
        for number in (1, 2):
            axes[number].plot(*signal[f'outer{number}'].T,
                label=f'c={signal["case"]["metrics"]["gap_um"]:g} um')
    for number in (1, 2):
        axes[number].set(xlabel=f'O{number}-H X (um)', ylabel=f'O{number}-H Y (um)',
                         title=f'Front outer ring {number}, last 3 turns', aspect='equal')
        axes[number].legend()
    save(fig, 'bearing_load_and_outer_orbits.png')

    for name, unit in (('velocity', 'mm/s'), ('acceleration', 'm/s2')):
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex='col')
        peak = 0.
        for signal in signals:
            use = signal['time'] >= signal['end']-3*signal['period']
            peak = max(peak, float(np.max(np.abs(signal[name][use]))))
            frequency, psd = periodogram(signal[name], fs=signal['fs'], window='hann', axis=0)
            for axis in range(2):
                label = f'c={signal["case"]["metrics"]["gap_um"]:g} um'
                axes[axis, 0].plot((signal['time'][use]-signal['end'])*1000, signal[name][use, axis], label=label)
                axes[axis, 1].semilogy(frequency, np.maximum(psd[:, axis], 1e-24), label=label)
        for axis in range(2):
            axes[axis, 0].set(ylabel=f'H {"XY"[axis]} ({unit})', ylim=(-1.1*peak, 1.1*peak))
            axes[axis, 1].set(ylabel=f'PSD (({unit})2/Hz)', xlim=(0, 1100))
            for ax in axes[axis]: ax.legend()
        axes[1, 0].set_xlabel('Time relative end (ms)'); axes[1, 1].set_xlabel('Frequency (Hz)')
        fig.suptitle(f'Same U: housing {name}, primary 10-900 Hz')
        save(fig, f'same_U_housing_{name}.png')

    history = root/'calibration_history.csv'
    if history.exists():
        with history.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        # Accept the names used in either the runner or final human-readable history.
        x, y = [], []
        for row in rows:
            force = row.get('force_N', row.get('F_N'))
            e = row.get('E_dir_um', row.get('E_um'))
            if force not in (None, '') and e not in (None, ''):
                x.append(float(force)); y.append(float(e))
        if x:
            fig, ax = plt.subplots(figsize=(7.2, 4))
            ax.scatter(x, y)
            for i, (a, b) in enumerate(zip(x, y), 1): ax.annotate(str(i), (a, b), xytext=(4, 4), textcoords='offset points', fontsize=8)
            ax.axhspan(1.9, 2.1, color='green', alpha=.15)
            ax.set(xlabel='Equivalent rotating-force peak at 7000 rpm (N)', ylabel='Healthy E_dir (um)',
                   title='Health calibration attempts at fixed c=2 um; labels show order')
            save(fig, 'health_calibration_history.png')
    (output/'plot_sources.json').write_text(json.dumps(dict(fixed_U_kg_m=healthy_u,
        health_case=health['case'], largest_representative=largest['case'], no_amplitude_scaling=True,
        primary_band_hz=[10, 900], figures=generated), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    plot(parser.parse_args().root)
