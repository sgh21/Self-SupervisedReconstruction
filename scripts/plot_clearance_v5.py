"""Plot V5 physical responses without combining different excitation levels."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram


def read_cases(root):
    cases = []
    for file in (root/'data').glob('F_*N/gap_*/metrics.json'):
        native = file.with_name('native.npz')
        if not native.exists():
            continue
        metrics = json.loads(file.read_text(encoding='utf-8'))
        cfg = json.loads(file.with_name('actual_config.json').read_text(encoding='utf-8'))
        omega = cfg['parameters']['rpm']*2*np.pi/60
        force = cfg['excitation']['U_kg_m']*omega**2
        cases.append(dict(force=force, gap=metrics['gap_um'], metrics=metrics,
                          native=native, config=cfg))
    return sorted(cases, key=lambda x: (x['force'], x['gap']))


def plot(root, force_N=None):
    root = Path(root)
    out = root/'figures'
    out.mkdir(exist_ok=True)
    cases = read_cases(root)
    if not cases:
        raise ValueError('No completed data/F_*N/gap_* cases')
    acceptance_file = root/'recomputed/acceptance.json'
    accepted = set()
    if acceptance_file.exists():
        acceptance = json.loads(acceptance_file.read_text(encoding='utf-8'))
        accepted = {item['case'] for item in acceptance['cases'] if item.get('main_valid')}
    for case in cases:
        case['stationary_domain'] = bool(case['metrics'].get('steady_pass') and case['metrics'].get('domain_pass'))
        case['main_valid'] = str(case['native'].parent.relative_to(root/'data')).replace('\\', '/') in accepted
    forces = sorted(set(round(c['force'], 8) for c in cases))
    selected_force = max(forces) if force_N is None else force_N
    chosen = [c for c in cases if np.isclose(c['force'], selected_force)]
    if not chosen:
        raise ValueError(f'No completed cases at {selected_force:g} N')
    healthy = next((c for c in chosen if c['gap'] == 0), None)
    if healthy is None:
        raise ValueError('Same-force healthy control is required')
    degraded = [c for c in chosen if c['gap'] > 0 and c['stationary_domain']]
    candidate = max(degraded, key=lambda c: c['metrics']['E_dir_um']) if degraded else healthy
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 140,
                         'axes.grid': True, 'grid.alpha': .25})
    generated = []

    def save(fig, name):
        fig.text(.5, .012, 'Filled: steady and in domain | Red x: excluded, unsteady/out of domain | Black ring: main metric accepted | Dashed lines: guides',
                 ha='center', fontsize=7.3)
        fig.tight_layout(rect=(0, .055, 1, 1))
        fig.savefig(out/name)
        plt.close(fig)
        generated.append(name)

    def levels(ax):
        for value in (5, 10, 15, 20):
            ax.axhline(value, color='.7', lw=.7, ls='--')

    def curve(ax, group, x, key, label):
        xx = np.array([c[x] for c in group])
        yy = np.array([c['metrics'][key] for c in group])
        line, = ax.plot(xx, yy, '--', alpha=.4, linewidth=1, label=label)
        valid = np.array([c['stationary_domain'] for c in group], dtype=bool)
        main = np.array([c['main_valid'] for c in group], dtype=bool)
        ax.scatter(xx[valid], yy[valid], c=line.get_color(), marker='o', s=27, zorder=3)
        ax.scatter(xx[~valid], yy[~valid], c='crimson', marker='x', s=48, linewidths=1.5, zorder=4)
        ax.scatter(xx[main], yy[main], facecolors='none', edgecolors='black', s=74, linewidths=1, zorder=5)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for gap in (0, 10):
        group = [c for c in cases if c['gap'] == gap]
        curve(axes[0], group, 'force', 'E_dir_um', f'c={gap:g} um')
    for force in forces:
        group = [c for c in cases if np.isclose(c['force'], force)]
        curve(axes[1], group, 'gap', 'E_dir_um', f'{force:g} N fixed')
    for ax in axes:
        levels(ax)
        ax.set_ylabel('Mean per-turn directional PP (um), includes 1X')
        ax.legend()
    axes[0].set(xlabel='Rotating-force peak (N)', title='Paired endpoint experiments')
    axes[1].set(xlabel='Common single-side radial gap (um)', title='Separate curves at each fixed force')
    save(fig, 'excitation_and_gap_responses.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in [('E_x_um', 'X PP'), ('E_y_um', 'Y PP'), ('E_dir_um', 'E_dir')]:
        curve(axes[0], chosen, 'gap', key, label)
    levels(axes[0])
    axes[0].set(xlabel='Common radial gap (um)', ylabel='Mean per-turn PP (um)', title=f'Fixed force {selected_force:g} N')
    for number, prefix in ((1, 'fit_'), (2, 'fit2_')):
        curve(axes[1], chosen, 'gap', prefix+'contact_fraction', f'Fit {number}')
    axes[1].set(xlabel='Common radial gap (um)', ylabel='Contact time fraction', ylim=(-.03, 1.03))
    for ax in axes:
        ax.legend()
    save(fig, 'fixed_force_range_contact.png')

    controls = [c for c in cases if c['gap'] == 0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for key, label in [('E_x_um', 'X'), ('E_y_um', 'Y')]:
        curve(axes[0], controls, 'force', key, label)
    for axis in 'xy':
        for ax, key, unit in [(axes[1], f'H_v_rms_{axis}_mm_s', 'H velocity RMS (mm/s)'),
                              (axes[2], f'H_a_rms_{axis}_m_s2', 'H acceleration RMS (m/s2)')]:
            curve(ax, controls, 'force', key, axis.upper())
            ax.set_ylabel(unit)
    axes[0].set_ylabel('Healthy per-turn PP (um)')
    for ax in axes:
        ax.set_xlabel('Rotating-force peak (N), c=0')
        ax.legend()
    save(fig, 'healthy_response_vs_force.png')

    pair = [healthy] if candidate is healthy else [healthy, candidate]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    signals = []
    absolute, centered = [], []
    for c in pair:
        with np.load(c['native']) as data:
            metadata = json.loads(str(data['metadata_json']))
            end = metadata['export_end_s']
            period = 60/abs(c['config']['parameters']['rpm'])
            t = data['t_raw']
            use = (t >= end-3*period) & (t < end)
            xy = data['r_m_raw_m'][use]*1e6
            center = xy-xy.mean(axis=0)
            absolute.append(xy)
            centered.append(center)
            axes[0].plot(*xy.T, label=f'c={c["gap"]:g} um')
            axes[1].plot(*center.T, label=f'c={c["gap"]:g} um')
            signals.append(dict(case=c, end=end, period=period, t=data['t_out'].copy(),
                                v=data['H_velocity_mm_s'].copy(),
                                a=data['H_acceleration_m_s2'].copy()))
    for ax, groups, title in zip(axes, (absolute, centered),
                                 ('Absolute M relative fixed probe', 'Mean removed for display; 1X retained')):
        values = np.concatenate(groups)
        span = max(np.ptp(values, axis=0).max(), 1e-8)*1.1
        midpoint = (values.min(axis=0)+values.max(axis=0))/2
        ax.set(xlabel='M X (um)', ylabel='M Y (um)', title=title,
               xlim=midpoint[0]+np.array([-.5, .5])*span,
               ylim=midpoint[1]+np.array([-.5, .5])*span)
        ax.set_aspect('equal')
        ax.legend()
    fig.suptitle(f'Same-force comparison, {selected_force:g} N, last 3 turns')
    save(fig, 'same_force_trajectories.png')

    for signal, unit, filename in [('v', 'mm/s', 'same_force_housing_velocity.png'),
                                    ('a', 'm/s2', 'same_force_housing_acceleration.png')]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex='col')
        signal_peak, psd_peak = 0., 0.
        for item in signals:
            t, values, end = item['t'], item[signal], item['end']
            use = t >= end-3*item['period']
            fs = item['case']['config']['simulation']['fs_out_hz']
            frequency, psd = periodogram(values, fs=fs, window='hann', axis=0, scaling='density')
            signal_peak = max(signal_peak, np.max(np.abs(values[use])))
            psd_peak = max(psd_peak, np.max(psd))
            for axis in range(2):
                label = f'c={item["case"]["gap"]:g} um'
                axes[axis, 0].plot((t[use]-end)*1000, values[use, axis], label=label)
                axes[axis, 1].semilogy(frequency, np.maximum(psd[:, axis], 1e-24), label=label)
        for axis in range(2):
            axes[axis, 0].set(ylabel=f'H {"XY"[axis]} ({unit})', ylim=(-1.1*signal_peak, 1.1*signal_peak))
            axes[axis, 1].set(ylabel=f'PSD (({unit})2/Hz)', xlim=(0, 2048),
                               ylim=(max(psd_peak*1e-10, 1e-24), psd_peak*3))
            for ax in axes[axis]:
                ax.legend()
        axes[1, 0].set_xlabel('Time relative window end (ms)')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        fig.suptitle(f'Same-force H comparison: {selected_force:g} N; common physical scales')
        save(fig, filename)
    details = dict(selected_force_N=selected_force, candidate_gap_um=candidate['gap'],
                   selection='Largest steady, in-domain degraded E at selected force; refinement validity reported separately',
                   no_amplitude_scaling=True, figures=generated,
                   cases=[dict(force_N=c['force'], gap_um=c['gap'], native=str(c['native'].relative_to(root)),
                               steady_pass=c['metrics'].get('steady_pass'),
                               domain_pass=c['metrics'].get('domain_pass'),
                               main_valid=c['main_valid']) for c in cases])
    (out/'plot_sources.json').write_text(json.dumps(details, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--force-N', type=float)
    arguments = parser.parse_args()
    plot(arguments.root, arguments.force_N)
