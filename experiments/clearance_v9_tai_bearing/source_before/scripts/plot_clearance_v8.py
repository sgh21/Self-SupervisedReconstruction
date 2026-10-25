"""Separate fixed-old-U mechanism comparisons and recalibrated V8 curves."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.plot_clearance_v6 import plot as common_plots


def plot(root):
    root = Path(root); output = root/'figures'; output.mkdir(exist_ok=True)
    if (root/'recomputed/acceptance.json').exists():
        common_plots(root, 'data_B_recalibrated')
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for label, group in [('V6 radial only', 'old'), ('V8 relative tilt', 'new')]:
        paths = {p.parent.name: p for p in (root/'data_A_same_U'/group).glob('gap*/metrics.json')}
        if group == 'old':
            paths.update({p.parent.name: p for p in (root/'data_A_same_U/old_matched').glob('gap*/metrics.json')})
        rows = sorted((read(p) for p in paths.values()), key=lambda row: row['gap_um'])
        if not rows:
            continue
        for ax, key in zip(axes, ('E_dir_um', 'H_v_rms_x_mm_s', 'H_v_rms_y_mm_s')):
            ax.plot([r['gap_um'] for r in rows], [r[key] for r in rows], 'o--', label=label)
    for ax, title, unit in zip(axes, ('Formal E_dir (1X retained)', 'Housing X, 10–900 Hz', 'Housing Y, 10–900 Hz'),
                              ('Mean per-turn PP (um)', 'Velocity RMS (mm/s)', 'Velocity RMS (mm/s)')):
        ax.set(xlabel='Single-side radial gap (um)', ylabel=unit, title=title)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('A: identical old U = 26.89135 g mm; no health recalibration')
    fig.tight_layout(); fig.savefig(output/'A_same_U_mechanism.png', dpi=150); plt.close(fig)
    rows = sorted((read(p) for p in (root/'data_B_recalibrated').glob('gap*/metrics.json')),
                  key=lambda row: row['gap_um'])
    if rows:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, name in zip(axes.flat, ('F1', 'F2', 'R1', 'R2')):
            for axis in 'xy':
                ax.plot([r['gap_um'] for r in rows],
                    [r[f'bearing_{name}_local_moment_rms_{axis}_Nm'] for r in rows],
                    'o--', label=axis.upper())
            ax.set(title=name, xlabel='Common radial gap (um)', ylabel='Local restoring moment RMS (N m)')
            ax.grid(alpha=.25); ax.legend()
        fig.suptitle('B: one frozen recalibrated U; local contact moments')
        fig.tight_layout(); fig.savefig(output/'B_local_moments.png', dpi=150); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for gap in (2, 10):
            folder = root/'data_B_recalibrated'/f'gap_{gap}um'
            if not (folder/'native.npz').exists():
                continue
            with np.load(folder/'native.npz') as data:
                meta = json.loads(str(data['metadata_json'])); t = data['t_raw']
                end = meta['export_end_s']; use = (t >= end-3*60/7000) & (t < end)
                for axis, ax in enumerate(axes):
                    ax.plot((t[use]-end)*1000, data['F1_local_restoring_Nm'][use,axis],
                            label=f'F1 local, c={gap} um')
            with np.load(folder/'constraint_reactions.npz') as data:
                for axis, ax in enumerate(axes):
                    ax.plot((t[use]-end)*1000, data['H_bearing_load_constraint_Nm'][use,axis],
                            '--', label=f'H constraint, c={gap} um')
        for ax, axis in zip(axes, 'XY'):
            ax.set(xlabel='Time relative window end (ms)', ylabel=f'{axis} moment (N m)')
            ax.grid(alpha=.25); ax.legend(fontsize=7)
        fig.suptitle('Local contact and constrained housing loads; different reference points')
        fig.tight_layout(); fig.savefig(output/'B_local_and_constraint_moments.png', dpi=150); plt.close(fig)
    return sorted(p.name for p in output.glob('*.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    print(json.dumps(plot(parser.parse_args().root)))
