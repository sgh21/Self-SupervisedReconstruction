"""Comparable V3 figures from saved native data; no signal scaling or fitting."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram


def plot_v3(output: Path, cases: list[Path], summaries: list[dict]):
    """Write figures and a Markdown legend; summaries and paths share order.

    The main displacement includes 1X. Mean removal and exact-frequency 1X
    subtraction are explicitly labelled display/diagnostic operations only.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    order = np.argsort([row['gap_um'] for row in summaries])
    rows = [summaries[i] for i in order]
    records = []
    for i in order:
        with np.load(cases[i], allow_pickle=False) as archive:
            record = {key: archive[key] for key in (
                't_raw', 'q_raw_m', 'v_raw_m_s', 'a_raw_m_s2', 'r_m_raw_m',
                't_out', 'H_velocity_mm_s', 'H_acceleration_m_s2',
                'fit_force_N', 'healthy_support_forces_N',
                'turn_start_s', 'turn_end_s', 'turn_pp_um')}
            meta = json.loads(str(archive['metadata_json']))
            keep = ((record['t_raw'] >= meta['export_start_s']) &
                    (record['t_raw'] < meta['export_end_s']))
            for key in ('t_raw', 'q_raw_m', 'v_raw_m_s', 'a_raw_m_s2',
                        'r_m_raw_m', 'fit_force_N', 'healthy_support_forces_N'):
                record[key] = record[key][keep]
            record['export_end_s'] = meta['export_end_s']
            records.append(record)
    colors = plt.cm.viridis(np.linspace(.08, .88, len(rows)))
    labels = [f'c = {row["gap_um"]:g} µm' for row in rows]
    paths = []

    def save(fig, name):
        path = output / name
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    def finish(axes):
        for ax in np.asarray(axes).flat:
            ax.grid(alpha=.22)
            ax.ticklabel_format(axis='both', style='plain', useOffset=False)

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'lines.linewidth': 1.0, 'axes.spines.top': False,
                         'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
        for d, color, label in zip(records, colors, labels):
            r = d['r_m_raw_m'] * 1e6
            axes[0].plot(r[:, 0], r[:, 1], color=color, label=label)
            dynamic = r - r.mean(axis=0)
            axes[1].plot(dynamic[:, 0], dynamic[:, 1], color=color)
        for ax in axes:
            ax.set(xlabel='M − fixed probe: X (µm)', ylabel='Y (µm)')
            ax.set_aspect('equal', adjustable='datalim')
        axes[0].set_title('Absolute reference; static offset retained')
        axes[1].set_title('Mean removed for display; 1X retained')
        axes[0].legend(fontsize=8)
        finish(axes)
        save(fig, 'v3_trajectories.png')

        fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, layout='constrained')
        for d, color, label in zip(records, colors, labels):
            t = d['t_out']
            duration = 3 * np.mean(d['turn_end_s'] - d['turn_start_s'])
            end = d['export_end_s']
            keep = t >= end - duration
            for channel in range(2):
                axes[0, channel].plot(t[keep] - end,
                    d['H_velocity_mm_s'][keep, channel], color=color, label=label)
                axes[1, channel].plot(t[keep] - end,
                    d['H_acceleration_m_s2'][keep, channel], color=color)
        for channel, direction in enumerate('XY'):
            axes[0, channel].set(title=f'H {direction}; final three turns', ylabel='Velocity (mm/s)')
            axes[1, channel].set(xlabel='Time relative to export end (s)', ylabel='Acceleration (m/s²)')
        axes[0, 0].legend(fontsize=8)
        finish(axes)
        save(fig, 'v3_housing_waveforms.png')

        fig, axes = plt.subplots(4, 2, figsize=(11, 10), layout='constrained')
        for d, color, label in zip(records, colors, labels):
            t = d['t_out']
            fs = 1 / np.median(np.diff(t))
            frequency = 1 / np.mean(d['turn_end_s'] - d['turn_start_s'])
            design = np.column_stack((np.ones(len(t)),
                np.cos(2*np.pi*frequency*t), np.sin(2*np.pi*frequency*t)))
            velocity = d['H_velocity_mm_s']
            residual = velocity - design @ np.linalg.lstsq(design, velocity, rcond=None)[0]
            fs_raw = 1 / np.median(np.diff(d['t_raw']))
            f_raw, psd_raw = periodogram(d['v_raw_m_s'][:, 4:6]*1000,
                fs_raw, window='hann', detrend='constant', axis=0)
            _, acc_psd = periodogram(d['a_raw_m_s2'][:, 4:6],
                fs_raw, window='hann', detrend='constant', axis=0)
            f, psd = periodogram(velocity, fs, window='hann', detrend='constant', axis=0)
            _, residual_psd = periodogram(residual, fs, window='hann', detrend=False, axis=0)
            for channel in range(2):
                axes[0, channel].semilogy(f_raw[1:], np.maximum(psd_raw[1:, channel], 1e-30),
                                          color=color, label=label)
                axes[1, channel].semilogy(f[1:], np.maximum(psd[1:, channel], 1e-30), color=color)
                axes[2, channel].semilogy(f[1:], np.maximum(residual_psd[1:, channel], 1e-30), color=color)
                axes[3, channel].semilogy(f_raw[1:], np.maximum(acc_psd[1:, channel], 1e-30), color=color)
        for channel, direction in enumerate('XY'):
            axes[0, channel].set(title=f'H {direction}: full native velocity band', xlim=(0, fs_raw/2))
            axes[1, channel].set(title='0–500 Hz; 1X retained', xlim=(0, min(500, fs/2)))
            axes[2, channel].set(title='Exact 1X removed: diagnostic only', xlim=(0, min(500, fs/2)))
            axes[3, channel].set(title='Full native acceleration band', xlim=(0, fs_raw/2))
        for ax in axes.flat:
            ax.set(xlabel='Frequency (Hz)', ylabel='PSD ((mm/s)²/Hz)')
            ax.grid(alpha=.22)
        for ax in axes[3]:
            ax.set_ylabel('PSD ((m/s²)²/Hz)')
        axes[0, 0].legend(fontsize=8)
        save(fig, 'v3_housing_spectra.png')

        gaps = np.array([row['gap_um'] for row in rows])
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), layout='constrained')
        for key, label in [('E_x_um', 'Ex'), ('E_y_um', 'Ey'), ('E_dir_um', 'E_dir')]:
            axes[0, 0].plot(gaps, [r[key] for r in rows], 'o--', label=label)
        for key, label in [('H_v_rms_x_mm_s', 'Hx'), ('H_v_rms_y_mm_s', 'Hy')]:
            axes[0, 1].plot(gaps, [r[key] for r in rows], 'o--', label=label)
        axes[1, 0].plot(gaps, [100*r['fit_contact_fraction'] for r in rows], 'o--')
        axes[1, 1].plot(gaps, [r['E_dir_um'] for r in rows], 'o', label='Computed E_dir')
        axes[1, 1].axhline(20, color='.4', ls='--', label='20 µm research target')
        axes[0, 0].set(ylabel='Direction PP (µm)', title='Native trajectory; includes 1X')
        axes[0, 1].set(ylabel='H velocity RMS (mm/s)', title='Same frozen physical parameters')
        axes[1, 0].set(ylabel='Target fit contact rate (%)', ylim=(-2, 102))
        axes[1, 1].set(ylabel='Absolute E_dir (µm)', ylim=(0, max(21, 1.05*max(r['E_dir_um'] for r in rows))))
        for ax in axes.flat:
            ax.set_xlabel('Single-sided radial fit clearance c (µm)')
        for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
            ax.legend(fontsize=8)
        finish(axes)
        save(fig, 'v3_gap_comparison.png')

        fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), layout='constrained')
        support_count = records[0]['healthy_support_forces_N'].shape[1]
        support_names = ['Healthy front F2', 'Healthy rear R1']
        axes[0, 0].plot(gaps, [d['fit_force_N'][:, 1].mean() for d in records], 'o--', label='Target fit')
        axes[0, 1].plot(gaps, [np.linalg.norm(d['fit_force_N'], axis=1).max() for d in records], 'o--', label='Target fit')
        for j in range(support_count):
            axes[0, 0].plot(gaps, [d['healthy_support_forces_N'][:, j, 1].mean() for d in records], 'o--', label=support_names[j])
            axes[0, 1].plot(gaps, [np.linalg.norm(d['healthy_support_forces_N'][:, j], axis=1).max() for d in records], 'o--', label=support_names[j])
        for d, color, label in zip(records, colors, labels):
            t = d['t_raw']
            duration = 3*np.mean(d['turn_end_s']-d['turn_start_s'])
            end = d['export_end_s']
            keep = t >= end-duration
            axes[1, 0].plot(t[keep]-end, np.linalg.norm(d['fit_force_N'][keep], axis=1), color=color, label=label)
            axes[1, 1].plot(t[keep]-end, np.linalg.norm(d['q_raw_m'][keep, 6:8]-d['q_raw_m'][keep, 4:6], axis=1)*1e6, color=color)
        axes[0, 0].set(ylabel='Mean force on H: Y (N)', xlabel='Clearance c (µm)', title='Load sharing; signed force')
        axes[0, 1].set(ylabel='Peak resultant force (N)', xlabel='Clearance c (µm)')
        axes[1, 0].set(ylabel='Target fit resultant (N)', xlabel='Time relative to export end (s)')
        axes[1, 1].set(ylabel='O − H radius (µm)', xlabel='Time relative to export end (s)')
        axes[0, 0].legend(fontsize=8)
        axes[1, 0].legend(fontsize=8)
        finish(axes)
        save(fig, 'v3_support_forces.png')

    note = '''# V3 图表说明

所有图使用保存的原生数据/正式输出和固定物理单位，无逐工况归一化或幅值缩放。NPZ原生数组含滤波保护区，绘图按metadata的完整正式导出起止时间裁除保护区；比较同一时间/转角窗。每张比较图的同一坐标轴对全部间隙共用。主轨迹为指定M截面相对固定探头支架，单位µm；主方向峰峰值保留1×。

- `v3_trajectories.png`：左图保留静态偏置，右图仅为看清轨迹而去均值，未修改数据或峰峰值。绝对位移与动态尺度不同，两个视图各自使用明确坐标。
- `v3_housing_waveforms.png`：正式抗混叠输出的最后三转，速度mm/s、加速度m/s²；时间零点是各输出末点，激励没有重调相位。
- `v3_housing_spectra.png`：第一行原生H速度完整Nyquist频带，第二行正式输出0–500Hz局部PSD；Hann窗、单边密度，单位(mm/s)²/Hz。第三行仅为诊断减去已知转频的截距/正弦/余弦最小二乘拟合，原数据与主指标仍保留1×。第四行原生H加速度完整Nyquist频带，单位(m/s²)²/Hz。对数图低于1e−30的值仅在显示中截底，绝非物理噪声底。采样的完整频带不是无限频带，按原生采样及导出抗混叠元数据解释。
- `v3_gap_comparison.png`：Ex/Ey/E_dir、H速度RMS、接触率和20µm目标使用实际求解点。虚线只引导阅读，不表示已验证连续覆盖或严格单调。
- `v3_support_forces.png`：目标配合与健康前后支承传给H的带符号平均Y力、峰值合力，以及末三转配合力与O−H半径。活动离散方向数量不等同接触面积。

图表本身不赋予数值或实机验收通过状态，判定以对应验收JSON和报告为准。
'''
    (output / 'README.md').write_text(note, encoding='utf-8')
    return paths
