"""Ideal-drive race pits and single-ball deformation: generate, plot, export.

No real measurements or inverse forcing. Source and limitations are documented
in docs/reports/bearing_inner_and_deformation_20261021.md.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram, find_peaks

from spindle.dynamics_model import (default_parameters, simulate, cage_speed,
                                    ball_spin_speed, inner_race_pit_geometry,
                                    outer_race_pit_geometry)
from utils.data_io import save_json
from utils.pitting_analysis import resample_to

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#737C87', '#228C9A', '#D28B26', '#B43B55']


def cases(cfg):
    p = default_parameters()
    p['rpm'] = cfg['simulation']['rpm']
    yield 'healthy', 'healthy', 0., p
    for race in ('outer', 'inner'):
        pit = cfg.get(race + '_pit')
        if pit is None:
            continue
        for area in pit['areas_mm2']:
            fault = {race + '_pit_' + key: value for key, value in dict(
                area_m2=area*1e-6, width_m=pit['width_mm']*1e-3, depth_m=pit['depth_mm']*1e-3,
                center_rad=float(np.deg2rad(pit['initial_center_deg']))).items()}
            yield f'{race}_{area:g}mm2', race, area, dict(p, **fault)
    ball = cfg['ball_deformation']
    if ball['fixed_waviness_order'] != 2:
        raise ValueError('Only the documented w=2 ball deformation is implemented')
    for amplitude in ball['amplitudes_um']:
        yield f'ball_{amplitude:g}um', 'ball', amplitude, dict(
            p, ball_deformation_amplitude_m=amplitude*1e-6,
            ball_deformation_index=ball['ball_index_zero_based'],
            ball_deformation_phase_rad=float(np.deg2rad(ball['phase_deg'])))


def characteristic_frequencies(p):
    shaft = p['rpm']/60
    cage = cage_speed(p)/(2*np.pi)
    spin = ball_spin_speed(p)/(2*np.pi)
    return dict(shaft_hz=shaft, cage_hz=cage, bpfo_hz=p['balls']*cage,
                bpfi_hz=p['balls']*(shaft-cage), bsf_hz=spin, twice_bsf_hz=2*spin)


def spectrum(values, fs):
    """One-sided Hann amplitude spectrum; off-bin lines have scalloping loss."""
    f, power = periodogram(values, fs=fs, window='hann', detrend='constant',
                           scaling='spectrum', axis=0)
    amplitude = np.sqrt(2*power)
    amplitude[0] /= np.sqrt(2)
    if len(values) % 2 == 0:
        amplitude[-1] /= np.sqrt(2)
    return f, amplitude


def processed(raw, cfg, source_fs=None, duration=None):
    s = cfg['simulation']
    source_fs = s['native_fs_hz'] if source_fs is None else source_fs
    duration = s['duration_s'] if duration is None else duration
    start, count = round(s['guard_s']*s['display_fs_hz']), round(duration*s['display_fs_hz'])
    return {k: resample_to(raw[k][:-1].T, source_fs, s['display_fs_hz']).T[start:start+count]
            for k in ('q', 'v', 'acc')}


def run_case(p, cfg, duration=None, refinement=1):
    s, d = cfg['simulation'], cfg['drive']
    duration = s['duration_s'] if duration is None else duration
    return simulate(p, 0., np.array([d['frequency_hz']]),
                    d['amplitude_N']*np.array([[1., -1j]]),
                    duration=duration+2*s['guard_s'], warmup=s['warmup_s'],
                    fs=s['native_fs_hz']*refinement, max_step=s['max_step_s']/refinement,
                    pit_steps_per_passage=s['pit_steps_per_passage']*refinement)


def generate(out, cfg):
    rows = []
    fs = cfg['simulation']['display_fs_hz']
    for name, kind, severity, p in cases(cfg):
        started = time.perf_counter()
        print(f'Integrating {name}', flush=True)
        raw = run_case(p, cfg)
        data = processed(raw, cfg)
        t = raw['t'][0]+cfg['simulation']['guard_s']+np.arange(len(data['q']))/fs
        phase = 2*np.pi*cfg['drive']['frequency_hz']*t
        force = cfg['drive']['amplitude_N']*np.column_stack([np.cos(phase), np.sin(phase)])
        np.savez_compressed(out/f'{name}.npz', **raw, display_time_s=t, display_force_N=force,
                            **{f'display_{k}':v for k,v in data.items()}, display_fs_hz=fs,
                            parameters_json=json.dumps(p))
        columns = ['time_s', 'force_x_N', 'force_y_N']
        columns += [f'{node}_{key}{axis}_{unit}' for key, unit in [('q','um'),('v','mm_s'),('a','m_s2')]
                    for node in ('shaft','holder','housing') for axis in 'xy']
        np.savetxt(out/f'{name}.csv', np.column_stack([t, force, data['q']*1e6, data['v']*1e3, data['acc']]),
                   delimiter=',', header=','.join(columns), comments='')
        f, a = spectrum(data['acc'], fs)
        np.savetxt(out/f'{name}_acc_spectrum.csv', np.column_stack([f,a]), delimiter=',',
                   header='frequency_hz,'+','.join(columns[-6:]), comments='')
        peaks = find_peaks(a[:,5])[0]
        peaks = peaks[np.argsort(a[peaks,5])[-12:][::-1]]
        dynamic = data['q']-data['q'].mean(0)
        row = dict(case=name, kind=kind, severity=severity, parameters=p,
                   geometry=(outer_race_pit_geometry(p) if kind=='outer' else inner_race_pit_geometry(p)),
                   frequencies=characteristic_frequencies(p),
                   H_acc_rms_xy=np.sqrt(np.mean(data['acc'][:,4:]**2,axis=0)).tolist(),
                   H_velocity_rms_xy_mm_s=(np.sqrt(np.mean(data['v'][:,4:]**2,axis=0))*1e3).tolist(),
                   T_dynamic_radial_rms_um=float(np.sqrt(np.mean(np.sum(dynamic[:,2:4]**2,axis=1)))*1e6),
                   T_acc_rms_xy=np.sqrt(np.mean(data['acc'][:,2:4]**2,axis=0)).tolist(),
                   H_y_largest_spectral_peaks=[dict(hz=float(f[i]),amplitude_m_s2=float(a[i,5])) for i in peaks],
                   finite=all(bool(np.isfinite(v).all()) for v in data.values()),
                   equilibrium_residual_N=raw['equilibrium_residual_N'], nfev=raw['nfev'],
                   elapsed_s=time.perf_counter()-started)
        rows.append(row)
        save_json(out/'summary.json', dict(config=cfg, cases=rows))
        print(f"Completed {name}: H acceleration RMS XY={row['H_acc_rms_xy']}, {row['elapsed_s']:.1f} s", flush=True)
    return rows


def figures(out, cfg, rows):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
                         'axes.unicode_minus':False, 'font.size':10})
    fs = cfg['simulation']['display_fs_hz']
    arrays = {}
    for row in rows:
        with np.load(out/f"{row['case']}.npz") as saved:
            arrays[row['case']] = {k:saved[f'display_{k}'] for k in ('q','v','acc')}
    for kind, title, unit in [('inner','内圈点蚀','mm²'),('ball','滚动体二阶失圆','μm')]:
        group = [r for r in rows if r['kind'] in ('healthy',kind)]
        fig, axes = plt.subplots(len(group),2,figsize=(12,9),layout='constrained')
        ymax = max(np.abs(arrays[r['case']]['acc'][:,5]).max() for r in group)*1.1
        amax = max(spectrum(arrays[r['case']]['acc'][:,5],fs)[1].max() for r in group)*1.12
        fr = group[-1]['frequencies']
        carrier = fr['bpfi_hz'] if kind=='inner' else fr['twice_bsf_hz']
        side = fr['shaft_hz'] if kind=='inner' else fr['cage_hz']
        for i,r in enumerate(group):
            y = arrays[r['case']]['acc'][:,5]
            t = np.arange(len(y))/fs*1000
            f,a = spectrum(y,fs)
            axes[i,0].plot(t,y,color=COLORS[i],lw=.8)
            axes[i,1].plot(f,a,color=COLORS[i],lw=.9)
            label = '健康' if r['kind']=='healthy' else f"{r['severity']:g} {unit}"
            axes[i,0].set(xlim=(0,40),ylim=(-ymax,ymax),ylabel=f'{label}\n外壳 Y 加速度 (m/s²)',xlabel='时间 (ms)')
            axes[i,1].set(xlim=(0,3500),ylim=(0,amax),ylabel='峰值幅度 (m/s²)',xlabel='频率 (Hz)')
            offsets = (-2,-1,0,1,2) if kind=='inner' else (-1,0,1)
            for freq in (carrier+n*side for n in offsets):
                axes[i,1].axvline(freq,color='.5',lw=.7,ls='--',alpha=.6)
            axes[i,0].text(.02,.92,f"RMS {r['H_acc_rms_xy'][1]:.3f} m/s²",transform=axes[i,0].transAxes,va='top')
        for ax in axes.flat:
            ax.grid(alpha=.17)
        feature = 'BPFI 及 ±1、±2 倍转频边带' if kind=='inner' else '2BSF ± 保持架频率'
        drive, duration = cfg['drive'],cfg['simulation']['duration_s']
        fig.suptitle(f"{title}｜相同 {drive['frequency_hz']:g} Hz、{drive['amplitude_N']:g} N 旋转正弦力；各档统一纵轴\n"
                     f'虚线：{feature}；热身后截取 {duration:g} s，频率间隔 {1/duration:g} Hz')
        fig.savefig(out/f'{kind}_time_spectrum.png',dpi=160)
        plt.close(fig)
        # All nodes are generated; show H velocity and the requested holder center.
        fig,axes = plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
        for i,r in enumerate(group):
            data = arrays[r['case']]
            label = '健康' if r['kind']=='healthy' else f"{r['severity']:g} {unit}"
            f,a = spectrum(data['v'][:,5]*1e3,fs)
            axes[0].plot(f,a,color=COLORS[i],label=label,lw=1)
            f,a = spectrum(data['acc'][:,3],fs)
            axes[1].plot(f,a,color=COLORS[i],label=label,lw=1)
            center=(data['q'][:,2:4]-data['q'][:,2:4].mean(0))*1e6
            axes[2].plot(*center[:round(.05*fs)].T,color=COLORS[i],label=label,lw=.7,alpha=.8)
        axes[0].set(xlim=(0,1600),xlabel='频率 (Hz)',ylabel='外壳 Y 速度峰值 (mm/s)')
        axes[1].set(xlim=(0,3500),xlabel='频率 (Hz)',ylabel='刀柄 Y 加速度峰值 (m/s²)')
        axes[2].set(xlabel='刀柄中心 X (μm)',ylabel='刀柄中心 Y (μm)',aspect='equal')
        for ax in axes:
            ax.grid(alpha=.2)
        axes[0].legend(frameon=False,fontsize=9)
        fig.suptitle(f'{title}｜同步输出外壳 H 与刀柄中心 T；轨迹去均值，显示 50 ms')
        fig.savefig(out/f'{kind}_velocity_and_center.png',dpi=160)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/bearing_faults_ideal.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=True)
    if args.plots_only:
        saved=json.loads((args.output/'summary.json').read_text(encoding='utf-8'))
        cfg,rows=saved['config'],saved['cases']
    else:
        if (args.output/'summary.json').exists():
            raise FileExistsError('Use a new --output directory to preserve previous results')
        save_json(args.output/'config.json',cfg)
        rows=generate(args.output,cfg)
    figures(args.output,cfg,rows)


if __name__=='__main__':
    main()
