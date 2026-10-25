"""Fixed-input S/T/H/O clearance experiment and reproducible acceptance checks."""
import argparse
import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.signal.windows import hann

from spindle.dynamics_model import (equilibrium_continuation, linear_matrices,
                                    simulate)
from utils.clearance_validation import validate_mechanics
from utils.pitting_analysis import resample_to
from utils.runout_labels import complete_turn_runout


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('Ex_mean_um', 'Ey_mean_um', 'H_vx_rms_mm_s', 'H_vy_rms_mm_s',
           'H_vx_1x_peak_mm_s', 'H_vy_1x_peak_mm_s')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_metrics(base, fine):
    checks = {}
    for key in METRICS:
        tolerance = max(.02*abs(fine[key]), .01 if key.endswith('_um') else .001)
        delta = abs(base[key]-fine[key])
        checks[key] = dict(base=base[key], fine=fine[key], difference=delta,
                           tolerance=tolerance, passed=delta <= tolerance)
    return dict(passed=all(x['passed'] for x in checks.values()), metrics=checks)


def velocity_output(result, config):
    """Common polyphase AA lowpass; retain absolute phase and trim true guards."""
    raw_fs = config['simulation']['fs_raw_hz']
    out_fs = config['simulation']['fs_out_hz']
    velocity = resample_to(result['v'][:, 4:6].T, raw_fs, out_fs).T*1000
    time = result['t'][0] + np.arange(len(velocity))/out_fs
    guard = config['simulation']['edge_discard_s']
    keep = (time >= result['t'][0]+guard) & (time <= result['t'][-1]-guard)
    return time[keep], velocity[keep]


def measure(result, config, start=None, end=None):
    p = config['parameters']
    time, velocity = velocity_output(result, config)
    start = time[0] if start is None else start
    end = time[-1] if end is None else end
    keep = (time >= start-1e-12) & (time < end-1e-12)
    t, vh = time[keep], velocity[keep]
    relative_um = (result['q'][:, :2]-result['q'][:, 4:6])*1e6
    turns = complete_turn_runout(result['t'], relative_um, p['rpm'], start, end,
                                 config['excitation']['phase_rad'])
    pp = turns['peak_to_peak_um']
    if len(pp) == 0:
        raise ValueError('No complete revolutions in measurement interval')
    mean, std = pp.mean(axis=0), pp.std(axis=0)
    raw_keep = (result['t'] >= start) & (result['t'] < end)
    center = relative_um[raw_keep].mean(axis=0)
    rms = np.sqrt(np.mean((vh-vh.mean(axis=0))**2, axis=0))
    theta = p['rpm']*2*np.pi/60*t+config['excitation']['phase_rad']
    design = np.column_stack((np.ones(len(t)), np.cos(theta), np.sin(theta)))
    coefficients = np.linalg.lstsq(design, vh, rcond=None)[0]
    peak = np.linalg.norm(coefficients[1:], axis=0)
    metrics = dict(clearance_um=p['fit_clearance_um'], rpm=p['rpm'],
                   Ex_mean_um=float(mean[0]), Ey_mean_um=float(mean[1]),
                   E_dir_um=float(max(mean)), Ex_std_um=float(std[0]), Ey_std_um=float(std[1]),
                   Ex_min_um=float(pp[:, 0].min()), Ex_max_um=float(pp[:, 0].max()),
                   Ey_min_um=float(pp[:, 1].min()), Ey_max_um=float(pp[:, 1].max()),
                   mean_x_um=float(center[0]), mean_y_um=float(center[1]),
                   H_vx_rms_mm_s=float(rms[0]), H_vy_rms_mm_s=float(rms[1]),
                   H_vx_1x_peak_mm_s=float(peak[0]), H_vy_1x_peak_mm_s=float(peak[1]),
                   equilibrium_residual_N=float(result['equilibrium_residual_N']),
                   complete_turns=len(pp), measurement_start_s=float(start),
                   measurement_end_s=float(end))
    return metrics, turns, (t, vh)


def steady_check(result, config):
    t, _ = velocity_output(result, config)
    p = config['parameters']
    rate = abs(p['rpm'])/60
    phase = np.sign(p['rpm'])*config['excitation']['phase_rad']/(2*np.pi)
    end = (np.floor(t[-1]*rate+phase)-phase)/rate
    width = 50/rate
    previous = measure(result, config, end-2*width, end-width)[0]
    last = measure(result, config, end-width, end)[0]
    comparison = compare_metrics(previous, last)
    # The plan's stationarity criterion includes PP and RMS, not fitted 1X.
    comparison['passed'] = all(comparison['metrics'][k]['passed'] for k in METRICS[:4])
    comparison['window_edges_s'] = [end-2*width, end-width, end]
    comparison['turns_per_window'] = 50
    return comparison


def run_case(config, q0, required_end=None):
    p, settings = config['parameters'], config['simulation']
    fs = settings['fs_raw_hz']
    window = settings['retained_s']+2*settings['edge_discard_s']
    initial_end = settings['warmup_s']+window if required_end is None else required_end
    result = simulate(p, 0, [], [], duration=initial_end-settings['warmup_s'],
                      warmup=settings['warmup_s'], fs=fs,
                      max_step=settings['max_step_s'], q_initial=q0,
                      unbalance=(config['excitation']['U_kg_m'], config['excitation']['phase_rad']))
    history = []
    while True:
        # Keep the same final duration for every statistic; no transient rescaling.
        begin = np.searchsorted(result['t'], result['t'][-1]-window-1e-12)
        for key in ('t', 'q', 'v', 'acc', 'fit_force_N', 'fit_active_count'):
            result[key] = result[key][begin:]
        check = steady_check(result, config)
        history.append(dict(total_time_s=float(result['t'][-1]), **check))
        if check['passed'] or result['t'][-1] >= settings['maximum_time_s']-1/fs or required_end:
            break
        extension = min(settings['extension_s'], settings['maximum_time_s']-result['t'][-1])
        tail = simulate(p, 0, [], [], duration=extension, warmup=0, fs=fs,
                        max_step=settings['max_step_s'], q_initial=q0,
                        unbalance=(config['excitation']['U_kg_m'], config['excitation']['phase_rad']),
                        t_start=float(result['t'][-1]), state_initial=result['state_final'])
        for key in ('t', 'q', 'v', 'acc', 'fit_force_N', 'fit_active_count'):
            result[key] = np.concatenate((result[key], tail[key][1:]))
        result['state_final'] = tail['state_final']
        result['nfev'] += tail['nfev']
    metrics, turns, output = measure(result, config)
    metrics.update(steady_pass=check['passed'], numerical_pass=False,
                   total_time_s=float(result['t'][-1]), nfev=result['nfev'])
    return result, metrics, turns, output, history


def save_case(folder, config, run):
    result, metrics, turns, output, history = run
    tag = f"clearance_{config['parameters']['fit_clearance_um']:g}um"
    t_out, vh = output
    window = hann(len(vh), sym=False)
    spectrum = 2*np.abs(np.fft.rfft((vh-vh.mean(axis=0))*window[:, None], axis=0))/window.sum()
    spectrum[0] /= 2
    if len(vh) % 2 == 0:
        spectrum[-1] /= 2
    frequencies = np.fft.rfftfreq(len(vh), 1/config['simulation']['fs_out_hz'])
    metadata = dict(config=config, metrics=metrics, stationarity_history=history,
                    source_version='experiment root source_final snapshot; no file hash used',
                    coordinates=['Sx','Sy','Tx','Ty','Hx','Hy','Ox','Oy'],
                    units=dict(t_raw_s='s', q_raw_m='m', v_raw_m_s='m/s',
                               rS_raw_m='m', fit_force_N='N', H_velocity_mm_s='mm/s'),
                    definitions=dict(E_dir='larger of the mean per-turn X/Y peak-to-peak values',
                                     reference='S relative H; H velocity relative fixed ground',
                                     spectrum='demean, periodic Hann, 2/sum(window) one-sided peak amplitude',
                                     one_x='least squares intercept+cos+sin at exact signed rotation frequency',
                                     contact_count='positive normal contact force; native samples, not filtered'),
                    processing='resample_poly Kaiser beta=5 default FIR, padtype=line; common guard trimmed; no bandpass')
    np.savez_compressed(folder/f'{tag}.npz', t_raw_s=result['t'], q_raw_m=result['q'],
                        v_raw_m_s=result['v'], rS_raw_m=result['q'][:,:2]-result['q'][:,4:6],
                        rO_raw_m=result['q'][:,6:8]-result['q'][:,4:6],
                        fit_force_N=result['fit_force_N'], fit_active_count=result['fit_active_count'],
                        q_equilibrium_m=result['q_equilibrium'], t_out_s=t_out,
                        H_velocity_mm_s=vh, frequency_hz=frequencies,
                        H_spectrum_peak_mm_s=spectrum,
                        turn_id=turns['turn_id'], turn_start_s=turns['start_time_s'],
                        turn_end_s=turns['end_time_s'], turn_pp_um=turns['peak_to_peak_um'],
                        metadata_json=json.dumps(metadata, ensure_ascii=False))
    write_csv(folder/f'{tag}_turns.csv', [dict(turn_id=int(k), start_s=float(a), end_s=float(b),
                   Ex_um=float(x), Ey_um=float(y)) for k,a,b,(x,y) in zip(turns['turn_id'],
                   turns['start_time_s'],turns['end_time_s'],turns['peak_to_peak_um'])])
    return metrics


def plot_results(folder, rows):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = plt.get_cmap('viridis')(np.linspace(.1,.85,len(rows)))
    fig, axes = plt.subplots(1, len(rows), figsize=(4*len(rows),4), constrained_layout=True)
    figv, av = plt.subplots(2, 1, figsize=(10,6), constrained_layout=True)
    figs, asp = plt.subplots(2, 2, figsize=(12,7), constrained_layout=True)
    orbit_limit = 0.0
    for ax, row, color in zip(np.atleast_1d(axes), rows, colors):
        c = row['clearance_um']
        with np.load(folder/f'clearance_{c:g}um.npz') as data:
            raw_t, relative = data['t_raw_s'], data['rS_raw_m']*1e6
            turn_end = data['turn_end_s'][-1]
            use = (raw_t >= turn_end-5*60/abs(row['rpm'])) & (raw_t <= turn_end)
            centered = relative-relative.mean(axis=0)
            orbit_limit = max(orbit_limit,float(np.abs(centered[use]).max()))
            ax.plot(centered[use,0],centered[use,1],color=color,lw=.9)
            ax.set(title=f'c = {c:g} um',xlabel='S-H X, mean removed (um)',ylabel='S-H Y, mean removed (um)')
            ax.set_aspect('equal',adjustable='datalim')
            # Align display by shaft phase; stored data retain absolute time.
            end = data['turn_end_s'][-1]
            tv = data['t_out_s']-end
            show = (tv >= -.05) & (tv <= 0)
            for i in range(2):
                av[i].plot(tv[show]*1000,data['H_velocity_mm_s'][show,i],color=color,label=f'{c:g} um',lw=1)
                for j in range(2):
                    asp[i,j].plot(data['frequency_hz'],data['H_spectrum_peak_mm_s'][:,i],color=color,label=f'{c:g} um',lw=1)
    for ax in np.atleast_1d(axes):
        ax.set(xlim=(-1.1*orbit_limit,1.1*orbit_limit),ylim=(-1.1*orbit_limit,1.1*orbit_limit))
        ax.set_aspect('equal',adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(3))
    fig.savefig(folder/'orbits.png',dpi=180)
    for i in range(2):
        av[i].set(xlabel='Time relative to final turn boundary (ms)',ylabel=f'H {"XY"[i]} velocity (mm/s)')
        av[i].legend(ncol=4)
        for j in range(2):
            asp[i,j].set(xlim=(0,4096) if j==0 else (10,900),xlabel='Frequency (Hz)',
                         ylabel=f'H {"XY"[i]} peak amplitude (mm/s)')
            asp[i,j].legend()
    figv.savefig(folder/'housing_velocity.png',dpi=180)
    figs.savefig(folder/'housing_spectra.png',dpi=180)
    fig, ax = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    c = [r['clearance_um'] for r in rows]
    for key,label in [('Ex_mean_um','X per-turn PP mean'),('Ey_mean_um','Y per-turn PP mean'),('E_dir_um','Larger directional PP mean')]:
        ax[0].plot(c,[r[key] for r in rows],'o-',label=label)
    for key,label in [('H_vx_rms_mm_s','H X'),('H_vy_rms_mm_s','H Y')]:
        ax[1].plot(c,[r[key] for r in rows],'o-',label=label)
    ax[0].set(xlabel='Radial fit clearance (um)',ylabel='S-H directional peak-to-peak (um)')
    ax[1].set(xlabel='Radial fit clearance (um)',ylabel='Demeaned velocity RMS (mm/s)')
    for a in ax:
        a.legend(); a.grid(alpha=.2)
    fig.savefig(folder/'clearance_metrics.png',dpi=180)
    plt.close('all')


def audit_exports(folder, rows):
    common = []
    evidence = []
    for row in rows:
        with np.load(folder/f"clearance_{row['clearance_um']:g}um.npz") as data:
            meta = json.loads(str(data['metadata_json']))
            fixed = meta['config']
            fixed['parameters'].pop('fit_clearance_um')
            common.append(fixed)
            consistent = np.array_equal(data['rS_raw_m'],data['q_raw_m'][:,:2]-data['q_raw_m'][:,4:6])
            finite = all(np.isfinite(data[k]).all() for k in
                         ('q_raw_m','v_raw_m_s','fit_force_N','H_velocity_mm_s','turn_pp_um'))
            evidence.append(dict(clearance_um=row['clearance_um'],passed=bool(consistent and finite),
                raw_samples=len(data['t_raw_s']),output_samples=len(data['t_out_s']),
                active_count_min=int(data['fit_active_count'].min()),
                active_count_max=int(data['fit_active_count'].max()),
                contact_fraction=float(np.mean(data['fit_active_count']>0)),
                fit_force_max_N=float(np.linalg.norm(data['fit_force_N'],axis=1).max()),
                O_relative_H_radius_max_um=float(np.linalg.norm(data['rO_raw_m'],axis=1).max()*1e6)))
    identical = all(p==common[0] for p in common)
    return dict(passed=identical and all(e['passed'] for e in evidence),
                common_inputs_except_clearance=identical,cases=evidence)


def write_report(folder, rows, coverage, convergence):
    lines = ['# 配合间隙实际运行结果', '',
             '固定 S/T/H/O 八自由度演示；参数、单位及来源见 frozen_config.json。', '',
             f"模型/力学检查：通过；共同输入检查：通过；数值验收：{'通过' if coverage['numerical_pass'] else '未通过或未完成'}；范围覆盖：{'通过' if coverage['coverage_pass'] else '覆盖目标未达到'}。", '',
             '| 径向间隙 µm | X逐转峰峰均值 µm | Y逐转峰峰均值 µm | H X RMS mm/s | H Y RMS mm/s |',
             '|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['clearance_um']:g} | {r['Ex_mean_um']:.9f} | {r['Ey_mean_um']:.9f} | {r['H_vx_rms_mm_s']:.9f} | {r['H_vy_rms_mm_s']:.9f} |")
    lines += ['',f"有效 E_dir 范围：{coverage['actual_min_um']}–{coverage['actual_max_um']} µm。各5 µm区间的参数点数量：{coverage['counts_by_bin']}。",
              'E_dir 是两个测量方向中的较大峰峰均值，保留1×；0间隙是理想对照，不是已知实机健康值。离散点不证明连续覆盖。', '',
              '原生位移计算完整转次；速度经统一 polyphase 抗混叠降至8192 Hz，舍弃两端0.02 s，不逐窗缩放。',
              '稳态比较末尾两个50转窗口；联合加密将最大步长减半、原生采样率及接触方向数加倍。详细误差见 convergence.json。',
              '接触率及外圈相对运动见 export_validation.json。无接触的工况不能解释为间隙撞击。',
              '完整来源与机理边界见 docs/reports/clearance_sources_20261023.md；源代码版本由 source_final 原文件快照追溯。', '',
              '复现（在仓库根目录，使用新输出目录）：',
              '```powershell',
              'python -B -m scripts.run_clearance_sweep --config config/clearance_config.json --output experiments/clearance_sweep_replay --validate',
              '```']
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/clearance_config.json')
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/clearance_sweep_20261023')
    parser.add_argument('--validate',action='store_true',help='Also run representative joint convergence check')
    parser.add_argument('--clearance-um',type=float)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    folder = args.output
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.csv').exists():
        raise FileExistsError('Choose a fresh --output; existing results are retained')
    snapshot = folder/'source_final'
    snapshot.mkdir(exist_ok=True)
    for name in ('src/spindle/dynamics_model.py','scripts/run_clearance_sweep.py',
                 'utils/clearance_validation.py','utils/runout_labels.py','utils/pitting_analysis.py'):
        shutil.copy2(ROOT/name,snapshot/Path(name).name)
    write_json(folder/'frozen_config.json',config)
    checks = validate_mechanics(config['parameters'],config['baseline_old_parameters'])
    checks['allocation'] = config['baseline_allocation']
    write_json(folder/'mechanics_validation.json',checks)
    if not checks['passed']:
        raise RuntimeError('Mechanical validation failed; see mechanics_validation.json')
    clearances = config['clearances_um'] if args.clearance_um is None else [args.clearance_um]
    jobs, equilibria = [], []
    q0 = None
    previous_c = 0.0
    for c in sorted(clearances):
        local = json.loads(json.dumps(config))
        local['parameters']['fit_clearance_um'] = c
        q0, residual = equilibrium_continuation(local['parameters'],initial_q=q0,
                                                start_clearance_um=previous_c)
        m, _, k = linear_matrices(local['parameters'],0,initial_q=q0)
        fmax = np.sqrt(np.linalg.eigvalsh(k/np.sqrt(np.outer(np.diag(m),np.diag(m)))).max())/(2*np.pi)
        if local['simulation']['max_step_s'] > min(1/32768,1/(20*fmax))*(1+1e-10):
            raise ValueError('Frozen max_step is too large for this tangent; refreeze common settings')
        equilibria.append(dict(clearance_um=c,residual_N=residual,q_m=q0.tolist(),fmax_hz=float(fmax)))
        jobs.append((local,q0.copy()))
        previous_c = c
    write_json(folder/'equilibria.json',equilibria)
    runs = {}
    with ProcessPoolExecutor(max_workers=config['simulation']['workers']) as pool:
        futures = {pool.submit(run_case,local,q):local for local,q in jobs}
        for future in as_completed(futures):
            local = futures[future]
            run = future.result()
            c = local['parameters']['fit_clearance_um']
            save_case(folder,local,run)
            runs[c] = run
            print(json.dumps(run[1]),flush=True)
    convergence = dict(passed=False,status='not run; use --validate')
    if args.validate:
        stable = [c for c in runs if runs[c][1]['steady_pass']]
        if not stable:
            raise RuntimeError('No statistically stable case for convergence testing')
        representative = max(stable)
        local,q0 = next((local,q) for local,q in jobs if local['parameters']['fit_clearance_um']==representative)
        fine = json.loads(json.dumps(local))
        fine['parameters']['N'] *= 2
        fine['simulation']['max_step_s'] /= 2
        fine['simulation']['fs_raw_hz'] *= 2
        qfine,_ = equilibrium_continuation(fine['parameters'],initial_q=q0,start_clearance_um=representative)
        refined = run_case(fine,qfine,required_end=runs[representative][1]['total_time_s'])
        convergence = compare_metrics(runs[representative][1],refined[1])
        convergence.update(clearance_um=representative, fine_steady_pass=refined[1]['steady_pass'],
                           changes=dict(N=fine['parameters']['N'],fs_raw_hz=fine['simulation']['fs_raw_hz'],
                                        max_step_s=fine['simulation']['max_step_s']))
        convergence['passed'] &= refined[1]['steady_pass']
        refined[1]['numerical_pass'] = convergence['passed']
        fine_folder = folder/'convergence_joint'
        fine_folder.mkdir(exist_ok=True)
        save_case(fine_folder,fine,refined)
        # A separated largest-gap case does not exercise fit discretization.
        # Also refine the contacting zero-gap reference when it is available.
        reference = min(stable)
        if reference != representative and not np.any(runs[representative][0]['fit_active_count']):
            contact_config, contact_q = next((local,q) for local,q in jobs
                if local['parameters']['fit_clearance_um']==reference)
            contact_config = json.loads(json.dumps(contact_config))
            contact_config['parameters']['N'] *= 2
            contact_config['simulation']['max_step_s'] /= 2
            contact_config['simulation']['fs_raw_hz'] *= 2
            contact_q,_ = equilibrium_continuation(contact_config['parameters'],initial_q=contact_q,
                                                   start_clearance_um=reference)
            contact_run = run_case(contact_config,contact_q,
                                  required_end=runs[reference][1]['total_time_s'])
            contact_check = compare_metrics(runs[reference][1],contact_run[1])
            contact_check.update(clearance_um=reference,fine_steady_pass=contact_run[1]['steady_pass'])
            contact_check['passed'] &= contact_run[1]['steady_pass']
            contact_run[1]['numerical_pass'] = contact_check['passed']
            contact_folder = folder/'convergence_contacting'
            contact_folder.mkdir(exist_ok=True)
            save_case(contact_folder,contact_config,contact_run)
            write_json(folder/'convergence_contacting.json',contact_check)
            convergence['additional_contacting_case'] = contact_check
            convergence['passed'] &= contact_check['passed']
        write_json(folder/'convergence.json',convergence)
        print('Joint convergence: '+str(convergence['passed']),flush=True)
    rows = []
    for local,_ in jobs:
        c = local['parameters']['fit_clearance_um']
        run = runs[c]
        run[1]['numerical_pass'] = bool(checks['passed'] and run[1]['steady_pass'] and convergence['passed'])
        run[1]['E_dir_increment_um'] = run[1]['E_dir_um']-runs[min(runs)][1]['E_dir_um']
        if convergence['passed']:
            uncertainty = max(convergence['metrics'][k]['difference'] for k in METRICS[:2])
            run[1]['gap_effect_resolved'] = abs(run[1]['E_dir_increment_um']) > uncertainty
        rows.append(save_case(folder,local,run))
    write_csv(folder/'summary.csv',rows)
    exports = audit_exports(folder,rows)
    write_json(folder/'export_validation.json',exports)
    if not exports['passed']:
        raise RuntimeError('Saved-state or common-input audit failed')
    values = np.asarray([r['E_dir_um'] for r in rows if r['numerical_pass']])
    coverage = dict(definition='larger of mean per-revolution X/Y PP, includes 1X',
                    baseline_clearance_um=min(runs), baseline_is_measured_healthy=False,
                    numerical_pass=all(r['numerical_pass'] for r in rows),
                    actual_min_um=float(values.min()) if len(values) else None,
                    actual_max_um=float(values.max()) if len(values) else None,
                    counts_by_bin={f'{a}-{a+5} um':int(np.sum((values>=a)&(values<(a+5) if a<15 else values<=20)))
                                   for a in (0,5,10,15)},
                    coverage_pass=bool(len(values) and values.max()>=20 and values.min()<=.01),
                    note='Discrete parameter points do not prove continuous coverage; never rescaled.')
    write_json(folder/'coverage.json',coverage)
    plot_results(folder,rows)
    write_report(folder,rows,coverage,convergence)
    print(json.dumps(coverage,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
