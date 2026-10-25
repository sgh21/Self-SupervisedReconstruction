"""Compare real healthy acceleration/velocity with matched pit-area simulations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from spindle.dynamics_model import outer_race_pit_displacement
from utils.data_io import save_json
from utils.pitting_analysis import acceleration_metrics


COLORS = ['#516475', '#008ca6', '#dc8a22', '#ba2458']


def healthy_time_alignment(reference, baseline, fs, max_shift_s=.05):
    """One physical time shift, fitted to healthy XY only, for all display panels."""
    start, end = round(.1*fs), round(.9*fs)
    target = reference[:, start:end]
    target = target-target.mean(axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1)
    lags = np.arange(-round(max_shift_s*fs), round(max_shift_s*fs)+1)
    correlations = []
    for lag in lags:
        candidate = baseline[:, start+lag:end+lag]
        candidate = candidate-candidate.mean(axis=-1, keepdims=True)
        correlations.append(np.sum(candidate*target, axis=-1)
                            / (np.linalg.norm(candidate, axis=-1)*target_norm))
    correlations = np.asarray(correlations)
    best = int(np.argmax(correlations.mean(axis=1)))
    return dict(method='maximize mean XY normalized correlation on healthy acceleration only',
                lag_samples=int(lags[best]), lag_s=float(lags[best]/fs), fs_hz=fs,
                search_limit_s=max_shift_s, fit_interval_s=[.1,.9],
                correlation_before_xy=correlations[lags==0][0].tolist(),
                correlation_after_xy=correlations[best].tolist(),
                convention='display simulation(t + lag_s) against real(t); no wrap, scaling or resynthesis',
                scope='one shared shift for XY, acceleration, velocity, and every pit area; plots only')


def comparison_figure(cases, signals, real, axis, key, fs, high, unit, seconds, target,
                      bpfo=None, lag_s=0):
    """Common axes/units; healthy-only time alignment never alters spectra or metrics."""
    fig, axes = plt.subplots(len(cases), 2, figsize=(11, 9), constrained_layout=True)
    reference = real[0, axis]
    f, real_psd = welch(real[:, axis], fs=fs, nperseg=fs//4, axis=-1)
    psd = real_psd.mean(0)
    largest_psd = float(np.max(psd[(f >= 10) & (f <= high)]))
    for c in cases:
        fx, sx = welch(signals[c['case']][key][axis], fs=fs, nperseg=fs//4)
        largest_psd = max(largest_psd, float(np.max(sx[(fx >= 10) & (fx <= high)])))
    psd_upper = 10**np.ceil(np.log10(largest_psd*1.1))
    time = np.arange(round(seconds*fs))/fs
    # Shift the excerpt without wrapping or modifying the stored waveform.
    excerpt = slice(round(.1*fs), round(.1*fs)+len(time))
    shifted_excerpt = slice(excerpt.start+round(lag_s*fs), excerpt.stop+round(lag_s*fs))
    span = max(np.max(np.abs(reference[excerpt])),
               *(np.max(np.abs(signals[c['case']][key][axis, shifted_excerpt])) for c in cases))*1.08
    for i, case in enumerate(cases):
        x = signals[case['case']][key][axis]
        label = f"Simulation, pit {case['area_mm2']:g} mm$^2$"
        axes[i,0].plot(time, reference[excerpt], color='.65', lw=.8, label='Real healthy reference')
        axes[i,0].plot(time, x[shifted_excerpt], color=COLORS[i], lw=.85, label=label)
        axes[i,0].set(xlabel='Relative time (s)', ylabel=f'H {"XY"[axis]} ({unit})', ylim=(-span, span))
        axes[i,0].legend(fontsize=7, loc='upper right')
        fx, sx = welch(x, fs=fs, nperseg=fs//4)
        axes[i,1].semilogy(f, psd, color='.65', lw=.9, label='Real healthy validation mean')
        axes[i,1].semilogy(fx, sx, color=COLORS[i], lw=1, label=label)
        axes[i,1].set(xlim=(10, high), xlabel='Frequency (Hz)', ylabel=f'PSD [({unit})$^2$/Hz]')
        if bpfo:
            for harmonic in range(1, int(high/bpfo)+1):
                axes[i,1].axvline(harmonic*bpfo, color='#945b9b', alpha=.35, lw=.7, ls='--')
            axes[i,1].text(.98, .95, 'Dashed: BPFO harmonics', ha='right', va='top', fontsize=7,
                           transform=axes[i,1].transAxes)
        axes[i,1].set_ylim(1e-11 if bpfo else 1e-10, psd_upper)
    quantity = 'acceleration' if 'acceleration' in key else 'velocity'
    fig.suptitle(f'Matched excitation, phase_0; housing {"XY"[axis]}; 10–{high} Hz {quantity}\n'
                 f'Healthy-based time alignment: simulation t {lag_s*1000:+.2f} ms; spectra unchanged', fontsize=11)
    fig.savefig(target, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    out = args.results
    cfg = json.loads((out/'config.json').read_text(encoding='utf-8'))
    processing = cfg.get('processing')
    afs = cfg['simulation']['comparison_fs_hz']
    vfs = processing['fs_hz'] if processing else 2048
    ahigh = cfg['simulation']['acceleration_band_hz'][1]
    vhigh = processing['band_hz'][1] if processing else 900
    akey = 'H_acceleration_band_m_s2' if processing else 'H_acceleration_band_m_s2_25600'
    vkey = 'H_velocity_band_mm_s' if processing else 'H_velocity_band_mm_s_2048'
    generation = json.loads((out/'generation.json').read_text(encoding='utf-8'))
    if not generation['completed']:
        raise RuntimeError('Generation is incomplete')
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    cases = generation['cases']
    representatives = [c for c in cases if c['group'] == 'phase_0']
    bpfo = generation['bpfo_hz']
    signals = {}
    for case in cases:
        with np.load(out/'signals'/f"{case['case']}.npz", allow_pickle=False) as f:
            signals[case['case']] = {k:f[k] for k in (akey, vkey)}
            for key in ('q', 'v', 'acc', 'T_center_m'):
                assert np.isfinite(f[key]).all(), (case['case'], key)
    with np.load(out/'healthy_acceleration_reference.npz', allow_pickle=False) as f:
        measured = f['band_m_s2']
    provenance = json.loads((out/'healthy_acceleration_provenance.json').read_text(encoding='utf-8'))
    validation = measured[[i for i,r in enumerate(provenance) if r['split'] == 'validation']]
    with np.load(out/'healthy_velocity_reference.npz', allow_pickle=False) as f:
        velocity_refs = {split:f[split] for split in ('train','validation')}
        measured_v = velocity_refs['validation']
    healthy_case = next(c for c in representatives if c['area_mm2']==0)
    alignment = healthy_time_alignment(validation[0], signals[healthy_case['case']][akey], afs)
    alignment['reference'] = next(r for r in provenance if r['split']=='validation')
    lag_s = alignment['lag_s']
    excerpt_start, excerpt_size = round(.1*afs), round(.02*afs)
    model_start = excerpt_start+alignment['lag_samples']
    baseline_acc = signals[healthy_case['case']][akey]
    alignment['displayed_acceleration_excerpt_rms_xy_m_s2'] = dict(
        duration_s=excerpt_size/afs,
        real=np.sqrt(np.mean(validation[0,:,excerpt_start:excerpt_start+excerpt_size]**2,axis=-1)).tolist(),
        healthy_simulation=np.sqrt(np.mean(baseline_acc[:,model_start:model_start+excerpt_size]**2,axis=-1)).tolist())
    alignment['reference_full_record_rms_xy_m_s2'] = np.sqrt(np.mean(validation[0]**2,axis=-1)).tolist()
    for axis in (0,1):
        comparison_figure(representatives, signals, validation, axis, akey,
                          afs, ahigh, 'm/s²', .02, figures/f'acceleration_{"xy"[axis]}.png', bpfo, lag_s)
        comparison_figure(representatives, signals, measured_v, axis, vkey,
                          vfs, vhigh, 'mm/s', .12, figures/f'velocity_{"xy"[axis]}.png',
                          bpfo if processing else None, lag_s)
    # Geometry profile, independent of the dynamic trajectory.
    fig, axes = plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    largest = representatives[-1]['geometry']
    coordinate = np.linspace(-.6e-3, .6e-3, 1001)
    for i,c in enumerate(representatives[1:], 1):
        phi = c['geometry']['center_rad'] + coordinate/largest['track_radius_m']
        axes[0].plot(coordinate*1e3, outer_race_pit_displacement(phi, c['parameters'])*1e6,
                     color=COLORS[i], label=f"{c['area_mm2']:g} mm²")
    axes[0].set(xlabel='Circumferential arc coordinate (mm)', ylabel='Normal approach loss (µm)',
                title='Prescribed pit passage profile; width fixed at 1 mm')
    axes[0].legend()
    for group in sorted({c['group'] for c in cases}):
        selected = [c for c in cases if c['group']==group]
        axes[1].plot([c['area_mm2'] for c in selected], [c['H_acc_y_rms_m_s2'] for c in selected],
                     'o-' if group=='phase_0' else 's--', label=group)
    axes[1].set(xlabel='Outer-race pit area (mm²)', ylabel='H Y acceleration RMS (m/s²)',
                title=f'10–{ahigh} Hz response; two excitation phases')
    axes[1].legend()
    fig.savefig(figures/'geometry_and_response.png',dpi=180)
    plt.close(fig)
    # Local spectrum makes the narrow fault line visible on the same physical scale.
    fig, axes = plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    reference_excerpt = slice(round(.1*afs),round(.1*afs)+round(.01*afs))
    axes[0].plot(np.arange(round(.01*afs))/afs, validation[0,1,reference_excerpt],
                 color='.65', lw=.8, label='Real healthy')
    rf, rs = welch(validation[:,1],fs=afs,nperseg=afs,axis=-1)
    axes[1].semilogy(rf,rs.mean(0),color='.65',label='Real healthy mean')
    for i,c in enumerate(representatives):
        a = signals[c['case']][akey][1]
        start = round(.1*afs)+round(lag_s*afs)
        excerpt = slice(start,start+round(.01*afs))
        axes[0].plot(np.arange(round(.01*afs))/afs, a[excerpt], color=COLORS[i], lw=.9, label=f"{c['area_mm2']:g} mm²")
        f,s = welch(a,fs=afs,nperseg=afs)
        axes[1].semilogy(f,s,color=COLORS[i],label=f"{c['area_mm2']:g} mm²")
    axes[0].set(xlabel='Relative time (s)',ylabel='H Y acceleration (m/s²)',
                title=f'Healthy-based alignment: {lag_s*1000:+.2f} ms')
    axes[0].legend()
    axes[1].axvline(bpfo,color='.4',ls='--',lw=.8)
    axes[1].set(xlim=(950,1080),ylim=(1e-11 if processing else 1e-16,10),xlabel='Frequency (Hz)',ylabel='PSD [(m/s²)²/Hz]',title=f'BPFO = {bpfo:.2f} Hz; 1 Hz bins')
    axes[1].legend()
    fig.savefig(figures/'fault_signature.png',dpi=180)
    plt.close(fig)
    spectral_energy = []
    for c in cases:
        f, psd = welch(signals[c['case']][akey][1], fs=afs, nperseg=afs//4)
        total = float(psd[(f>=10)&(f<=ahigh)].sum()*(f[1]-f[0]))
        fundamental = float(psd[np.abs(f-bpfo)<=8].sum()*(f[1]-f[0]))
        spectral_energy.append(dict(case=c['case'], area_mm2=c['area_mm2'],
                                    total_band_power_m2_s4=total, bpfo_band_power_m2_s4=fundamental,
                                    bpfo_band_power_percent=100*fundamental/total))
    selected_energy = [s for s in spectral_energy if s['case'] in {c['case'] for c in representatives}]
    fig, axes = plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    x = np.arange(len(selected_energy))
    axes[0].bar(x-.16,[s['total_band_power_m2_s4'] for s in selected_energy],.32,label=f'Total 10–{ahigh} Hz')
    axes[0].bar(x+.16,[s['bpfo_band_power_m2_s4'] for s in selected_energy],.32,label='BPFO ±8 Hz')
    axes[0].set(ylabel='Y acceleration mean-square [(m/s²)²]',title='Welch band power; linear scale')
    axes[0].legend()
    axes[1].bar(x,[s['bpfo_band_power_percent'] for s in selected_energy],color=COLORS)
    axes[1].set(ylabel='BPFO band / total band power (%)',ylim=(0,100),title='Energy concentrated near the fault frequency')
    for axis in axes:
        axis.set(xticks=x,xticklabels=[f"{s['area_mm2']:g}" for s in selected_energy],xlabel='Pit area (mm²)')
    fig.savefig(figures/'spectral_energy.png',dpi=180)
    plt.close(fig)
    peaks = []
    for c in cases:
        if c['area_mm2'] == 0:
            continue
        baseline = next(b for b in cases if b['group']==c['group'] and b['area_mm2']==0)
        a = signals[c['case']][akey][1]-signals[baseline['case']][akey][1]
        f,s = welch(a,fs=afs,nperseg=afs)
        near = np.flatnonzero(np.abs(f-bpfo)<30)
        peak = float(f[near[np.argmax(s[near])]])
        peaks.append(dict(case=c['case'], peak_hz=peak, quantity='matched fault-minus-healthy Y acceleration',
                          within_one_hz=abs(peak-bpfo)<=1))
    edge_checks = []
    for c in cases:
        baseline = next(b for b in cases if b['group']==c['group'] and b['area_mm2']==0)
        b = signals[baseline['case']][vkey][1]
        v = signals[c['case']][vkey][1]
        interior = slice(round(.1*vfs), round(.9*vfs))
        edge_checks.append(dict(case=c['case'], interval_s=[.1,.9],
            Y_velocity_rms_ratio_to_matched_healthy=float(np.linalg.norm(v[interior])/np.linalg.norm(b[interior])),
            Y_velocity_paired_relative_l2=float(np.linalg.norm(v[interior]-b[interior])/np.linalg.norm(b[interior]))))
    summary = dict(bpfo_hz=bpfo, bpfo_local_peaks=peaks, velocity_edge_check=edge_checks,
                   displayed_time_offset_s=.1, time_alignment=alignment, spectral_energy=spectral_energy,
                   all_native_states_finite=True, independent_phase_families=2,
                   real_acceleration_validation=acceleration_metrics(validation, bpfo, afs),
                   by_area=[], real_per_run=[], healthy_baseline_comparison=[])
    for area in sorted({c['area_mm2'] for c in cases}):
        same = [c for c in cases if c['area_mm2']==area]
        summary['by_area'].append(dict(area_mm2=area, **{
            key: dict(mean=float(np.mean([c[key] for c in same])),
                      min=float(min(c[key] for c in same)), max=float(max(c[key] for c in same)))
            for key in ('H_acc_x_rms_m_s2','H_acc_y_rms_m_s2','H_bpfo_y_rms_m_s2',
                        'H_vel_x_rms_mm_s','H_vel_y_rms_mm_s','T_dynamic_radial_rms_um')}))
    for run_id in dict.fromkeys(r['run_id'] for r in provenance):
        select = [i for i,r in enumerate(provenance) if r['run_id']==run_id]
        summary['real_per_run'].append(dict(run_id=run_id, split=provenance[select[0]]['split'],
                                           **acceleration_metrics(measured[select],bpfo,afs)))
    model = np.stack([signals[c['case']][akey] for c in cases if c['area_mm2']==0])
    model_rms = np.sqrt(np.mean(model**2,axis=(0,2)))
    model_velocity = np.stack([signals[c['case']][vkey] for c in cases if c['area_mm2']==0])
    model_vrms = np.sqrt(np.mean(model_velocity**2,axis=(0,2)))
    frequency, model_psd = welch(model,fs=afs,nperseg=afs//4,axis=-1)
    band = (frequency>=10)&(frequency<=ahigh)
    for label in ('train','validation'):
        select = [i for i,r in enumerate(provenance) if r['split']==label]
        values = measured[select]
        rms = np.sqrt(np.mean(values**2,axis=(0,2)))
        _, psd = welch(values,fs=afs,nperseg=afs//4,axis=-1)
        error = np.sqrt(np.mean((10*np.log10(model_psd.mean(0)[:,band])
                                -10*np.log10(psd.mean(0)[:,band]))**2,axis=-1))
        summary['healthy_baseline_comparison'].append(dict(split=label, reference_windows=len(select),
                    measured_acc_rms_xy_m_s2=rms.tolist(), model_acc_rms_xy_m_s2=model_rms.tolist(),
                    relative_rms_error_percent=(100*(model_rms/rms-1)).tolist(),
                    measured_velocity_rms_xy_mm_s=np.sqrt(np.mean(velocity_refs[label]**2,axis=(0,2))).tolist(),
                    model_velocity_rms_xy_mm_s=model_vrms.tolist(),
                    velocity_rms_error_percent=(100*(model_vrms/np.sqrt(np.mean(velocity_refs[label]**2,axis=(0,2)))-1)).tolist(),
                    log_psd_rmse_xy_db=error.tolist()))
    for run in summary['real_per_run']:
        rms = np.sqrt(np.mean(np.asarray(run['rms_xy_m_s2'])**2,axis=0))
        run['pooled_rms_xy_m_s2'] = rms.tolist()
        run['healthy_model_rms_error_percent'] = (100*(model_rms/rms-1)).tolist()
    save_json(out/'analysis.json', summary)
    print(json.dumps(summary['by_area'],indent=2),flush=True)


if __name__ == '__main__':
    main()
