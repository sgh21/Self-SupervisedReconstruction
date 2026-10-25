"""Common-window statistics, stationarity and independent V3 convergence.

Primary displacement keeps 1X. Harmonic removal is diagnostic only. Contact
impulse means integral of the resultant force magnitude over one revolution.
"""
import numpy as np
from scipy.signal import periodogram

from utils.pitting_analysis import resample_to
from utils.runout_labels import complete_turn_runout


BANDS = ((10, 50), (50, 100), (100, 900), (900, 2048))


def _harmonics(time, signal, omega, phase, prefix, unit, energy_unit, fs):
    angle = omega*time+phase
    design = np.column_stack((np.ones(len(time)), np.cos(angle), np.sin(angle)))
    coef = np.linalg.lstsq(design, signal, rcond=None)[0]
    residual = signal-design@coef
    frequencies, psd = periodogram(residual, fs=fs, window='hann', detrend=False,
                                  scaling='density', axis=0)
    metrics = {}
    for name, values in [('rms', np.sqrt(np.mean(signal**2, axis=0))),
                         ('1x', np.linalg.norm(coef[1:], axis=0)),
                         ('residual', np.sqrt(np.mean(residual**2, axis=0)))]:
        for axis, value in zip('xy', values):
            metrics[f'{prefix}_{name}_{axis}_{unit}'] = float(value)
    for low, high in BANDS:
        band = (frequencies >= low) & (frequencies < high)
        energy = psd[band].sum(axis=0)*(frequencies[1]-frequencies[0])
        for axis, value in zip('xy', energy):
            metrics[f'{prefix}_residual_energy_{low}_{high}_{axis}_{energy_unit}'] = float(value)
    return metrics


def _window(raw, output, config, begin, end):
    time, q, relative, force, active = raw
    tout, velocity, acceleration = output
    p = config['parameters']
    phase = config['excitation']['phase_rad']
    turns = complete_turn_runout(time, relative*1e6, p['rpm'], begin, end,
                                 phase_at_zero_rad=phase)
    pp = turns['peak_to_peak_um']
    if not len(pp):
        raise ValueError('V3 analysis needs complete revolutions in its requested window')
    mean = pp.mean(axis=0)
    metrics = dict(E_x_um=float(mean[0]), E_y_um=float(mean[1]), E_dir_um=float(max(mean)),
                   complete_turns=len(pp), measurement_start_s=float(begin), measurement_end_s=float(end))
    for axis, values in zip('xy', pp.T):
        for stat, value in [('std', np.std(values)), ('min', min(values)), ('max', max(values))]:
            metrics[f'E_{axis}_{stat}_um'] = float(value)
    use = (time >= begin-1e-12) & (time < end-1e-12)
    center = relative[use].mean(axis=0)*1e6
    outer = (q[use, 6:8]-q[use, 4:6]).mean(axis=0)*1e6
    metrics.update(mean_x_um=float(center[0]), mean_y_um=float(center[1]),
                   OH_mean_x_um=float(outer[0]), OH_mean_y_um=float(outer[1]),
                   fit_contact_fraction=float(np.mean(active[use] > 0)),
                   fit_peak_N=float(force[use].max()), fit_mean_N=float(force[use].mean()))
    per_impulse, per_peak, per_contact = [], [], []
    for left, right in zip(turns['start_time_s'], turns['end_time_s']):
        interior = (time > left) & (time < right)
        local_t = np.r_[left, time[interior], right]
        local_force = np.r_[np.interp(left, time, force), force[interior], np.interp(right, time, force)]
        per_impulse.append(np.trapezoid(local_force, local_t))
        per_peak.append(local_force.max())
        per_contact.append(np.mean(active[(time >= left) & (time < right)] > 0))
    turns.update(fit_impulse_N_s=np.asarray(per_impulse), fit_peak_N=np.asarray(per_peak),
                 fit_contact_fraction=np.asarray(per_contact))
    metrics['fit_impulse_N_s'] = float(np.mean(per_impulse))
    use_out = (tout >= begin-1e-12) & (tout < end-1e-12)
    omega = p['rpm']*2*np.pi/60
    fs = config['simulation']['fs_out_hz']
    metrics.update(_harmonics(tout[use_out], velocity[use_out], omega, phase, 'H_v', 'mm_s', 'mm2_s2', fs))
    metrics.update(_harmonics(tout[use_out], acceleration[use_out], omega, phase, 'H_a', 'm_s2', 'm2_s4', fs))
    return metrics, turns


def _tolerance(key, reference, config, dynamic_scale, steady=False):
    t = config['numerical_thresholds']
    if key in ('mean_x_um', 'mean_y_um', 'OH_mean_x_um', 'OH_mean_y_um'):
        return max(t['position_um'], t['relative_main']*dynamic_scale)
    if key.startswith('E_'):
        return max(t['position_um'], t['relative_main']*abs(reference))
    if key.startswith('H_v_'):
        floor = t['velocity_mm_s']
    elif key.startswith('H_a_'):
        floor = t['acceleration_m_s2']
    elif key == 'fit_contact_fraction':
        return t['contact_rate']
    elif key == 'fit_impulse_N_s':
        return max(t['contact_impulse_N_s'], t['relative_contact_impulse']*abs(reference))
    elif key in ('fit_peak_N', 'fit_mean_N'):
        return max(t['contact_peak_N'], t['relative_contact_peak']*abs(reference))
    relative = t['relative_residual_steady'] if steady and '_residual_' in key else t['relative_main']
    return max(floor, relative*abs(reference))


def _comparison_keys(metrics):
    return [key for key in metrics if key in ('E_x_um', 'E_y_um', 'E_dir_um', 'mean_x_um',
        'mean_y_um', 'OH_mean_x_um', 'OH_mean_y_um', 'fit_contact_fraction', 'fit_impulse_N_s',
        'fit_peak_N', 'fit_mean_N') or (key.startswith(('H_v_', 'H_a_')) and '_energy_' not in key)]


def _categories(checks):
    groups = {}
    for label, prefix in [('motion', ('E_', 'mean_', 'OH_')), ('velocity', ('H_v_',)),
                          ('acceleration', ('H_a_',)), ('contact', ('fit_',))]:
        groups[label+'_pass'] = bool(all(item['passed'] for key, item in checks.items() if key.startswith(prefix)))
    return groups


def compare_v3(base, fine, config):
    """Compare scalar summaries on exactly the same time/phase window."""
    same_window = (abs(base['measurement_start_s']-fine['measurement_start_s']) < 1e-10
                   and abs(base['measurement_end_s']-fine['measurement_end_s']) < 1e-10)
    checks = {}
    for key in _comparison_keys(fine):
        tolerance = _tolerance(key, fine[key], config, fine['E_dir_um'])
        difference = abs(base[key]-fine[key])
        checks[key] = dict(base=base[key], fine=fine[key], difference=difference,
                           tolerance=tolerance, passed=bool(difference <= tolerance))
    groups = _categories(checks)
    return dict(passed=bool(same_window and all(groups.values())), same_window=same_window,
                metrics=checks, **groups)


def analyze_v3(result, config, start, end, stride=1):
    """Analyze all exported samples and every consecutive acceptance subwindow.

    stride=2 samples the same retained dense solution at half the native rate.
    Both branches use true guards before resampling; no display downsampling is
    used to compute runout. start/end must already be within those guards.
    """
    settings = config['simulation']
    time, q = result['t'][::stride], result['q'][::stride]
    relative = result['measured_relative_xy_m'][::stride]
    force = np.linalg.norm(result['fit_force_N'][::stride], axis=1)
    raw = time, q, relative, force, result['fit_active_count'][::stride]
    source_fs = settings['fs_raw_hz']/stride
    fs_out = settings['fs_out_hz']
    velocity = resample_to(result['v'][::stride, 4:6].T, source_fs, fs_out).T*1e3
    acceleration = resample_to(result['acc'][::stride, 4:6].T, source_fs, fs_out).T
    tout = time[0]+np.arange(len(velocity))/fs_out
    if start < time[0]+settings['guard_s']-1/source_fs or end > time[-1]-settings['guard_s']+1/source_fs:
        raise ValueError('Export interval needs real raw guards on both sides')
    output = tout, velocity, acceleration
    metrics, turns = _window(raw, output, config, start, end)
    edges = np.linspace(start, end, settings['subwindows']+1)
    windows = [_window(raw, output, config, left, right)[0] for left, right in zip(edges[:-1], edges[1:])]
    checks = {}
    for key in _comparison_keys(metrics):
        values = np.array([window[key] for window in windows])
        tolerance = _tolerance(key, max(abs(values)), config, metrics['E_dir_um'], steady=True)
        spread = float(np.ptp(values))
        checks[key] = dict(minimum=float(min(values)), maximum=float(max(values)),
                           difference=spread, tolerance=tolerance, passed=bool(spread <= tolerance))
    bands = {}
    for key in metrics:
        if '_energy_' not in key:
            continue
        values = np.array([window[key] for window in windows])
        floor = (config['numerical_thresholds']['residual_band_energy_floor_mm2_s2']
                 if key.startswith('H_v_') else config['numerical_thresholds']['acceleration_m_s2']**2)
        tolerance = max(floor, config['numerical_thresholds']['relative_residual_steady']*max(values))
        decays = bool(max(values) > floor and values[0]-values[-1] > tolerance and np.all(np.diff(values) < 0))
        bands[key] = dict(values=values.tolist(), floor=floor, drop=float(values[0]-values[-1]),
                          tolerance=tolerance, significant_monotonic_decay=decays, passed=not decays)
    groups = _categories(checks)
    enough_turns = all(window['complete_turns'] >= 20 for window in windows)
    use = (tout >= start-1e-12) & (tout < end-1e-12)
    covered = sum(int(np.sum((tout >= left-1e-12) & (tout < right-1e-12)))
                  for left, right in zip(edges[:-1], edges[1:]))
    stationarity = dict(passed=bool(all(groups.values()) and enough_turns
                       and covered == int(use.sum()) and all(item['passed'] for item in bands.values())),
        window_edges_s=edges.tolist(), subwindow_metrics=windows, checks=checks,
        band_decay_checks=bands, each_subwindow_at_least_20_turns=enough_turns,
        export_sample_count=int(use.sum()), covered_sample_count=covered, **groups)
    metrics.update(clearance_um=config['parameters']['fit_clearance_um'],
                   rpm=config['parameters']['rpm'], steady_pass=stationarity['passed'])
    return metrics, turns, (tout[use], velocity[use], acceleration[use]), stationarity
