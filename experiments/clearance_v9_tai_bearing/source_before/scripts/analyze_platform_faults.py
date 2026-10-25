"""Physical-unit comparison of a shared-drive spindle/platform fault series."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import welch

from utils.data_io import save_json


KINDS = ('acceleration', 'velocity', 'displacement')
UNITS = {'acceleration': 'm/s²', 'velocity': 'mm/s', 'displacement': 'µm'}
CHANNELS = ('ACC2: housing X', 'ACC1: housing Y', 'ACC3: platform Y')
COLORS = ('#526875', '#168b9c', '#d39431', '#b44762', '#775f9c')
REAL_COLOR = '#8c9398'
DAY_COLORS = ('#477b95', '#76a196', '#b29161')


def configure_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'legend.frameon': False})


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_figure(fig, folder, name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder/f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(folder/f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def correlation(reference, modeled):
    reference = reference-reference.mean(-1, keepdims=True)
    modeled = modeled-modeled.mean(-1, keepdims=True)
    return np.sum(reference*modeled, axis=-1)/np.sqrt(
        np.sum(reference**2, axis=-1)*np.sum(modeled**2, axis=-1))


def one_x_phasor(values, time_s, frequency_hz):
    z = np.exp(2j*np.pi*frequency_hz*np.asarray(time_s))
    design = np.column_stack((z.real, -z.imag, np.ones(len(z))))
    fit = np.linalg.lstsq(design, np.asarray(values).T, rcond=None)[0]
    return fit[0]+1j*fit[1]


def healthy_phase_registration(reference_velocity, healthy_spline, origin_s, fs, frequency_hz):
    """Reuse the joint-1X first-half registration, with one lag for all outputs.

    Housing X/Y determine the lag. Platform Y is an independent response check;
    every fault case inherits this healthy lag without another optimization.
    """
    time_s = np.arange(fs//2)/fs
    target = reference_velocity[:2, :fs//2].T
    target = target-target.mean(0)
    norm = np.linalg.norm(target, axis=0)

    def score(lags):
        predicted = healthy_spline(origin_s+time_s[None, :]+np.atleast_1d(lags)[:, None])[..., :2]
        predicted = predicted-predicted.mean(1, keepdims=True)
        return (np.einsum('stc,tc->sc', predicted, target)/
                (np.linalg.norm(predicted, axis=1)*norm)).mean(1)

    lags = np.linspace(-.5/frequency_hz, .5/frequency_hz, 256, endpoint=False)
    best = int(np.argmax(score(lags)))
    step = 1/(frequency_hz*len(lags))
    result = minimize_scalar(lambda lag: -score(lag)[0],
                             bounds=(lags[best]-step, lags[best]+step),
                             method='bounded', options={'xatol': 1e-10})
    return float(result.x), float(-result.fun)


def band_psd(values, fs):
    """Same Welch estimator for every case; no spectrum normalization."""
    return welch(values, fs=fs, nperseg=min(2*fs, values.shape[-1]),
                 noverlap=min(fs, values.shape[-1]//2), axis=-1)


def load_cases(output, fs, frequency_hz):
    summary = read_json(output/'generation_summary.json')
    entries = summary if isinstance(summary, list) else summary['cases']
    cases = []
    for entry in entries:
        with np.load(output/entry['path']) as data:
            values = {key: data[key] for key in
                      ('t', *KINDS, 'holder_absolute_um', 'holder_relative_platform_um')}
        case = dict(entry, values=values, spline={}, psd={}, one_x_rms={})
        for kind in KINDS:
            case['spline'][kind] = CubicSpline(values['t'], values[kind].T)
            f, case['psd'][kind] = band_psd(values[kind], fs)
            case['one_x_rms'][kind] = abs(one_x_phasor(
                values[kind], values['t'], frequency_hz))/np.sqrt(2)
        case['frequency_hz'] = f
        cases.append(case)
    return cases


def project_one_x(values, time_s, frequency_hz):
    phasor = one_x_phasor(values, time_s, frequency_hz)
    return (phasor[:, None]*np.exp(2j*np.pi*frequency_hz*time_s)).real


def compare_healthy(refs, runs, healthy, fs, frequency_hz):
    time_s = np.arange(fs)/fs
    origin = .5
    aligned = {kind: np.empty_like(refs[kind]) for kind in KINDS}
    registrations, rows = [], []
    for i, run in enumerate(runs):
        for segment in range(refs['velocity'].shape[1]):
            lag, score = healthy_phase_registration(
                refs['velocity'][i, segment], healthy['spline']['velocity'],
                origin, fs, frequency_hz)
            registrations.append(dict(run_id=run['run_id'], split=run['split'],
                segment=segment, lag_s=lag, registration_xy_correlation=score))
            for kind in KINDS:
                simulated = healthy['spline'][kind](origin+time_s+lag).T
                aligned[kind][i, segment] = simulated
                real = refs[kind][i, segment, :, fs//2:]
                model = simulated[:, fs//2:]
                real_1x = project_one_x(real, time_s[fs//2:], frequency_hz)
                model_1x = project_one_x(model, time_s[fs//2:], frequency_hz)
                corr, corr_1x = correlation(real, model), correlation(real_1x, model_1x)
                nrmse = np.linalg.norm(real-model, axis=-1)/np.linalg.norm(
                    real-real.mean(-1, keepdims=True), axis=-1)
                rms_real = np.sqrt(np.mean(real**2, axis=-1))
                rms_model = np.sqrt(np.mean(model**2, axis=-1))
                for channel in range(3):
                    rows.append(dict(run_id=run['run_id'], day=run['day'],
                        split=run['split'], segment=segment, kind=kind,
                        channel=CHANNELS[channel], unit=UNITS[kind], lag_s=lag,
                        evaluation_correlation=float(corr[channel]),
                        evaluation_1x_correlation=float(corr_1x[channel]),
                        evaluation_nrmse=float(nrmse[channel]),
                        real_rms=float(rms_real[channel]), model_rms=float(rms_model[channel])))
    return aligned, registrations, rows


def display_run_indices(runs, split):
    days = sorted({run['day'] for run in runs if run['split'] == split})
    return [next(i for i, run in enumerate(runs) if run['day'] == day) for day in days]


def healthy_time_figures(refs, runs, aligned, folder, fs, frequency_hz):
    time_s = np.arange(fs)/fs
    excerpt = (time_s >= .5) & (time_s < .56)
    for split in ('train', 'validation'):
        indices = display_run_indices(runs, split)
        for kind in KINDS:
            for only_1x in (False, True):
                fig, axes = plt.subplots(len(indices), 3,
                    figsize=(13.5, max(3.2, 2.7*len(indices))),
                    layout='constrained', squeeze=False)
                for row, i in enumerate(indices):
                    real = refs[kind][i, 0]
                    projection = project_one_x(real[:, fs//2:], time_s[fs//2:], frequency_hz)
                    for channel, ax in enumerate(axes[row]):
                        if not only_1x:
                            ax.plot((time_s[excerpt]-.5)*1000, real[channel, excerpt],
                                    color=REAL_COLOR, lw=.7, label='Measured healthy, full band')
                        ax.plot((time_s[excerpt]-.5)*1000,
                            projection[channel, time_s[fs//2:] < .56],
                            color=DAY_COLORS[0], lw=1, ls='--', label='Measured healthy, 1X projection')
                        ax.plot((time_s[excerpt]-.5)*1000, aligned[kind][i, 0, channel, excerpt],
                            color=COLORS[1], lw=1.1, label='Frozen healthy model')
                        ax.set(xlabel='Time into evaluation half (ms)',
                            ylabel=f'{kind.title()} ({UNITS[kind]})',
                            title=f'{runs[i]["day"][4:6]}/{runs[i]["day"][6:]} | {CHANNELS[channel]}')
                for channel in range(3):
                    limits = [ax.get_ylim() for ax in axes[:, channel]]
                    extent = max(max(abs(a), abs(b)) for a, b in limits)
                    for ax in axes[:, channel]:
                        ax.set_ylim(-extent, extent)
                fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
                           bbox_to_anchor=(.5, -.08), ncol=3, fontsize=8)
                fig.suptitle(f'{split.title()} healthy | '+
                    ('measured 1X vs frozen model' if only_1x else '10–2048 Hz')+
                    ' | one housing XY lag from preceding 0.5 s; no amplitude rescaling')
                suffix = '_1x' if only_1x else ''
                save_figure(fig, folder, f'healthy_{split}_{kind}{suffix}_time')


def psd_panel(ax, curves, frequency_hz, unit, xlim, rotation_hz):
    mask = (frequency_hz >= xlim[0]) & (frequency_hz <= xlim[1])
    highest = max(float(curve['psd'][mask].max()) for curve in curves)
    for curve in curves:
        ax.semilogy(frequency_hz[mask], np.maximum(curve['psd'][mask], 1e-30),
                    color=curve['color'], lw=curve.get('lw', 1),
                    ls=curve.get('ls', '-'), label=curve['label'])
        if 'range' in curve:
            ax.fill_between(frequency_hz[mask], curve['range'][0, mask],
                            curve['range'][1, mask], color=curve['color'], alpha=.12)
    ax.axvline(rotation_hz, color='.6', ls=':', lw=.7)
    ax.set(xlim=xlim, ylim=(highest*1e-8, highest*3),
           xlabel='Frequency (Hz)', ylabel=f'PSD (({unit})²/Hz)')


def healthy_spectrum_figures(refs, runs, healthy, folder, frequency_hz):
    f = refs['frequency_hz']
    for split in ('train', 'validation'):
        days = sorted({run['day'] for run in runs if run['split'] == split})
        fig, axes = plt.subplots(3, 6, figsize=(22, 9), layout='constrained')
        for row, kind in enumerate(KINDS):
            for channel in range(3):
                curves = []
                for day_index, day in enumerate(days):
                    indices = [i for i, run in enumerate(runs) if run['day'] == day]
                    powers = refs[kind+'_psd'][indices, channel]
                    curves.append(dict(psd=powers.mean(0), range=np.quantile(powers, [.1, .9], axis=0),
                        color=DAY_COLORS[day_index], label=f'Measured {day[4:6]}/{day[6:]}'))
                curves.append(dict(psd=healthy['psd'][kind][channel], color=COLORS[1],
                                   label='Frozen healthy', ls='--', lw=1.2))
                for view, xlim in enumerate(((10, 2048), (100, 134))):
                    ax = axes[row, 2*channel+view]
                    psd_panel(ax, curves, f, UNITS[kind], xlim, frequency_hz)
                    ax.set_title(CHANNELS[channel]+(' | full band' if view == 0 else ' | 1X'))
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f'{split.title()} healthy spectra | rows: acceleration / velocity / dynamic displacement\n'
                     '2 s Hann; measured date means and 10–90% run ranges; numerical floor clipped at eight decades')
        save_figure(fig, folder, f'healthy_{split}_spectra')


def fault_figures(cases, healthy, refs, runs, registrations, folder, fs, frequency_hz,
                  characteristic_frequencies):
    families = sorted({case['fault_type'] for case in cases if case is not healthy})
    train = [i for i, run in enumerate(runs) if run['split'] == 'train']
    representative = train[0]
    lag = next(item['lag_s'] for item in registrations
               if item['run_id'] == runs[representative]['run_id'] and item['segment'] == 0)
    time_s = np.arange(fs)/fs
    excerpt = (time_s >= .5) & (time_s < .56)
    for family in families:
        members = [healthy]+sorted([case for case in cases if case['fault_type'] == family],
                                   key=lambda item: item['severity'])
        fig, axes = plt.subplots(2, len(members), figsize=(4*len(members), 7.5),
                                 layout='constrained', squeeze=False)
        extent = 0
        for row, (key, title) in enumerate(zip(
                ('holder_relative_platform_um', 'holder_absolute_um'),
                ('Relative to platform', 'Absolute coordinates'))):
            for index, case in enumerate(members):
                ax = axes[row, index]
                values = case['values'][key]
                values = values-values.mean(-1, keepdims=True)
                last = case['values']['t'] >= case['values']['t'][-1]-.25
                extent = max(extent, float(np.max(abs(values[:, last]))))
                ax.plot(values[0, last], values[1, last], color=COLORS[index],
                        lw=.7)
                ax.set(xlabel='Holder X (µm)', ylabel=title+'\nHolder Y (µm)',
                       title=case['label'], aspect='equal')
        for ax in axes.flat:
            ax.set(xlim=(-extent*1.08, extent*1.08), ylim=(-extent*1.08, extent*1.08))
        fig.suptitle(f'{family}: holder center trajectories | final 0.25 s; common physical scales\n'
                     'Mean-removed direct states; not accelerometer-integrated motion or an ISO accuracy grade')
        save_figure(fig, folder, f'{family}_holder_orbits')

        for kind in KINDS:
            fig, axes = plt.subplots(3, 2, figsize=(13.5, 9), layout='constrained')
            for channel in range(3):
                ax = axes[channel, 0]
                ax.plot((time_s[excerpt]-.5)*1000,
                        refs[kind][representative, 0, channel, excerpt], color=REAL_COLOR,
                        lw=.7, alpha=.8, label='Measured healthy, 07/22 first run')
                powers = refs[kind+'_psd'][train, channel]
                curves = [dict(psd=powers.mean(0), range=np.quantile(powers, [.1, .9], axis=0),
                    color=REAL_COLOR, label='Measured healthy, 07/22–24')]
                for index, case in enumerate(members):
                    value = case['spline'][kind](.5+time_s[excerpt]+lag)[:, channel]
                    ax.plot((time_s[excerpt]-.5)*1000, value, color=COLORS[index],
                            lw=1, label=case['label'])
                    curves.append(dict(psd=case['psd'][kind][channel], color=COLORS[index],
                                       label=case['label'], lw=1))
                ax.set(xlabel='Time into evaluation half (ms)',
                       ylabel=f'{kind.title()} ({UNITS[kind]})', title=CHANNELS[channel])
                psd_panel(axes[channel, 1], curves, refs['frequency_hz'],
                          UNITS[kind], (10, 2048), frequency_hz)
                cf = characteristic_frequencies
                markers = {
                    'outer': [(cf['outer_pass'], 'BPFO'), (2*cf['outer_pass'], '2BPFO')],
                    'inner': [(cf['inner_pass']-frequency_hz, 'BPFI−1X'),
                              (cf['inner_pass'], 'BPFI'), (cf['inner_pass']+frequency_hz, 'BPFI+1X')],
                    'ball': [(cf['ball_waviness_2x']-cf['cage'], '2BSF−FTF'),
                             (cf['ball_waviness_2x'], '2BSF'),
                             (cf['ball_waviness_2x']+cf['cage'], '2BSF+FTF')],
                }.get(family, [])
                for value, label in markers:
                    axes[channel, 1].axvline(value, color='.4', ls='--', lw=.6)
                    axes[channel, 1].text(value, .97, f' {label}: {value:.1f} Hz',
                        transform=axes[channel, 1].get_xaxis_transform(),
                        ha='right', va='top', fontsize=7, rotation=90)
                axes[channel, 1].set_title(CHANNELS[channel]+' | 2 s Hann PSD')
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
                       bbox_to_anchor=(.5, -.035), ncol=3, fontsize=8)
            fig.suptitle(f'{family}: {kind} | same force and healthy-fitted lag for every level\n'
                         'Measured curves are healthy references, not fault ground truth; no amplitude normalization')
            save_figure(fig, folder, f'{family}_{kind}')


def one_x_diagnostics(refs, runs, healthy, folder, frequency_hz):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    targets = refs['target_acceleration_rms']
    for channel, ax in enumerate(axes):
        for split, color in [('train', DAY_COLORS[0]), ('validation', DAY_COLORS[2])]:
            indices = [i for i, run in enumerate(runs) if run['split'] == split]
            ax.scatter(indices, targets[indices, channel], s=20, color=color, label=split.title())
        ax.axhline(healthy['one_x_rms']['acceleration'][channel], color=COLORS[1],
                   label='Frozen healthy model')
        ax.set(xlabel='Run index, date/time order', ylabel='Net 1X acceleration RMS (m/s²)',
               title=CHANNELS[channel])
    axes[0].legend(fontsize=8)
    fig.suptitle('Three-channel amplitude alignment | measured ±4 Hz band minus local background')
    save_figure(fig, folder, 'healthy_one_x_amplitudes')

    phasors = refs['velocity_1x_phasor']
    simulated = one_x_phasor(healthy['values']['velocity'], healthy['values']['t'],
                            frequency_hz)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    diagnostics = {}
    for ax, (a, b, name) in zip(axes, ((1, 0, 'Housing Y minus X'), (2, 1, 'Platform Y minus housing Y'))):
        ratio = phasors[..., a]/phasors[..., b]
        phase = np.angle(ratio, deg=True)
        model_phase = float(np.angle(simulated[a]/simulated[b], deg=True))
        item = dict(model_phase_deg=model_phase, measured_segment_phase_deg=phase.tolist())
        for split, color in [('train', DAY_COLORS[0]), ('validation', DAY_COLORS[2])]:
            indices = [i for i, run in enumerate(runs) if run['split'] == split]
            days = sorted({runs[i]['day'] for i in indices})
            weights = np.array([1/(len(days)*sum(runs[j]['day'] == runs[i]['day'] for j in indices))
                                for i in indices])
            circle = ratio[indices]/abs(ratio[indices])
            circular_mean = np.sum(weights*np.mean(circle, axis=1))
            error = np.rad2deg(np.angle(circle*np.exp(-1j*np.deg2rad(model_phase))))
            item[split] = dict(circular_mean_deg=float(np.angle(circular_mean, deg=True)),
                concentration=float(abs(circular_mean)),
                phase_error_rms_deg=float(np.sqrt(np.sum(weights*np.mean(error**2, axis=1)))),
                phasor_amplitude_ratio_geomean=float(np.exp(np.sum(weights*np.mean(np.log(abs(ratio[indices])), axis=1)))))
            for segment in range(phase.shape[1]):
                ax.scatter(np.array(indices)+(segment-1)*.15, phase[indices, segment],
                           color=color, s=13, alpha=.65, label=split.title() if segment == 0 else None)
        ax.axhline(model_phase, color=COLORS[1], lw=1.2, label='Frozen healthy model')
        ax.set(xlabel='Run index, date/time order', ylabel='Relative 1X phase (deg)',
               ylim=(-180, 180), title=name)
        diagnostics[name] = item
    axes[0].legend(fontsize=8)
    fig.suptitle('Relative phase remains an independent check | common registration cannot change these phases\n'
                 'Angles wrapped to [−180°, 180°]; ±180° are adjacent; summaries use circular statistics')
    save_figure(fig, folder, 'healthy_relative_phase')
    return diagnostics


def separated_fault_figures(cases, output, fs):
    """One fault level per panel; reuse saved signals and the original common lag."""
    folder = output/'figures_fault_levels'
    registration = read_json(output/'alignment.json')['items'][0]
    start = 1.+registration['lag_s']
    time_s = np.arange(round(.06*fs))/fs
    families = sorted({case['fault_type'] for case in cases
                       if case['fault_type'] != 'healthy'})
    exported = []
    for family in families:
        members = sorted([case for case in cases if case['fault_type'] == family],
                         key=lambda case: case['severity'])
        for kind in KINDS:
            for domain in ('time', 'spectrum'):
                fig, axes = plt.subplots(3, len(members), figsize=(13.5, 8.3),
                    sharex=True, sharey='row', layout='constrained', squeeze=False)
                for channel in range(3):
                    curves = []
                    for column, case in enumerate(members):
                        ax = axes[channel, column]
                        if domain == 'time':
                            values = case['spline'][kind](start+time_s)[:, channel]
                            ax.plot(time_s*1000, values, color=COLORS[column+1], lw=.85)
                        else:
                            f = case['frequency_hz']
                            band = (f >= 10) & (f <= 2048)
                            values = case['psd'][kind][channel, band]
                            ax.semilogy(f[band], np.maximum(values, 1e-30),
                                        color=COLORS[column+1], lw=.85)
                        curves.append(values)
                        ax.grid(axis='y', color='.92', lw=.6)
                        if channel == 0:
                            ax.set_title(case['label'], fontsize=11, pad=10)
                        if column == 0:
                            unit = (f'{kind.title()} ({UNITS[kind]})' if domain == 'time'
                                    else f'PSD (({UNITS[kind]})²/Hz)')
                            ax.set_ylabel(CHANNELS[channel]+'\n'+unit)
                        if channel == 2:
                            ax.set_xlabel('Time (ms)' if domain == 'time' else 'Frequency (Hz)')
                    peak = max(float(np.max(abs(values))) for values in curves)
                    axes[channel, 0].set(
                        xlim=(0, 60) if domain == 'time' else (10, 2048),
                        ylim=(-1.08*peak, 1.08*peak) if domain == 'time'
                             else (peak*1e-8, peak*3))
                title = {'inner':'Inner-race pitting', 'outer':'Outer-race pitting',
                         'ball':'Ball waviness', 'taper':'Taper contact-area loss'}[family]
                view = 'time signals' if domain == 'time' else 'power spectra'
                fig.suptitle(f'{title} | {kind} {view}\n'
                    'Fault simulations only · one level per column · common scale within each row',
                    fontsize=12)
                name = f'{family}_{kind}_{domain}'
                save_figure(fig, folder, name)
                exported.append(name)
    save_json(folder/'figure_sources.json', dict(
        source='Existing accepted cases; plotting only, no new simulation or parameter fit.',
        cases=[case['path'] for case in cases if case['fault_type'] != 'healthy'],
        columns='Increasing fault severity; healthy simulation and measured records excluded.',
        rows=list(CHANNELS), units=UNITS, time_start_in_processed_record_s=start,
        time_duration_s=.06, common_lag_source=registration,
        spectrum='Existing 2 s Hann Welch PSD, 50% overlap, displayed at 10–2048 Hz.',
        scaling='No normalization; common y limits across levels for each channel; PSD shows eight decades.',
        figures=exported))
    print(f'Saved {len(exported)} separate-level figures (PNG/SVG): {folder}', flush=True)


def build_summary(refs, runs, cases, healthy, rows, frequency_hz):
    f = refs['frequency_hz']
    band = (f >= 10) & (f <= 2048)
    one_x_band = abs(f-frequency_hz) <= 4
    df = f[1]-f[0]
    summary = dict(rotation_hz=frequency_hz, fs_hz=8192, band_hz=[10, 2048],
                   channel_order=list(CHANNELS), units=UNITS, groups={}, cases=[],
                   metrics_weighting='Correlations: equal run and segment weight; 1X objective: equal date weight.',
                   displacement_note='Measured dynamic displacement is acceleration-derived double integration; holder motion is directly simulated.')
    for split in ('train', 'validation'):
        indices = [i for i, run in enumerate(runs) if run['split'] == split]
        days = sorted({runs[i]['day'] for i in indices})
        targets = refs['target_acceleration_rms'][indices]
        errors = 20*np.log10(healthy['one_x_rms']['acceleration']/targets)
        date_errors = [errors[[runs[i]['day'] == day for i in indices]] for day in days]
        summary['groups'][split] = dict(run_count=len(indices), days=days,
            target_acceleration_rms_1x_geomean_run_weighted=np.exp(np.log(targets).mean(0)).tolist(),
            target_acceleration_rms_1x_geomean_day_weighted=np.exp(np.mean([
                np.log(refs['target_acceleration_rms'][[i for i in indices if runs[i]['day'] == day]]).mean(0)
                for day in days], axis=0)).tolist(),
            one_x_error_db_rms_day_balanced=float(np.sqrt(np.mean([np.mean(item**2) for item in date_errors]))),
            one_x_error_db_rms_by_channel_run_weighted=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
            one_x_error_db_rms_by_channel_day_weighted=np.sqrt(np.mean([
                np.mean(item**2, axis=0) for item in date_errors], axis=0)).tolist())
        for kind in KINDS:
            powers = refs[kind+'_psd'][indices]
            fields = ('evaluation_correlation', 'evaluation_1x_correlation', 'evaluation_nrmse')
            metrics = {field: [float(np.mean([row[field] for row in rows if row['split'] == split
                and row['kind'] == kind and row['channel'] == channel])) for channel in CHANNELS]
                for field in fields}
            metrics.update(measured_full_band_rms_mean=np.sqrt(powers[..., band].sum(-1)*df).mean(0).tolist(),
                measured_1x_band_energy_fraction=(powers[..., one_x_band].sum(-1)/powers[..., band].sum(-1)).mean(0).tolist())
            summary['groups'][split][kind] = metrics
    for case in cases:
        item = {key: case[key] for key in ('case_id', 'fault_type', 'severity', 'label', 'path')}
        for kind in KINDS:
            item[kind] = dict(full_band_rms=np.sqrt(case['psd'][kind][:, band].sum(-1)*df).tolist(),
                              one_x_rms=case['one_x_rms'][kind].tolist())
        for key in ('holder_relative_platform_um', 'holder_absolute_um'):
            dynamic = case['values'][key]-case['values'][key].mean(-1, keepdims=True)
            item[key] = dict(dynamic_radial_rms=float(np.sqrt(np.mean(np.sum(dynamic**2, axis=0)))),
                              peak_to_peak_xy=np.ptp(dynamic, axis=-1).tolist(),
                              max_radius=float(np.max(np.linalg.norm(dynamic, axis=0))))
        summary['cases'].append(item)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/
                        'experiments/platform_faults_20261022')
    parser.add_argument('--fault-only-separated', action='store_true',
                        help='Plot separate fault levels without measured or simulated healthy traces.')
    args = parser.parse_args()
    output = args.output
    configure_style()
    if args.fault_only_separated:
        config = read_json(output/'frozen_config.json')
        fs = int(config['processing']['fs_hz'])
        cases = load_cases(output, fs, float(config['excitation']['frequency_hz']))
        separated_fault_figures(cases, output, fs)
        return
    manifest = read_json(output/'manifest.json')
    runs = manifest['runs']
    refs = dict(np.load(output/'references.npz'))
    fs = int(manifest['processing']['fs_hz'])
    frequency_hz = float(manifest['velocity_1x_phasor']['frequency_hz'])
    cases = load_cases(output, fs, frequency_hz)
    config = read_json(output/'frozen_config.json')
    healthy = next(case for case in cases if case['case_id'] == 'healthy')
    aligned, registrations, rows = compare_healthy(refs, runs, healthy, fs, frequency_hz)
    summary = build_summary(refs, runs, cases, healthy, rows, frequency_hz)
    folder = output/'figures'
    healthy_time_figures(refs, runs, aligned, folder, fs, frequency_hz)
    healthy_spectrum_figures(refs, runs, healthy, folder, frequency_hz)
    fault_figures(cases, healthy, refs, runs, registrations, folder, fs, frequency_hz,
                  config['characteristic_frequencies_hz'])
    summary['relative_phase'] = one_x_diagnostics(refs, runs, healthy, folder, frequency_hz)
    with (output/'healthy_comparison_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output/'analysis_summary.json', summary)
    save_json(output/'alignment.json', dict(items=registrations,
        simulation_origin_s=.5, registration_interval_s=[0, .5], evaluation_interval_s=[.5, 1],
        method='One housing XY velocity-correlation lag per real run/segment; same lag for all channels, quantities and fault cases.',
        scope='Signal registration only, not mechanical keyphase; validation only estimates its lag, never force or structural parameters.'))
    save_json(output/'figure_source_map.json', dict(
        source_files=['manifest.json', 'references.npz', 'generation_summary.json',
                      'frozen_config.json', *[case['path'] for case in cases]],
        display_runs={split: [runs[i]['run_id'] for i in display_run_indices(runs, split)]
                      for split in ('train', 'validation')},
        display_segment=0, time_excerpt_s=[.5, .56], orbit_excerpt='final 0.25 s',
        selection='First predetermined run/segment per date, not selected by fit quality.',
        spectrum='2 s Hann Welch, 50% overlap. Measured shading: 10–90% run quantiles, not confidence intervals.',
        spectrum_display_floor='Axes stop eight decades below largest plotted PSD; lower numerical leakage is clipped, not asserted to be zero.',
        holder_motion='Mean-removed direct states at 8192 Hz; resampling antialias only, not 10–2048 Hz vibration filtering.',
        scaling='No amplitude normalization. Holder orbit axes equal and common across levels and reference frames within each family.',
        units=UNITS, backend='Python Matplotlib; PNG and SVG'))
    print(json.dumps(summary['groups'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
