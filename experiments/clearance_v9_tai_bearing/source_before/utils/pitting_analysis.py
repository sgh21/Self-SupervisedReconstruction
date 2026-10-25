"""Read-only raw acceleration references and common pit signal measurements."""
import csv
import io
import lzma
from math import gcd
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import fftconvolve, firwin, resample, resample_poly, welch

from utils.data_io import bandlimit


def audit_steady_references(records, data_config, tolerance_rpm=10, max_telemetry_gap_s=1):
    """Verify entire selected intervals against raw speed telemetry, before fitting."""
    audit = dict(tolerance_rpm=tolerance_rpm, minimum_valid_fraction=1.0,
                 max_telemetry_gap_s=max_telemetry_gap_s, runs=[])
    for relative in data_config['train_runs']+data_config['validation_runs']:
        with h5py.File(Path(data_config['root'])/relative, 'r') as h5:
            rid = str(h5.attrs['run_id'])
            target = float(h5.attrs['target_speed_rpm'])
            timing = {k:float(h5.attrs[k]) for k in
                      ('platform_start_s','stable_wait_s','stable_cutoff_s','platform_end_s')}
        selected = [r for r in records if r['run_id']==rid]
        path = Path(selected[0]['path']).parent.parent/'spindle_telemetry.csv'
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        time = np.array([float(r['time_s']) for r in rows])
        rpm = np.array([float(r['actual_speed_rpm']) for r in rows])
        good = np.array([r['speed_ok'].lower() in ('true','1')
                         and abs(float(r['target_rpm'])-target)<.01 for r in rows])
        good &= np.isfinite(rpm) & (np.abs(rpm-target)<=tolerance_rpm)
        segments = []
        for record in selected:
            start, end = record['start_s'], record['start_s']+record['duration_s']
            # Include the telemetry sample bracketing each boundary, so a gap or
            # speed transition at the edge cannot be hidden by interior samples.
            left = max(0, np.searchsorted(time,start,side='right')-1)
            right = min(len(time), np.searchsorted(time,end,side='left')+1)
            t, speed = time[left:right], rpm[left:right]
            passed = (len(t)>1 and t[0]<=start and t[-1]>=end
                      and start>=timing['stable_cutoff_s'] and end<=timing['platform_end_s']
                      and good[left:right].all() and np.max(np.diff(t))<=max_telemetry_gap_s)
            if not passed:
                raise ValueError(f'Calibration/comparison interval is not verified steady: {rid} {start}–{end} s')
            segments.append(dict(start_s=start, end_s=end, telemetry_samples=len(t),
                                 actual_speed_range_rpm=[float(speed.min()),float(speed.max())],
                                 actual_speed_mean_rpm=float(speed.mean()),
                                 maximum_telemetry_gap_s=float(np.max(np.diff(t))), passed=True))
        audit['runs'].append(dict(run_id=rid,split=selected[0]['split'],telemetry_path=str(path),
                                 target_rpm=target,**timing,segments=segments))
    audit['passed'] = True
    return audit


def read_raw_references(data_config, raw_root, provenance, count=3, full_segments=False,
                        sensor_ids=('ACC2', 'ACC1')):
    """Read selected valid windows or their full 10 s parents, default X=ACC2/Y=ACC1.

    Reuses the upstream prepare_data.py payload/channel contract. Full-run
    allocation and valid steady-state selection come from read_healthy_runs.
    Raw filenames are resolved locally, ignoring stale upstream absolute paths.
    sensor_ids optionally selects additional recorded sensors in the given order.
    """
    signals, records = [], []
    for record in provenance:
        h5_path = Path(record['path'])
        run = Path(raw_root)/h5_path.parent.name/h5_path.stem
        with (run/'sensor_info.csv').open(encoding='utf-8-sig', newline='') as f:
            mapping = {r['sensor_id']: r['channel'] for r in csv.DictReader(f)
                       if r['signal_type'] == 'acceleration'}
        with (run/'segment_records.csv').open(encoding='utf-8-sig', newline='') as f:
            segments = {int(r['segment_index']): r for r in csv.DictReader(f)
                        if r['signal_type'] == 'acceleration'}
        with h5py.File(h5_path, 'r') as h5:
            for selected in np.linspace(0, len(record['indices'])-1, count, dtype=int):
                index = record['indices'][selected]
                segment = segments[int(h5['windows_1s/source_segment_index'][index])]
                path = run/'acceleration_25600Hz_256000samples'/segment['data_file']
                with lzma.open(path, 'rb') as f, np.load(io.BytesIO(f.read()), allow_pickle=False) as raw:
                    channels = raw['channels'].astype(str).tolist()
                    fs = int(raw['sample_rate_hz'].item())
                    t = raw['time_s']
                    values = raw['data']
                    if (fs != 25600 or str(raw['unit'].item()) != 'g' or values.shape != (3, 256000)
                            or len(channels) != 3 or len(t) != values.shape[1]
                            or not np.allclose(np.diff(t), 1/fs, rtol=0, atol=1e-10)):
                        raise ValueError(f'Unexpected raw acquisition: {path}')
                    start = float(h5['windows_1s/start_time_s'][index])
                    offset = 0 if full_segments else round((start-t[0])*fs)
                    length = 10*fs if full_segments else fs
                    if full_segments:
                        same_parent = h5['windows_1s/source_segment_index'][:] == int(segment['segment_index'])
                        if not np.all(h5['windows_1s/sample_valid'][:][same_parent]):
                            raise ValueError(f'Parent contains invalid windows: {path}')
                        start = float(t[0])
                    x = values[[channels.index(mapping[s]) for s in sensor_ids], offset:offset+length]
                    if x.shape != (len(sensor_ids), length) or not np.isfinite(x).all():
                        raise ValueError(f'Incomplete or invalid reference: {path}')
                    signals.append(x.astype(np.float64)*9.80665)
                records.append(dict(path=str(path), run_id=record['run_id'], split=record['split'],
                                    h5_window_index=int(index), start_s=start,
                                    duration_s=length/fs, segment_index=int(segment['segment_index']),
                                    channel_order=list(sensor_ids), unit='m/s^2', fs_hz=fs))
    return np.asarray(signals), records


def resample_to(signal, source_fs, target_fs):
    divisor = gcd(int(source_fs), int(target_fs))
    return resample_poly(signal, int(target_fs)//divisor, int(source_fs)//divisor,
                         axis=-1, padtype='line')


def filter_vibration(signal, source_fs, processing):
    """Shared linear-phase FIR for real and simulated data, with caller-owned guards.

    The reported band edges are firwin half-amplitude cutoffs, not ideal brick
    walls. Centered convolution removes group delay; keep only guarded samples.
    """
    values = resample_to(signal, source_fs, processing['fs_hz'])
    taps = firwin(processing['fir_taps'], processing['band_hz'], pass_zero=False,
                  fs=processing['fs_hz'], window=('kaiser', processing['kaiser_beta']))
    return fftconvolve(values, taps.reshape((1,)*(values.ndim-1)+(-1,)), mode='same', axes=-1)


def prepare_broadband_references(raw, processing, source_fs=25600):
    """Integrate full real records in frequency domain; filter before center cropping."""
    f = np.fft.rfftfreq(raw.shape[-1], 1/source_fs)
    spectrum = np.fft.rfft(raw, axis=-1)
    spectrum[..., 0] = 0
    spectrum[..., 1:] /= 2j*np.pi*f[1:]
    velocity = np.fft.irfft(spectrum, n=raw.shape[-1], axis=-1)
    acceleration = filter_vibration(raw, source_fs, processing)
    velocity = filter_vibration(velocity, source_fs, processing)*1000
    fs = processing['fs_hz']
    begin = (acceleration.shape[-1]-fs)//2
    # Excitation is calibrated to the original band, before the display FIR.
    calibration = bandlimit(resample_to(raw, source_fs, fs), fs, *processing['band_hz'])
    return acceleration[..., begin:begin+fs], velocity[..., begin:begin+fs], calibration


def process_pit_signals(result, simulation, processing=None):
    """Common physical units; no per-window normalization or appended shocks.

    New runs use common FIR filtering and explicit guards. Historical configs
    without processing retain their original hard FFT path. Native acceleration
    is oversampled; output sampling convergence is checked separately.
    """
    fs = simulation['output_fs_hz']
    if processing is not None:
        target_fs = processing['fs_hz']
        guard = processing['guard_s']
        if guard*target_fs < (processing['fir_taps']-1)/2+16:
            raise ValueError('Guard must cover FIR support plus resampling boundary')
        n = round((simulation['duration_s']+2*guard)*fs)
        begin, size = round(guard*target_fs), round(simulation['duration_s']*target_fs)
        a = filter_vibration(result['H_acceleration_m_s2'][:n].T, fs, processing)
        v = filter_vibration(result['H_velocity_m_s'][:n].T, fs, processing)
        return dict(H_acceleration_band_m_s2=a[:,begin:begin+size],
                    H_velocity_band_mm_s=v[:,begin:begin+size]*1000,
                    processed_fs_hz=target_fs, processed_band_hz=np.asarray(processing['band_hz']),
                    processed_start_time_s=float(result['t'][0]+guard))
    n = round(simulation['duration_s']*fs)
    acceleration = bandlimit(result['H_acceleration_m_s2'][:n].T, fs,
                             *simulation['acceleration_band_hz'])
    velocity = bandlimit(result['H_velocity_m_s'][:n].T, fs)
    return dict(H_acceleration_band_m_s2_25600=resample(acceleration, round(n/fs*25600), axis=-1),
                H_velocity_band_mm_s_2048=resample(velocity, round(n/fs*2048), axis=-1)*1000)


def acceleration_metrics(acceleration, bpfo, fs=25600):
    frequencies, psd = welch(acceleration, fs=fs, nperseg=min(fs//4, acceleration.shape[-1]), axis=-1)
    band = np.abs(frequencies-bpfo) <= 8
    line_rms = np.sqrt(np.sum(psd[..., band], axis=-1)*(frequencies[1]-frequencies[0]))
    return dict(rms_xy_m_s2=np.sqrt(np.mean(acceleration**2, axis=-1)).tolist(),
                bpfo_band_rms_xy_m_s2=line_rms.tolist(), bpfo_band_halfwidth_hz=8)
