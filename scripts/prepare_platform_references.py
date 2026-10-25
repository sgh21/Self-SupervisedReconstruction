"""Prepare three-channel, read-only real references for platform/fault analysis."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

from utils.data_io import save_json
from utils.pitting_analysis import (
    audit_steady_references,
    filter_vibration,
    read_raw_references,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('D:/WorkSpace/ZMProject/Self-SupervisedReconstruction/data')
H5 = DATA/'organized_data/reconstruction_velocity_temperature/7000rpm'
SELECTION = ROOT/'experiments/harmonic_excitation_20261022/manifest.json'
OUTPUT = ROOT/'experiments/platform_faults_20261022'
CHANNELS = ('ACC2', 'ACC1', 'ACC3')
PROCESS = dict(fs_hz=8192, band_hz=[10, 2048], fir_taps=4097,
               kaiser_beta=8.6, guard_s=.5)
FS = PROCESS['fs_hz']
RAW_FS = 25600
FR = 7000/60
UNITS = dict(acceleration='m/s^2', velocity='mm/s', displacement='um')


def integrate_acceleration(raw):
    """FFT integrals in SI; restrict displacement integration to f >= 10 Hz.

    Double integration amplifies slow sensor drift enough to survive the finite
    FIR stopband. Exclude that unmeasured quasi-static displacement explicitly,
    using the entire 10 s record, never a separately normalized 1 s window.
    """
    frequency = np.fft.rfftfreq(raw.shape[-1], 1/RAW_FS)
    spectrum = np.fft.rfft(raw, axis=-1)
    spectrum[..., 0] = 0
    spectrum[..., 1:] /= 2j*np.pi*frequency[1:]
    velocity = np.fft.irfft(spectrum, n=raw.shape[-1], axis=-1)
    spectrum[..., frequency < PROCESS['band_hz'][0]] = 0
    spectrum[..., 1:] /= 2j*np.pi*frequency[1:]
    displacement = np.fft.irfft(spectrum, n=raw.shape[-1], axis=-1)
    return velocity, displacement


def phasor(values):
    """Peak complex coefficient: y(t)=Re(c*exp(i*2*pi*fr*t))+DC."""
    time = np.arange(values.shape[-1])/FS
    z = np.exp(2j*np.pi*FR*time)
    design = np.column_stack([z.real, -z.imag, np.ones(len(z))])
    fitted = np.linalg.lstsq(design, values.reshape(-1, len(time)).T, rcond=None)[0]
    return (fitted[0]+1j*fitted[1]).reshape(values.shape[:-1])


def integration_check(raw, velocity, displacement):
    """Check the two integral spectra against the original acceleration in-band."""
    frequency = np.fft.rfftfreq(raw.shape[-1], 1/RAW_FS)
    band = (frequency >= 10) & (frequency <= 2048)
    omega = 2*np.pi*frequency[band]
    acceleration = np.fft.rfft(raw, axis=-1)[..., band]
    v = np.fft.rfft(velocity, axis=-1)[..., band]
    d = np.fft.rfft(displacement, axis=-1)[..., band]
    scale = np.linalg.norm(acceleration)
    return dict(
        velocity_derivative_relative_l2=float(np.linalg.norm(1j*omega*v-acceleration)/scale),
        displacement_second_derivative_relative_l2=float(np.linalg.norm(-omega**2*d-acceleration)/scale),
    )


def prepare(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = [run for run in json.loads(SELECTION.read_text(encoding='utf-8'))['runs']
            if run['day'] in ('20260722', '20260723', '20260724', '20260725')
            and run['role'] != 'challenge']
    for run in runs:
        run['role'] = run['split'] = 'validation' if run['day'] == '20260725' else 'train'
    manifest = dict(
        runs=runs, processing=PROCESS, channel_order=list(CHANNELS), units=UNITS,
        channel_meanings=['front housing X', 'front housing Y', 'table Y reference'],
        train_days=['20260722', '20260723', '20260724'], validation_days=['20260725'],
        selection_source=str(SELECTION), segments_per_run=3, raw_segment_duration_s=10,
        center_interval_s=[4.5, 5.5], psd_usable_interval_s=[.5, 9.5],
        psd=dict(method='Welch', nperseg=2*FS, noverlap=FS, average='mean',
                 segment_aggregation='arithmetic mean over three 10 s parent records'),
        integration='Full raw 10 s FFT division by i*2*pi*f or (i*2*pi*f)^2; DC=0; displacement additionally excludes f<10 Hz before integration; then common resampling/FIR.',
        displacement_integration_min_hz=PROCESS['band_hz'][0],
        displacement_note='Dynamic displacement derived from accelerometers; not a direct spindle error-motion measurement.',
        velocity_1x_phasor=dict(frequency_hz=FR, unit='mm/s peak',
            time_origin='start of central 1 s window, separately for each segment',
            convention='Re(c*exp(i*2*pi*fr*t)); joint sinusoid/cosinusoid/DC least squares',
            phase_use='Simultaneous channel relative phase only; no mechanical keyphase.'),
        target_acceleration_rms=dict(halfwidth_hz=4, background_offsets_hz=[6, 12],
            estimator='Integrated 1X band PSD minus local median PSD times band width.',
            net_power_floor_fraction_of_gross=.001),
        no_july26_data=True, new_blind_test=False,
        note='Previously inspected July 25 is frozen validation, not a new blind test.',
    )
    centers = {kind: [] for kind in UNITS}
    powers = {kind: [] for kind in UNITS}
    records, audits, checks = [], [], []
    for index, run in enumerate(runs):
        cfg = dict(root=str(H5), train_runs=[str(Path(run['path']).relative_to(H5))],
                   validation_runs=[])
        raw, provenance = read_raw_references(
            cfg, DATA/'runs', [run], count=3, full_segments=True, sensor_ids=CHANNELS)
        audit = audit_steady_references(provenance, cfg)
        audits.extend(audit['runs'])
        records.extend(provenance)
        velocity, displacement = integrate_acceleration(raw)
        checks.append(dict(run_id=run['run_id'], **integration_check(raw, velocity, displacement)))
        for kind, values, multiplier in [('acceleration', raw, 1),
                                          ('velocity', velocity, 1000),
                                          ('displacement', displacement, 1e6)]:
            filtered = filter_vibration(values*multiplier, RAW_FS, PROCESS)
            usable = filtered[..., FS//2:-FS//2]
            frequency, psd = welch(usable, fs=FS, nperseg=2*FS, noverlap=FS, axis=-1)
            centers[kind].append(filtered[..., int(4.5*FS):int(5.5*FS)])
            powers[kind].append(psd.mean(axis=0))
        print(f'Prepared {index+1}/{len(runs)} {run["run_id"]}', flush=True)

    payload = dict(frequency_hz=frequency)
    for kind in centers:
        payload[kind] = np.asarray(centers[kind])
        payload[kind+'_psd'] = np.asarray(powers[kind])
    payload['velocity_1x_phasor'] = phasor(payload['velocity'])
    band = abs(frequency-FR) <= 4
    side = (abs(frequency-FR) >= 6) & (abs(frequency-FR) <= 12)
    psd = payload['acceleration_psd']
    df = frequency[1]-frequency[0]
    gross = psd[..., band].sum(-1)*df
    background = np.median(psd[..., side], axis=-1)*band.sum()*df
    net = np.maximum(gross-background, gross*.001)
    payload.update(target_acceleration_rms=np.sqrt(net), gross_1x_acceleration_power=gross,
                   estimated_background_1x_power=background)
    finite = all(np.isfinite(value).all() for value in payload.values())
    check_passed = max(max(c['velocity_derivative_relative_l2'],
                          c['displacement_second_derivative_relative_l2']) for c in checks) < 1e-8
    if not finite or not check_passed:
        raise ValueError('Reference integration or finite-value check failed')
    manifest['verification'] = dict(passed=True, finite=True, integration_checks=checks,
        shapes={key: list(value.shape) for key, value in payload.items()})
    np.savez_compressed(output/'references.npz', **payload)
    save_json(output/'manifest.json', manifest)
    save_json(output/'raw_provenance.json', records)
    save_json(output/'telemetry_audit.json', dict(
        passed=True, tolerance_rpm=audit['tolerance_rpm'],
        minimum_valid_fraction=audit['minimum_valid_fraction'],
        max_telemetry_gap_s=audit['max_telemetry_gap_s'], runs=audits))
    print(f'Saved {len(runs)} runs ({sum(r["role"] == "train" for r in runs)} train) to {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    prepare(parser.parse_args().output)
