"""Numerical and train-run checks for the shared physical velocity conversion."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import firwin, freqz, welch

from utils.velocity_data import (PROCESSING, SOURCE_FS, acceleration_to_velocity,
                                 read_acceleration, read_run_index, steady_interval, telemetry_state)


def relative_error(values, reference):
    return float(np.linalg.norm(values - reference) / np.linalg.norm(reference))


def synthetic_checks():
    fs = PROCESSING["fs_hz"]
    time = np.arange(30 * SOURCE_FS) / SOURCE_FS
    out_time = np.arange(10 * fs, 20 * fs) / fs
    frequencies = [10, 20, 116 + 2 / 3, 900, 1500, 2048]
    phases = [0.1, 0.7, -0.3]
    taps = firwin(PROCESSING["fir_taps"], PROCESSING["band_hz"], pass_zero=False,
                  fs=fs, window=("kaiser", PROCESSING["kaiser_beta"]))
    results = []
    for frequency in frequencies:
        raw = np.cos(2 * np.pi * frequency * time[None, :] + np.array(phases)[:, None]) / 9.80665
        converted = acceleration_to_velocity(raw)[:, 10 * fs:20 * fs]
        _, response = freqz(taps, worN=[frequency], fs=fs)
        gain = float(abs(response[0]))
        expected = gain * 1000 / (2 * np.pi * frequency) * np.sin(
            2 * np.pi * frequency * out_time[None, :] + np.array(phases)[:, None])
        basis = np.stack([np.sin(2 * np.pi * frequency * out_time),
                          np.cos(2 * np.pi * frequency * out_time), np.ones(len(out_time))], axis=1)
        coefficients = np.linalg.lstsq(basis, converted.T, rcond=None)[0]
        amplitudes = np.hypot(coefficients[0], coefficients[1])
        actual_phase = np.arctan2(coefficients[1], coefficients[0])
        phase_error = np.angle(np.exp(1j * (actual_phase - phases)))
        results.append(dict(frequency_hz=frequency, fir_gain=gain,
                            amplitude_relative_error=(amplitudes / (gain * 1000 / (2 * np.pi * frequency)) - 1).tolist(),
                            phase_error_deg=np.rad2deg(phase_error).tolist(),
                            waveform_relative_error=relative_error(converted, expected),
                            center_mean_mm_s=converted.mean(axis=-1).tolist()))
    alias_input = np.cos(2 * np.pi * 10000 * time)[None, :] / 9.80665
    alias = acceleration_to_velocity(alias_input)[:, 10 * fs:20 * fs]
    ref_input = np.cos(2 * np.pi * 1808 * time)[None, :] / 9.80665
    reference = acceleration_to_velocity(ref_input)[:, 10 * fs:20 * fs]
    rejection_db = float(20 * np.log10(np.linalg.norm(alias) / np.linalg.norm(reference)))
    return dict(tones=results, alias_10000_to_1808_hz_rejection_db=rejection_db,
                passed=all(max(abs(np.array(row["amplitude_relative_error"]))) < .01
                           and max(abs(np.array(row["phase_error_deg"]))) < .1
                           and row["waveform_relative_error"] < .02 for row in results)
                and rejection_db < -60)


def real_checks(run_dir):
    mapping, rows = read_run_index(run_dir)
    state = telemetry_state(run_dir)
    eligible = [index for index, row in enumerate(rows)
                if steady_interval(state, float(row["time_start_s"]), float(row["time_start_s"]) + 10)]
    index = eligible[len(eligible) // 2]
    raw = np.concatenate([read_acceleration(run_dir, rows[item], mapping)[0]
                          for item in range(index - 2, index + 4)], axis=-1)
    fs = PROCESSING["fs_hz"]
    reference, acceleration = acceleration_to_velocity(raw, return_acceleration=True)
    center_reference = reference[:, 20 * fs:30 * fs]
    guards = []
    for guard in (1, 5, 10, 20):
        begin, end = (20 - guard) * SOURCE_FS, (30 + guard) * SOURCE_FS
        converted = acceleration_to_velocity(raw[:, begin:end])[:, guard * fs:(guard + 10) * fs]
        guards.append(dict(guard_s=guard, relative_l2=relative_error(converted, center_reference),
                           channel_relative_l2=[relative_error(x, y) for x, y in zip(converted, center_reference)]))
    adjacent = np.concatenate([acceleration_to_velocity(raw[:, begin * SOURCE_FS:(begin + 30) * SOURCE_FS])
                               [:, 10 * fs:20 * fs] for begin in (10, 20)], axis=-1)
    boundary_reference = reference[:, 20 * fs:40 * fs]
    boundary_index = 10 * fs
    boundary_difference = adjacent[:, boundary_index] - adjacent[:, boundary_index - 1]
    expected_difference = boundary_reference[:, boundary_index] - boundary_reference[:, boundary_index - 1]
    frequency, power_v = welch(center_reference, fs=fs, nperseg=8192, axis=-1)
    _, power_a = welch(acceleration[:, 20 * fs:30 * fs], fs=fs, nperseg=8192, axis=-1)
    band = (frequency >= 20) & (frequency <= 2048)
    predicted_v = power_a[:, band] * (1000 / (2 * np.pi * frequency[band])) ** 2
    spectrum_relation = [relative_error(x, y) for x, y in zip(power_v[:, band], predicted_v)]
    high = (frequency >= 900) & (frequency <= 2048)
    full = (frequency >= 10) & (frequency <= 2048)
    fractions = power_v[:, high].sum(axis=-1) / power_v[:, full].sum(axis=-1)
    return dict(run=Path(run_dir).name, source_center_segment=int(rows[index]["segment_index"]),
                guard_comparison_to_60s_record=guards,
                adjacent_10s_relative_l2=relative_error(adjacent, boundary_reference),
                boundary_extra_jump_mm_s=(boundary_difference - expected_difference).tolist(),
                velocity_acceleration_psd_relative_l2=spectrum_relation,
                fraction_velocity_energy_900_2048_hz=fractions.tolist(),
                passed=next(row for row in guards if row["guard_s"] == 10)["relative_l2"] < .02
                and relative_error(adjacent, boundary_reference) < .02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=Path(
        "data/runs/20260722/run_20260722_212303_7000rpm_20min_0722_night1"))
    args = parser.parse_args()
    if any(item in args.run.name for item in ("221740", "204016", "225918", "210750", "215234")):
        raise ValueError("Conversion algorithm checks must not use protected test waveforms")
    result = dict(processing=PROCESSING, synthetic=synthetic_checks(), real=real_checks(args.run))
    result["passed"] = result["synthetic"]["passed"] and result["real"]["passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Velocity validation failed; inspect checks before batch conversion")


if __name__ == "__main__":
    main()
