"""Physical observations used by data generation and RMS calibration.

Hann energy integration preserves the frozen RMS convention. No PSD or plots
are exported. Runout uses unfiltered native displacement relative to platform.
"""

import numpy as np
from scipy.signal import get_window, periodogram, csd, welch

CHANNELS = ("ACC2_Hx", "ACC1_Hy", "ACC3_Py")
FS = 8192
BAND = (10.0, 900.0)
HALF_WIDTH_HZ = 2.0
OBSERVATION_VERSION = "coherent_hann_svd_rms_only_v6"


def physical_psd(velocity_mm_s, fs_hz=FS):
    values = np.asarray(velocity_mm_s, dtype=float)
    values = values - np.mean(values, axis=-1, keepdims=True)
    frequency, psd = periodogram(
        values, fs=fs_hz, window="hann", detrend=False, scaling="density", axis=-1
    )
    return frequency, psd, values


def feature_summary(velocity_mm_s, fs_hz=FS, rpm=7000, half_width_hz=HALF_WIDTH_HZ):
    """Physical features of a guarded, already processed 3 x N velocity record.

    All vector keys follow ACC2/Hx, ACC1/Hy, ACC3/Py:
    band_rms_mm_s, one_x_rms_mm_s, low_non1x_rms_mm_s, high_rms_mm_s,
    non1x_rms_mm_s, non1x_fraction_power, one_x_snr_power,
    time_rms_mm_s, hann_weighted_rms_mm_s, parseval_relative_error,
    boxcar_parseval_relative_error. Scalars include P_over_Hy_band_ratio,
    P_over_Hy_1x_ratio, duration_s, df_hz, f1_hz, one_x_half_width_hz.
    Parseval uses the exact window-weighted time energy for Hann PSD.
    """
    frequency, psd, centered = physical_psd(velocity_mm_s, fs_hz)
    duration = centered.shape[-1] / fs_hz
    f1 = abs(float(rpm)) / 60
    df = frequency[1] - frequency[0]
    one = np.abs(frequency - f1) <= half_width_hz
    band = (frequency >= BAND[0]) & (frequency <= BAND[1])
    low = (frequency >= BAND[0]) & (frequency < 200) & ~one
    high = (frequency >= 200) & (frequency <= BAND[1]) & ~one
    power = lambda mask: np.sum(psd[:, mask], axis=-1) * df
    total, p1 = power(band), power(band & one)
    residual = power(band & ~one)
    side = (np.abs(frequency - f1) >= 5) & (np.abs(frequency - f1) <= 15)
    local_floor = np.median(psd[:, side], axis=-1) * np.sum(one) * df
    snr = p1 / np.maximum(local_floor, np.finfo(float).tiny)
    window = get_window("hann", centered.shape[-1])
    weighted = np.sum((centered * window) ** 2, axis=-1) / np.sum(window**2)
    integrated = np.sum(psd, axis=-1) * df
    _, rectangular = periodogram(
        centered, fs=fs_hz, window="boxcar", detrend=False, scaling="density", axis=-1
    )
    time_power = np.mean(centered**2, axis=-1)
    values = dict(
        channel_order=list(CHANNELS),
        unit="mm/s",
        fs_hz=fs_hz,
        duration_s=duration,
        df_hz=df,
        f1_hz=f1,
        one_x_half_width_hz=half_width_hz,
        band_rms_mm_s=np.sqrt(total).tolist(),
        rms_mm_s=np.sqrt(total).tolist(),
        one_x_rms_mm_s=np.sqrt(p1).tolist(),
        low_non1x_rms_mm_s=np.sqrt(power(low)).tolist(),
        high_rms_mm_s=np.sqrt(power(high)).tolist(),
        non1x_rms_mm_s=np.sqrt(residual).tolist(),
        non1x_fraction_power=(residual / total).tolist(),
        one_x_snr_power=snr.tolist(),
        time_rms_mm_s=np.sqrt(time_power).tolist(),
        hann_weighted_rms_mm_s=np.sqrt(weighted).tolist(),
        parseval_relative_error=(np.abs(integrated - weighted) / weighted).tolist(),
        boxcar_parseval_relative_error=(
            np.abs(np.sum(rectangular, axis=-1) * df - time_power) / time_power
        ).tolist(),
        P_over_Hy_band_ratio=float(np.sqrt(total[2] / total[1])),
        P_over_Hy_1x_ratio=float(np.sqrt(p1[2] / p1[1])),
    )
    # Welch averaging gives a diagnostic coherence, unlike a single FFT pair.
    if duration >= 4:
        nperseg = 2 * int(fs_hz)
        f, cross = csd(centered[1], centered[2], fs=fs_hz, nperseg=nperseg)
        _, hy = welch(centered[1], fs=fs_hz, nperseg=nperseg)
        _, py = welch(centered[2], fs=fs_hz, nperseg=nperseg)
        index = int(np.argmin(np.abs(f - f1)))
        values.update(
            P_over_Hy_1x_phase_deg=float(np.angle(cross[index], deg=True)),
            P_Hy_1x_coherence=float(abs(cross[index]) ** 2 / (hy[index] * py[index])),
        )
    return values


def design_matrix(t_s, f1_hz, origin_s, orders=7):
    phase = 2 * np.pi * float(f1_hz) * (np.asarray(t_s) - float(origin_s))
    basis = np.ones((len(phase), 1 + 2 * orders))
    for k in range(1, orders + 1):
        basis[:, 2 * k - 1] = np.cos(k * phase)
        basis[:, 2 * k] = np.sin(k * phase)
    return basis


def fit_joint_rectangular(t_s, velocity_mm_s, f1_hz, origin_s=None, orders=7):
    """Legacy rectangular fit, retained only for before/after evidence."""
    t_s = np.asarray(t_s, dtype=float)
    velocity = np.atleast_2d(np.asarray(velocity_mm_s, dtype=float))
    origin_s = float(t_s[0]) if origin_s is None else float(origin_s)
    basis = design_matrix(t_s, f1_hz, origin_s, orders)
    coeff = np.linalg.solve(basis.T @ basis, basis.T @ velocity.T)
    z = (coeff[1::2] - 1j * coeff[2::2]).T
    residual = velocity - (basis @ coeff).T
    return dict(
        coefficients=coeff,
        z_complex=z,
        amplitude_rms=np.abs(z) / np.sqrt(2),
        phase_rad=np.angle(z),
        offset=coeff[0],
        residual=residual,
        f1_hz=float(f1_hz),
        origin_s=origin_s,
    )


def fit_joint_hann(t_s, velocity_mm_s, f1_hz, origin_s=None, orders=7):
    """Hann-weighted joint least squares, including DC and harmonic terms."""
    t_s = np.asarray(t_s, dtype=float)
    velocity = np.atleast_2d(np.asarray(velocity_mm_s, dtype=float))
    origin_s = float(t_s[0]) if origin_s is None else float(origin_s)
    basis = design_matrix(t_s, f1_hz, origin_s, orders)
    weight = get_window("hann", len(t_s))

    root = np.sqrt(weight)[:, None]
    coeff, _, _, _ = np.linalg.lstsq(root * basis, root * velocity.T, rcond=None)
    z = (coeff[1::2] - 1j * coeff[2::2]).T
    residual = velocity - (basis @ coeff).T
    amplitude_rms = abs(z) / np.sqrt(2)
    return dict(
        coefficients=coeff,
        z_complex=z,
        amplitude_rms=amplitude_rms,
        phase_rad=np.angle(z),
        offset=coeff[0],
        residual=residual,
        f1_hz=float(f1_hz),
        origin_s=origin_s,
    )


def fit_joint(t_s, velocity_mm_s, f1_hz, origin_s=None, orders=7):
    """Default coherent estimator: Hann-weighted joint DC and harmonics."""
    return fit_joint_hann(t_s, velocity_mm_s, f1_hz, origin_s, orders)


def complete_turn_runout(
    time_s, position_um, rpm=None, start_s=None, end_s=None, phase_at_zero_rad=0.0, angle_rad=None
):
    """Use both interpolated phase boundaries and every native interior sample.

    ``position_um`` is [sample, coordinate], without bandpass or integration.
    Turns count in the direction of rotation, even when rpm is negative.
    An optional strictly monotone unwrapped ``angle_rad`` supports known
    variable-speed trajectories. Partial revolutions at either end are omitted.
    Labels are sampled-state
    extrema, not an analytic search of the continuous ODE dense solution.
    """
    time_s = np.asarray(time_s, dtype=float)
    position_um = np.asarray(position_um, dtype=float)
    if (
        time_s.ndim != 1
        or len(time_s) < 2
        or position_um.shape[0] != len(time_s)
        or position_um.ndim != 2
        or np.any(np.diff(time_s) <= 0)
    ):
        raise ValueError("Need increasing sample times and [sample,coordinate] positions")
    start_s = float(time_s[0] if start_s is None else start_s)
    end_s = float(time_s[-1] if end_s is None else end_s)
    if start_s < time_s[0] - 1e-12 or end_s > time_s[-1] + 1e-12 or end_s <= start_s:
        raise ValueError("Requested interval must lie inside the recorded states")
    if angle_rad is None:
        if rpm is None or rpm == 0:
            raise ValueError("Supply nonzero constant rpm or known monotone unwrapped angles")
        rate = abs(float(rpm)) / 60
        phase_cycles = np.sign(rpm) * phase_at_zero_rad / (2 * np.pi)
        first = int(np.ceil(start_s * rate + phase_cycles - 1e-10))
        last = int(np.floor(end_s * rate + phase_cycles + 1e-10))
        edges = (np.arange(first, last + 1) - phase_cycles) / rate
    else:
        angle_rad = np.asarray(angle_rad, dtype=float)
        if angle_rad.shape != time_s.shape:
            raise ValueError("Angle history must match position sample times")
        cycles = np.sign(angle_rad[-1] - angle_rad[0]) * angle_rad / (2 * np.pi)
        if np.any(np.diff(cycles) <= 0):
            raise ValueError("Unwrapped angle must be strictly monotone")
        first = int(np.ceil(np.interp(start_s, time_s, cycles) - 1e-10))
        last = int(np.floor(np.interp(end_s, time_s, cycles) + 1e-10))
        edges = np.interp(np.arange(first, last + 1), cycles, time_s)
    turn_id = np.arange(first, last, dtype=np.int64)
    boundary_positions = np.column_stack(
        [np.interp(edges, time_s, position_um[:, axis]) for axis in range(position_um.shape[1])]
    )
    minimum, maximum = [], []
    for index, (begin, finish) in enumerate(zip(edges[:-1], edges[1:])):
        left = np.searchsorted(time_s, begin, side="right")
        right = np.searchsorted(time_s, finish, side="left")
        boundary = boundary_positions[index : index + 2]
        values = np.concatenate((boundary[:1], position_um[left:right], boundary[1:]))
        minimum.append(values.min(axis=0))
        maximum.append(values.max(axis=0))
    shape = (len(turn_id), position_um.shape[1])
    minimum = np.asarray(minimum).reshape(shape)
    maximum = np.asarray(maximum).reshape(shape)
    return dict(
        turn_id=turn_id,
        start_time_s=edges[:-1],
        end_time_s=edges[1:],
        min_um=minimum,
        max_um=maximum,
        peak_to_peak_um=maximum - minimum,
    )


def signal_features(t, velocity, f1, nominal_duration_s=None):
    """Input: absolute seconds and already guarded/filtered [channel,time]."""
    t, velocity = np.asarray(t), np.asarray(velocity)
    basic = feature_summary(velocity, fs_hz=8192, rpm=f1 * 60)
    legacy_joint = fit_joint_rectangular(t, velocity, f1, origin_s=0.0)
    legacy_single = fit_joint_rectangular(t, velocity, f1, origin_s=0.0, orders=1)
    joint = fit_joint(t, velocity, f1, origin_s=0.0)
    single = fit_joint(t, velocity, f1, origin_s=0.0, orders=1)
    centered = velocity - velocity.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=1))
    peak = np.max(abs(centered), axis=1)
    valid = rms > 0
    crest = np.divide(peak, rms, out=np.full(3, np.nan), where=valid)
    kurt = np.divide(np.mean(centered**4, axis=1), rms**4, out=np.full(3, np.nan), where=valid)
    spectral_count = len(t)
    spectral_trim = 0
    spectral_centered = centered
    if nominal_duration_s is not None:
        spectral_count = int(np.floor(float(nominal_duration_s) * 8192 + 1e-9))
        spectral_trim = len(t) - spectral_count
        if spectral_trim not in (0, 1):
            raise ValueError(
                f"Nominal spectral window requires trimming {spectral_trim} samples; "
                "expected an exact or one-sample endpoint correction"
            )
        spectral_centered = velocity[:, :spectral_count]
        spectral_centered = spectral_centered - spectral_centered.mean(axis=1, keepdims=True)
    frequency, energy = periodogram(
        spectral_centered, fs=8192, window="hann", detrend=False, scaling="density", axis=-1
    )
    df = frequency[1] - frequency[0]
    orders = np.arange(1, 8)
    bands = np.stack([abs(frequency - order * f1) <= 2 for order in orders])
    fractional = np.array([0.5, 1.5, 2.5])
    fraction_bands = np.stack([abs(frequency - order * f1) <= 2 for order in fractional])
    order_rms = np.sqrt(energy @ bands.T * df)
    residual_band = (frequency >= 10) & (frequency <= 900) & ~bands.any(axis=0)
    return dict(
        band_rms_mm_s=np.asarray(basic["band_rms_mm_s"]),
        one_x_rms_mm_s=np.asarray(basic["one_x_rms_mm_s"]),
        one_x_snr_power=np.asarray(basic["one_x_snr_power"]),
        time_rms_mm_s=rms,
        peak_to_peak_mm_s=np.ptp(centered, axis=1),
        absolute_peak_mm_s=peak,
        crest_factor=crest,
        kurtosis_ordinary=kurt,
        ratio_feature_defined=valid,
        coherent_order_rms_mm_s=joint["amplitude_rms"],
        order_peak_mm_s=np.abs(joint["z_complex"]),
        order_phase_rad=joint["phase_rad"],
        legacy_rectangular=dict(
            coherent_order_rms_mm_s=legacy_joint["amplitude_rms"],
            order_peak_mm_s=np.abs(legacy_joint["z_complex"]),
            order_phase_rad=legacy_joint["phase_rad"],
            residual_rms_mm_s=np.sqrt(np.mean(legacy_joint["residual"] ** 2, axis=1)),
            residual_after_1x_rms_mm_s=np.sqrt(np.mean(legacy_single["residual"] ** 2, axis=1)),
            coherent_definition="rectangular joint DC+1..7 LS; absolute t=0",
        ),
        phase_origin_absolute_s=0.0,
        narrowband_order_rms_mm_s=order_rms,
        fractional_orders=fractional,
        fractional_band_rms_mm_s=np.sqrt(energy @ fraction_bands.T * df),
        noninteger_band_rms_mm_s=np.sqrt(energy[:, residual_band].sum(axis=1) * df),
        residual_after_1x_rms_mm_s=np.sqrt(np.mean(single["residual"] ** 2, axis=1)),
        residual_rms_mm_s=np.sqrt(np.mean(joint["residual"] ** 2, axis=1)),
        duration_s=len(t) / 8192,
        df_hz=df,
        f1_hz=f1,
        observation_version=OBSERVATION_VERSION,
        spectral_window_nominal_duration_s=nominal_duration_s,
        spectral_window_sample_count=int(spectral_count),
        spectral_window_trimmed_tail_samples=int(spectral_trim),
        spectral_window_policy=(
            "periodogram uses floor(nominal_duration_s*8192+1e-9) "
            "original samples; at most one terminal sample is omitted; all other "
            "features use the full input record"
            if nominal_duration_s is not None
            else "legacy full-record periodogram; no nominal duration supplied"
        ),
        rms_definition="inherited V10 Hann energy sum, 10-900 Hz; orders +/-2 Hz",
        coherent_definition=(
            "Hann-weighted joint DC+1..7 LS; sum(hann*residual^2); "
            "peak/sqrt(2), phase relative to absolute t=0"
        ),
        residual_note="LS projection residual; nonorthogonal components not assumed energy-additive",
    )


def runout_features(t, relative, rpm, phase, start, end):
    turns = complete_turn_runout(
        t, np.asarray(relative) * 1e6, rpm, start, end, phase_at_zero_rad=phase
    )
    pp = turns["peak_to_peak_um"]
    mean = pp.mean(axis=0)
    use = (t >= start) & (t < end)
    result = dict(
        E_dir_um=float(max(mean)),
        Ex_um=float(mean[0]),
        Ey_um=float(mean[1]),
        complete_turns=len(pp),
        pp_std_um=pp.std(axis=0),
        pp_q05_q50_q95_um=np.quantile(pp, [0.05, 0.5, 0.95], axis=0),
        static_mean_relative_um=np.asarray(relative)[use].mean(axis=0) * 1e6,
    )
    return result, turns


def contact_features(t, fit, rolling, count, compression, start, end, applicable):
    use = (t >= start) & (t < end)
    selected_t = np.asarray(t)[use]
    result = {}
    for name, force in (("fit", fit), ("rolling", rolling)):
        selected = np.asarray(force)[use]
        norm = np.linalg.norm(selected, axis=-1)
        result[name + "_dynamic_force_rms_N"] = np.sqrt(
            np.mean((selected - selected.mean(axis=0)) ** 2, axis=0)
        )
        result[name + "_peak_force_N"] = np.max(norm, axis=0)
        result[name + "_mean_force_N"] = selected.mean(axis=0)
        result[name + "_impulse_N_s"] = np.trapezoid(norm, selected_t, axis=0)
    result.update(
        fit_detached_fraction=np.mean(np.asarray(compression)[use] <= 0, axis=0),
        fit_force_free_fraction=np.mean(np.asarray(count)[use] == 0, axis=0),
        fit_active_count_mean=np.mean(np.asarray(count)[use], axis=0),
        fit_max_compression_m=np.max(np.asarray(compression)[use], axis=0),
        fit_applicable=np.asarray(applicable, dtype=bool),
    )
    for key in result:
        if key.startswith("fit_") and key != "fit_applicable":
            result[key] = np.where(
                np.asarray(applicable).reshape((4,) + (1,) * (result[key].ndim - 1)),
                result[key],
                np.nan,
            )
    return result


LIMITS = {
    "band_rms_mm_s": (1e-4, 0.02),
}


def compare_features(a, b, *, relative_limits=None):
    checks, missing = {}, []
    for name, (floor, relative) in LIMITS.items():
        if relative_limits is not None:
            relative = relative_limits.get(name, relative)
        if name not in a or name not in b:
            missing.append(name)
            continue
        av, bv = np.asarray(a[name], dtype=float), np.asarray(b[name], dtype=float)
        applicable = np.isfinite(av) & np.isfinite(bv)
        expected = np.ones(av.shape, dtype=bool)
        limit = np.maximum(floor, relative * np.maximum(abs(av), abs(bv)))
        difference = abs(av - bv)
        checks[name] = dict(
            passed=bool(
                np.all(applicable[expected]) and np.all(difference[expected] <= limit[expected])
            ),
            original=av,
            compared=bv,
            absolute_difference=difference,
            limit=limit,
            applicable=expected,
        )
    return dict(
        passed=not missing and all(x["passed"] for x in checks.values()),
        missing=missing,
        checks=checks,
        relative_limit_overrides=relative_limits or {},
        criterion="abs(a-b)<=max(floor,rel*max(abs(a),abs(b)))",
        threshold_sources="band RMS only; explicit override or default 2%",
    )
