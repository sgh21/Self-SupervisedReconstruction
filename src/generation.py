"""Run one physical case and export native states, sensors and turn labels."""

from copy import deepcopy
import itertools
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.signal import resample_poly
from spindle.installation_dynamics import InstallationModel, IntegrationFailure
from utils import metrics as physical_metrics
from utils.io import load_json, write_json, write_csv, serial, crc_file, config_crc
from utils.metrics import (
    CHANNELS,
    signal_features,
    runout_features,
    contact_features,
    compare_features,
)
from utils.processing import process_velocity


def source_versions():
    source = Path(__file__).parent
    utilities = Path(physical_metrics.__file__).parent
    files = [
        *source.glob("*.py"),
        *source.joinpath("spindle").glob("*.py"),
        *utilities.glob("*.py"),
    ]
    return {p.parent.name + "/" + p.name: crc_file(p) for p in sorted(files)}


def continuation_physical_inputs(config):
    """Only the front gaps may change along a phase-continuous branch."""
    p = deepcopy(config["base_v8"]["parameters"])
    metadata_keys = {
        "machine_model",
        "model_status",
        "calibration_status",
        "parameter_status",
        "fit_parameter_status",
    }
    for key in tuple(p):
        if (
            key.startswith("_")
            or key in metadata_keys
            or key in ("front_group_clearance_um", "fit_clearance_um", "fit2_clearance_um", "N")
        ):
            p.pop(key)
    for ring in p.get("rear_outer_rings", {}).values():
        ring.pop("fit_direction_count", None)
        ring.pop("parameter_status", None)
    model = InstallationModel(config)
    return serial(
        dict(
            mechanical_parameters=p,
            probe_reference=config["probe_reference"],
            excitation={key: config["excitation"][key] for key in ("U_kg_m", "phase_rad")},
            installation_mC_kg=config["installation"]["mC_kg"],
            installation_mP_kg=config["installation"]["mP_kg"],
            actual_link_stiffness_N_m=model.link_stiffness,
            actual_link_damping_N_s_m=model.link_damping,
        )
    )


def window_features(t, velocity, native, config, start, end):
    p = config["base_v8"]["parameters"]
    use = (t >= start) & (t < end)
    result = signal_features(
        t[use], velocity[:, use], abs(p["rpm"]) / 60.0, nominal_duration_s=end - start
    )
    result["observation_source_crc32"] = {
        "generation.py": crc_file(Path(__file__)),
        "metrics.py": crc_file(Path(physical_metrics.__file__)),
    }
    runout, turns = runout_features(
        native["t_s"],
        native["measured_relative_xy_m"],
        p["rpm"],
        config["excitation"]["phase_rad"],
        start,
        end,
    )
    result.update(runout)
    result.update(
        contact_features(
            native["t_s"],
            native["all_fit_force_N"],
            native["rolling_contact_force_N"],
            native["all_fit_active_count"],
            native["all_fit_max_compression_m"],
            start,
            end,
            native["fit_applicable"],
        )
    )
    result.update(start_s=float(start), end_s=float(end))
    return result, turns


def stationarity(t, velocity, native, config, start, end):
    edges = np.linspace(start, end, 5)
    configured = config.get("acceptance", {}).get("stationarity_relative_limits", {})
    relative_limits = ({"band_rms_mm_s": configured["band_rms_mm_s"]}
                       if "band_rms_mm_s" in configured else {})
    blocks = []
    for left, right in zip(edges[:-1], edges[1:]):
        use = (t >= left) & (t < right)
        frequency, psd, _ = physical_metrics.physical_psd(velocity[:, use])
        band = (frequency >= physical_metrics.BAND[0]) & (frequency <= physical_metrics.BAND[1])
        rms = np.sqrt(np.sum(psd[:, band], axis=-1) * (frequency[1] - frequency[0]))
        blocks.append(dict(start_s=float(left), end_s=float(right), band_rms_mm_s=rms))
    comparisons = {
        f"{i + 1}_vs_{j + 1}": compare_features(blocks[i], blocks[j], relative_limits=relative_limits)
        for i, j in itertools.combinations(range(4), 2)
    }
    passed = all(comparison["passed"] for comparison in comparisons.values())
    return dict(
        passed=bool(passed),
        status="STEADY" if passed else "NOT_STEADY",
        classification="PASS" if passed else "FAIL_RMS_STATIONARITY",
        acceptance_scope="source_10_900_hz_band_rms_only",
        stationarity_relative_limit_overrides=relative_limits or {},
        window_edges_s=edges,
        subwindows=blocks,
        pairwise_comparisons=comparisons,
        criterion="All six comparisons of four complete 120-turn blocks; 240-turn blocks for 960-turn runs",
    )


def native_payload(result, model, config, start, end, *, validate=True):
    t, q, v = result["t"], result["q"], result["v"]
    p = model.base_parameters
    names = np.asarray(model.state_names)
    units_q = np.asarray(["rad" if name.startswith("theta") else "m" for name in names])
    units_v = np.asarray(["rad/s" if name.startswith("theta") else "m/s" for name in names])
    angle = p["rpm"] * 2 * np.pi / 60 * t + config["excitation"]["phase_rad"]
    native = dict(
        t_s=t,
        q=q,
        v=v,
        acc=result["acc"],
        q_equilibrium=result["q_equilibrium"],
        state_final=result["state_final"],
        final_absolute_time=np.asarray(result["integration_end_s"]),
        rpm_signed=np.asarray(p["rpm"]),
        rotor_phase=angle,
        per_revolution_index=np.floor(np.sign(p["rpm"]) * angle / (2 * np.pi)).astype(np.int64),
        state_names=names,
        units_by_state=units_q,
        velocity_units_by_state=units_v,
        base_mechanical_dof=np.asarray(model.base_mechanical_dof),
        structural_dof=np.asarray(model.structural_dof),
        sensor_velocity_mm_s=result["sensor_velocity_m_s"] * 1000,
        channel_names=np.asarray(CHANNELS),
        sensor_indices=np.asarray(model.sensor_indices),
        measured_section_xy_m=result["measured_section_xy_m"],
        platform_xy_m=q[:, model.P],
        measured_relative_xy_m=result["measured_relative_xy_m"],
        measurement_start_s=np.asarray(start),
        measurement_end_s=np.asarray(end),
        formal_mask=(t >= start) & (t < end),
    )
    for key in (
        "all_fit_force_N",
        "all_fit_active_count",
        "all_fit_max_compression_m",
        "all_fit_detached_flag",
        "all_fit_force_free_flag",
        "rolling_contact_force_N",
        "outer_relative_H_xy_m",
        "fit_applicable",
        "rear_bound_rolling_force_N",
        "integration_accepted_steps_s",
        "integration_accepted_times_s",
        "solver_success",
    ):
        if key in result:
            native[key] = result[key]
    if validate:
        validate_native(native)
    return native


def validate_native(native):
    """RMS cannot be calculated from nonfinite time or sensor velocity."""
    required = ("t_s", "sensor_velocity_mm_s")
    if any(not np.isfinite(native[key]).all() for key in required):
        raise ValueError("Nonfinite time or sensor velocity prevents RMS calculation")


def export_case(folder, native=None, config=None):
    folder = Path(folder)
    config = config or load_json(folder / "actual_config.json")
    if native is None:
        with np.load(folder / "native.npz", allow_pickle=False) as source:
            native = {key: source[key] for key in source.files}
    validate_native(native)
    tr = native["t_s"]
    fs = int(config["simulation"]["fs_raw_hz"])
    velocity = process_velocity(native["sensor_velocity_mm_s"].T, fs)
    t = tr[0] + np.arange(velocity.shape[-1]) / 8192.0
    start, end = float(native["measurement_start_s"]), float(native["measurement_end_s"])
    formal = (t >= start) & (t < end)
    features, turns = window_features(t, velocity, native, config, start, end)
    steady = stationarity(t, velocity, native, config, start, end)
    # Keep analysis-level trajectories for every node, separately named from native states.
    divisor = np.gcd(fs, 8192)
    aq = resample_poly(native["q"], 8192 // divisor, fs // divisor, axis=0, padtype="line")
    av = resample_poly(native["v"], 8192 // divisor, fs // divisor, axis=0, padtype="line")
    identity = config["comparison_identity"]
    solver_success = bool(native["solver_success"]) if "solver_success" in native else None
    classification = (
        "FAIL_SOLVER" if solver_success is False
        else "INCONCLUSIVE_SOLVER" if solver_success is None
        else "FAIL_RMS_STATIONARITY" if not steady["passed"]
        else "PASS"
    )
    metadata = dict(
        case_id=config["case_id"],
        group_id=config["group_id"],
        model_mode=config["model_mode"],
        parameter_id=config["parameter_id"],
        **identity,
        base_mechanical_dof=int(native["base_mechanical_dof"]),
        structural_dof=int(native["structural_dof"]),
        state_names=native["state_names"].tolist(),
        sensor_indices=native["sensor_indices"].tolist(),
        channel_order=list(CHANNELS),
        units="mm/s",
        fs_hz=8192,
        raw_fs_hz=fs,
        nominal_rpm=config["base_v8"]["parameters"]["rpm"],
        formal_start_s=start,
        formal_end_s=end,
        formal_sample_count=int(formal.sum()),
        formal_turns=features["complete_turns"],
        guard_before_s=start - t[0],
        guard_after_s=t[-1] - end,
        steady_pass=bool(steady["passed"]),
        rms_stationarity_pass=bool(steady["passed"]),
        solver_success=solver_success,
        classification=classification,
        acceptance_scope="source_10_900_hz_band_rms_only",
        displacement_reference="M-P",
        output_gain=1.0,
        beta_applied=False,
        E_dir_um=features["E_dir_um"],
        Ex_um=features["Ex_um"],
        Ey_um=features["Ey_um"],
        native_path="native.npz",
        native_crc32=crc_file(folder / "native.npz"),
        configuration_path="actual_config.json",
        configuration_crc32=config_crc(serial(config)),
        observation_version=features["observation_version"],
        observation_source_crc32=features["observation_source_crc32"],
        numerical_check_status="NOT_REQUESTED",
        processing=config["processing_contract"],
        initialization=config["initialization"],
    )
    np.savez_compressed(
        folder / "signal.npz",
        t_s=t,
        velocity_mm_s=velocity,
        formal_mask=formal,
        channel_names=np.asarray(CHANNELS),
        fs_hz=np.asarray(8192.0),
        analysis_q=aq,
        analysis_v=av,
        state_names=native["state_names"],
        sensor_indices=native["sensor_indices"],
        units_by_state=native["units_by_state"],
    )
    np.savez_compressed(folder / "per_revolution.npz", **turns)
    write_csv(
        folder / "per_revolution.csv",
        [
            dict(
                turn_id=int(k),
                start_s=float(left),
                end_s=float(right),
                Ex_um=float(pp[0]),
                Ey_um=float(pp[1]),
            )
            for k, left, right, pp in zip(
                turns["turn_id"],
                turns["start_time_s"],
                turns["end_time_s"],
                turns["peak_to_peak_um"],
            )
        ],
    )
    write_json(folder / "signal.json", metadata)
    write_json(folder / "features.json", features)
    write_json(folder / "stationarity.json", steady)
    return metadata


def execute(config, folder, initial_source=None, *, max_attempts=None, cold_start_state=None):
    """Never replace a case produced from different physical/numerical inputs."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if max_attempts is None:
        max_attempts = 1
    requested = deepcopy(config)
    versions = source_versions()
    initial = Path(initial_source).resolve() if initial_source else None
    if initial is not None and cold_start_state is not None:
        raise ValueError("A common cold state cannot also be a continuation")
    requested["execution_inputs"] = dict(
        source_crc32=versions,
        initial_native_crc32=crc_file(initial / "native.npz") if initial else None,
        initial_source=initial.as_posix() if initial else None,
        max_attempts=max_attempts,
        cold_start_state=serial(cold_start_state),
    )
    request_path = folder / "requested_config.json"
    if request_path.exists():
        if load_json(request_path) != requested:
            raise ValueError(
                f"Existing case has different effective request: {folder}; use a new variant/output"
            )
        if (folder / "signal.json").exists():
            existing = load_json(folder / "signal.json")
            if existing["native_crc32"] != crc_file(folder / "native.npz"):
                raise ValueError(f"Native cache contents changed: {folder}")
            return existing
    write_json(request_path, requested)
    actual = deepcopy(config)
    actual["source_fingerprint"] = versions
    write_json(folder / "actual_config.json", actual)
    model = InstallationModel(actual)
    q0, residual = model.equilibrium()
    force_residual = model.internal_force(0.0, q0, np.zeros(model.structural_dof)) - model.gravity
    equilibrium = dict(
        residual_norm_equivalent_N=residual,
        maximum_force_residual_N=float(np.max(abs(np.delete(force_residual, [8, 9])))),
        maximum_moment_residual_Nm=float(np.max(abs(force_residual[8:10]))),
        q_equilibrium=q0,
        frozen_phase_t_s=0.0,
    )
    write_json(folder / "equilibrium.json", equilibrium)
    actual["actual_link_stiffness_N_m"] = model.link_stiffness.tolist()
    actual["actual_link_damping_N_s_m"] = model.link_damping.tolist()
    settings = actual["simulation"]
    t0, state0 = 0.0, None
    if cold_start_state is not None:
        state0 = np.asarray(cold_start_state, dtype=float).copy()
        if state0.shape != (2 * model.structural_dof,) or not np.isfinite(state0).all():
            raise ValueError("Common cold state must be a finite complete q/v state")
    if initial:
        previous_config = load_json(initial / "actual_config.json")
        if continuation_physical_inputs(previous_config) != continuation_physical_inputs(actual):
            raise ValueError(
                "Continuation must preserve all rear gaps, masses, actual K/C, bearings and excitation"
            )
        with np.load(initial / "native.npz") as data:
            state0 = data["state_final"].copy()
            t0 = float(data["final_absolute_time"])
        if len(state0) != 2 * model.structural_dof:
            raise ValueError("Continuation source has another state dimension")
    actual["initialization"] = dict(
        policy=("neighbor_state_continuation" if initial is not None
                else "cold_common_state" if cold_start_state is not None else "cold_equilibrium"),
        source=None if initial is None else initial.as_posix(),
        absolute_start_s=t0,
        phase_policy="Carry complete q/v and absolute time; no phase reset",
        state0=np.r_[q0, np.zeros(model.structural_dof)] if state0 is None else state0,
        physical_inputs_crc32=config_crc(continuation_physical_inputs(actual)),
    )
    history, began = [], time.perf_counter()
    warmup = settings["warmup_s"]
    accumulated_warmup = warmup
    while True:
        rate = abs(model.base_parameters["rpm"]) / 60.0
        phase = (
            np.sign(model.base_parameters["rpm"]) * actual["excitation"]["phase_rad"] / (2 * np.pi)
        )
        start = (np.ceil((t0 + warmup) * rate + phase) - phase) / rate
        end = start + settings["retained_turns"] / rate
        fs, guard = int(settings["fs_raw_hz"]), float(settings["guard_s"])
        keep = np.floor((start - guard) * fs) / fs
        finish = np.ceil((end + guard) * fs) / fs
        write_json(
            folder / "progress.json",
            dict(
                status="RUNNING",
                attempt=len(history) + 1,
                case_id=actual["case_id"],
                formal_start_s=start,
                formal_end_s=end,
                wall_elapsed_s=time.perf_counter() - began,
            ),
        )
        # Effective inputs must survive a failure before simulate() returns.
        actual["source_fingerprint"] = versions
        actual["actual_window"] = dict(start_s=start, end_s=end, source_sampling_guard_s=guard)
        write_json(folder / "actual_config.json", actual)
        try:
            result = model.simulate(
                duration=finish - keep,
                warmup=keep - t0,
                fs=fs,
                q_initial=q0,
                state_initial=state0,
                t_start=t0,
                max_step=settings["max_step_s"],
                rtol=settings["rtol"],
                method=settings["method"],
            )
        except IntegrationFailure as error:
            partial = dict(error.partial_native)
            partial["data_status"] = np.asarray("PARTIAL_NOT_FORMAL")
            partial["partial_is_formal_complete"] = np.asarray(False)
            np.savez_compressed(folder / "partial_native.npz", **partial)
            write_json(folder / "solver_failure.json", error.diagnostics)
            write_json(folder / "failure.json", dict(
                status="FAIL_SOLVER", data_status="PARTIAL_NOT_FORMAL",
                partial_native="partial_native.npz", diagnostics=error.diagnostics,
                source_crc32=versions, actual_config="actual_config.json",
                exception=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())))
            actual["actual_solver_settings"] = error.diagnostics
            actual["stationarity_status"] = "NOT_EVALUATED_SOLVER_FAILURE"
            write_json(folder / "actual_config.json", actual)
            write_json(folder / "progress.json", dict(status="FAIL_SOLVER", case_id=actual["case_id"],
                       solver_success=False, partial_native="partial_native.npz"))
            # Preserve partial states before reconstructing any contact observations.
            try:
                partial.update(model.observations(partial["q"], partial["v"]))
                partial["sensor_velocity_mm_s"] = partial.pop("sensor_velocity_m_s") * 1000
                partial["acc"] = np.asarray([
                    model.rhs(t, np.r_[q, v])[model.structural_dof:]
                    for t, q, v in zip(partial["t_s"], partial["q"], partial["v"])
                ])
                partial.update(model.contact_observations(partial["t_s"], partial["q"], partial["v"]))
                np.savez_compressed(folder / "partial_native.npz", **partial)
            except Exception as diagnostic_error:
                write_json(folder / "partial_contact_failure.json", dict(type=type(diagnostic_error).__name__,
                           message=str(diagnostic_error), contact_status="UNKNOWN"))
            raise
        native = native_payload(result, model, actual, start, end, validate=False)
        actual["actual_window"] = dict(start_s=start, end_s=end, source_sampling_guard_s=guard)
        actual["actual_solver_settings"] = result["actual_solver_settings"]
        actual["state_schema"] = dict(
            state_names=list(model.state_names),
            base_mechanical_dof=model.base_mechanical_dof,
            structural_dof=model.structural_dof,
            sensor_state_indices=list(model.sensor_indices),
            angular_indices=[8, 9],
        )
        actual["source_fingerprint"] = versions
        actual["stationarity_status"] = "PENDING_OBSERVATION"
        # Persist the unmodified solver arrays before any observation can fail.
        np.savez_compressed(folder / "native.npz", **native)
        write_json(folder / "actual_config.json", actual)
        validate_native(native)
        velocity = process_velocity(native["sensor_velocity_mm_s"].T, fs)
        t = native["t_s"][0] + np.arange(velocity.shape[-1]) / 8192.0
        steady = stationarity(t, velocity, native, actual, start, end)
        history.append(
            dict(
                attempt=len(history) + 1,
                transition_warmup_s=warmup,
                accumulated_transition_warmup_s=accumulated_warmup,
                formal_start_s=start,
                formal_end_s=end,
                steady_pass=steady["passed"],
                wall_elapsed_s=time.perf_counter() - began,
                solver_statistics=result["actual_solver_settings"],
            )
        )
        write_json(folder / "warmup_history.json", history)
        write_json(folder / "stationarity.json", steady)
        print(
            json.dumps(
                dict(
                    case_id=actual["case_id"],
                    attempt=len(history),
                    steady=steady["passed"],
                    wall_elapsed_s=round(time.perf_counter() - began, 2),
                )
            ),
            flush=True,
        )
        if (
            steady["passed"]
            or accumulated_warmup >= settings["maximum_warmup_s"]
            or (max_attempts is not None and len(history) >= max_attempts)
        ):
            break
        state0 = result["state_final"].copy()
        t0 = float(result["integration_end_s"])
        warmup = min(2.0, settings["maximum_warmup_s"] - accumulated_warmup)
        if warmup <= guard:
            break
        accumulated_warmup += warmup
        del result, native, velocity
    actual["solver_wall_elapsed_s"] = time.perf_counter() - began
    actual["stationarity_status"] = steady["status"]
    if versions != source_versions():
        raise RuntimeError(
            "Source code changed while this case was running; result is not accepted"
        )
    write_json(folder / "actual_config.json", actual)
    metadata = export_case(folder, native, actual)
    write_json(
        folder / "progress.json",
        dict(
            status="COMPLETE" if metadata["classification"] == "PASS" else metadata["classification"],
            case_id=actual["case_id"],
            classification=metadata["classification"],
            wall_elapsed_s=time.perf_counter() - began,
            E_dir_um=metadata["E_dir_um"],
            numerical_check_status=metadata["numerical_check_status"],
        ),
    )
    return metadata
