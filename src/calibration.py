"""RMS-only installation calibration. Runout and 1X never enter the loss or selection."""

from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from utils.configuration import PARAMETER_NAMES, decode_unit, encode_unit, update_installation
from generation import execute
from utils.io import load_json, write_json, crc_file


def rms_score(features, reference):
    ratios = np.asarray(features["band_rms_mm_s"]) / np.asarray(reference)
    logs = np.log(np.maximum(ratios, 1e-30))
    central = float(np.mean(logs**2))
    return dict(
        ratios=ratios,
        central_loss=central,
        objective=central,
    )


def calibrate(config, targets_path, output, max_evaluations=30, fidelity="low"):
    """One bounded start; fit targets select parameters, validation is checked only afterward."""
    identity = config["comparison_identity"]
    active_gaps = [
        value for key, value in identity.items() if key.startswith("c_") and value is not None
    ]
    if not active_gaps or any(value != 1.0 for value in active_gaps):
        raise ValueError("Health calibration requires every active fit gap to equal 1 um")
    targets = load_json(targets_path)
    if targets.get("H_test_used") is not False or targets.get("beta_used") is not False:
        raise ValueError("Targets must explicitly exclude held-out test data and beta")
    if targets["channel_order"] != ["ACC2_Hx", "ACC1_Hy", "ACC3_Py"] or targets["band_hz"] != [
        10,
        900,
    ]:
        raise ValueError("Targets require ACC2/Hx, ACC1/Hy, ACC3/Py and 10-900 Hz")
    for split in ("fit", "val"):
        values = np.asarray(targets["targets"][split]["band_rms_mm_s"])
        if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"Invalid {split} reference RMS")
    folder = Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("Calibration output must be a new or empty directory")
    folder.mkdir(parents=True, exist_ok=True)
    prior = encode_unit(config)
    write_json(folder / "input_config.json", config)
    write_json(folder / "targets.json", targets)
    write_json(
        folder / "objective_contract.json",
        dict(
            mode="rms_only",
            parameters=PARAMETER_NAMES,
            band_hz=[10, 900],
            reference_gap_um=1.0,
            beta_used=False,
            runout_in_objective=False,
            one_x_in_objective=False,
            selection_split="fit",
            validation_used_for_selection=False,
            H_test_used=False,
            objective="mean squared log RMS ratio",
            fidelity=fidelity,
            targets_crc32=crc_file(targets_path),
        ),
    )
    history, cache = [], {}

    def objective(unit):
        key = tuple(np.round(unit, 12))
        if key in cache:
            return cache[key]["score"]["objective"]
        candidate = update_installation(config, decode_unit(unit, config))
        candidate["case_id"] = f"eval_{len(history):04d}"
        candidate["calibration"] = dict(objective_mode="rms_only", fidelity=fidelity)
        if fidelity == "low":
            candidate["simulation"].update(
                fs_raw_hz=16384,
                retained_turns=160,
                warmup_s=1.5,
                maximum_warmup_s=1.5,
                max_step_s=4 * config["simulation"]["max_step_s"],
            )
            p = candidate["base_v8"]["parameters"]
            p["N"] = 128
            for ring in p.get("rear_outer_rings", {}).values():
                ring["fit_direction_count"] = 128
        out = folder / "search" / candidate["case_id"]
        row = dict(
            index=len(history),
            unit_coordinates=unit,
            parameters=candidate["installation"]["parameters"],
            output=str(out),
        )
        try:
            meta = execute(candidate, out, max_attempts=1)
            features = load_json(out / "features.json")
            score = rms_score(features, targets["targets"]["fit"]["band_rms_mm_s"])
            row.update(score=score, solver_success=meta["solver_success"], steady_pass=meta["steady_pass"])
        except Exception as error:
            row.update(
                error=str(error),
                solver_success=False,
                steady_pass=False,
                score=dict(
                    objective=1e4,
                    central_loss=1e4,
                ),
            )
            write_json(out / "failure.json", row)
        history.append(row)
        cache[key] = row
        write_json(folder / "history.json", history)
        print(f"evaluation={row['index']} RMS_loss={row['score']['objective']:.6g}", flush=True)
        return row["score"]["objective"]

    result = minimize(
        objective,
        prior,
        method="COBYQA",
        bounds=[(0.0, 1.0)] * 6,
        options=dict(maxfev=max_evaluations, initial_tr_radius=0.14, final_tr_radius=0.003),
    )
    eligible = [row for row in history if row.get("solver_success") is True and "error" not in row]
    if not eligible:
        write_json(
            folder / "result.json",
            dict(
                status="FAILED_ALL_CANDIDATES",
                history_count=len(history),
                optimizer_success=bool(result.success),
                termination=str(result.message),
            ),
        )
        raise RuntimeError(
            "No candidate completed; see history.json"
        )
    best = min(
        eligible,
        key=lambda row: row["score"]["central_loss"],
    )
    selected = update_installation(config, best["parameters"])
    selected["case_id"] = "selected_formal"
    selected["calibration"] = dict(objective_mode="rms_only")
    write_json(folder / "candidate_config.json", selected)
    # Restore caller's full simulation settings for the one selected candidate.
    try:
        meta = execute(selected, folder / "formal", max_attempts=1)
    except Exception as error:
        write_json(
            folder / "result.json",
            dict(status="FORMAL_FAILED", error=str(error), selected_index=best["index"]),
        )
        raise
    features = load_json(folder / "formal/features.json")
    scores = {
        split: rms_score(features, targets["targets"][split]["band_rms_mm_s"])
        for split in ("fit", "val")
    }
    passed = bool(meta["solver_success"] is True and meta["steady_pass"])
    status = "RMS_STATIONARITY_PASS" if passed else "RMS_STATIONARITY_OR_SOLVER_FAILED"
    selected["calibration"].update(
        status=status,
        beta_used=False,
        runout_in_objective=False,
        one_x_in_objective=False,
    )
    write_json(folder / "candidate_config.json", selected)
    if passed:
        write_json(folder / "frozen_config.json", selected)
    summary = dict(
        status=status,
        passed=passed,
        optimizer_success=bool(result.success),
        termination=str(result.message),
        evaluations=len(history),
        selected_index=best["index"],
        scores=scores,
        steady_pass=meta["steady_pass"],
        E_dir_um_report_only=features["E_dir_um"],
        one_x_rms_mm_s_report_only=features["one_x_rms_mm_s"],
    )
    write_json(folder / "result.json", summary)
    return summary
