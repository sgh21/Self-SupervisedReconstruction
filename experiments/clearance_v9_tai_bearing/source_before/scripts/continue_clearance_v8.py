"""Run phase-preserving V8 gap branches from a frozen healthy state.

The ascending path starts from the independently run B healthy case and
continues through c=3, 5, 7.5 and 10 um. ``--reverse`` optionally starts from
the ascending c=10 state and returns through c=7.5 and 5 um. Every handoff
uses the predecessor's complete 24-state endpoint and absolute integration
time, so unbalance and cage phases remain continuous.

This entry requires the numerically confirmed V8 calibration file. It never
changes U between gaps and uses each available cold B case's retained-turn and
warmup settings. If a high-gap reference is still running, it defaults to 480
turns there and the branch audit marks that comparison pending.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from scripts.run_clearance_v3 import write_json
from scripts.run_clearance_v6 import case_config, execute


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = ROOT / "experiments/clearance_v8_tilt_contact"
ASCENDING_GAPS = (2.0, 3.0, 5.0, 7.5, 10.0)
REVERSE_GAPS = (7.5, 5.0)
DEFAULT_LONG_GAPS = (7.5, 10.0)
CALIBRATED_STATUS = "numerically_confirmed_conditional_calibration"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _case_name(gap_um):
    return f"gap_{gap_um:g}um"


def _frozen_u(config):
    if config.get("analysis_version") != "v8":
        raise ValueError("Continuation requires analysis_version='v8'.")
    if config.get("calibration", {}).get("status") != CALIBRATED_STATUS:
        raise ValueError(
            "U is not frozen: use parameters_v8_calibrated.json after independent certification."
        )
    if config["parameters"].get("bearing_contact_model") != "relative_tilt_v8":
        raise ValueError("Continuation requires bearing_contact_model='relative_tilt_v8'.")
    excitation = config["excitation"]
    value = excitation.get("U_kg_m")
    frozen = config["calibration"].get("U_star_kg_m")
    if value is None or frozen is None or float(value) != float(frozen):
        raise ValueError("Frozen excitation.U_kg_m must exactly match calibration.U_star_kg_m.")
    return float(value)


def _load_reference(root, reference_group, gap_um, require_native=False):
    folder = Path(root) / reference_group / _case_name(gap_um)
    config_path = folder / "actual_config.json"
    metrics_path = folder / "metrics.json"
    native_path = folder / "native.npz"
    reference_config = None
    if config_path.exists():
        try:
            reference_config = _read_json(config_path)
        except json.JSONDecodeError:
            # A cold case may be writing actual_config.json concurrently.
            reference_config = None
    try:
        reference_metrics = _read_json(metrics_path) if metrics_path.exists() else None
    except json.JSONDecodeError:
        reference_metrics = None
    if require_native and (reference_config is None or reference_metrics is None or not native_path.exists()):
        raise FileNotFoundError(f"Cold B healthy seed is incomplete: {folder}")
    return folder, reference_config, reference_metrics


def _target_config(frozen_config, reference_config, gap_um, frozen_u, long_gaps):
    """Use the frozen physics and that gap's cold-run window policy."""
    target = case_config(frozen_config, gap_um)
    simulation = target["simulation"]
    if reference_config is not None:
        reference = reference_config["simulation"]
        simulation["retained_turns"] = int(reference["retained_turns"])
        # The saved cold actual_config contains its final warmup after any
        # adaptive extension. Reuse it as a relative warmup after the gap
        # switch; execute() adds the predecessor's absolute epoch.
        simulation["warmup_s"] = float(reference["warmup_s"])
    elif float(gap_um) in long_gaps:
        # Start the high-gap branch with the planned long statistical window
        # even if its independent cold reference has not finished yet.
        simulation["retained_turns"] = 480
    maximum = float(target["simulation"].get("maximum_warmup_s", 20.0))
    if simulation["warmup_s"] > maximum + 1e-12:
        raise ValueError(
            f"Cold B gap {gap_um:g} um warmup exceeds the frozen 20 s allowance."
        )
    if float(target["excitation"]["U_kg_m"]) != frozen_u:
        raise ValueError("Target case changed the frozen new U.")
    if reference_config is not None:
        reference_p = reference_config["parameters"]
        target_p = target["parameters"]
        gap_fields = {"front_group_clearance_um", "fit_clearance_um", "fit2_clearance_um"}
        changed = sorted(key for key in (set(reference_p) | set(target_p))
                         if key not in gap_fields and reference_p.get(key) != target_p.get(key))
        if changed:
            raise ValueError(f"Cold B reference physics differs from frozen config: {changed}")
        if reference_config["excitation"] != target["excitation"]:
            raise ValueError("Cold B reference excitation differs from the frozen U/config.")
        if reference_config["processing"] != target["processing"]:
            raise ValueError("Cold B reference signal processing differs from frozen config.")
    return target


def _predecessor_epoch(folder):
    with np.load(Path(folder) / "native.npz") as data:
        state = np.asarray(data["state_final"])
        epoch = float(data["integration_end_s"])
    if state.shape != (24,):
        raise ValueError(f"Expected a 24-state V8 endpoint in {folder}; got {state.shape}.")
    return state, epoch


def _prepare_existing_case(config, output_folder, predecessor_folder):
    """Guard reuse against the V6 runner's seed-blind existing-case shortcut."""
    output_folder = Path(output_folder)
    if not output_folder.exists():
        return config
    if not (output_folder / "metrics.json").exists():
        raise FileExistsError(f"Partial output folder already exists: {output_folder}")
    seed_path = output_folder / "initial_seed.npz"
    saved_config_path = output_folder / "actual_config.json"
    history_path = output_folder / "warmup_history.json"
    if not seed_path.exists() or not saved_config_path.exists() or not history_path.exists():
        raise ValueError(f"Existing branch is missing seed/config/warmup evidence: {output_folder}")
    expected_state, expected_epoch = _predecessor_epoch(predecessor_folder)
    with np.load(seed_path) as seed:
        same_state = np.array_equal(seed["state"], expected_state)
        same_epoch = float(seed["absolute_time_s"]) == expected_epoch
    if not (same_state and same_epoch):
        raise ValueError(f"Existing branch was seeded from a different predecessor: {output_folder}")
    saved = _read_json(saved_config_path)
    history = _read_json(history_path).get("history", [])
    if not history:
        raise ValueError(f"Existing branch has no warmup history: {output_folder}")
    requested_start = float(config["simulation"]["warmup_s"]) + expected_epoch
    actual_start = float(history[0]["warmup_s"])
    if not np.isclose(actual_start, requested_start, rtol=0, atol=1e-10):
        raise ValueError(f"Existing branch used a different relative warmup: {output_folder}")
    # execute() shifts this setting by the seed's absolute epoch. Reproduce that
    # expected value only when reusing a proven exact seed; fresh runs must pass
    # the unshifted relative maximum so execute() adds the epoch once.
    expected_maximum = float(config["simulation"]["maximum_warmup_s"]) + expected_epoch
    actual_maximum = float(saved["simulation"]["maximum_warmup_s"])
    if not np.isclose(actual_maximum, expected_maximum, rtol=0, atol=1e-12):
        raise ValueError(f"Existing branch has an unexpected warmup limit: {output_folder}")
    reusable = json.loads(json.dumps(config))
    reusable["simulation"]["maximum_warmup_s"] = actual_maximum
    return reusable


def _run_sequence(frozen_config, root, gaps, output_group, reference_group,
                  initial_seed_folder, frozen_u, long_gaps):
    predecessor = Path(initial_seed_folder)
    output_base = Path(root) / output_group
    records = []
    for gap_um in gaps:
        reference_folder, reference_config, reference_metrics = _load_reference(
            root, reference_group, gap_um, require_native=(gap_um == 2.0 and output_group.endswith("ascending"))
        )
        if reference_metrics is not None and float(reference_metrics["gap_um"]) != float(gap_um):
            raise ValueError(f"Reference folder has the wrong clearance: {reference_folder}")
        if reference_metrics is not None and float(reference_metrics["U_kg_m"]) != frozen_u:
            raise ValueError(f"Cold B reference does not use frozen U: {reference_folder}")
        if reference_config is not None and float(reference_config["excitation"]["U_kg_m"]) != frozen_u:
            raise ValueError(f"Cold B reference config does not use frozen U: {reference_folder}")
        target = _target_config(frozen_config, reference_config, gap_um, frozen_u, long_gaps)
        output_folder = output_base / _case_name(gap_um)
        call_config = copy.deepcopy(_prepare_existing_case(target, output_folder, predecessor))
        result = execute(call_config, output_folder, predecessor)
        records.append(dict(gap_um=gap_um, folder=str(output_folder),
            reference_folder=str(reference_folder), predecessor_folder=str(predecessor),
            reference_config_available=reference_config is not None,
            reference_result_available=reference_metrics is not None,
            retained_turns=target["simulation"]["retained_turns"],
            relative_warmup_s=target["simulation"]["warmup_s"],
            steady_pass=bool(result["steady_pass"]), domain_pass=bool(result["domain_pass"]),
            sampling_pass=bool(result["sampling_pass"]), E_dir_um=result["E_dir_um"]))
        write_json(output_base / "continuation_progress.json", dict(
            direction=output_group.rsplit("/", 1)[-1], frozen_U_kg_m=frozen_u,
            records=records, scope="Sequential phase-preserving continuation; no uniqueness claim."))
        predecessor = output_folder
    return records


def continue_branch(config, root=DEFAULT_EXPERIMENT_ROOT, reference_group="data_B_recalibrated",
                    reverse=False, long_gaps=DEFAULT_LONG_GAPS):
    """Run one serialized ascending or optional descending continuation path."""
    root = Path(root)
    frozen_u = _frozen_u(config)
    if reverse:
        start = root / "initial_condition/ascending/gap_10um"
        if not (start / "native.npz").exists():
            raise FileNotFoundError("--reverse requires the completed ascending c=10 um case.")
        gaps = REVERSE_GAPS
        group = "initial_condition/descending"
        seed = start
    else:
        gaps = ASCENDING_GAPS
        group = "initial_condition/ascending"
        seed = root / reference_group / "gap_2um"
    records = _run_sequence(config, root, gaps, group, reference_group, seed, frozen_u,
                            {float(value) for value in long_gaps})
    return dict(direction="descending" if reverse else "ascending",
        frozen_U_kg_m=frozen_u, seed_source=str(seed), records=records,
        uniqueness_claim="not_inferred_from_branch/cold agreement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
        help="Frozen parameters_v8_calibrated.json; unconfirmed U is rejected.")
    parser.add_argument("--root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--reference-group", default="data_B_recalibrated")
    parser.add_argument("--reverse", action="store_true",
        help="After ascending is complete, run c=10 -> 7.5 -> 5 um.")
    parser.add_argument("--long-gaps", type=float, nargs="*", default=list(DEFAULT_LONG_GAPS),
        help="Use 480 turns when a high-gap cold reference config is not available (default: 7.5 10).")
    args = parser.parse_args()
    config = _read_json(args.config)
    report = continue_branch(config, args.root, args.reference_group, args.reverse, args.long_gaps)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
