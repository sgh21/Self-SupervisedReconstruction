"""Audit cold-versus-continuation V8 branches without equating their clocks.

The V6 metric comparator supplies feature tolerances. Its ``same_window``
flag is recorded, then excluded from the feature-only result because cold and
continued branches necessarily use different absolute windows. Missing error
budget terms stay null; this audit does not invent zero uncertainty for a
continuation that has no independent refinement.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.run_clearance_v3 import write_json
from utils.clearance_v6_analysis import compare_metrics, keys as metric_keys, load_case


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = ROOT / "experiments/clearance_v8_tilt_contact"


def _read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def _case_name(gap_um):
    return f"gap_{gap_um:g}um"


def _load_turns(folder):
    path = Path(folder) / "turns.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {int(round(float(row["turn_id"]))):
            (float(row["Ex_um"]), float(row["Ey_um"])) for row in rows}


def _difference_stats(reference, branch):
    values = np.asarray(branch, dtype=float) - np.asarray(reference, dtype=float)
    if values.ndim == 1:
        return dict(mean=float(values.mean()), std=float(values.std()),
                    rms=float(np.sqrt(np.mean(values**2))),
                    max_abs=float(np.max(np.abs(values))),
                    p95_abs=float(np.percentile(np.abs(values), 95)))
    return {axis: _difference_stats(reference[:, index], branch[:, index])
            for index, axis in enumerate(("x", "y"))}


def _pp_lag_statistics(reference_folder, branch_folder):
    reference = _load_turns(reference_folder)
    branch = _load_turns(branch_folder)
    if not reference or not branch:
        return dict(status="not_run", reason="Missing per-turn PP rows.")
    reference_ids, branch_ids = set(reference), set(branch)
    offsets = sorted({bid-rid for bid in branch_ids for rid in reference_ids})
    minimum_pairs = max(30, min(len(reference), len(branch))//4)

    def paired(offset):
        ids = sorted(rid for rid in reference_ids if rid+offset in branch_ids)
        if len(ids) < minimum_pairs:
            return None
        base = np.asarray([reference[turn] for turn in ids])
        other = np.asarray([branch[turn+offset] for turn in ids])
        return ids, base, other

    comparisons = []
    for offset in offsets:
        values = paired(offset)
        if values is None:
            continue
        ids, base, other = values
        diff = other-base
        comparisons.append((float(np.sqrt(np.mean(diff**2))), offset, ids, base, other))
    if not comparisons:
        return dict(status="not_comparable", minimum_pairs=minimum_pairs,
                    reference_turns=len(reference), branch_turns=len(branch))
    comparisons.sort(key=lambda row: (row[0], abs(row[1]), row[1]))
    rms, offset, ids, base, other = comparisons[0]
    zero = paired(0)
    zero_summary = None
    if zero is not None:
        zero_ids, zero_base, zero_other = zero
        zero_summary = dict(pair_count=len(zero_ids), difference=_difference_stats(zero_base, zero_other))
    return dict(status="computed", phase_reference="turn_id labels from the common signed unbalance phase",
        minimum_pairs=minimum_pairs, reference_turns=len(reference), branch_turns=len(branch),
        best_integer_turn_lag=int(offset),
        positive_lag_definition="branch turn_id = cold reference turn_id + lag",
        best_lag_pair_count=len(ids), best_lag_rms_vector_difference_um=rms,
        best_lag_difference=_difference_stats(base, other),
        best_lag_reference_E_dir_mean_um=float(np.mean(np.max(base, axis=1))),
        best_lag_branch_E_dir_mean_um=float(np.mean(np.max(other, axis=1))),
        zero_lag=zero_summary,
        note="Lag alignment is descriptive for PP sequences; agreement after an integer-turn shift does not establish a unique branch.")


def _stroboscopic_state(folder):
    folder = Path(folder)
    config = _read_json(folder / "actual_config.json")
    p, excitation = config["parameters"], config["excitation"]
    with np.load(folder / "native.npz") as data:
        time = np.asarray(data["t_raw"], dtype=float)
        q = np.asarray(data["q_raw_m"], dtype=float)
    rpm = float(p["rpm"])
    rate = abs(rpm)/60
    phase_cycles = np.sign(rpm)*float(excitation["phase_rad"])/(2*np.pi)
    drive_turn = rate*time+phase_cycles
    first = int(np.ceil(drive_turn[0]-1e-10))
    last = int(np.floor(drive_turn[-1]+1e-10))
    turn_ids = np.arange(first, last+1, dtype=int)
    if len(turn_ids) < 5:
        return dict(turn_ids=np.empty(0, dtype=int), q=np.empty((0, 12)),
                    l_ref=float(p.get("L_ref", .03)))
    state = np.column_stack([np.interp(turn_ids, drive_turn, q[:, index])
                             for index in range(q.shape[1])])
    return dict(turn_ids=turn_ids, q=state, l_ref=float(p.get("L_ref", .03)))


def _scaled_q(q, l_ref):
    scaled = np.asarray(q, dtype=float).copy()
    scaled[:, 8:10] *= l_ref
    return scaled


def _q_repeat_rows(series, maximum_lag=4, maximum_pairs=120):
    q = _scaled_q(series["q"], series["l_ref"])
    rows = []
    for lag in range(1, maximum_lag+1):
        if len(q) <= lag:
            continue
        pairs = min(len(q)-lag, maximum_pairs)
        later = q[-pairs:]
        earlier = q[-pairs-lag:-lag]
        diff = later-earlier
        rows.append(dict(turn_lag=lag, compared_pairs=pairs,
            RMS_equivalent_translation_m=float(np.sqrt(np.mean(diff**2))),
            maximum_absolute_coordinate_difference_m=float(np.max(np.abs(diff))),
            per_coordinate_RMS_m=np.sqrt(np.mean(diff**2, axis=0)).tolist()))
    return rows


def _cross_run_q(reference_folder, branch_folder, turn_lag):
    reference = _stroboscopic_state(reference_folder)
    branch = _stroboscopic_state(branch_folder)
    rmap = {int(turn): index for index, turn in enumerate(reference["turn_ids"])}
    bmap = {int(turn): index for index, turn in enumerate(branch["turn_ids"])}
    ids = sorted(turn for turn in rmap if turn+turn_lag in bmap)
    if not ids:
        return dict(status="not_comparable", pair_count=0,
                    reason="No shared stroboscopic states at the PP-selected integer lag.")
    base = np.asarray([reference["q"][rmap[turn]] for turn in ids])
    other = np.asarray([branch["q"][bmap[turn+turn_lag]] for turn in ids])
    diff = _scaled_q(other, branch["l_ref"])-_scaled_q(base, reference["l_ref"])
    return dict(status="computed", pair_count=len(ids), integer_turn_lag=int(turn_lag),
        RMS_equivalent_translation_m=float(np.sqrt(np.mean(diff**2))),
        maximum_absolute_coordinate_difference_m=float(np.max(np.abs(diff))),
        per_coordinate_RMS_m=np.sqrt(np.mean(diff**2, axis=0)).tolist())


def _load_budget_file(root, explicit=None):
    candidates = ([Path(explicit)] if explicit else []) + [
        Path(root)/"recomputed/numerical_error_estimates.json",
        Path(root)/"data_B_recalibrated/recomputed/numerical_error_estimates.json",
    ]
    for path in candidates:
        if path.exists():
            return path, _read_json(path, {})
    return None, {}


def _gap_budget(budgets, gap_um):
    name = _case_name(gap_um)
    for key in (name, f"data_B_recalibrated/{name}", f"data/{name}"):
        row = budgets.get(key)
        if isinstance(row, dict):
            return row
    return {}


def _phase_differences(reference_metrics, branch_metrics):
    keys = set(metric_keys(reference_metrics)) & set(metric_keys(branch_metrics))
    result = {}
    for key in sorted(keys):
        if "_phase_" not in key:
            continue
        delta = float(branch_metrics[key])-float(reference_metrics[key])
        wrapped = float(np.arctan2(np.sin(delta), np.cos(delta)))
        result[key] = dict(reference_rad=float(reference_metrics[key]),
            branch_rad=float(branch_metrics[key]), raw_difference_rad=delta,
            wrapped_difference_rad=wrapped, threshold="not_assigned")
    return result


def _physics_flags(reference_config, branch_config, frozen_u):
    allowed_gap_fields = {"front_group_clearance_um", "fit_clearance_um", "fit2_clearance_um"}
    rp, bp = reference_config["parameters"], branch_config["parameters"]
    parameter_changes = sorted(key for key in (set(rp) | set(bp))
                               if key not in allowed_gap_fields and rp.get(key) != bp.get(key))
    excitation_same = reference_config["excitation"] == branch_config["excitation"]
    processing_same = reference_config["processing"] == branch_config["processing"]
    ignored_clock = {"warmup_s", "maximum_warmup_s"}
    rs, bs = reference_config["simulation"], branch_config["simulation"]
    simulation_changes = sorted(key for key in (set(rs) | set(bs))
                                 if key not in ignored_clock and rs.get(key) != bs.get(key))
    return dict(parameters_match_except_gap=not parameter_changes,
        changed_parameter_keys=parameter_changes,
        excitation_matches=excitation_same,
        branch_U_matches_frozen=bool(float(branch_config["excitation"]["U_kg_m"]) == frozen_u),
        processing_matches=processing_same,
        simulation_matches_except_absolute_warmup=not simulation_changes,
        changed_simulation_keys=simulation_changes)


def _seed_flags(branch_folder, predecessor_folder):
    branch_folder, predecessor_folder = Path(branch_folder), Path(predecessor_folder)
    seed_path = branch_folder / "initial_seed.npz"
    with np.load(predecessor_folder / "native.npz") as predecessor:
        previous_state = np.asarray(predecessor["state_final"])
        previous_epoch = float(predecessor["integration_end_s"])
    if not seed_path.exists():
        return dict(seed_record_exists=False, exact_state_match=False,
            exact_absolute_time_match=False, predecessor_state_size=int(previous_state.size))
    with np.load(seed_path) as seed:
        state = np.asarray(seed["state"])
        epoch = float(seed["absolute_time_s"])
    with np.load(branch_folder / "native.npz") as output:
        branch_end = float(output["integration_end_s"])
        endpoint_shape = np.asarray(output["state_final"]).shape
    return dict(seed_record_exists=True, predecessor_state_size=int(previous_state.size),
        seed_state_shape=list(state.shape), endpoint_state_shape=list(endpoint_shape),
        exact_state_match=bool(np.array_equal(state, previous_state)),
        exact_absolute_time_match=bool(epoch == previous_epoch),
        seed_epoch_s=epoch, predecessor_integration_end_s=previous_epoch,
        branch_integration_end_s=branch_end,
        branch_time_advances=bool(branch_end > epoch),
        phase_preserving_handoff=bool(np.array_equal(state, previous_state)
            and epoch == previous_epoch and state.shape == (24,) and endpoint_shape == (24,)))


def _direct_flags(case, folder, physics, seed):
    metrics = case["metrics"]
    inputs = case.get("input_checks", {})
    checks = dict(solver_pass=bool(case["solver_pass"]), input_pass=bool(case["input_pass"]),
        sampling_identity_pass=bool(case["sampling_identity_pass"]),
        steady_pass=bool(metrics.get("steady_pass", False)),
        domain_pass=bool(metrics.get("domain_pass", case.get("domain", {}).get("passed", False))),
        sampling_pass=bool(metrics.get("sampling_pass", False)),
        physics_parameters_match=bool(physics["parameters_match_except_gap"]),
        fixed_excitation_match=bool(physics["excitation_matches"] and physics["branch_U_matches_frozen"]),
        processing_match=bool(physics["processing_matches"]),
        solver_settings_match=bool(physics["simulation_matches_except_absolute_warmup"]),
        exact_seed_state_match=bool(seed.get("exact_state_match", False)),
        exact_seed_absolute_time_match=bool(seed.get("exact_absolute_time_match", False)),
        seed_is_24_state=bool(seed.get("seed_state_shape") == [24]
            and seed.get("endpoint_state_shape") == [24]),
        branch_time_advances=bool(seed.get("branch_time_advances", False)))
    return dict(checks=checks, input_formula_checks=inputs,
        all_required_pass=bool(all(checks.values()) and case["input_pass"]))


def audit_record(root, gap_um, branch_folder, reference_folder, predecessor_folder,
                 frozen_u, budget_entry):
    branch_folder, reference_folder = Path(branch_folder), Path(reference_folder)
    branch_case = load_case(branch_folder, replay=False)
    reference_case = load_case(reference_folder, replay=False)
    branch_metrics = branch_case["metrics"]
    reference_metrics = reference_case["metrics"]
    comparison = compare_metrics(reference_metrics, branch_metrics, branch_case["config"])
    rows = comparison["metrics"]
    reference_keys = set(metric_keys(reference_metrics))
    branch_keys = set(metric_keys(branch_metrics))
    missing_in_branch = sorted(reference_keys-branch_keys)
    missing_in_reference = sorted(branch_keys-reference_keys)
    feature_only_pass = bool(rows and not missing_in_branch and not missing_in_reference
                             and all(row["passed"] for row in rows.values()))
    compared_keys = set(rows)
    candidates = reference_keys | branch_keys
    non_phase_uncompared = sorted(key for key in candidates
        if key not in compared_keys and "_phase_" not in key
        and key in reference_metrics and key in branch_metrics)
    phase = _phase_differences(reference_metrics, branch_metrics)
    physics = _physics_flags(reference_case["config"], branch_case["config"], frozen_u)
    seed = _seed_flags(branch_folder, predecessor_folder)
    direct = _direct_flags(branch_case, branch_folder, physics, seed)

    budget = {}
    missing = []
    for key, row in rows.items():
        entry = budget_entry.get(key, {}) if isinstance(budget_entry, dict) else {}
        epsilon = entry.get("epsilon") if isinstance(entry, dict) else None
        if epsilon is None:
            missing.append(key)
        difference = row["difference"]
        budget[key] = dict(reference_epsilon=epsilon,
            branch_epsilon=None, combined_epsilon=None,
            difference_over_reference_epsilon=(float(difference)/float(epsilon)
                if epsilon is not None and float(epsilon) > 0 else None),
            reference_budget_complete=bool(entry.get("complete", False)) if isinstance(entry, dict) else False,
            interpretation="Reference-only ratio; no independent continuation epsilon is available.")

    reference_window = (reference_metrics.get("measurement_start_s"), reference_metrics.get("measurement_end_s"))
    branch_window = (branch_metrics.get("measurement_start_s"), branch_metrics.get("measurement_end_s"))
    pp_lag = _pp_lag_statistics(reference_folder, branch_folder)
    q_repeat = dict(reference_within_run=_q_repeat_rows(_stroboscopic_state(reference_folder)),
        branch_within_run=_q_repeat_rows(_stroboscopic_state(branch_folder)),
        cross_run=(_cross_run_q(reference_folder, branch_folder,
            pp_lag["best_integer_turn_lag"]) if pp_lag.get("status") == "computed" else None),
        caveat=("q is sampled at integer signed-unbalance phase. Nonzero turn-lag differences can reflect "
                "multi-turn/contact-phase dynamics; low differences do not prove a unique or exactly periodic branch."))
    return dict(gap_um=float(gap_um), folder=str(branch_folder),
        reference_folder=str(reference_folder), predecessor_folder=str(predecessor_folder),
        reference_window_s=list(reference_window), branch_window_s=list(branch_window),
        direct_flags=direct, physics_comparison=physics,
        seed_handoff=seed,
        statistics_comparison=dict(same_window=bool(comparison["same_window"]),
            v6_compare_metrics_pass=bool(comparison["passed"]),
            passed_ignoring_window_only=feature_only_pass,
            feature_count=len(rows), feature_checks=rows,
            missing_in_branch_feature_keys=missing_in_branch,
            missing_in_reference_feature_keys=missing_in_reference,
            non_phase_uncompared_keys=non_phase_uncompared,
            phase_differences=phase,
            note="Only same_window is ignored in the feature-only result; absolute window times remain recorded."),
        reference_numerical_error_budget=dict(features=budget,
            missing_reference_epsilon_keys=missing,
            branch_epsilon="not_available; no independent refinement was run on this continuation branch",
            combined_epsilon="not_available; missing branch epsilon is not treated as zero"),
        PP_lag_statistics=pp_lag, full_q_phase_repeat=q_repeat,
        uniqueness_claim="not_inferred_from_branch/cold agreement")


def audit_branches(root=DEFAULT_EXPERIMENT_ROOT, reference_group="data_B_recalibrated",
                   groups=("ascending", "descending"), budget_path=None,
                   output_path=None, frozen_config_path=None):
    root = Path(root)
    budgets_path, budgets = _load_budget_file(root, budget_path)
    frozen_config = _read_json(frozen_config_path) if frozen_config_path else None
    frozen_u = (float(frozen_config["excitation"]["U_kg_m"]) if frozen_config else None)
    records = []
    group_records = {}
    for group in groups:
        branch_base = root / "initial_condition" / group
        if group == "ascending":
            gaps = (2.0, 3.0, 5.0, 7.5, 10.0)
            first_predecessor = root / reference_group / _case_name(2.0)
        elif group == "descending":
            gaps = (7.5, 5.0)
            first_predecessor = root / "initial_condition/ascending/gap_10um"
        else:
            raise ValueError(f"Unknown continuation direction {group!r}.")
        current_predecessor = first_predecessor
        group_rows = []
        for gap_um in gaps:
            branch_folder = branch_base / _case_name(gap_um)
            reference_folder = root / reference_group / _case_name(gap_um)
            if not (branch_folder / "metrics.json").exists():
                row = dict(status="not_run", direction=group, gap_um=float(gap_um),
                    folder=str(branch_folder), reference_folder=str(reference_folder),
                    predecessor_folder=str(current_predecessor),
                    reason=("Continuation output is incomplete; metrics.json is missing."
                        if branch_folder.exists() else "Continuation case has not been run yet."))
                group_rows.append(row); records.append(row)
                break
            reference_complete = all((reference_folder/name).exists()
                for name in ("metrics.json", "actual_config.json", "native.npz", "turns.csv"))
            if not reference_complete:
                row = dict(status="not_run", direction=group, gap_um=float(gap_um),
                    folder=str(branch_folder), reference_folder=str(reference_folder),
                    predecessor_folder=str(current_predecessor),
                    reason="Cold B reference is not complete yet.")
                group_rows.append(row); records.append(row)
                current_predecessor = branch_folder
                continue
            branch_config = _read_json(branch_folder / "actual_config.json")
            if frozen_u is None:
                frozen_u = float(branch_config["excitation"]["U_kg_m"])
            row = audit_record(root, gap_um, branch_folder, reference_folder,
                current_predecessor, frozen_u, _gap_budget(budgets, gap_um))
            row["direction"] = group
            row["status"] = "audited"
            group_rows.append(row); records.append(row)
            current_predecessor = branch_folder
        statuses = {row["status"] for row in group_rows}
        group_status = ("not_run" if not group_rows or statuses == {"not_run"}
                        else "partial" if "not_run" in statuses else "audited")
        group_records[group] = dict(status=group_status,
            cases=[row["folder"] for row in group_rows])
    statuses = {row["status"] for row in records}
    output = Path(output_path) if output_path else root / "validation/branches.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report_status = ("not_run" if not records or statuses == {"not_run"}
                     else "partial" if "not_run" in statuses else "audited")
    report = dict(schema="clearance_v8_branch_audit_v1", status=report_status,
        frozen_U_kg_m=frozen_u, seed_source=str(root/reference_group/"gap_2um"),
        reference_group=reference_group, branch_groups=group_records,
        numerical_error_estimates_path=str(budgets_path) if budgets_path else None,
        records=records,
        caveats=["Cold and continuation windows differ in absolute time; same_window is reported but ignored only for per-feature statistics.",
            "No independent continuation refinement was run here, so branch and combined numerical epsilons remain null.",
            "PP lag alignment and q stroboscopic repeat diagnostics do not establish branch uniqueness."],
        scope="Branch/path audit only; not a new calibration or uniqueness proof.")
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--reference-group", default="data_B_recalibrated")
    parser.add_argument("--group", action="append", choices=("ascending", "descending"),
        help="Repeatable; defaults to both branch directions.")
    parser.add_argument("--budgets", type=Path,
        help="Optional path to recomputed/numerical_error_estimates.json.")
    parser.add_argument("--config", type=Path,
        help="Frozen config, used to record and cross-check U.")
    parser.add_argument("--output", type=Path,
        help="Defaults to ROOT/validation/branches.json.")
    args = parser.parse_args()
    report = audit_branches(args.root, args.reference_group,
        tuple(args.group) if args.group else ("ascending", "descending"),
        args.budgets, args.output, args.config)
    print(json.dumps(dict(status=report["status"], frozen_U_kg_m=report["frozen_U_kg_m"],
        record_count=len(report["records"]), output=str(args.output or args.root/"validation/branches.json")),
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
