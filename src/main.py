"""Package entry point: generation and RMS-only parameter calibration."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import traceback
from utils.configuration import load_config, with_clearances
from utils.io import write_json


def save_generation_status(output, plan, status):
    """Record progress and each requested case's acceptance result."""
    plan["status"] = status
    write_json(output / "generation_plan.json", plan)
    write_json(
        output / "generation_summary.json",
        dict(
            status=status,
            run_started_at=plan["run_started_at"],
            qualification=plan["qualification"],
            passed_case_indices=[
                item["case_index"] for item in plan["cases"] if item["status"] == "PASS"
            ],
            failed_case_indices=[
                item["case_index"] for item in plan["cases"] if item["status"] == "FAIL"
            ],
            not_run_case_indices=[
                item["case_index"] for item in plan["cases"] if item["status"] == "NOT_RUN"
            ],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="16/20-DOF spindle model")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate", help="Export all requested cases and record their acceptance results"
    )
    calibrate = commands.add_parser(
        "calibrate", help="Fit installation parameters using only pre-beta 10-900 Hz RMS"
    )
    for command in (generate, calibrate):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    generate.add_argument("--gaps", nargs="+", type=float, help="Front pair gaps; 20 DOF requires --rear-gaps")
    generate.add_argument(
        "--rear-gaps", nargs=2, type=float, help="Explicit fixed R1/R2 override; requires --gaps"
    )
    generate.add_argument("--turns", type=int)
    generate.add_argument("--warmup", type=float)
    generate.add_argument("--max-attempts", type=int, default=1)
    calibrate.add_argument("--targets", type=Path, required=True)
    calibrate.add_argument("--max-evaluations", type=int, default=30)
    calibrate.add_argument("--fidelity", choices=["low", "formal"], default="low")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "calibrate":
        from calibration import calibrate as fit

        if args.max_evaluations < 1:
            parser.error("--max-evaluations must be positive")
        result = fit(config, args.targets, args.output, args.max_evaluations, args.fidelity)
        print(result["status"])
        return 0 if result["passed"] else 2
    from generation import execute

    if args.rear_gaps is not None and args.gaps is None:
        parser.error("--rear-gaps requires --gaps")
    if config["structural_dof"] == 20 and args.gaps is not None and args.rear_gaps is None:
        parser.error("20 DOF --gaps requires explicit --rear-gaps R1 R2")
    if args.turns is not None:
        if args.turns < 32 or args.turns % 4:
            parser.error("--turns must be a multiple of 4 and at least 32")
        config["simulation"]["retained_turns"] = args.turns
    if args.warmup is not None:
        if args.warmup <= config["simulation"]["guard_s"]:
            parser.error("--warmup must exceed the filter guard duration")
        config["simulation"]["warmup_s"] = args.warmup
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    gaps = [None] if args.gaps is None else args.gaps
    if len(set(gaps)) != len(gaps):
        parser.error("Duplicate gaps would reuse the same output; supply unique values")
    plan = dict(
        run_started_at=datetime.now(timezone.utc).isoformat(),
        qualification="SOURCE_WINDOW_RMS_ONLY; SOLVER_MUST_COMPLETE",
        cases=[
            dict(
                case_index=index,
                front_gap_um=gap,
                output=args.output if gap is None else args.output / f"gap_{gap:g}um",
                status="NOT_RUN",
            )
            for index, gap in enumerate(gaps, start=1)
        ],
    )
    save_generation_status(args.output, plan, "RUNNING")
    for item in plan["cases"]:
        gap, out = item["front_gap_um"], item["output"]
        item["status"] = "RUNNING"
        save_generation_status(args.output, plan, "RUNNING")
        try:
            case = config if gap is None else with_clearances(config, gap, args.rear_gaps)
            item["clearances_um"] = case["comparison_identity"].copy()
            meta = execute(case, out, max_attempts=args.max_attempts)
            required_gates = ("solver_success", "steady_pass")
            gates = {name: meta.get(name) for name in (*required_gates, "classification")}
            failed = [
                name for name in required_gates
                if gates[name] is not True
            ]
        except Exception as error:
            failure = dict(
                status="EXECUTION_FAILED",
                case_index=item["case_index"],
                front_gap_um=gap,
                output=out,
                clearances_um=item.get("clearances_um"),
                failed_at=datetime.now(timezone.utc).isoformat(),
                exception=dict(
                    type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
                ),
            )
            item["status"] = "FAIL"
            write_json(out / "entry_failure.json", failure)
            save_generation_status(args.output, plan, "EXECUTION_FAILED")
            raise
        item["gates"] = gates
        print(f"{out}: E_dir={meta.get('E_dir_um')} um, gates={gates}", flush=True)
        if failed:
            item["status"] = "FAIL"
            item["failed_gates"] = failed
        else:
            item["status"] = "PASS"
        save_generation_status(args.output, plan, "RUNNING")
    any_failed = any(item["status"] == "FAIL" for item in plan["cases"])
    save_generation_status(args.output, plan, "COMPLETED_WITH_FAILURES" if any_failed else "ALL_CASES_PASSED")
    return 2 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
