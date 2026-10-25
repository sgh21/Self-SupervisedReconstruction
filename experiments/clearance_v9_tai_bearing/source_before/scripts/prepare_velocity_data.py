"""Convert raw g to guarded 8192 Hz, 10–2048 Hz, ACC2/ACC1/ACC3 mm/s HDF5."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from utils.velocity_data import DEFAULT_SPLITS, audit_collection, prepare_run, run_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional JSON data.train_runs/validation_runs/test_runs")
    parser.add_argument("--runs", nargs="*", help="Run-name fragments; default existing two train and two validation runs")
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--all-conditions", action="store_true", help="Keep nonsteady conditions with healthy_eligible=false")
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--audit-only", action="store_true", help="Read saved provenance and metadata without converting waveforms")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.raw_root.resolve()):
        raise ValueError("Output cannot be inside read-only raw runs")
    if args.audit_only:
        result = audit_collection(args.output)
        print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
        return
    splits = DEFAULT_SPLITS
    if args.config:
        data = json.loads(args.config.read_text(encoding="utf-8"))["data"]
        splits = {split: data.get(f"{split}_runs", []) for split in ("train", "validation", "test")}
    requested = args.runs or splits["train"] + splits["validation"]
    runs = [path.parent for path in sorted(args.raw_root.glob("*/*/segment_records.csv"))
            if args.all_runs or any(Path(item).stem in path.parent.name or item in path.parent.name for item in requested)]
    if not runs:
        raise ValueError("No runs matched")
    args.output.mkdir(parents=True, exist_ok=True)
    selection = [dict(run=str(run.resolve()), split=run_split(run.name, splits)) for run in runs]
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare_run, run, args.output, item["split"], args.max_segments,
                               args.all_conditions) for run, item in zip(runs, selection)]
        for future in futures:
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
            (args.output / "conversion_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    audit_collection(args.output)


if __name__ == "__main__":
    main()
