"""Explicit V8 entry; reuse the V6 solver, native evidence and signal processing."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts.run_clearance_v3 import write_json
from scripts.run_clearance_v6 import case_config, execute
from utils.clearance_v8_parameters import layout_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/clearance_v8_config.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', default='data_B_recalibrated')
    parser.add_argument('--gaps', type=float, nargs='+', default=[2, 3, 5, 7.5, 10])
    parser.add_argument('--forces', type=float, nargs='+')
    parser.add_argument('--model', choices=['radial_only_v6', 'relative_tilt_v8'])
    parser.add_argument('--layout', choices=['DB_DB', 'DB_DF', 'DF_DB', 'DF_DF'])
    parser.add_argument('--directions', type=int)
    parser.add_argument('--step-factor', type=float, default=1.)
    parser.add_argument('--warmup', type=float)
    parser.add_argument('--maximum-warmup', type=float)
    parser.add_argument('--turns', type=int)
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    baseline = json.loads(args.config.read_text(encoding='utf-8'))
    if args.layout:
        baseline = layout_config(baseline, args.layout)
    if args.model:
        baseline['parameters']['bearing_contact_model'] = args.model
    if baseline.get('analysis_version') != 'v8':
        raise ValueError('Use an explicit V8 configuration with analysis_version=v8.')
    jobs = []
    for force in args.forces or [None]:
        for gap in args.gaps:
            cfg = case_config(baseline, gap, force, args.directions, args.step_factor)
            for option, key in [('warmup','warmup_s'), ('maximum_warmup','maximum_warmup_s'),
                                ('turns','retained_turns')]:
                value = getattr(args, option)
                if value is not None:
                    cfg['simulation'][key] = value
            parent = args.output/args.group
            if force is not None:
                parent = parent/f'F_{force:.12g}N'
            jobs.append((cfg, parent/f'gap_{gap:g}um', args.seed))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, *job): job[1] for job in jobs}
        for future in as_completed(futures):
            folder = futures[future]
            try:
                result = future.result()
                print(json.dumps(dict(folder=str(folder), E_um=result['E_dir_um'],
                    steady=result['steady_pass'], domain=result['domain_pass'])), flush=True)
            except Exception as error:
                folder.mkdir(parents=True, exist_ok=True)
                write_json(folder/'failed.json', dict(error=str(error)))
                raise


if __name__ == '__main__':
    main()
