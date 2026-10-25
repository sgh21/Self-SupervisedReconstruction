"""Independent V8 cold baselines, half steps and doubled fit-direction counts."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts.run_clearance_v3 import write_json
from scripts.run_clearance_v6 import case_config, execute
from utils.clearance_v6_analysis import assemble


def certify(config, root, gaps, workers=3, long_gaps=()):
    jobs = []
    for gap in gaps:
        for group, step, count in (('data_B_recalibrated', 1, 512),
                ('convergence/h_half', .5, 512), ('convergence/n1024', 1, 1024)):
            cfg = case_config(config, gap, directions=count, step_factor=step)
            if gap in long_gaps:
                cfg['simulation']['retained_turns'] = 480
            jobs.append((cfg, root/group/f'gap_{gap:g}um'))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, *job): job[1] for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps(dict(folder=str(futures[future]), E_um=result['E_dir_um'],
                steady=result['steady_pass'], sampling=result['sampling_pass'])), flush=True)
    return assemble(root, base_group='data_B_recalibrated',
        integration_group='convergence/h_half', angular_group='convergence/n1024')


def freeze(root, results):
    selection = json.loads((root/'calibration_selection.json').read_text(encoding='utf-8'))
    health = next(row for row in results['records'] if row['metrics']['gap_um'] == 2)
    epsilon = health['epsilon']['E_dir_um']['epsilon']
    difference = abs(health['metrics']['E_dir_um']-selection['selected']['E_um'])
    checks = dict(healthy_range=1.9 <= health['metrics']['E_dir_um'] <= 2.1,
        certified=health['full_waveform_pass'], independent_repeat=epsilon is not None and difference <= epsilon,
        precise_health=epsilon is not None and epsilon < .01,
        neighbors_valid=selection['all_neighbors_valid'], force_ceiling=health['metrics']['force_N'] <= 30)
    confirmation = dict(status='pass' if all(checks.values()) else 'fail', checks=checks,
        independent_repeat_difference_um=difference, epsilon_E_um=epsilon,
        healthy_E_um=health['metrics']['E_dir_um'], scope='Conditional U calibration; no independent machine validation.')
    (root/'validation').mkdir(exist_ok=True)
    write_json(root/'validation/health_confirmation.json', confirmation)
    if not all(checks.values()):
        raise RuntimeError('V8 healthy certification failed; U has not been frozen.')
    config = json.loads((root/'parameters_v8_candidate.json').read_text(encoding='utf-8'))
    config['calibration'].update(status='numerically_confirmed_conditional_calibration',
        healthy_E_um=health['metrics']['E_dir_um'], epsilon_E_um=epsilon,
        real_machine_validation='not_validated')
    write_json(root/'parameters_v8_calibrated.json', config)
    return confirmation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gaps', type=float, nargs='+', required=True)
    parser.add_argument('--long-gaps', type=float, nargs='*', default=[])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    result = certify(json.loads(args.config.read_text(encoding='utf-8')),
        args.output, args.gaps, args.workers, args.long_gaps)
    if args.freeze:
        print(json.dumps(freeze(args.output, result)), flush=True)
