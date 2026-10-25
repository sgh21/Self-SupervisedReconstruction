"""Measure warmed V8/V9 RHS cost on the same saved states, without integration."""
import argparse
import copy
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from spindle.fast_rhs import prepare_fast_rhs


def campaign_costs(root, output=None):
    """Count exported local iterations separately from uninstrumented RHS work."""
    root = Path(root)
    selection = json.loads((root/'case_selection.json').read_text(encoding='utf-8'))
    cases = {}
    for variant, gaps in selection.get('A', {}).items():
        for gap, path in gaps.items():
            cases.setdefault(path, []).append(f'A/{variant}/c{gap}')
    for gap, path in selection.get('B', {}).items():
        cases.setdefault(path, []).append(f'B/N3/c{gap}')
    rows = []
    for relative, roles in cases.items():
        folder = root/relative
        metrics = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        config = json.loads((folder/'actual_config.json').read_text(encoding='utf-8'))
        stats_path = folder/'per_ball_statistics.csv'
        stats = []  # Frozen V8 runner did not export local iterative roots.
        if stats_path.exists():
            with stats_path.open(encoding='utf-8-sig', newline='') as stream:
                stats = list(csv.DictReader(stream))
        iteration_rows = [r for r in stats if r['field']=='iterations']
        samples = sum(int(r['finite_samples']) for r in iteration_rows)
        iterations = sum(round(float(r['mean'])*int(r['finite_samples']))
                         for r in iteration_rows)
        evaluations = sum(round(float(r['mean'])*int(r['finite_samples']))
                          for r in stats if r['field']=='evaluations')
        status_rows = [r for r in stats if r['field']=='status_code']
        retry_count = (sum(round(float(r['mean'])*int(r['finite_samples']))
                           for r in status_rows)
                       if status_rows and all(float(r['minimum'])>=0 and
                                              float(r['maximum'])<=1 for r in status_rows)
                       else None)
        max_value = lambda suffix: max((float(v) for k,v in metrics.items()
                                        if k.endswith(suffix)), default=None)
        rows.append(dict(case=relative, roles=';'.join(roles),
            variant=config.get('mechanism_variant','B0'), gap_um=metrics['gap_um'],
            nfev=metrics['nfev'], wall_seconds=metrics['wall_seconds'],
            simulated_seconds=metrics.get('simulated_seconds'),
            seconds_per_simulated_second=metrics.get('seconds_per_simulated_second'),
            exported_ball_solves=samples or None,
            exported_iterations_total=iterations if samples else None,
            exported_evaluations_total=evaluations if samples else None,
            exported_iterations_mean=iterations/samples if samples else None,
            exported_iterations_max=max_value('_max_iterations'),
            exported_retry_count=retry_count,
            max_force_residual_N=max_value('_max_force_residual_N'),
            max_force_tolerance_ratio=max_value('_max_force_residual_tolerance_ratio'),
            max_geometry_normalized=max_value('_max_geometry_residual_normalized')))
    destination = Path(output) if output is not None else root/'validation'
    destination.mkdir(parents=True, exist_ok=True)
    record = dict(rows=rows,
        iteration_scope='Exact sums over exported native-time ball solves, reconstructed from each ball mean times its finite sample count. These are postprocessing calls, not all integrator RHS calls.',
        all_rhs_local_iterations_total=None,
        all_rhs_iteration_status='Not instrumented in this campaign; nfev is the actual integrator RHS count. Do not multiply sampled mean iterations into a claimed measured RHS total.',
        cost_scope='Per-case wall time includes integration and analysis up to metric construction, excluding the final native/diagnostic compression and file writes. Resumed histories include inherited wall time. CPU contention differs across cases. The separate warmed identical-state RHS benchmark is the controlled V8 cost comparison.')
    (destination/'campaign_costs.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    if rows:
        with (destination/'campaign_costs.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    return record


def benchmark(root, samples=1000, repetitions=5):
    root = Path(root)
    rows = []
    for gap in (2, 10):
        folder = root/'data_A'/'B0'/f'gap_{gap}um'
        base = json.loads((folder/'actual_config.json').read_text(encoding='utf-8'))
        with np.load(folder/'native.npz') as data:
            t = data['t_raw']
            indices = np.linspace(0, len(t)-1, samples, dtype=int)
            q, v = data['q_raw_m'][indices], data['v_raw_m_s'][indices]
            t = t[indices]
        states = np.concatenate((q, v), axis=1)
        configs = {'B0': base}
        for variant in ('B0d', 'N1', 'N2', 'N3'):
            path = root/'data_A'/variant/f'gap_{gap}um'/'actual_config.json'
            if path.exists():
                configs[variant] = json.loads(path.read_text(encoding='utf-8'))
        functions = {}
        for variant, config in configs.items():
            p = copy.deepcopy(config['parameters'])
            drive = config['excitation']
            rhs = prepare_fast_rhs(p, (drive['U_kg_m'], drive['phase_rad']))
            for j in range(min(10, samples)):
                rhs(t[j], states[j])  # Compile and warm caches before timing.
            functions[variant] = rhs
        records = {variant: [] for variant in functions}
        for repeat in range(repetitions):
            for variant, rhs in functions.items():
                wall, cpu = time.perf_counter(), time.process_time()
                checksum = 0.
                for j in range(samples):
                    checksum += float(rhs(t[j], states[j])[-1])
                records[variant].append(dict(
                    wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu,
                    checksum=checksum, repetition=repeat))
        baseline = np.median([x['wall_s'] for x in records['B0']])
        for variant, measurements in records.items():
            median = float(np.median([x['wall_s'] for x in measurements]))
            rows.append(dict(gap_um=gap, variant=variant, calls_per_repetition=samples,
                repetitions=repetitions, measurements=measurements,
                median_wall_us_per_rhs=median/samples*1e6,
                median_cpu_us_per_rhs=float(np.median([x['cpu_s'] for x in measurements]))/samples*1e6,
                ratio_to_v8_wall=median/baseline, state_source=str(folder/'native.npz')))
    record = dict(rows=rows, platform=platform.platform(),
        scope='Warmed complete RHS on identical saved V8 states at each gap; no new trajectory.',
        concurrency_note='Other campaign workers may run concurrently; wall and process CPU times are both retained. Whole-case costs are recorded separately in metrics.json.')
    output = root/'validation'/'rhs_performance.json'
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--output', type=Path,
        help='Output directory for --costs-only; RHS benchmark output remains ROOT/validation.')
    parser.add_argument('--costs-only', action='store_true',
        help='Aggregate saved campaign costs without evaluating an RHS.')
    args = parser.parse_args()
    if args.costs_only:
        record = campaign_costs(args.root, args.output)
        print(json.dumps(dict(cases=len(record['rows']),scope=record['iteration_scope'])))
        return
    if args.output is not None:
        parser.error('--output can only be used with --costs-only')
    record = benchmark(args.root, args.samples, args.repetitions)
    print(json.dumps([{k:r[k] for k in ('gap_um','variant','median_wall_us_per_rhs','ratio_to_v8_wall')}
                      for r in record['rows']]))


if __name__ == '__main__':
    main()
