"""Offline per-turn PP distributions and short-lag repeat diagnostics for V6.

Repeating PP is weaker evidence than a periodic full-state trajectory. The cage
keeps rotating; this helper does not assert a strict system period.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from utils.clearance_v5_analysis import _read, _write, _csv


def analyze(root, groups=('data', 'certification/h_half', 'certification/n1024')):
    root = Path(root)
    out = root/'validation'; out.mkdir(exist_ok=True)
    records, turns, statistics = [], [], []
    for group in groups:
        for file in sorted((root/group).rglob('turns.csv')):
            with file.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            pp = np.array([[float(row['Ex_um']), float(row['Ey_um'])] for row in rows])
            metrics = _read(file.with_name('metrics.json'))
            case = file.parent.relative_to(root/group).as_posix()
            mean = pp.mean(axis=0)
            formal = float(max(mean))
            lag_errors = []
            for lag in range(1, 7):
                error = abs(pp[lag:]-pp[:-lag])
                lag_errors.append(dict(lag_turns=lag, maximum_X_um=float(error[:, 0].max()),
                    maximum_Y_um=float(error[:, 1].max()), maximum_XY_um=float(error.max()),
                    rms_X_um=float(np.sqrt(np.mean(error[:, 0]**2))),
                    rms_Y_um=float(np.sqrt(np.mean(error[:, 1]**2)))))
            short = next((row['lag_turns'] for row in lag_errors if row['maximum_XY_um'] <= .005), None)
            record = dict(group=group, case=case, gap_um=metrics['gap_um'], complete_turns=len(pp),
                E_x_mean_um=float(mean[0]), E_y_mean_um=float(mean[1]), E_dir_um=formal,
                official_metric_replay_difference_um=abs(formal-metrics['E_dir_um']),
                maximum_single_turn_X_um=float(pp[:, 0].max()), maximum_single_turn_Y_um=float(pp[:, 1].max()),
                maximum_single_turn_directional_PP_um=float(pp.max()),
                mean_per_turn_max_direction_PP_um=float(pp.max(axis=1).mean()),
                approximate_PP_repeat_turns=short, diagnostic_repeat_limit_um=.005,
                classification=f'near_{short}_turn_PP_repeat' if short is not None else 'no_1_to_6_turn_PP_repeat_within_diagnostic_limit',
                lag_errors=lag_errors,
                note='The 0.005 um criterion is a descriptive PP repetition screen, not an added model acceptance rule or proof of a strict full-system period.')
            with np.load(file.with_name('native.npz')) as native:
                t, xy = native['t_raw'], native['r_m_raw_m']*1e6
                meta=json.loads(str(native['metadata_json']))
                left,right=meta['export_start_s'],meta['export_end_s']
                use=(t>left)&(t<right)
                boundary=np.array([[np.interp(edge,t,xy[:,axis]) for axis in (0,1)] for edge in (left,right)])
                envelope=np.ptp(np.vstack((boundary[:1],xy[use],boundary[1:])),axis=0)
            record.update(full_window_coordinate_envelope_X_um=float(envelope[0]),
                          full_window_coordinate_envelope_Y_um=float(envelope[1]))
            records.append(record)
            for i, row in enumerate(rows):
                turns.append(dict(group=group, case=case, gap_um=metrics['gap_um'],
                    turn_index_in_window=i+1, absolute_turn_id=float(row['turn_id']),
                    start_time_s=float(row['start_time_s']), end_time_s=float(row['end_time_s']),
                    Ex_um=pp[i,0], Ey_um=pp[i,1], max_direction_this_turn_um=float(pp[i].max()),
                    formal_E_dir_um=formal))
            for axis, values in zip(('X', 'Y', 'max_direction_this_turn'), (pp[:,0],pp[:,1],pp.max(axis=1))):
                statistics.append(dict(group=group, case=case, gap_um=metrics['gap_um'], axis=axis,
                    count=len(values), mean_um=float(values.mean()), std_um=float(values.std()),
                    minimum_um=float(values.min()), p05_um=float(np.quantile(values,.05)),
                    median_um=float(np.median(values)), p95_um=float(np.quantile(values,.95)),
                    maximum_um=float(values.max()), formal_E_dir_um=formal,
                    approximate_PP_repeat_turns=short))
    result = dict(scope='Saved per-turn displacement PP only; no simulation performed',
        formal_definition='max(mean(per_turn_X_PP),mean(per_turn_Y_PP)); retains 1X',
        not_formal_metrics=['maximum single-turn PP', 'mean(max(X_PP,Y_PP) per turn)', 'full-window coordinate envelope'],
        full_system_periodicity_claimed=False, cases=records,
        all_formal_metrics_reproduced=bool(records and all(r['official_metric_replay_difference_um']<1e-10 for r in records)))
    _write(out/'response_classification.json',result)
    _csv(out/'per_turn_PP.csv',turns)
    _csv(out/'per_turn_PP_distribution.csv',statistics)
    print(json.dumps(dict(cases=len(records), all_formal_metrics_reproduced=result['all_formal_metrics_reproduced'],
        main=[{k:r[k] for k in ('gap_um','E_dir_um','maximum_single_turn_directional_PP_um','approximate_PP_repeat_turns')}
              for r in records if r['group']=='data']),indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('experiments/clearance_v6_GMN'))
    analyze(parser.parse_args().root)
