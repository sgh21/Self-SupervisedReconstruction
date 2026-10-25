"""Recompute representative E and H features from the small V5 review ZIP."""
import argparse
import json
from pathlib import Path

import numpy as np
from utils.runout_labels import complete_turn_runout
from utils.clearance_v3_analysis import _harmonics


def review(root):
    root = Path(root)
    index = json.loads((root/'review_representatives/index.json').read_text(encoding='utf-8'))
    rows = []
    for item in index['cases']:
        c = json.loads((root/item['actual_config']).read_text(encoding='utf-8'))
        p, s, e = c['parameters'], c['simulation'], c['excitation']
        with np.load(root/item['file']) as d:
            turns = complete_turn_runout(d['t_M_s'], d['M_relative_probe_m']*1e6,
                p['rpm'], item['start_s'], item['end_s'], phase_at_zero_rad=e['phase_rad'])
            mean = turns['peak_to_peak_um'].mean(axis=0)
            metrics = dict(E_x_um=float(mean[0]), E_y_um=float(mean[1]), E_dir_um=float(max(mean)))
            t = d['t_H_s']; use = (t>=item['start_s']-1e-12)&(t<item['end_s']-1e-12)
            for key, prefix, unit, energy in (
                    ('H_velocity_mm_s', 'H_v', 'mm_s', 'mm2_s2'),
                    ('H_acceleration_m_s2', 'H_a', 'm_s2', 'm2_s4')):
                metrics.update(_harmonics(t[use], d[key][use], p['rpm']*2*np.pi/60,
                    e['phase_rad'], prefix, unit, energy, s['fs_out_hz']))
        reference = item['reference_metrics']
        differences = {key:abs(value-reference[key]) for key,value in metrics.items() if key in reference}
        rows.append(dict(case=item['case'], complete_turns=len(turns['turn_id']),
            recomputed=metrics, max_absolute_metric_difference=max(differences.values()),
            passed=bool(len(turns['turn_id'])==120 and max(differences.values())<1e-9)))
    result = dict(passed=bool(rows and all(row['passed'] for row in rows)), cases=rows,
        scope='Representative full-120-turn E and saved filtered H only; not independent full-window contact/convergence acceptance.')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    review(parser.parse_args().root)
