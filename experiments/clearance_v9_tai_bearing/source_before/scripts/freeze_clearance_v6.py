"""Freeze U only after the healthy cold run and independent numerical checks."""
import argparse
import json
from pathlib import Path
from scripts.run_clearance_v3 import write_json
from utils.clearance_v6_analysis import assemble


def freeze(root):
    root=Path(root)
    results=assemble(root)
    acceptance=results['acceptance']
    health=next(row for row in results['records'] if row['metrics']['gap_um']==2)
    selected=json.loads((root/'calibration_selection.json').read_text(encoding='utf-8'))
    difference=abs(health['metrics']['E_dir_um']-selected['selected']['E_um'])
    epsilon=health['epsilon']['E_dir_um']['epsilon']
    checks=dict(healthy_target_and_independent_refinements=acceptance['health_target_numerically_certified'],
        cold_start_reproducible=epsilon is not None and difference<=epsilon,
        neighbors_valid=selected['all_neighbors_valid'],
        primary_H_RMS_certified=all(health['feature_status'][k]=='pass'
            for k in ('H_v_rms_x_mm_s','H_v_rms_y_mm_s')),
        force_below_ceiling=health['metrics']['force_N']<=30.)
    confirmation=dict(passed=all(checks.values()),checks=checks,
        healthy_E_um=health['metrics']['E_dir_um'],epsilon_E_um=epsilon,
        independent_cold_start_difference_um=difference,neighbors=selected['neighbors'],
        scope='Conditional healthy calibration; not independent real-machine validation')
    write_json(root/'validation/health_confirmation.json',confirmation)
    if not confirmation['passed']:
        raise RuntimeError('Healthy confirmation failed; no calibrated configuration written.')
    config=json.loads((root/'parameters_v6_candidate.json').read_text(encoding='utf-8'))
    config['calibration'].update(status='numerically_confirmed_conditional_calibration',
        healthy_E_um=health['metrics']['E_dir_um'],epsilon_E_um=epsilon,
        real_machine_validation='not_validated')
    write_json(root/'parameters_v6_calibrated.json',config)
    print(json.dumps(confirmation,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('experiments/clearance_v6_GMN'))
    freeze(parser.parse_args().root)
