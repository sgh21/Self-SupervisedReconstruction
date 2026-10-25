"""Recalibrate only U after the saved old-U mechanism comparison is complete."""
import argparse
import json
from pathlib import Path

from scripts.run_clearance_v3 import write_csv, write_json
from scripts.run_clearance_v6 import case_config, execute


def calibrate(baseline, root):
    root = Path(root)
    reference = root/'data_A_same_U/new/gap_2um'
    old = json.loads((reference/'metrics.json').read_text(encoding='utf-8'))
    history = []
    candidates = 0

    def record(result, config, folder, reason, reused=False):
        row = dict(F_N=config['excitation']['peak_force_N'],
            U_kg_m=config['excitation']['U_kg_m'], E_um=result['E_dir_um'],
            steady=result['steady_pass'], domain=result['domain_pass'],
            sampling=result['sampling_pass'],
            usable=bool(result['steady_pass'] and result['domain_pass'] and result['sampling_pass']),
            reused_A_result=reused, reason=reason, folder=str(folder))
        history.append(row)
        write_csv(root/'calibration_history.csv', history)
        write_json(root/'calibration_history.json', history)
        return row

    def evaluate(force, reason):
        nonlocal candidates
        if candidates >= 12:
            raise RuntimeError('V8 maximum of 12 new calibration candidates reached.')
        candidates += 1
        config = case_config(baseline, 2, force)
        folder = root/'calibration'/f'F_{force:.12g}N'/'gap_2um'
        return record(execute(config, folder), config, folder, reason)

    old_config = json.loads((reference/'actual_config.json').read_text(encoding='utf-8'))
    first = record(old, old_config, reference, 'Start from the required fixed-old-U experiment A.', True)
    selected = first if first['usable'] and 1.975 <= first['E_um'] <= 2.025 else None
    reason = None
    if selected is None and not first['usable']:
        reason = 'Old-U healthy branch failed stationarity/domain/sampling; no interpolation across it.'
    elif selected is None:
        bound = evaluate(30. if first['E_um'] < 2 else 0., 'Establish a bounded stable bracket with the old-U healthy point.')
        ordered = sorted([first, bound], key=lambda r: r['F_N'])
        lower, upper = ordered
        if not bound['usable']:
            reason = 'Bracket endpoint is not stable and numerically usable; preserve branch evidence.'
        elif not lower['E_um'] <= 2 <= upper['E_um']:
            reason = 'No monotone stable target bracket within the unchanged 0–30 N boundary.'
        else:
            while candidates < 10 and selected is None:
                fraction = (2-lower['E_um'])/(upper['E_um']-lower['E_um'])
                force = lower['F_N']+max(.1, min(.9, fraction))*(upper['F_N']-lower['F_N'])
                row = evaluate(force, 'Safeguarded interpolation inside the observed stable monotone bracket.')
                if not row['usable'] or not lower['E_um'] <= row['E_um'] <= upper['E_um']:
                    reason = 'Unstable/nonmonotone candidate; no root interpolation across a possible branch jump.'
                    break
                if abs(row['E_um']-2) <= .025:
                    selected = row
                elif row['E_um'] < 2:
                    lower = row
                else:
                    upper = row
    if selected is None and reason is None:
        valid = [row for row in history if row['usable'] and 1.9 <= row['E_um'] <= 2.1]
        selected = min(valid, key=lambda row: abs(row['E_um']-2)) if valid else None
    if selected is None:
        write_json(root/'calibration_selection.json', dict(status='fail', reason=reason or 'No valid target candidate.',
            new_candidate_count=candidates, history=history))
        return None
    neighbors = []
    for factor in (.95, 1.05):
        if selected['F_N']*factor <= 30 and candidates < 12:
            neighbors.append(evaluate(selected['F_N']*factor, 'Local U ±5% branch continuity check.'))
    config = case_config(baseline, 2, selected['F_N'])
    config['calibration'].update(status='candidate_requires_cold_repeat_and_independent_h_N',
        U_old_kg_m=first['U_kg_m'], U_star_kg_m=selected['U_kg_m'],
        F_star_N_at_7000=selected['F_N'], healthy_E_um=selected['E_um'])
    write_json(root/'parameters_v8_candidate.json', config)
    write_json(root/'calibration_selection.json', dict(status='candidate', selected=selected,
        neighbors=neighbors, all_neighbors_valid=all(row['usable'] for row in neighbors),
        history=history, new_candidate_count=candidates,
        scope='Conditional calibration on the observed branch; independent certification remains required.'))
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/clearance_v8_config.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    calibrate(json.loads(args.config.read_text(encoding='utf-8')), args.output)
