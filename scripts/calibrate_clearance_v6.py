"""Bounded one-parameter health calibration; candidates are not independent validation."""
import argparse
import json
from pathlib import Path

from scripts.run_clearance_v3 import write_json, write_csv
from scripts.run_clearance_v6 import case_config, execute


def calibrate(baseline, root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    history = []

    def evaluate(force, kind='search'):
        config = case_config(baseline, 2., float(force))
        folder = root/'calibration'/f'F_{force:g}N'/'gap_2um'
        result = execute(config, folder)
        valid = result['steady_pass'] and result['domain_pass'] and result['sampling_pass']
        row = dict(kind=kind, F_N=float(force), U_kg_m=config['excitation']['U_kg_m'],
            E_um=result['E_dir_um'], steady=result['steady_pass'], domain=result['domain_pass'],
            sampling=result['sampling_pass'], usable=bool(valid), folder=str(folder),
            rejection='' if valid else 'stationarity/domain/sampling not all passed')
        history.append(row)
        write_csv(root/'calibration_history.csv', history)
        return row

    zero = evaluate(0.)
    if zero['E_um'] >= 1.9:
        write_json(root/'calibration_selection.json', dict(status='failed', reason='U=0 already near/above target', history=history))
        return
    selected = None
    for force in (2., 5., 10., 20., 30.):
        row = evaluate(force)
        if row['usable'] and abs(row['E_um']-2.) <= .025:
            selected = row; break
        brackets = [(a,b) for a,b in zip(history[:-1],history[1:])
                    if a['usable'] and b['usable'] and (a['E_um']-2)*(b['E_um']-2)<0]
        if brackets:
            lower, upper = brackets[0]
            while len(history) < 10:
                fraction = (2.-lower['E_um'])/(upper['E_um']-lower['E_um'])
                candidate = lower['F_N']+fraction*(upper['F_N']-lower['F_N'])
                row = evaluate(candidate)
                if not row['usable']:
                    break  # do not interpolate through an unsteady response
                if abs(row['E_um']-2.) <= .025:
                    selected = row; break
                if (row['E_um']-2)*(lower['E_um']-2)>0:
                    lower=row
                else:
                    upper=row
            break
    if selected is None:
        valid = [r for r in history if r['usable'] and 1.9 <= r['E_um'] <= 2.1]
        if valid:
            selected = min(valid, key=lambda r: abs(r['E_um']-2))
    if selected is None:
        write_json(root/'calibration_selection.json', dict(status='failed', reason='No valid 1.9-2.1 um candidate at <=30 N',history=history))
        return
    neighbors=[]
    for factor in (.95, 1.05):
        force=selected['F_N']*factor
        if force<=30 and len(history)<12:
            neighbors.append(evaluate(force,'neighbor'))
    candidate=case_config(baseline,2.,selected['F_N'])
    candidate['calibration'].update(U_star_kg_m=selected['U_kg_m'],F_star_N_at_7000=selected['F_N'],
        status='candidate_requires_independent_cold_start_and_h_N_certification')
    write_json(root/'parameters_v6_candidate.json',candidate)
    write_json(root/'calibration_selection.json',dict(status='candidate',selected=selected,
        neighbors=neighbors,all_neighbors_valid=all(r['usable'] for r in neighbors),
        local_branch_only=True,all_search_history=history))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/clearance_v6_config.json'))
    parser.add_argument('--output',type=Path,default=Path('experiments/clearance_v6_GMN'))
    args=parser.parse_args()
    calibrate(json.loads(args.config.read_text(encoding='utf-8')),args.output)
