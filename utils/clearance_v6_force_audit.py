"""Audit saved V6 dynamic force/moment budgets without rerunning integration."""
import argparse
import json
from pathlib import Path

import numpy as np

from spindle import dynamics_model as model
from scripts.run_clearance_v3 import write_csv, write_json


def _end_value(t, values, when):
    index = np.searchsorted(t, when)
    if t[index] == when:
        return values[index]
    fraction = (when-t[index-1])/(t[index]-t[index-1])
    return values[index-1]+fraction*(values[index]-values[index-1])


def _mean(t, values, start, end):
    use = (t > start) & (t < end)
    times = np.r_[start, t[use], end]
    bounded = np.concatenate((_end_value(t, values, start)[None], values[use],
                              _end_value(t, values, end)[None]), axis=0)
    return np.trapezoid(bounded, times, axis=0)/(end-start)


def audit_case(folder):
    folder = Path(folder)
    config = json.loads((folder/'actual_config.json').read_text(encoding='utf-8'))
    p = config['parameters']
    with np.load(folder/'native.npz') as native:
        t, q, v, acc = (native[k] for k in ('t_raw', 'q_raw_m', 'v_raw_m_s', 'a_raw_m_s2'))
        b1, b2, rear = (native[k] for k in ('bearing1_force_N', 'bearing2_force_N', 'rear_force_N'))
        fit1, fit2 = native['fit_force_N'], native['fit2_force_N']
        local_moments = {name: native[name+'_local_restoring_Nm']
            for name in ('F1','F2','R1','R2') if name+'_local_restoring_Nm' in native}
        metadata = json.loads(str(native['metadata_json']))
    with np.load(folder/'actual_input.npz') as actual:
        input_t, xy, generalized = actual['t_raw'], actual['force_xy_N'], actual['generalized_rotor_force']
    start, end = metadata['export_start_s'], metadata['export_end_s']
    main = (t >= start) & (t < end)
    masses, gravity = model.mass_and_gravity(p)
    link = model.area_and_link(p, 0)[1]*(model.rotor_section(q, p['z_T'])-q[:, 2:4])
    link += p['cl']*(model.rotor_section(v, p['z_T'])-v[:, 2:4])
    ground = p['kh']*q[:, 4:6]+p['ch']*v[:, 4:6]
    restoring = model.generalized_point_force(b1, p['z_f'], 12)
    restoring += model.generalized_point_force(b2, p['z_f2'], 12)
    restoring += model.generalized_point_force(link, p['z_T'], 12)
    for index, support in enumerate(p['rear_supports']):
        restoring += model.generalized_point_force(rear[:, index], support['z_m'], 12)
    for values in local_moments.values():
        restoring[:, 8:10] += values
    restoring[:, 2:4] = -link
    restoring[:, 4:6] = -rear.sum(axis=1)-fit1-fit2+ground
    restoring[:, 6:8] = -b1+fit1
    restoring[:, 10:12] = -b2+fit2
    gyro = v@model.gyro_matrix(p).T
    restoring += gyro
    drive = np.zeros_like(q)
    drive[:, model.ROTOR] = generalized
    residual = masses*acc-gravity-drive+restoring
    translation_indices = np.r_[0:8, 10:12]
    force_max = float(abs(residual[main][:, translation_indices]).max())
    torque_max = float(abs(residual[main, 8:10]).max())
    mean_residual = _mean(t, residual, start, end)
    average_inertia = _mean(t, masses*acc, start, end)
    boundary_momentum_rate = masses*(_end_value(t, v, end)-_end_value(t, v, start))/(end-start)
    # Integrating acceleration on the output grid has its own quadrature error;
    # report that discrepancy separately from the instantaneous ODE identity.
    quadrature_difference = average_inertia-boundary_momentum_rate
    angle = p['rpm']*2*np.pi/60*t+config['excitation']['phase_rad']
    expected_xy = config['excitation']['U_kg_m']*(p['rpm']*2*np.pi/60)**2*np.column_stack((np.cos(angle), np.sin(angle)))
    formula_error = float(abs(expected_xy-xy).max())
    generalized_error = float(abs(xy@model.section_matrix(p['z_u'])-generalized).max())
    # Global translational momentum counts each lumped body once per direction.
    whole_inertia = sum(masses[index]*acc[:, index:index+2] for index in (0, 2, 4, 6, 10))
    whole_gravity = sum(gravity[index:index+2] for index in (0, 2, 4, 6, 10))
    whole_residual = whole_inertia-whole_gravity-xy+ground
    diagnostics = {}
    for name, values in [('F1_to_O1', b1), ('F2_to_O2', b2), ('R1_to_H', rear[:, 0]),
                         ('R2_to_H', rear[:, 1]), ('fit1_to_H', fit1), ('fit2_to_H', fit2),
                         ('rotor_to_holder', link), ('H_ground_restoring', ground), ('unbalance_to_rotor', xy)]:
        average = _mean(t, values, start, end)
        diagnostics[name] = dict(mean_X_N=float(average[0]), mean_Y_N=float(average[1]),
            peak_resultant_N=float(np.linalg.norm(values[main], axis=1).max()))
    rotor_terms = dict(inertial=average_inertia[model.ROTOR],
        gyro=_mean(t, gyro, start, end)[model.ROTOR],
        F1=model.section_matrix(p['z_f']).T@_mean(t, b1, start, end),
        F2=model.section_matrix(p['z_f2']).T@_mean(t, b2, start, end),
        R1=model.section_matrix(p['rear_supports'][0]['z_m']).T@_mean(t, rear[:, 0], start, end),
        R2=model.section_matrix(p['rear_supports'][1]['z_m']).T@_mean(t, rear[:, 1], start, end),
        holder=model.section_matrix(p['z_T']).T@_mean(t, link, start, end),
        gravity_and_static_load=gravity[model.ROTOR],
        actual_unbalance=_mean(t, generalized, start, end))
    for name, values in local_moments.items():
        rotor_terms[name+'_local_moment'] = np.r_[0., 0., _mean(t, values, start, end)]
    checks = dict(native_input_time_identity=np.array_equal(t, input_t),
        force_formula=formula_error <= 1e-10,
        force_generalization=generalized_error <= 1e-10,
        instantaneous_force_identity=force_max <= 1e-7,
        instantaneous_moment_identity=torque_max <= 1e-8,
        whole_body_translation=float(abs(whole_residual[main]).max()) <= 1e-7)
    report = dict(passed=all(checks.values()), checks=checks,
        window_s=[start, end], turns=(end-start)*abs(p['rpm'])/60,
        maximum_translational_residual_N=force_max, maximum_moment_residual_Nm=torque_max,
        time_weighted_mean_residual_mixed_SI=mean_residual.tolist(),
        mean_inertia_mixed_SI=average_inertia.tolist(), boundary_momentum_rate_mixed_SI=boundary_momentum_rate.tolist(),
        acceleration_quadrature_difference_mixed_SI=quadrature_difference.tolist(),
        maximum_whole_body_residual_N=float(abs(whole_residual[main]).max()),
        time_weighted_whole_body_inertia_N=_mean(t, whole_inertia, start, end).tolist(),
        whole_body_gravity_N=whole_gravity.tolist(),
        reaction_forces=diagnostics,
        time_weighted_rotor_budget={name: value.tolist() for name, value in rotor_terms.items()},
        rotor_budget_units=['N', 'N', 'N m', 'N m'],
        rotor_budget_equation='inertial+gyro+F1+F2+R1+R2+holder+sum(local_moments_if_enabled)=gravity_and_static_load+actual_unbalance',
        scope='Saved-state equation identity and whole-window load budget; not independent real-machine validation.')
    row = dict(gap_um=p['front_group_clearance_um'], F_N=config['excitation']['peak_force_N'],
        U_kg_m=config['excitation']['U_kg_m'], passed=report['passed'],
        force_residual_N=force_max, moment_residual_Nm=torque_max,
        **{f'{name}_mean_{axis}_N': info[f'mean_{axis}_N']
           for name, info in diagnostics.items() for axis in ('X', 'Y')})
    return report, row


def audit(root, group='data', output=None):
    root = Path(root)
    output = Path(output) if output else root/'validation'
    output.mkdir(parents=True, exist_ok=True)
    reports, rows = {}, []
    for path in sorted((root/group).rglob('metrics.json')):
        name = path.parent.relative_to(root).as_posix()
        report, row = audit_case(path.parent)
        reports[name] = report
        rows.append(dict(case=name, **row))
    result = dict(passed=bool(reports and all(r['passed'] for r in reports.values())),
                  cases=reports, averaging='Time weighted with interpolated exact revolution-window boundaries.')
    write_json(output/'dynamic_balance.json', result)
    if rows:
        write_csv(output/'dynamic_balance.csv', rows)
    print(json.dumps(dict(passed=result['passed'], cases=len(reports))), flush=True)
    return result


def audit_initial_condition(root, reference='data/gap_5um',
                            continued='initial_condition/from_c10/gap_5um',
                            seed='data/gap_10um'):
    """Compare response statistics at different absolute phase-preserving times."""
    from utils import clearance_v6_analysis as analysis
    root = Path(root)
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    cold, reverse = (analysis.load_case(root/path) for path in (reference, continued))
    comparison = analysis.compare_metrics(cold['metrics'], reverse['metrics'], cold['config'])
    budget_file = root/'recomputed/numerical_error_estimates.json'
    budgets = read(budget_file).get(reference.removeprefix('data/'), {}) if budget_file.exists() else {}
    rows = comparison['metrics']
    for key, item in rows.items():
        epsilon = budgets.get(key, {}).get('epsilon')
        item['cold_branch_epsilon'] = epsilon
        item['two_times_cold_branch_epsilon'] = None if epsilon is None else 2*epsilon
        item['within_two_cold_budgets'] = None if epsilon is None else item['difference'] <= 2*epsilon
    with np.load(root/continued/'initial_seed.npz') as initial, np.load(root/seed/'native.npz') as previous:
        exact_seed = np.array_equal(initial['state'], previous['state_final'])
        exact_epoch = float(initial['absolute_time_s']) == float(previous['integration_end_s'])
        epoch = float(initial['absolute_time_s'])
    configs_identical = cold['config']['parameters'] == reverse['config']['parameters']
    configs_identical = configs_identical and cold['config']['excitation'] == reverse['config']['excitation']
    configs_identical = configs_identical and cold['config']['processing'] == reverse['config']['processing']
    checks = dict(seed_state_exact=exact_seed, seed_epoch_exact=exact_epoch,
        physical_parameters_identical=configs_identical,
        force_phase_formula=cold['input_pass'] and reverse['input_pass'],
        both_steady=cold['steady']['passed'] and reverse['steady']['passed'],
        both_domain=cold['domain']['passed'] and reverse['domain']['passed'],
        solver=cold['solver_pass'] and reverse['solver_pass'],
        statistics_acceptance=all(row['passed'] for row in rows.values()))
    main_keys = analysis.MAIN_KEYS+analysis.CONTACT_KEYS
    budget_ready = all(rows[k]['within_two_cold_budgets'] is not None for k in main_keys)
    main_matches = all(rows[k]['within_two_cold_budgets'] for k in main_keys) if budget_ready else None
    certified_features = {key: row['within_two_cold_budgets'] for key, row in rows.items()
                          if key.startswith(('H_v_', 'H_a_'))}
    complete = budget_ready and all(value is not None for value in certified_features.values())
    phases = {key: float(np.angle(np.exp(1j*(reverse['metrics'][key]-cold['metrics'][key]))))
              for key in analysis.keys(cold['metrics']) if '_phase_' in key}
    result = dict(status='passed' if complete and main_matches and all(checks.values())
                  and all(certified_features.values()) else 'pending_budget' if not complete else 'different_or_unresolved',
        checks=checks, budget_complete=complete, main_and_contact_within_budget=main_matches,
        feature_within_budget=certified_features,
        cold_case=reference, continued_case=continued, seed_case=seed, seed_absolute_time_s=epoch,
        cold_window_s=[cold['metrics']['measurement_start_s'], cold['metrics']['measurement_end_s']],
        continued_window_s=[reverse['metrics']['measurement_start_s'], reverse['metrics']['measurement_end_s']],
        cold_E_um=cold['metrics']['E_dir_um'], continued_E_um=reverse['metrics']['E_dir_um'],
        E_difference_um=reverse['metrics']['E_dir_um']-cold['metrics']['E_dir_um'],
        phase_difference_rad=phases, metrics=rows,
        criterion='Statistics compared despite distinct absolute windows. Agreement uses twice the certified cold-branch numerical budget; continuation is not separately h/N certified.',
        conclusion_scope='A repeated branch supports local initial-state robustness; it does not rule out all other stable branches.')
    write_json(root/'validation/initial_condition.json', result)
    write_csv(root/'validation/initial_condition.csv', [dict(feature=key, **row) for key, row in rows.items()])
    print(json.dumps(dict(status=result['status'], E_difference_um=result['E_difference_um'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiments/clearance_v6_GMN'))
    parser.add_argument('--group', default='data')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--initial-condition', action='store_true')
    args = parser.parse_args()
    if args.initial_condition:
        audit_initial_condition(args.root)
    else:
        audit(args.root, args.group, args.output)
