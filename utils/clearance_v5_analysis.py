"""V5 layered acceptance and offline evidence, using the V4 signal metrics.

Error budgets are engineering estimates, not confidence intervals. Missing
independent refinement evidence stays unknown rather than contributing zero.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from utils.clearance_v4_analysis import analyze as analyze_v4, keys as metric_keys


MAIN_KEYS = ('E_x_um', 'E_y_um', 'E_dir_um')
CONTACT_KEYS = tuple(prefix + suffix for prefix in ('fit_', 'fit2_')
                     for suffix in ('contact_fraction', 'peak_N', 'impulse_N_s'))


def absolute_floor(key, config):
    t = config['numerical_thresholds']
    if '_phase_' in key:
        return None  # phases are retained diagnostics, without an invented limit
    if '_energy_' in key:
        return (t['acceleration_m_s2'] if key.startswith('H_a_') else t['velocity_mm_s'])**2
    if key.startswith('H_v_'):
        return t['velocity_mm_s']
    if key.startswith('H_a_'):
        return t['acceleration_m_s2']
    if 'contact_fraction' in key:
        return t['contact_rate']
    if 'impulse' in key:
        return t['contact_impulse_N_s']
    if key.startswith(('fit_', 'fit2_')):
        return t['contact_peak_N']
    return t['position_um']


def tolerance(key, base, fine, config):
    """The V5 max(floor, relative*max(abs(values))) rule, in each unit."""
    t = config['numerical_thresholds']
    if 'contact_fraction' in key:
        return t['contact_rate']
    if 'impulse' in key:
        relative = t['relative_contact_impulse']
    elif key.startswith(('fit_', 'fit2_')):
        relative = t['relative_contact_peak']
    elif any(part in key for part in ('_residual_', '_2x_', '_3x_', '_energy_')):
        relative = t.get('relative_residual_convergence', .05)
    else:
        relative = t['relative_main']
    return max(absolute_floor(key, config), relative * max(abs(base), abs(fine)))


def _layers(checks):
    selected = lambda names: bool(all(k in checks and checks[k]['passed'] for k in names))
    features = {k: v['passed'] for k, v in checks.items() if k.startswith(('H_v_', 'H_a_'))}
    return dict(main_metric_pass=selected(MAIN_KEYS), contact_pass=selected(CONTACT_KEYS),
                feature_pass=bool(features and all(features.values())), features=features)


def compare_metrics(base, fine, config):
    checks = {}
    for key in metric_keys(fine):
        if key not in base or absolute_floor(key, config) is None:
            continue
        delta = abs(base[key] - fine[key])
        limit = tolerance(key, base[key], fine[key], config)
        checks[key] = dict(base=base[key], fine=fine[key], difference=delta,
                           tolerance=limit, passed=bool(delta <= limit))
    same = all(abs(base[k] - fine[k]) < 1e-10
               for k in ('measurement_start_s', 'measurement_end_s'))
    layers = _layers(checks)
    return dict(passed=bool(same and checks and all(x['passed'] for x in checks.values())),
                same_window=same, metrics=checks, **layers)


def analyze(result, config, start, end, stride=1):
    """Retain V4 metrics/turns and apply V5 full-window acceptance limits."""
    metrics, turns, output, steady = analyze_v4(result, config, start, end, stride)
    checks = {}
    for key in metric_keys(metrics):
        if absolute_floor(key, config) is None:
            continue
        values = np.asarray([m[key] for m in steady['subwindow_metrics']])
        limit = tolerance(key, float(np.min(values)), float(np.max(values)), config)
        # A static center must not hide drift behind the gravity displacement.
        if key.startswith(('mean_', 'OH_mean_')):
            limit = max(config['numerical_thresholds']['position_um'],
                        config['numerical_thresholds']['relative_main']*metrics['E_dir_um'])
        checks[key] = dict(values=values.tolist(), difference=float(np.ptp(values)),
                           tolerance=limit, passed=bool(np.ptp(values) <= limit))
    layers = _layers(checks)
    expected = config['simulation']['retained_turns']
    complete = bool(expected >= 120 and expected % 4 == 0 and metrics['complete_turns'] == expected
                    and steady['subwindow_complete_turns'] == [expected//4]*4)
    drift_pass = all(item['passed'] for key, item in checks.items()
                     if key.startswith(('mean_', 'OH_mean_')))
    steady.update(checks=checks, **layers)
    steady['main_stationarity_pass'] = bool(complete and layers['main_metric_pass'] and drift_pass)
    steady['passed'] = bool(complete and all(v['passed'] for v in checks.values())
                            and all(v['passed'] for v in steady['band_drift_checks'].values()))
    metrics['steady_pass'] = steady['passed']
    metrics['force_N'] = config['excitation']['peak_force_N']
    metrics['U_kg_m'] = config['excitation']['U_kg_m']
    # Phase convention: A*cos(Omega*t+input_phase+reported_phase).
    time, hv, ha = output
    angle = config['parameters']['rpm']*2*np.pi/60*time + config['excitation']['phase_rad']
    design = np.column_stack((np.ones(len(angle)), np.cos(angle), np.sin(angle)))
    for signal, prefix in ((hv, 'H_v'), (ha, 'H_a')):
        coefficient = np.linalg.lstsq(design, signal, rcond=None)[0]
        for axis, value in zip('xy', np.arctan2(-coefficient[2], coefficient[1])):
            metrics[f'{prefix}_1x_phase_{axis}_rad'] = float(value)
    return metrics, turns, output, steady


def error_budget(metrics, steady, sampling, time_comparison, angular_comparison, config):
    output = {}
    for key in metric_keys(metrics):
        floor = absolute_floor(key, config)
        if floor is None:
            continue
        parts = {}
        for name, evidence in (('time', time_comparison), ('angular', angular_comparison),
                               ('sampling', sampling)):
            item = (evidence or {}).get('metrics', {}).get(key)
            usable = evidence and evidence.get('same_window', True)
            usable = usable and evidence.get('independent_refinement_applied', True)
            parts[name] = item['difference'] if item and usable else None
        item = (steady or {}).get('checks', {}).get(key)
        parts['subwindow'] = item['difference'] if item else None
        complete = all(value is not None for value in parts.values())
        output[key] = dict(epsilon=max(floor, sum(parts.values())) if complete else None,
                           absolute_floor=floor, complete=complete, components=parts)
    return output


def _read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _csv(path, rows):
    if not rows:
        return
    names = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def load_case(folder, replay=False):
    """Read a completed case; optional replay rebuilds signal metrics from native data."""
    folder = Path(folder)
    config = _read(folder/'actual_config.json')
    metrics = _read(folder/'metrics.json')
    steady = _read(folder/'stationarity.json', {})
    sampling = _read(folder/'sampling_convergence.json', {})
    replay_difference = None
    if replay:
        names = dict(t='t_raw', q='q_raw_m', v='v_raw_m_s', acc='a_raw_m_s2',
                     measured_relative_xy_m='r_m_raw_m', fit_force_N='fit_force_N',
                     fit_active_count='fit_active_count', fit2_force_N='fit2_force_N',
                     fit2_active_count='fit2_active_count')
        with np.load(folder/'native.npz') as data:
            result = {key: data[name] for key, name in names.items()}
            meta = json.loads(str(data['metadata_json']))
        rebuilt, _, _, steady = analyze(result, config, meta['export_start_s'], meta['export_end_s'])
        coarse, _, _, _ = analyze(result, config, meta['export_start_s'], meta['export_end_s'], stride=2)
        sampling = compare_metrics(coarse, rebuilt, config)
        replay_difference = max(abs(rebuilt[k]-metrics[k]) for k in metric_keys(rebuilt)
                                if k in metrics and '_phase_' not in k)
        metrics = dict(metrics, **rebuilt)
    p, s, e = config['parameters'], config['simulation'], config['excitation']
    metrics.update(force_N=e['peak_force_N'], U_kg_m=e['U_kg_m'], gap_um=p['front_group_clearance_um'])
    omega = p['rpm']*2*np.pi/60
    aq = np.full(12, s['atol_translation_m']); aq[8:10] = s['atol_angle_rad']
    av = np.full(12, s['atol_velocity_m_s']); av[8:10] = s['atol_angle_rate_rad_s']
    solver = _read(folder/'actual_solver_settings.json', [])
    checks = [item['method'] == s['method'] and item['rtol'] == s['rtol']
              and item.get('rhs_backend', 'python') == p.get('rhs_backend', 'python')
              and item['max_step_s'] == s['max_step_s']
              and item['fit_direction_count'] == p['N']
              and item['output_fs_hz'] == s['fs_raw_hz']
              and np.array_equal(item['atol'], np.r_[aq, av])
              and item['accepted_steps']['max_s'] <= s['max_step_s']*(1+1e-8)
              for item in solver]
    inputs = dict(force_U_consistent=bool(abs(e['U_kg_m']*omega**2-e['peak_force_N'])
                                         <= 1e-10*max(1, abs(e['peak_force_N']))),
                  common_gap=bool(p['fit_clearance_um'] == p['fit2_clearance_um'] == p['front_group_clearance_um']),
                  gap_range=bool(0 <= p['front_group_clearance_um'] <= 10),
                  dof_12=p['model_dof'] == 12)
    drive_path = folder/'actual_input.npz'
    saved_drive = _read(folder/'input_check.json', {})
    inputs['saved_input_completed'] = saved_drive.get('status') == 'completed'
    drive_errors = None
    if drive_path.exists():
        with np.load(drive_path) as drive:
            time = drive['t_raw']; angle = drive['angle_rad']
            xy = drive['force_xy_N']; generalized = drive['generalized_rotor_force']
        expected_angle = omega*time+e['phase_rad']
        expected_xy = e['U_kg_m']*omega**2*np.column_stack((np.cos(expected_angle), np.sin(expected_angle)))
        expected_q = np.column_stack((xy, -p['z_u']*xy[:, 1], p['z_u']*xy[:, 0]))
        drive_errors = dict(norm_N=float(np.max(abs(np.linalg.norm(xy, axis=1)-e['peak_force_N']))),
                            force_formula_N=float(np.max(abs(xy-expected_xy))),
                            force_generalization_N=float(np.max(abs(generalized[:, :2]-expected_q[:, :2]))),
                            torque_Nm=float(np.max(abs(generalized[:, 2:]-expected_q[:, 2:]))),
                            angle_rad=float(np.max(abs(angle-expected_angle))))
        force_limit = 1e-10*abs(e['peak_force_N'])
        torque_limit = force_limit*max(abs(p['z_u']), 1e-15)
        inputs.update(actual_force_norm=drive_errors['norm_N'] <= force_limit,
                      actual_force_formula=drive_errors['force_formula_N'] <= force_limit,
                      actual_force_projection=drive_errors['force_generalization_N'] <= force_limit,
                      actual_force_moment=drive_errors['torque_Nm'] <= torque_limit,
                      actual_signed_angle=drive_errors['angle_rad'] <= 1e-10,
                      generalized_indices=saved_drive.get('generalized_rotor_indices') == [0, 1, 8, 9],
                      saved_force_consistent=bool(np.isclose(saved_drive.get('force_N', np.nan),
                          e['peak_force_N'], rtol=1e-10, atol=0)))
    else:
        inputs['actual_input_available'] = False
    return dict(folder=str(folder), config=config, metrics=metrics, steady=steady,
                sampling=sampling, domain=_read(folder/'domain.json', {}),
                solver_pass=bool(checks and all(checks)), input_checks=inputs,
                input_pass=bool(all(inputs.values())),
                actual_input_errors=drive_errors,
                replay_max_difference=replay_difference,
                actual_max_steps_s=[item['accepted_steps']['max_s'] for item in solver])


def _refinement(base, fine, label):
    a, b = base['config'], fine['config']
    p, fp = a['parameters'], b['parameters']
    changed = {k for k in set(p)|set(fp) if p.get(k) != fp.get(k)}
    s, fs = a['simulation'], b['simulation']
    simulation_changes = {k for k in set(s)|set(fs) if s.get(k) != fs.get(k)}
    # Window alignment is checked independently. Warmup may reflect a saved continuation.
    simulation_changes -= {'warmup_s', 'workers'}
    if label == 'h_half':
        correct = not changed and simulation_changes <= {'max_step_s'}
        correct = correct and fs['max_step_s'] == s['max_step_s']/2
        correct = correct and bool(base['actual_max_steps_s'] and fine['actual_max_steps_s'])
        correct = correct and max(fine['actual_max_steps_s']) < .75*max(base['actual_max_steps_s'])
    else:
        correct = changed == {'N'} and fp['N'] == p['N']*2 and not simulation_changes
    correct = bool(correct and a['excitation'] == b['excitation'] and a['processing'] == b['processing'])
    compared = compare_metrics(base['metrics'], fine['metrics'], a)
    compared['independent_refinement_applied'] = correct
    compared['fine_solver_pass'] = fine['solver_pass']
    compared['fine_input_pass'] = fine['input_pass']
    compared['fine_domain_pass'] = fine['domain'].get('passed', False)
    compared['fine_steady_pass'] = fine['steady'].get('passed', False)
    compared['parameter_changes'] = sorted(changed)
    compared['simulation_changes'] = sorted(simulation_changes)
    compared['passed'] = bool(compared['passed'] and correct and fine['solver_pass'])
    return compared


def _status(items):
    if any(value is False for value in items):
        return 'fail'
    return 'not_evaluated' if any(value is None for value in items) else 'pass'


def _check(evidence, key):
    return (evidence or {}).get('metrics', {}).get(key, {}).get('passed')


def _steady_check(steady, key):
    return (steady or {}).get('checks', {}).get(key, {}).get('passed')


def _physical_differences(config, baseline):
    a, b = config['parameters'], baseline['parameters']
    allowed = {'front_group_clearance_um', 'fit_clearance_um', 'fit2_clearance_um'}
    changed = [key for key in set(a)|set(b) if key not in allowed and a.get(key) != b.get(key)]
    if config['excitation'] != baseline['excitation']:
        changed.append('excitation')
    if config['processing'] != baseline['processing']:
        changed.append('processing')
    return sorted(changed)


def _coverage(records):
    healthy = next((r for r in records if r['metrics']['gap_um'] == 0), None)
    valid = [r for r in records if r['main_valid']]
    evidence = []
    for label, low, high in (('healthy_le_2', None, 2), ('level_5', 4, 6),
                              ('level_10', 9, 11), ('level_15', 14, 16),
                              ('near_20', 20, 22)):
        cases = ([healthy] if healthy else []) if label == 'healthy_le_2' else valid
        names = []
        for case in cases:
            if not case or not case['main_valid']:
                continue
            e = case['metrics']['E_dir_um']; eps = case['epsilon']['E_dir_um']['epsilon']
            gap_resolvable = True
            if label != 'healthy_le_2':
                gap_resolvable = bool(healthy and healthy['main_valid'] and
                    abs(e-healthy['metrics']['E_dir_um']) >
                    5*(eps+healthy['epsilon']['E_dir_um']['epsilon']))
            if (low is None or e-eps >= low) and e+eps <= high and gap_resolvable:
                names.append(case['case'])
        evidence.append(dict(level=label, interval_um=[low, high], cases=names, passed=bool(names)))
    endpoint = [r['case'] for r in valid if r['metrics']['E_dir_um']-r['epsilon']['E_dir_um']['epsilon'] >= 20]
    maximum = max(valid, key=lambda r:r['metrics']['E_dir_um']) if valid else None
    return dict(endpoint_20um_pass=bool(endpoint), endpoint_cases=endpoint,
                full_range_pass=all(item['passed'] for item in evidence), bin_evidence=evidence,
                healthy_E_um=healthy['metrics']['E_dir_um'] if healthy else None,
                healthy_epsilon_um=healthy['epsilon']['E_dir_um']['epsilon'] if healthy else None,
                maximum_valid_E_um=maximum['metrics']['E_dir_um'] if maximum else None,
                maximum_valid_epsilon_um=maximum['epsilon']['E_dir_um']['epsilon'] if maximum else None,
                maximum_valid_gap_um=maximum['metrics']['gap_um'] if maximum else None,
                maximum_valid_case=maximum['case'] if maximum else None)


def assemble(root, output=None, base_group='data', integration_group='certification/h_half',
             angular_group='certification/n512', selected_force_N=None, replay=False):
    """Assemble per-force results without treating missing certification as zero error."""
    root = Path(root); output = Path(output) if output else root/'recomputed'
    output.mkdir(parents=True, exist_ok=True)
    data = {path.parent.relative_to(root/base_group).as_posix(): load_case(path.parent, replay)
            for path in sorted((root/base_group).glob('F_*N/gap_*/metrics.json'))}
    refined = {label:{name:load_case(root/group/name, replay) for name in data
                      if (root/group/name/'metrics.json').exists()}
               for label, group in (('h_half', integration_group), ('N_double', angular_group))}
    records = []; comparisons = {}; budgets = {}; summaries = []
    for name, case in data.items():
        m, c, steady = case['metrics'], case['config'], case['steady']
        group = name.split('/')[0]
        health = data.get(group+'/gap_0um')
        physics_changes = _physical_differences(c, health['config']) if health else ['missing_same_force_health']
        comp = {label:_refinement(case, values[name], label) for label, values in refined.items() if name in values}
        comparisons[name] = comp
        eps = error_budget(m, steady, case['sampling'], comp.get('h_half'), comp.get('N_double'), c)
        budgets[name] = eps
        common = [case['solver_pass'], case['input_pass'], not physics_changes]
        for label in ('h_half', 'N_double'):
            comparison = comp.get(label)
            common.append(None if comparison is None else bool(comparison['independent_refinement_applied']
                          and comparison['same_window'] and comparison['fine_solver_pass']
                          and comparison['fine_input_pass']))
            fine = refined[label].get(name)
            common.append(fine['steady'].get('main_stationarity_pass', fine['steady'].get('passed'))
                          if fine else None)
        main_items = list(common); contact_items = list(common)
        for key in MAIN_KEYS+CONTACT_KEYS:
            items = main_items if key in MAIN_KEYS else contact_items
            items.extend((_check(case['sampling'], key), _steady_check(steady, key)))
            for label in ('h_half', 'N_double'):
                items.append(_check(comp.get(label), key))
                fine = refined[label].get(name)
                items.append(_steady_check(fine['steady'], key) if fine else None)
        main_items.append(steady.get('main_stationarity_pass', steady.get('passed')))
        main_status, contact_status = _status(main_items), _status(contact_items)
        main_valid = bool(main_status == contact_status == 'pass' and case['domain'].get('passed', False)
                          and all(fine[name]['domain'].get('passed', False) for fine in refined.values() if name in fine)
                          and eps['E_dir_um']['complete'])
        feature_status = {}
        for key in metric_keys(m):
            if not key.startswith(('H_v_', 'H_a_')) or absolute_floor(key, c) is None:
                continue
            checks = common + [_check(case['sampling'], key), _steady_check(steady, key)]
            for label in ('h_half', 'N_double'):
                checks.append(_check(comp.get(label), key))
                fine = refined[label].get(name)
                checks.append(_steady_check(fine['steady'], key) if fine else None)
            feature_status[key] = _status(checks)
        record = dict(case=name, metrics=m, epsilon=eps, main_metric_status=main_status,
                      contact_status=contact_status, feature_status=feature_status,
                      main_valid=main_valid, domain_pass=case['domain'].get('passed', False),
                      full_waveform_pass=bool(main_valid and steady.get('passed') and
                                             all(value=='pass' for value in feature_status.values()) and
                                             case['sampling'].get('passed') and
                                             all(item['passed'] for item in comp.values())),
                      physics_freeze_pass=not physics_changes, physics_changes=physics_changes,
                      input_checks=case['input_checks'], solver_pass=case['solver_pass'],
                      actual_input_errors=case['actual_input_errors'],
                      domain=case['domain'], source=case['folder'],
                      replay_max_difference=case['replay_max_difference'])
        records.append(record)
        summaries.append(dict(m, case=name, F_N=m['force_N'], U_g_mm=m['U_kg_m']*1e6,
                              epsilon_E_um=eps['E_dir_um']['epsilon'],
                              **{'epsilon_'+key:budget['epsilon'] for key,budget in eps.items()
                                 if key.startswith('H_v_')},
                              **{'domain_'+key:value for key,value in case['domain'].get('values', {}).items()},
                              main_metric_status=main_status, contact_status=contact_status,
                              main_valid=main_valid, domain_pass=record['domain_pass'],
                              full_waveform_pass=record['full_waveform_pass']))
    pairs = []; force_health = []
    grouped = {}
    for record in records:
        grouped.setdefault(record['metrics']['force_N'], []).append(record)
    for force, members in grouped.items():
        health = next((r for r in members if r['metrics']['gap_um']==0), None)
        if not health:
            continue
        for member in members:
            if member is health:
                continue
            for key in MAIN_KEYS + tuple(k for k in health['feature_status'] if k.startswith('H_v_')):
                a, b = health['metrics'][key], member['metrics'][key]
                e0, ec = health['epsilon'][key]['epsilon'], member['epsilon'][key]['epsilon']
                combined = e0+ec if e0 is not None and ec is not None else None
                certified = health['main_valid'] and member['main_valid']
                if key not in MAIN_KEYS:
                    certified = certified and health['feature_status'][key] == member['feature_status'][key] == 'pass'
                distinguishable = bool(certified and abs(b-a)>5*combined) if combined is not None else None
                pairs.append(dict(force_N=force, case=member['case'], gap_um=member['metrics']['gap_um'],
                                  feature=key, healthy=a, degraded=b, difference=b-a,
                                  percent_change=100*(b-a)/abs(a) if e0 is not None and abs(a)>5*e0 else None,
                                  epsilon_healthy=e0, epsilon_gap=ec, epsilon_sum=combined,
                                  certified=certified, numerically_resolvable=distinguishable))
    health_cases = sorted((r for r in records if r['metrics']['gap_um']==0), key=lambda r:r['metrics']['force_N'])
    regression_health = [load_case(path.parent, replay)
                         for path in sorted((root/'regression').glob('F_*N/gap_0um/metrics.json'))]
    original_regression = min(regression_health, key=lambda r:r['metrics']['force_N'], default=None)
    if health_cases and original_regression:
        original = dict(metrics=original_regression['metrics'],
                        epsilon=error_budget(original_regression['metrics'], original_regression['steady'],
                            original_regression['sampling'], None, None, original_regression['config']),
                        feature_status={key: 'not_evaluated' for key in metric_keys(original_regression['metrics'])
                                        if key.startswith(('H_v_', 'H_a_')) and
                                        absolute_floor(key, original_regression['config']) is not None})
        for member in health_cases:
            for key in MAIN_KEYS+tuple(k for k in original['feature_status'] if k.startswith('H_v_')):
                a, b = original['metrics'][key], member['metrics'][key]
                e0, ec = original['epsilon'][key]['epsilon'], member['epsilon'][key]['epsilon']
                force_health.append(dict(original_force_N=original['metrics']['force_N'],
                                         force_N=member['metrics']['force_N'], feature=key,
                                         original_healthy=a, increased_force_healthy=b, difference=b-a,
                                         percent_change=100*(b-a)/abs(a) if e0 is not None and abs(a)>5*e0 else None,
                                         epsilon_sum=e0+ec if e0 is not None and ec is not None else None))
    coverage = {f'{force:g}N':_coverage(members) for force, members in grouped.items()}
    selected = next((value for force, value in coverage.items()
                     if np.isclose(float(force[:-1]), selected_force_N, rtol=1e-10, atol=1e-10)), None) \
                     if selected_force_N is not None else None
    maximum = max((r for r in records if r['main_valid']), key=lambda r:r['metrics']['E_dir_um'], default=None)
    candidate = max(records, key=lambda r:r['metrics']['E_dir_um'], default=None)
    acceptance = dict(schema_version='v5', selected_common_force_N=selected_force_N,
                      selected_common_U_kg_m=next((r['metrics']['U_kg_m'] for r in records
                          if selected_force_N is not None and np.isclose(r['metrics']['force_N'],
                              selected_force_N, rtol=1e-10, atol=1e-10)), None),
                      endpoint_20um_pass=any(c['endpoint_20um_pass'] for c in coverage.values()),
                      single_force_full_range_pass=selected['full_range_pass'] if selected else any(c['full_range_pass'] for c in coverage.values()),
                      selected_force_coverage=selected, coverage_by_force=coverage,
                      maximum_valid_E_um=maximum['metrics']['E_dir_um'] if maximum else None,
                      maximum_valid_case=maximum['case'] if maximum else None,
                      maximum_computed_candidate_E_um=candidate['metrics']['E_dir_um'] if candidate else None,
                      same_force_vibration_difference_pass=any(p['numerically_resolvable'] is True and p['feature'].startswith('H_v_') for p in pairs),
                      real_machine_parameter_support='not_validated', engineering_status='assumed_demo',
                      cases=records, note='Missing comparisons have null error budgets. Different forces are never merged into one gap curve.')
    _csv(output/'summary.csv', summaries)
    _csv(output/'same_force_differences.csv', pairs)
    _csv(output/'healthy_force_changes.csv', force_health)
    _write(output/'acceptance.json', acceptance)
    _write(output/'convergence.json', comparisons)
    _write(output/'numerical_error_estimates.json', budgets)
    details = [dict(case=name, refinement=label, feature=key,
                    independent_refinement_applied=item['independent_refinement_applied'],
                    same_window=item['same_window'], **values)
               for name, groups in comparisons.items() for label,item in groups.items()
               for key,values in item['metrics'].items()]
    _csv(output/'convergence_details.csv', details)
    return dict(records=records, acceptance=acceptance, epsilon=budgets,
                comparisons=comparisons, pairs=pairs, healthy_force_changes=force_health)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--base-group', default='data')
    parser.add_argument('--integration-group', default='certification/h_half')
    parser.add_argument('--angular-group', default='certification/n512')
    parser.add_argument('--selected-force-N', type=float)
    parser.add_argument('--replay', action='store_true', help='Recompute from native arrays, without ODE integration')
    args = parser.parse_args()
    result = assemble(args.root, args.output, args.base_group, args.integration_group,
                      args.angular_group, args.selected_force_N, args.replay)
    text = {key:value for key,value in result['acceptance'].items() if key != 'cases'}
    print(json.dumps(text, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
