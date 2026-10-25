"""Aggregate saved V8 evidence without running new dynamics simulations."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np


REQUIRED_GAPS = (2.0, 3.0, 5.0, 7.5, 10.0)
A_GAPS = (2.0, 5.0, 10.0)
DEFAULT_ROOT = Path('experiments/clearance_v8_tilt_contact')
DEFAULT_TEMPLATE = Path('GPTOrder/spindle_v8_tilt_contact_plan/templates/acceptance_result.template.json')


def _read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def _csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    fields = list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _gap_name(gap):
    return f'gap_{gap:g}um'


def _rel(root, path):
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _status(present, passed=None):
    if not present:
        return 'not_run'
    if passed is None:
        return 'not_run'
    return 'pass' if bool(passed) else 'fail'


def _check(status, evidence=(), max_error=None, tolerance=None, reason=''):
    return dict(status=status, evidence=list(evidence), max_error=max_error,
                tolerance=tolerance, reason=reason)


def _flatten(value, prefix=''):
    leaves = {}
    if isinstance(value, dict):
        for key, child in value.items():
            leaves.update(_flatten(child, f'{prefix}.{key}' if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            leaves.update(_flatten(child, f'{prefix}[{index}]'))
    else:
        leaves[prefix] = value
    return leaves


def _config_differences(left, right):
    a, b = copy.deepcopy(left), copy.deepcopy(right)
    for config in (a, b):
        config.get('parameters', {}).pop('bearing_contact_model', None)
    aa, bb = _flatten(a), _flatten(b)
    return [key for key in sorted(set(aa) | set(bb)) if aa.get(key) != bb.get(key)]


def _folder_case(folder):
    folder = Path(folder)
    metrics = _read(folder/'metrics.json')
    config = _read(folder/'actual_config.json')
    if metrics is None or config is None:
        return None
    return dict(folder=folder, metrics=metrics, config=config)


def _max_tilt(folder):
    path = Path(folder)/'native.npz'
    if not path.exists():
        return None
    with np.load(path) as data:
        q = data['q_raw_m']
        time = data['t_raw']
        if 'metadata_json' in data:
            meta = json.loads(str(data['metadata_json']))
            use = (time >= meta.get('export_start_s', time[0])-1e-12) & (time < meta.get('export_end_s', time[-1])+1e-12)
            if use.any():
                q = q[use]
        return float(np.linalg.norm(q[:, 8:10], axis=1).max())


def _dynamic_balance_index(root):
    result = {}
    for path in Path(root).rglob('dynamic_balance.json'):
        data = _read(path, {})
        for case, row in data.get('cases', {}).items():
            key = str(Path(case)).replace('\\', '/').strip('/')
            result[key] = row
        if data.get('scope') and 'passed' in data:
            result[_rel(root, path.parent)] = data
    return result


def _balance_for(folder, root, index):
    local = _read(Path(folder)/'dynamic_balance.json')
    if local is not None:
        return local
    rel = _rel(root, folder)
    for key, row in index.items():
        if key == rel or key.endswith('/'+rel) or rel.endswith('/'+key):
            return row
    return None


def _saved_balance_status(a_new, b_cases):
    a_rows = {case.get('case'): case.get('dynamic_balance') for case in a_new.values()}
    required_a = [a_rows.get(_gap_name(gap)) for gap in A_GAPS]
    required_b = [case.get('dynamic_balance') for case in b_cases]
    rows = required_a+required_b
    if any(row and row.get('passed') is False for row in rows):
        return 'fail', rows
    if len(required_a) != len(A_GAPS) or len(required_b) != len(REQUIRED_GAPS) \
            or any(row is None or row.get('passed') is None for row in rows):
        return 'not_run', rows
    return ('pass' if all(row.get('passed') is True for row in rows) else 'fail'), rows


def _contact_wrench_csv(root, records):
    columns = ('radial_N', 'local_Nm', 'lever_Nm')
    case_fields = ('gap_um', 'U_kg_m', 'E_x_um', 'E_y_um', 'E_dir_um',
        'E_x_min_um', 'E_x_max_um', 'E_y_min_um', 'E_y_max_um',
        'mean_x_um', 'mean_y_um', 'OH_mean_x_um', 'OH_mean_y_um',
        'OH2_mean_x_um', 'OH2_mean_y_um', 'H_v_rms_x_mm_s', 'H_v_rms_y_mm_s',
        'H_v_residual_x_mm_s', 'H_v_residual_y_mm_s', 'fit_contact_fraction',
        'fit2_contact_fraction', 'steady_pass', 'domain_pass', 'sampling_pass')
    b_records = {}
    for record in records:
        source = Path(record.get('source', ''))
        if not source.is_absolute() and not source.exists():
            source = Path(root)/source
        b_records[source.resolve()] = record
    rows = []
    for metrics_path in sorted(Path(root).rglob('metrics.json')):
        folder = metrics_path.parent
        if 'source_before' in folder.parts or 'recomputed' in folder.parts:
            continue
        metrics = _read(metrics_path, {})
        config = _read(folder/'actual_config.json', {})
        wrench = _read(folder/'wrench_diagnostics.json', {})
        diagnostics = _read(folder/'bearing_diagnostics.json', {})
        if not wrench or not diagnostics:
            continue
        full = b_records.get(folder.resolve(), {})
        excitation = config.get('excitation', {})
        base = dict(group=folder.parent.relative_to(root).as_posix(),
            case=folder.name, case_path=_rel(root, folder),
            U_kg_m=excitation.get('U_kg_m', metrics.get('U_kg_m')),
            theta_peak_rad=_max_tilt(folder),
            main_valid=full.get('main_valid'),
            full_waveform_pass=full.get('full_waveform_pass'))
        base.update({key: metrics.get(key) for key in case_fields})
        raw_names = ('native.npz','turns.csv','metrics.json','actual_config.json',
            'bearing_contacts.npz','bearing_diagnostics.json','wrench_diagnostics.json',
            'constraint_reactions.npz')
        base.update({'raw_'+name.replace('.', '_')+'_path':
            (_rel(root, folder/name) if (folder/name).exists() else None) for name in raw_names})
        h_constraint_rms = wrench.get('H_bearing_load_constraint_rms_Nm', (None, None))
        h_constraint_peak = wrench.get('H_bearing_load_constraint_peak_Nm', (None, None))
        base.update(H_bearing_load_constraint_rms_x_Nm=None,
            H_bearing_load_constraint_rms_y_Nm=None,
            H_bearing_load_constraint_peak_x_Nm=None,
            H_bearing_load_constraint_peak_y_Nm=None,
            constraint_scope=None)
        for name in ('F1','F2','R1','R2'):
            values = (wrench.get('bearings') or {}).get(name, {})
            diag = diagnostics.get(name, {})
            row = dict(base, bearing=name)
            for quantity in columns:
                stat = values.get(quantity, {})
                for label in ('mean','rms','peak','one_x_amplitude','one_x_phase_rad'):
                    pair = stat.get(label, (None,None))
                    row[f'{quantity}_{label}_x'] = pair[0]
                    row[f'{quantity}_{label}_y'] = pair[1]
            axial = values.get('axial_restoring_N', {})
            row.update(axial_restoring_mean_N=axial.get('mean'),
                axial_restoring_min_N=axial.get('minimum'),
                axial_restoring_max_N=axial.get('maximum'),
                axial_constraint_reaction_mean_N=(-float(axial['mean']) if axial.get('mean') is not None else None),
                outer_rotation_constraint_rms_x_Nm=(values.get('local_Nm', {}).get('rms', [None, None])[0]),
                outer_rotation_constraint_rms_y_Nm=(values.get('local_Nm', {}).get('rms', [None, None])[1]),
                outer_rotation_constraint_peak_x_Nm=(values.get('local_Nm', {}).get('peak', [None, None])[0]),
                outer_rotation_constraint_peak_y_Nm=(values.get('local_Nm', {}).get('peak', [None, None])[1]),
                outer_rotation_constraint_mean_x_Nm=(-values.get('local_Nm', {}).get('mean', [None, None])[0]
                    if values.get('local_Nm', {}).get('mean', [None, None])[0] is not None else None),
                outer_rotation_constraint_mean_y_Nm=(-values.get('local_Nm', {}).get('mean', [None, None])[1]
                    if values.get('local_Nm', {}).get('mean', [None, None])[1] is not None else None),
                ball_peak_N=metrics.get(f'bearing_{name}_ball_peak_N'),
                contact_fraction=metrics.get(f'bearing_{name}_contact_fraction'),
                Q_min_N=diag.get('Q_min_N'), Q_max_N=diag.get('Q_max_N'),
                eta_tilt_max=diag.get('eta_tilt_max'),
                eta_tilt_above_half=diag.get('eta_tilt_above_half'),
                delta_over_diameter_max=diag.get('delta_over_diameter_max'),
                delta_over_overlap_max=diag.get('delta_over_overlap_max'),
                maximum_actual_tilt_compression_over_overlap=diag.get('maximum_actual_tilt_compression_over_overlap'),
                outer_tilt_constraint=diag.get('outer_tilt_constraint'))
            rows.append(row)
        rows.append(dict(base, bearing='H_aggregate_constraint',
            H_bearing_load_constraint_rms_x_Nm=(h_constraint_rms[0] if h_constraint_rms else None),
            H_bearing_load_constraint_rms_y_Nm=(h_constraint_rms[1] if h_constraint_rms else None),
            H_bearing_load_constraint_peak_x_Nm=(h_constraint_peak[0] if h_constraint_peak else None),
            H_bearing_load_constraint_peak_y_Nm=(h_constraint_peak[1] if h_constraint_peak else None),
            constraint_scope='H reference-point aggregate; do not sum with per-bearing outer_rotation_constraint_Nm'))
    output = Path(root)/'reports'/'case_wrench_contact_tilt.csv'
    _write_csv(output, rows)
    return _rel(root, output), len(rows)


def _dynamic_case_inventory(root):
    root = Path(root)
    excluded = {'source_before', 'recomputed'}
    completed, incomplete = [], []
    for path in sorted(root.rglob('metrics.json')):
        folder = path.parent
        if excluded.intersection(folder.parts):
            continue
        if (folder/'native.npz').exists() and (folder/'actual_solver_settings.json').exists():
            completed.append(_rel(root, folder))
    for path in sorted(root.rglob('progress.json')):
        folder = path.parent
        if excluded.intersection(folder.parts):
            continue
        if not (folder/'metrics.json').exists() or not (folder/'native.npz').exists():
            incomplete.append(_rel(root, folder))

    attempts, failed, unknown, extended = 0, [], [], []
    for path in sorted(root.rglob('warmup_history.json')):
        if excluded.intersection(path.parts):
            continue
        history = _read(path, [])
        for index, item in enumerate(history):
            attempts += 1
            stationarity = item.get('stationarity', {})
            folder = path.parent
            row = dict(path=_rel(root, path), case_path=_rel(root, folder), attempt=index+1,
                warmup_s=item.get('warmup_s'), start_s=item.get('start_s'),
                end_s=item.get('end_s'), stationarity_pass=stationarity.get('passed'),
                native_path=(_rel(root, folder/'native.npz') if (folder/'native.npz').exists() else None),
                solver_settings_path=(_rel(root, folder/'actual_solver_settings.json')
                    if (folder/'actual_solver_settings.json').exists() else None))
            if stationarity.get('passed') is False:
                failed.append(row)
                accepted = next((later for later in history[index+1:]
                    if later.get('stationarity', {}).get('passed') is True
                    and float(later.get('warmup_s') or 0) > float(item.get('warmup_s') or 0)), None)
                if accepted:
                    extended.append(dict(**row, extended_to_warmup_s=accepted.get('warmup_s')))
            elif stationarity.get('passed') is not True:
                unknown.append(row)
    by_group = {}
    for case in completed:
        group = Path(case).parent.as_posix()
        by_group[group] = by_group.get(group, 0)+1
    log_paths = sorted(_rel(root, path) for path in (root/'logs').glob('*.log')) if (root/'logs').exists() else []
    return dict(completed_dynamic_case_result_directories=len(completed), completed_case_paths=completed,
        completed_dynamic_case_directories_by_group=by_group,
        started_but_incomplete_dynamic_case_directory_count=len(incomplete),
        started_but_incomplete_dynamic_case_directories=incomplete,
        warmup_stationarity_attempts=attempts,
        warmup_stationarity_pass_count=attempts-len(failed)-len(unknown),
        rejected_warmup_stationarity_count=len(failed), rejected_warmup_attempts=failed,
        extended_after_rejected_warmup_count=len(extended), extended_warmup_cases=extended,
        warmup_stationarity_unknown_count=len(unknown), warmup_stationarity_unknown_attempts=unknown,
        orchestration_log_paths=log_paths,
        counting_rule='One completed dynamic case result directory per case folder containing metrics.json, native.npz and actual_solver_settings.json. This is not a solver-trajectory count: warmup extensions can reuse prior state, and each warmup stationarity check is counted separately. P1 3 ms short snapshot regressions are listed separately, not included in these case directories. Started folders without metrics/native are not counted as completed.')


def _case_summary(root, folder, group, name, balance_index, record=None):
    loaded = _folder_case(folder)
    if loaded is None:
        return None
    metrics, config = loaded['metrics'], loaded['config']
    input_pass = sampling_identity_pass = None
    solver_pass = record.get('solver_pass') if record else None
    if record is not None:
        input_pass = record.get('input_pass')
        sampling_identity_pass = record.get('sampling_identity_pass')
    else:
        from utils.clearance_v6_analysis import load_case
        audited = load_case(folder, replay=False)
        solver_pass = audited.get('solver_pass')
        input_pass = audited.get('input_pass')
        sampling_identity_pass = audited.get('sampling_identity_pass')
    p = config.get('parameters', {})
    excitation = config.get('excitation', {})
    wrench = _read(Path(folder)/'wrench_diagnostics.json')
    bearing_diagnostics = _read(Path(folder)/'bearing_diagnostics.json')
    if wrench:
        for bearing in wrench.get('bearings', {}).values():
            axial = bearing.get('axial_restoring_N', {})
            if 'mean' in axial:
                bearing['axial_constraint_reaction_N'] = -float(axial['mean'])
    eps = record.get('epsilon', {}) if record else {}
    return dict(group=group, case=name, path=_rel(root, folder),
        layout=p.get('bearing_layout_name'), contact_model=p.get('bearing_contact_model'),
        gap_um=metrics.get('gap_um', p.get('front_group_clearance_um')),
        U_kg_m=excitation.get('U_kg_m', metrics.get('U_kg_m')),
        force_N=excitation.get('peak_force_N', metrics.get('force_N')),
        metrics=metrics, epsilon=eps,
        main_valid=record.get('main_valid') if record else None,
        full_waveform_pass=record.get('full_waveform_pass') if record else None,
        steady_pass=metrics.get('steady_pass'), domain_pass=metrics.get('domain_pass'),
        sampling_pass=metrics.get('sampling_pass'),
        solver_pass=solver_pass, input_pass=input_pass,
        sampling_identity_pass=sampling_identity_pass,
        theta_peak_rad=_max_tilt(folder), wrench=wrench,
        bearing_diagnostics=bearing_diagnostics,
        raw_files={name: (_rel(root, Path(folder)/name) if (Path(folder)/name).exists() else None)
            for name in ('native.npz','turns.csv','metrics.json','actual_config.json',
                'bearing_contacts.npz','bearing_diagnostics.json','wrench_diagnostics.json',
                'constraint_reactions.npz')},
        dynamic_balance=_balance_for(folder, root, balance_index),
        retained_turns=config.get('simulation', {}).get('retained_turns'),
        warmup_s=config.get('simulation', {}).get('warmup_s'),
        actual_config=_rel(root, Path(folder)/'actual_config.json'))


def _p1_map(p1):
    return {row.get('name'): row for row in (p1 or {}).get('checks', [])}


def _p1_status(p1map, *names):
    rows = [p1map.get(name) for name in names]
    if any(row is None for row in rows):
        return 'not_run'
    return 'pass' if all(row.get('passed') for row in rows) else 'fail'


def _run_pairs(root, group, model_names, balance_index):
    pairs = {}
    for mode in model_names:
        for gap in A_GAPS:
            folder = Path(root)/group/mode/_gap_name(gap)
            case = _case_summary(root, folder, f'{group}/{mode}', _gap_name(gap), balance_index)
            if case:
                pairs.setdefault(mode, {})[gap] = case
    return pairs


def _layout_cases(root, nominal, balance_index):
    cases = []
    for path in sorted(Path(root).rglob('metrics.json')):
        if 'source_before' in path.parts:
            continue
        folder = path.parent
        config = _read(folder/'actual_config.json', {})
        layout = config.get('parameters', {}).get('bearing_layout_name')
        if not layout or layout == nominal:
            continue
        if any(part in path.parts for part in ('data_A_same_U', 'data_B_recalibrated', 'convergence', 'calibration', 'initial_condition')):
            continue
        name = _rel(root, folder)
        case = _case_summary(root, folder, 'layout_sensitivity', name, balance_index)
        if case:
            cases.append(case)
    return cases


def _group_status(cases, required, attr):
    by_gap = {float(c['gap_um']): c for c in cases if c.get('gap_um') is not None}
    if any(gap not in by_gap for gap in required):
        return 'not_run'
    values = [by_gap[gap].get(attr) for gap in required]
    if any(value is False for value in values):
        return 'fail'
    if any(value is None for value in values):
        return 'not_run'
    return 'pass' if all(values) else 'fail'


def _convergence_status(comparisons, label, required=REQUIRED_GAPS):
    if not comparisons:
        return 'not_run'
    rows = []
    for gap in required:
        key = _gap_name(gap)
        item = comparisons.get(key, {}).get(label)
        if item is None:
            return 'not_run'
        rows.append(item)
    return 'pass' if all(row.get('passed') for row in rows) else 'fail'


def _metrics_for_gaps(records):
    return {round(float(row['metrics']['gap_um']), 8): row for row in records
            if row.get('metrics', {}).get('gap_um') is not None}


def _waveform_status(records, required=REQUIRED_GAPS):
    by_gap = _metrics_for_gaps(records)
    rows = []
    for gap in required:
        record = by_gap.get(round(float(gap), 8))
        if record is None:
            return 'not_run', rows
        rows.append(record)
    values = [row.get('full_waveform_pass') for row in rows]
    if any(value is False for value in values):
        return 'fail', rows
    if any(value is not True for value in values):
        return 'not_run', rows
    return 'pass', rows


def _waveform_case_audit(records):
    audit = {}
    for record in records:
        metrics = record.get('metrics', {})
        gap = metrics.get('gap_um')
        if gap is None:
            continue
        h_status = record.get('feature_status', {})
        bearing_status = record.get('bearing_feature_status', {})
        audit[_gap_name(float(gap))] = dict(
            main_valid=record.get('main_valid'),
            full_waveform_pass=record.get('full_waveform_pass'),
            steady_pass=metrics.get('steady_pass'),
            failed_H_features={key: value for key, value in h_status.items() if value != 'pass'},
            failed_bearing_features={key: value for key, value in bearing_status.items() if value != 'pass'},
            source=record.get('source'))
    return audit


def _branch_audit_pass(branch_evidence, required=REQUIRED_GAPS):
    """Require an audited ascending continuation with valid handoffs, not branch equality."""
    if branch_evidence.get('status') != 'audited':
        return False, []
    rows = [row for row in branch_evidence.get('records', [])
            if row.get('direction') == 'ascending' and row.get('gap_um') is not None]
    by_gap = {round(float(row['gap_um']), 8): row for row in rows}
    selected = [by_gap.get(round(float(gap), 8)) for gap in required]
    if any(row is None or row.get('status') != 'audited' for row in selected):
        return False, [row for row in selected if row is not None]
    checks_ok = all(row.get('direct_flags', {}).get('all_required_pass') is True
                    for row in selected)
    comparisons_present = all(isinstance(row.get('statistics_comparison'), dict)
                              and row['statistics_comparison'].get('feature_count', 0) > 0
                              for row in selected)
    return bool(checks_ok and comparisons_present), selected


def _layout_trigger(base, alt):
    if not base or not alt:
        return None
    a, b = base['metrics'], alt['metrics']
    features = ('E_dir_um', 'H_v_rms_x_mm_s', 'H_v_rms_y_mm_s')
    changes = {key: (None if not a.get(key) else (b.get(key)-a[key])/abs(a[key]))
               for key in features}
    trigger = any(value is not None and abs(value) > .05 for value in changes.values())
    trigger = trigger or base.get('steady_pass') != alt.get('steady_pass') or base.get('domain_pass') != alt.get('domain_pass')
    return dict(relative_changes=changes, triggers_remaining_layout_pairs=bool(trigger))


def _same_measurement_window(left, right):
    if not left or not right:
        return False
    a, b = left.get('metrics', {}), right.get('metrics', {})
    time_keys = ('measurement_start_s', 'measurement_end_s')
    same_times = all(a.get(key) is not None and b.get(key) is not None
        and abs(float(a[key])-float(b[key])) <= 1e-9 for key in time_keys)
    return bool(same_times and a.get('complete_turns') == b.get('complete_turns'))


def _legacy_v6_regression(root, old_cases):
    from utils.clearance_v6_analysis import compare_metrics

    excluded = {'bearing_contact_model', 'bearing_layout', 'bearing_layout_name',
        'bearing_layout_status', 'bearing_layout_basis', 'bearing_position_basis',
        'front_pair_arrangement_assumption', 'rear_pair_arrangement_assumption',
        'model_status', 'calibration_status'}
    reports = {}
    for gap in A_GAPS:
        case = old_cases.get(gap)
        previous = Path('experiments/clearance_v6_GMN/data')/_gap_name(gap)
        if case is None or not (previous/'metrics.json').exists() or not (previous/'actual_config.json').exists():
            reports[_gap_name(gap)] = dict(status='not_run', evidence=previous.as_posix())
            continue
        current_metrics = _read(Path(root)/case['path']/'metrics.json', {})
        current_config = _read(Path(root)/case['actual_config'], {})
        old_metrics = _read(previous/'metrics.json', {})
        old_config = _read(previous/'actual_config.json', {})
        param_differences = []
        old_p, new_p = old_config.get('parameters', {}), current_config.get('parameters', {})
        for key in sorted(set(old_p) | set(new_p)):
            if key in excluded or key.startswith('_'):
                continue
            if old_p[key] != new_p.get(key):
                param_differences.append(key)
        input_differences = []
        old_excitation = copy.deepcopy(old_config.get('excitation', {}))
        new_excitation = copy.deepcopy(current_config.get('excitation', {}))
        old_excitation.pop('source', None)
        new_excitation.pop('source', None)
        old_simulation = copy.deepcopy(old_config.get('simulation', {}))
        new_simulation = copy.deepcopy(current_config.get('simulation', {}))
        old_simulation.pop('maximum_warmup_s', None)
        new_simulation.pop('maximum_warmup_s', None)
        for section, left, right in (('excitation', old_excitation, new_excitation),
                                     ('simulation', old_simulation, new_simulation),
                                     ('processing', old_config.get('processing'), current_config.get('processing'))):
            if left != right:
                input_differences.append(section)
        comparison = compare_metrics(old_metrics, current_metrics, current_config)
        ratios = [row['difference']/row['tolerance'] for row in comparison['metrics'].values()
                  if row.get('tolerance')]
        passed = bool(comparison.get('passed') and not param_differences and not input_differences)
        reports[_gap_name(gap)] = dict(status='pass' if passed else 'fail',
            evidence=[previous.as_posix()+'/metrics.json', case['path']+'/metrics.json'],
            same_window=comparison.get('same_window'), shared_metric_count=len(comparison['metrics']),
            maximum_difference_to_tolerance=max(ratios) if ratios else None,
            failed_metrics=[key for key,row in comparison['metrics'].items() if not row.get('passed')],
            changed_shared_parameters=param_differences, changed_input_sections=input_differences,
            tolerance_basis='clearance_v6_analysis.compare_metrics; saved V6 acceptance tolerances')
    status = 'not_run' if any(row['status'] == 'not_run' for row in reports.values()) else \
        'pass' if all(row['status'] == 'pass' for row in reports.values()) else 'fail'
    return dict(status=status, cases=reports,
        scope='Current radial_only_v6 actual runs compared with the saved V6 data; c10 differences are judged against the established metric tolerances, not bitwise equality.')


def _one_check(result, key, status, evidence, max_error=None, tolerance=None, reason=''):
    result['checks'][key] = _check(status, evidence, max_error, tolerance, reason)


def build_acceptance(root=DEFAULT_ROOT, output=None, template=DEFAULT_TEMPLATE,
                     recompute=False, base_group='data_B_recalibrated',
                     integration_group='convergence/h_half', angular_group='convergence/n1024'):
    """Create the per-check V8 acceptance JSON strictly from saved evidence."""
    root = Path(root)
    output = Path(output) if output else root/'acceptance_result.json'
    template_data = _read(template)
    if template_data is None:
        raise FileNotFoundError(f'V8 acceptance template not found: {template}')

    b_paths = list((root/base_group).rglob('metrics.json')) if (root/base_group).exists() else []
    if recompute and b_paths:
        from utils.clearance_v6_analysis import assemble
        assemble(root, output=root/'recomputed', base_group=base_group,
                 integration_group=integration_group, angular_group=angular_group, replay=False)
    v6 = _read(root/'recomputed/acceptance.json', {})
    comparisons = _read(root/'recomputed/convergence.json', {})
    source = _read(root/'validation/source_checks.json', {})
    preflight = _read(root/'validation/static_layout_preflight.json', {})
    p1 = _read(root/'validation/production_mechanics.json', {})
    p1map = _p1_map(p1)
    policy = _read(root/'experiment_policy.json', {})
    environment = _read(root/'environment.json', {})
    selection = _read(root/'calibration_selection.json', {})
    calibration_history = _read(root/'calibration_history.json')
    if calibration_history is None:
        calibration_history = _csv(root/'calibration_history.csv')
    candidate = _read(root/'parameters_v8_candidate.json', {})
    calibrated = _read(root/'parameters_v8_calibrated.json', {})
    health_confirmation = _read(root/'validation/health_confirmation.json', {})
    balance_index = _dynamic_balance_index(root)

    v8base = _read('config/clearance_v8_config.json', {})
    base_p = v8base.get('parameters', {})
    nominal_layout = base_p.get('bearing_layout_name') or 'DB_DB'
    selected_layout = policy.get('layout_selected') or preflight.get('selected_max_sensitivity_layout')
    source_checks = source.get('comparisons', {})
    runtime_check = source_checks.get('v8_runtime_construction', {})
    old_u = source_checks.get('v8_runtime_construction', {}).get('U_old_kg_m')
    if old_u is None:
        old_u = v8base.get('excitation', {}).get('U_kg_m')
    new_config = calibrated or candidate
    new_u = (new_config.get('excitation', {}).get('U_kg_m')
             or new_config.get('calibration', {}).get('U_star_kg_m'))
    rpm = new_config.get('parameters', {}).get('rpm', base_p.get('rpm'))
    force_new = None if new_u is None or rpm is None else float(new_u)*(float(rpm)*2*np.pi/60)**2
    a_cases = _run_pairs(root, 'data_A_same_U', ('old', 'new'), balance_index)
    a_old_original = a_cases.get('old', {})
    a_old = dict(a_old_original)
    matched_old_c10 = _case_summary(root, root/'data_A_same_U/old_matched/gap_10um',
        'data_A_same_U/old_matched', 'gap_10um', balance_index)
    if matched_old_c10:
        a_old[10.0] = matched_old_c10
    a_cases['old'] = a_old
    a_new = a_cases.get('new', {})
    legacy_regression = _legacy_v6_regression(root, a_old_original)
    a_differences, a_windows, a_frozen = {}, {}, True
    for gap in A_GAPS:
        old_case = a_old.get(gap)
        new_case = a_new.get(gap)
        if not old_case or not new_case:
            a_differences[_gap_name(gap)] = None
            a_frozen = False
            continue
        old_cfg = _read(root/old_case['actual_config'], {})
        new_cfg = _read(root/new_case['actual_config'], {})
        diff = _config_differences(old_cfg, new_cfg)
        a_differences[_gap_name(gap)] = diff
        a_windows[_gap_name(gap)] = dict(same_window=_same_measurement_window(old_case, new_case),
            old_start_s=old_case['metrics'].get('measurement_start_s'),
            old_end_s=old_case['metrics'].get('measurement_end_s'),
            new_start_s=new_case['metrics'].get('measurement_start_s'),
            new_end_s=new_case['metrics'].get('measurement_end_s'),
            old_complete_turns=old_case['metrics'].get('complete_turns'),
            new_complete_turns=new_case['metrics'].get('complete_turns'))
        a_frozen = a_frozen and not diff
        a_frozen = a_frozen and old_case.get('U_kg_m') == new_case.get('U_kg_m')
        a_frozen = a_frozen and a_windows[_gap_name(gap)]['same_window']

    b_records = v6.get('cases', [])
    b_by_gap = _metrics_for_gaps(b_records)
    diagnostic_csv_path, diagnostic_csv_rows = _contact_wrench_csv(root, b_records)
    dynamic_case_inventory = _dynamic_case_inventory(root)
    b_cases = []
    for gap in REQUIRED_GAPS:
        record = b_by_gap.get(gap)
        folder = root/base_group/_gap_name(gap)
        summary = _case_summary(root, folder, base_group, _gap_name(gap), balance_index, record) if record else None
        if summary:
            b_cases.append(summary)
    alt_layout_cases = _layout_cases(root, nominal_layout, balance_index)
    saved_balance_status, saved_balance_rows = _saved_balance_status(a_new, b_cases)
    layout_pair_checks = {}
    nominal_b = {float(case['gap_um']): case for case in b_cases}
    for gap in (2.0, 10.0):
        alt = next((case for case in alt_layout_cases
                    if case.get('layout') == selected_layout and float(case.get('gap_um', -1)) == gap), None)
        layout_pair_checks[_gap_name(gap)] = _layout_trigger(nominal_b.get(gap), alt)
    trigger_remaining = any((row or {}).get('triggers_remaining_layout_pairs') for row in layout_pair_checks.values())
    layouts_present = sorted({case.get('layout') for case in alt_layout_cases})

    # Statuses stay individual; an absent run is never promoted to pass.
    _one_check(template_data, 'source_coordinate_mapping',
        _status(bool(source), bool(runtime_check.get('unchanged_non_v8_parameter_leaf_difference_count') == 0
            and runtime_check.get('analysis_version') == 'v8'
            and runtime_check.get('bearing_contact_model') == 'relative_tilt_v8'
            and runtime_check.get('support_reallocation_called') is False
            and source_checks.get('actual_v6_healthy_config_vs_plan_copy', {}).get('byte_identical')
            and source_checks.get('current_pre_edit_model_snapshot_vs_plan_baseline', {}).get('dynamics_model', {}).get('byte_identical')
            and source_checks.get('current_pre_edit_model_snapshot_vs_plan_baseline', {}).get('fast_rhs', {}).get('byte_identical'))),
        ['validation/source_checks.json'], reason='Actual V6 source, snapshot, and V8 parameter construction compared field by field; CRC32 only, no SHA-256.')
    fixed_pass = bool(p1.get('state_contract', {}).get('model_dof') == 12
        and p1.get('state_contract', {}).get('q_dof') == 12
        and p1.get('state_contract', {}).get('integrated_state_size') == 24)
    _one_check(template_data, 'fixed_12dof', _status(bool(p1), fixed_pass),
        ['validation/production_mechanics.json'], reason='Production state and mass/gravity dimensions.')
    _one_check(template_data, 'saved_state_force_moment_balance', saved_balance_status,
        ['validation/A_new/dynamic_balance.json']+
            (['validation/B/dynamic_balance.json'] if (root/'validation/B/dynamic_balance.json').exists() else [])+
            [f'{base_group}/{_gap_name(gap)}/dynamic_balance.json' for gap in REQUIRED_GAPS
             if (root/base_group/_gap_name(gap)/'dynamic_balance.json').exists()],
        reason='Requires the three A-new and all five B saved-state audits to pass translational and moment identities separately.')
    gradient = p1map.get('B07_B08_potential_gradient_tangent', {}).get('details', {})
    gradient_errors = [entry.get('potential_gradient_relative_error') for entry in gradient.values()]
    _one_check(template_data, 'algebra_energy_gradient',
        _p1_status(p1map, 'B07_B08_potential_gradient_tangent'),
        ['validation/production_mechanics.json'], max(gradient_errors) if gradient_errors else None,
        1e-5, 'Frozen-phase production potential gradient finite differences.')
    tangent_errors = [entry.get('tangent_relative_error') for entry in gradient.values()]
    symmetry_errors = [entry.get('tangent_symmetry_relative_error') for entry in gradient.values()]
    _one_check(template_data, 'algebra_tangent_psd',
        _p1_status(p1map, 'B07_B08_potential_gradient_tangent'),
        ['validation/production_mechanics.json'], max(tangent_errors+symmetry_errors) if tangent_errors else None,
        '1e-5 tangent; 1e-8 symmetry; PSD/rank checked at scaled coordinates',
        'Finite-difference tangent, symmetry and expected semidefinite rank are in the P1 evidence.')
    virtual = p1map.get('system_virtual_work_assembly', {}).get('details', {})
    _one_check(template_data, 'assembly_virtual_work', _p1_status(p1map, 'system_virtual_work_assembly'),
        ['validation/production_mechanics.json'], virtual.get('maximum_relative_power_difference'), 1e-8,
        'Random full-system virtual-displacement power identity.')
    assembly = p1map.get('system_force_moment_reaction_assembly', {}).get('details', {})
    _one_check(template_data, 'constraint_reactions', _p1_status(p1map, 'system_force_moment_reaction_assembly'),
        ['validation/production_mechanics.json'], dict(force_N=assembly.get('force_balance_error_N'),
            moment_Nm=assembly.get('moment_balance_error_Nm')),
        dict(force_N='P1 tolerance', moment_Nm='P1 tolerance'),
        'Includes housing torque, outer rotation and axial reactions with exact expected signs.')
    pressure_error = p1map.get('B09_pressure_center_equivalence', {}).get('details', {}).get('maximum_scaled_generalized_wrench_error')
    _one_check(template_data, 'no_pressure_center_double_count', _p1_status(p1map, 'B09_pressure_center_equivalence', 'B10_DB_DF_pair_stiffness'),
        ['validation/production_mechanics.json', 'validation/static_layout_preflight.json'], pressure_error,
        1e-10, 'Geometric-center force plus local moment was compared against the pressure-center generalized wrench; DB/DF stiffness checked separately.')
    backend = p1map.get('B14_python_numba_mixed_states', {}).get('details', {})
    backend_error = max(backend.get('maximum_normalized_force_error', 0), backend.get('maximum_normalized_moment_error', 0)) if backend else None
    _one_check(template_data, 'python_numba_equivalence', _p1_status(p1map, 'B14_python_numba_mixed_states'),
        ['validation/production_mechanics.json'], backend_error, 1e-10,
        '128 mixed states; N and N m use distinct normalization floors; raw maxima are preserved in P1 JSON.')

    a_complete = all(gap in a_old and gap in a_new for gap in A_GAPS)
    a_running = [case for group in (a_old, a_new) for case in group.values()]
    a_stable = bool(a_complete and all(case.get('steady_pass') is True and
        case.get('solver_pass') is True and case.get('input_pass') is True and
        case.get('sampling_pass') is True and case.get('sampling_identity_pass') is True
        for case in a_running))
    a_domain = bool(a_complete and all(case.get('domain_pass') is True for case in a_running))
    _one_check(template_data, 'same_U_comparison',
        'not_run' if not a_complete else 'pass' if a_frozen and a_stable and a_domain and legacy_regression['status'] == 'pass' else 'fail',
        [str(Path('data_A_same_U')/('old_matched' if gap == 10.0 else 'old')/_gap_name(gap)/'actual_config.json')
         for gap in A_GAPS if (root/'data_A_same_U'/('old_matched' if gap == 10.0 else 'old')/_gap_name(gap)/'actual_config.json').exists()]
        +[f'data_A_same_U/new/{_gap_name(gap)}/actual_config.json' for gap in A_GAPS
          if (root/'data_A_same_U/new'/_gap_name(gap)/'actual_config.json').exists()],
        reason='A uses the recorded U_old. The controlled c10 pair uses old_matched; original old c10 remains a separate unmatched source. Paired actual configs and absolute measurement windows must match except for bearing_contact_model. Legacy V6 regression uses the original old runs.')
    static_layouts = preflight.get('layouts', {})
    layout_status = 'not_run'
    layout_evidence = ['validation/static_layout_preflight.json']
    if preflight.get('status') == 'complete' and selected_layout == preflight.get('selected_max_sensitivity_layout') \
            and selected_layout and static_layouts and layout_pair_checks.get(_gap_name(2.0)) and layout_pair_checks.get(_gap_name(10.0)):
        pairs_complete = all(layout_pair_checks[_gap_name(gap)] is not None for gap in (2.0, 10.0))
        selected_cases = [case for case in alt_layout_cases if case.get('layout') == selected_layout
                          and float(case.get('gap_um', -1)) in (2.0, 10.0)]
        base_cases = [nominal_b.get(gap) for gap in (2.0, 10.0)]
        outcomes = selected_cases+base_cases
        expected_layouts = set(static_layouts)-{nominal_layout}
        other_layouts_complete = all(any(c.get('layout') == layout and float(c.get('gap_um', -1)) == gap for c in alt_layout_cases)
            for layout in expected_layouts for gap in (2.0, 10.0)) if trigger_remaining else True
        if trigger_remaining:
            outcomes += [case for case in alt_layout_cases
                if float(case.get('gap_um', -1)) in (2.0, 10.0)]
        outcomes_valid = all(case and case.get('steady_pass') is True
            and case.get('domain_pass') is True and case.get('solver_pass') is True
            and case.get('input_pass') is True and case.get('sampling_pass') is True
            and case.get('sampling_identity_pass') is True for case in outcomes)
        layout_status = 'pass' if pairs_complete and other_layouts_complete and outcomes_valid else 'fail' if pairs_complete and not outcomes_valid else 'not_run'
        layout_evidence += [case['path']+'/metrics.json' for case in alt_layout_cases]
    _one_check(template_data, 'layout_sensitivity', layout_status, layout_evidence,
        reason='Layout choice came from frozen-phase rotational compliance before dynamic outputs; DB/DF arrangement remains an assumption.')

    health_record = b_by_gap.get(2.0)
    health_pass = bool(health_confirmation.get('status') == 'pass'
        and calibrated and health_record and health_record.get('full_waveform_pass'))
    calibration_status = ('pass' if health_pass else
        'fail' if selection.get('status') == 'fail' or health_confirmation.get('status') == 'fail' else
        'not_run')
    _one_check(template_data, 'new_health_calibration', calibration_status,
        [path for path in ('calibration_history.json', 'calibration_selection.json',
            'parameters_v8_candidate.json', 'parameters_v8_calibrated.json',
            'validation/health_confirmation.json', 'data_B_recalibrated/gap_2um/actual_config.json')
         if (root/path).exists()],
        reason='Conditional U calibration is complete only after its cold repeat and independent h/N/sampling confirmation.')

    stationarity_status = _group_status(b_cases, REQUIRED_GAPS, 'steady_pass')
    _one_check(template_data, 'whole_window_stationarity', stationarity_status,
        [f'{base_group}/{_gap_name(gap)}/metrics.json' for gap in REQUIRED_GAPS
         if (root/base_group/_gap_name(gap)/'metrics.json').exists()],
        reason='Checks each required B point over its complete exported window and four consecutive subwindows.')
    waveform_status, waveform_records = _waveform_status(b_records)
    _one_check(template_data, 'full_waveform_certification', waveform_status,
        ['recomputed/acceptance.json', 'recomputed/convergence.json'],
        reason='Requires all five B points to pass the assembled full-waveform gate, including H and every bearing feature (local moments included).')
    h_status = _convergence_status(comparisons, 'h_half')
    n_status = _convergence_status(comparisons, 'N_double')
    _one_check(template_data, 'time_step_convergence', h_status,
        ['recomputed/convergence.json'], reason='Independent h/2 comparison on each required B point.')
    _one_check(template_data, 'contact_discretization_convergence', n_status,
        ['recomputed/convergence.json'], reason='Independent N512 to N1024 comparison on each required B point.')
    sampling_status = _group_status(b_cases, REQUIRED_GAPS, 'sampling_pass')
    if sampling_status == 'pass' and any(not row.get('sampling_identity_pass') for row in b_records):
        sampling_status = 'fail'
    _one_check(template_data, 'sampling_convergence', sampling_status,
        [f'{base_group}/{_gap_name(gap)}/sampling_convergence.json' for gap in REQUIRED_GAPS
         if (root/base_group/_gap_name(gap)/'sampling_convergence.json').exists()],
        reason='Same dense solution sampled at the declared base and 2x output rates.')

    long_cases = []
    for path in sorted((root/base_group).rglob('metrics.json')) if (root/base_group).exists() else []:
        record = b_by_gap.get(round(float(_read(path, {}).get('gap_um', -1)), 8))
        case = _case_summary(root, path.parent, base_group, path.parent.name,
            balance_index, record)
        if case and (case.get('retained_turns') or 0) >= 480:
            long_cases.append(case)
    branch_evidence = _read(root/'validation/branches.json', {})
    branch_status = branch_evidence.get('status') or branch_evidence.get('overall_status')
    long_by_gap = {float(c['gap_um']): c for c in long_cases if c.get('gap_um') is not None}
    long_complete = all(gap in long_by_gap and long_by_gap[gap].get('steady_pass') is True
                        and long_by_gap[gap].get('full_waveform_pass') is True
                        and long_by_gap[gap]['metrics'].get('complete_turns') == 480 for gap in (7.5, 10.0))
    branch_pass, ascending_branch_rows = _branch_audit_pass(branch_evidence)
    branch_check_status = ('pass' if long_complete and branch_pass
        else 'fail' if branch_status in ('fail', 'failed') or (branch_status == 'audited' and not branch_pass)
        else 'not_run')
    _one_check(template_data, 'long_window_branches',
        branch_check_status,
        ['validation/branches.json']+[case['path']+'/metrics.json' for case in long_cases],
        reason='Requires 480-turn, full-waveform-passing 7.5/10 um runs and an audited ascending continuation with valid seed/physics flags at all five gaps. Branch agreement is reported, not required; it does not prove uniqueness.')

    b_complete = all(gap in b_by_gap for gap in REQUIRED_GAPS)
    b_u_values = [b_by_gap[gap]['metrics'].get('U_kg_m') for gap in REQUIRED_GAPS if gap in b_by_gap]
    b_same_u = bool(b_complete and b_u_values and max(b_u_values)-min(b_u_values) <= max(abs(float(b_u_values[0]))*1e-12, 1e-20))
    sweep_status = 'not_run' if not b_complete else 'pass' if b_same_u else 'fail'
    _one_check(template_data, 'new_U_gap_sweep', sweep_status,
        [f'{base_group}/{_gap_name(gap)}/actual_config.json' for gap in REQUIRED_GAPS
         if (root/base_group/_gap_name(gap)/'actual_config.json').exists()],
        reason='Requires all five actual points at the same frozen U_new; numerical validity is listed separately.')

    main_valid = [row for row in b_records if row.get('main_valid')]
    full_waveform_valid = [row for row in b_records if row.get('full_waveform_pass') is True]
    steady_domain_candidates = [row for row in b_records
        if row.get('metrics', {}).get('steady_pass') is True
        and row.get('domain', {}).get('passed') is True]
    max_record = max(full_waveform_valid, key=lambda row: row['metrics']['E_dir_um'], default=None)
    health_valid = next((row for row in full_waveform_valid
                         if float(row['metrics']['gap_um']) == 2.0), None)
    single_turn = [max(row['metrics'].get('E_x_max_um', 0), row['metrics'].get('E_y_max_um', 0))
                   for row in full_waveform_valid]
    candidate_max = max(steady_domain_candidates,
        key=lambda row: row['metrics']['E_dir_um'], default=None)
    main_max = max(main_valid, key=lambda row: row['metrics']['E_dir_um'], default=None)
    template_data['range_coverage'].update(
        mean_Edir_max_um=max_record['metrics']['E_dir_um'] if max_record else None,
        maximum_single_turn_um=max(single_turn) if single_turn else None,
        delta_from_health_um=(max_record['metrics']['E_dir_um']-health_valid['metrics']['E_dir_um'])
            if max_record and health_valid else None,
        mean_20um_reached=(None if max_record is None else bool(max_record['metrics']['E_dir_um'] >= 20.0)),
        maximum_case=max_record.get('case') if max_record else None,
        maximum_single_turn_scope='Full-waveform-certified B runs only; one-turn X/Y max, not the formal mean E_dir.' if single_turn else 'No full-waveform-certified B run.',
        full_waveform_certified_gaps_um=sorted(float(row['metrics']['gap_um']) for row in full_waveform_valid),
        full_waveform_certified_case_count=len(full_waveform_valid),
        main_valid_candidate_gaps_um=sorted(float(row['metrics']['gap_um']) for row in main_valid),
        main_valid_candidate_Edir_max_um=main_max['metrics']['E_dir_um'] if main_max else None,
        main_valid_candidate_case=main_max.get('case') if main_max else None,
        maximum_steady_in_domain_candidate_Edir_um=candidate_max['metrics']['E_dir_um'] if candidate_max else None,
        maximum_steady_in_domain_candidate_case=candidate_max.get('case') if candidate_max else None)

    template_data.update(code_version='uncommitted workspace V8 source; see source_before snapshot and run evidence',
        config_sha256=None, nominal_layout=nominal_layout,
        layout_evidence=dict(preflight_path='validation/static_layout_preflight.json',
            selected_max_sensitivity_layout=selected_layout, dynamic_cases=layouts_present,
            basis=preflight.get('selection_rule')),
        U_old_kg_m=old_u, U_new_kg_m=new_u, F_new_N_at_7000=force_new,
        results_are_actual_runs=bool(a_cases or b_paths),
        real_machine_validation=dict(status='not_validated', reasons=[
            '安装配对方向未由实机资料确认', '外圈倾转固定约束及其承载能力未实测',
            '输入等效参数仅作方法原型，未完成本机独立识别', '没有同工况实测H/轴心/支架同步数据']),
        source_evidence=dict(source_checks='validation/source_checks.json',
            static_layout_preflight='validation/static_layout_preflight.json',
            production_mechanics='validation/production_mechanics.json',
            source_checks_status=source.get('schema'),
            no_sha256='配置/代码未计算SHA-256；按仓库规则仅保留原配置和差异证据。'),
        run_groups=dict(A_same_U={mode:{_gap_name(gap):case for gap,case in sorted(rows.items())}
                                  for mode,rows in a_cases.items()},
            A_old_unmatched={_gap_name(gap):case for gap,case in sorted(a_old_original.items())},
            A_config_differences=a_differences,
            A_measurement_windows=a_windows,
            A_legacy_v6_regression=legacy_regression,
            B_recalibrated={_gap_name(float(case['gap_um'])):case for case in b_cases},
            layout_sensitivity=alt_layout_cases,
            layout_dynamic_comparison=layout_pair_checks,
            calibration=dict(selection=selection, history=calibration_history,
                health_confirmation=health_confirmation, candidate_config=bool(candidate),
                calibrated_config=bool(calibrated),
                U_new_matches_U_old=(None if new_u is None or old_u is None else
                    abs(float(new_u)-float(old_u)) <= max(abs(float(old_u))*1e-12, 1e-20)),
                U_new_minus_U_old_kg_m=(None if new_u is None or old_u is None else float(new_u)-float(old_u))),
            convergence=comparisons,
            dynamic_balance_evidence=[_rel(root, path) for path in root.rglob('dynamic_balance.json')],
            saved_state_force_moment_balance=dict(status=saved_balance_status, cases=saved_balance_rows),
            case_wrench_contact_tilt_csv=dict(path=diagnostic_csv_path, rows=diagnostic_csv_rows),
            dynamic_case_inventory=dynamic_case_inventory,
            branch_evidence=branch_evidence,
            branch_ascending_required_cases=[row for row in ascending_branch_rows],
            B_waveform_certification=_waveform_case_audit(b_records),
            long_window_cases=long_cases),
        source_parameter_config='config/clearance_v8_config.json',
        runtime=environment,
        analysis_outputs=dict(recomputed_acceptance='recomputed/acceptance.json',
            convergence='recomputed/convergence.json', error_estimates='recomputed/numerical_error_estimates.json',
            gap_summary='recomputed/gap_summary.csv', vibration_differences='recomputed/same_U_vibration_differences.csv',
            case_wrench_contact_tilt=diagnostic_csv_path,
            dynamic_case_inventory='run_groups.dynamic_case_inventory'))

    _one_check(template_data, 'same_U_comparison', template_data['checks']['same_U_comparison']['status'],
        template_data['checks']['same_U_comparison']['evidence'],
        reason=template_data['checks']['same_U_comparison']['reason']+
            f' Actual physical-config differences after excluding the contact switch: {a_differences}.')
    not_passed = [name for name, row in template_data['checks'].items() if row['status'] != 'pass']
    template_data['remaining_issues'] = list(dict.fromkeys(
        ['未知實機配對方向、外圈固定邊界承載及本機等效參數需獨立確認。',
         '模型只輸出單一等效H節點，無空間傳遞或多點殼體驗證。']+
         ([f'20 μm平均E_dir未由全波形认证B点达到；认证最大值={template_data["range_coverage"]["mean_Edir_max_um"]} μm。']
          if template_data['range_coverage']['mean_20um_reached'] is False else [])+
         ([f'当前没有全波形认证B点；主量候选最大E_dir={template_data["range_coverage"]["main_valid_candidate_Edir_max_um"]} μm，不计作认证范围。']
          if template_data['range_coverage']['mean_20um_reached'] is None else [])+
        ([f'未通过或未执行的验收项：{", ".join(not_passed)}。'] if not_passed else [])))
    template_data['summary_states'] = dict(
        source_and_mechanism=template_data['checks']['source_coordinate_mapping']['status'],
        production_algebra=template_data['checks']['fixed_12dof']['status'] if all(
            template_data['checks'][name]['status'] == 'pass' for name in (
                'fixed_12dof','algebra_energy_gradient','algebra_tangent_psd',
                'assembly_virtual_work','constraint_reactions','no_pressure_center_double_count',
                'python_numba_equivalence')) else 'incomplete_or_failed',
        saved_state_force_moment_balance=template_data['checks']['saved_state_force_moment_balance']['status'],
        fixed_U_effect=template_data['checks']['same_U_comparison']['status'],
        new_health_calibration=template_data['checks']['new_health_calibration']['status'],
        numerical_reliability='pass' if all(template_data['checks'][key]['status'] == 'pass' for key in (
            'whole_window_stationarity','full_waveform_certification','time_step_convergence',
            'contact_discretization_convergence','sampling_convergence')) else 'incomplete_or_failed',
        new_U_gap_sweep=template_data['checks']['new_U_gap_sweep']['status'],
        range_20um=template_data['range_coverage']['mean_20um_reached'],
        real_machine_validation='not_validated')

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(template_data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return template_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument('--recompute', action='store_true', help='Reassemble saved B results using V6 analysis; does not run simulations.')
    args = parser.parse_args()
    result = build_acceptance(args.root, args.output, args.template, args.recompute)
    print(json.dumps(dict(output=str(args.output or args.root/'acceptance_result.json'),
        checks={name:item['status'] for name,item in result['checks'].items()},
        mean_20um_reached=result['range_coverage']['mean_20um_reached']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
