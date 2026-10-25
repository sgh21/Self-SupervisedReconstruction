"""Offline V9 acceptance aggregation from selected case folders and JSON evidence.

No simulation is run here. Paths in case_selection.json refer to case folders;
relative paths are resolved from the project root first, then the selection file.
For a flat ``refinements`` entry, comparisons default to A.N3 (old-U); set
``target_case`` to another case id such as ``A.B0`` or ``B`` when needed.
"""
import argparse
import copy
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from utils import clearance_v9_analysis as analysis


REFINEMENT_KINDS = ('h_half', 'root_tenth', 'n1024')
OLD_REFINEMENT_KINDS = ('h_half', 'n1024')
SIGNIFICANCE_EPSILON_FLOOR_UM = 1e-4
CONSERVATIVE_SIGNIFICANCE_FLOOR_UM = .01
H_VELOCITY_METRICS = (
    'H_v_rms_x_mm_s', 'H_v_rms_y_mm_s',
    'H_v_1x_x_mm_s', 'H_v_1x_y_mm_s',
    'H_v_residual_x_mm_s', 'H_v_residual_y_mm_s',
)
H_VELOCITY_EPSILON_FLOOR_MM_S = 1e-4
H_VELOCITY_EPSILON_FLOOR_SOURCE = (
    'Inherited V8 engineering floor: config/clearance_v8_config.json '
    'numerical_thresholds.velocity_mm_s=1e-4 mm/s; utils/clearance_v6_analysis.py '
    'error_budget uses clearance_v5_analysis.absolute_floor for non-position metrics.'
)
DEFAULT_EXPERIMENT = Path('experiments/clearance_v9_tai_bearing')
DEFAULT_SINGLE_BEARING_TESTS = 'validation/solver_retry_local/local_bearing_validation.json'
DEFAULT_CONTACT_BOUNDARY = 'validation/contact_boundary.json'
DEFAULT_INITIAL_STATE_COMPARISONS = (
    dict(comparison_id='initial_state_up_5_to_10_N3_gap10',
         reference_path='data_A/N3/gap_10um',
         branch_path='sensitivity/initial_state_up_5_to_10/N3/gap_10um',
         initial_seed_argument='experiments/clearance_v9_tai_bearing/data_A/N3/gap_5um',
         original_failure_evidence_path='sensitivity/initial_state_up_5_to_10/N3/run_failures.json'),
    dict(comparison_id='initial_state_down_10_to_5_retry_N3_gap5',
         reference_path='data_A_long/N3/gap_5um',
         branch_path='sensitivity/initial_state_down_10_to_5_retry/N3/gap_5um',
         initial_seed_argument='experiments/clearance_v9_tai_bearing/data_A/N3/gap_10um',
         original_failure_evidence_path='sensitivity/initial_state_down_10_to_5/N3/gap_5um/failed.json'),
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def _number(value):
    if isinstance(value, (int, float)):
        return float(value)
    matches = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)', str(value))
    if not matches:
        raise ValueError(f'Cannot parse clearance key: {value!r}')
    return float(matches[0])


def _gap_tag(value):
    return f'{_number(value):g}'


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _resolve_path(value, root, selection_parent=None):
    if value is None or value == '':
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [root / path]
    if not path.is_absolute() and selection_parent is not None:
        candidates.append(selection_parent / path)
    for candidate in candidates:
        if candidate.exists():
            if candidate.is_file():
                if candidate.name in ('metrics.json', 'actual_config.json', 'failed.json'):
                    return candidate.parent.resolve()
                return candidate.resolve()
            return candidate.resolve()
    return candidates[0].resolve()


def _load_case_json(folder):
    folder = Path(folder)
    result = {'folder': str(folder), 'exists': folder.is_dir()}
    if not folder.is_dir():
        result['status'] = 'pending'
        result['reason'] = 'case folder missing'
        return result
    for name, key in (('actual_config.json', 'config'), ('metrics.json', 'metrics'),
                      ('stationarity.json', 'stationarity'),
                      ('sampling_convergence.json', 'sampling'), ('domain.json', 'domain'),
                      ('failed.json', 'failure')):
        path = folder / name
        if path.is_file():
            try:
                result[key] = _read_json(path)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
                result[key + '_read_error'] = str(error)
    if result.get('failure') is not None and result.get('metrics') is None:
        result['status'] = 'failed'
    elif result.get('metrics') is None:
        result['status'] = 'pending'
        result['reason'] = 'metrics.json missing'
    else:
        result['status'] = 'available'
    return result


def _candidate_a_paths(root, variant, gap, declared):
    tag = _gap_tag(gap)
    standard = root / DEFAULT_EXPERIMENT
    candidates = [standard / 'data_A_long' / str(variant) / f'gap_{tag}um',
                  standard / 'data_A' / str(variant) / f'gap_{tag}um']
    if declared is not None:
        candidates.append(declared)
    unique = []
    seen = set()
    for path in candidates:
        p = Path(path).resolve()
        if str(p).casefold() not in seen:
            unique.append(p)
            seen.add(str(p).casefold())
    return unique


def _turn_count(path):
    try:
        metrics = _read_json(Path(path) / 'metrics.json')
        return int(metrics.get('complete_turns', 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def _select_a_path(root, variant, gap, declared):
    candidates = _candidate_a_paths(root, variant, gap, declared)
    for candidate in candidates:
        if _turn_count(candidate) >= 480:
            if candidate.name == f'gap_{_gap_tag(gap)}um' and candidate.parent.name == variant:
                source = 'preferred_480_turn_case'
                if 'data_A_long' in candidate.parts:
                    source = 'data_A_long_480_turn_priority'
                elif 'data_A' in candidate.parts:
                    source = 'data_A_480_turn_fallback'
            else:
                source = 'declared_480_turn_case'
            return candidate, source
    for candidate in candidates:
        if (candidate / 'metrics.json').is_file():
            return candidate, 'pilot_or_incomplete_metrics_retained_pending_480_turns'
    for candidate in candidates:
        if candidate.is_dir() and (candidate / 'actual_config.json').exists():
            return candidate, 'pilot_or_incomplete_case_retained_pending_480_turns'
    return (declared, 'declared_path_missing') if declared is not None else (None, 'no_case_path')


def _configuration_values(case):
    config = case.get('config') or {}
    metrics = case.get('metrics') or {}
    params = config.get('parameters') or {}
    sim = config.get('simulation') or {}
    excitation = config.get('excitation') or {}
    variant = metrics.get('mechanism_variant') or config.get('mechanism_variant')
    gap = metrics.get('gap_um')
    if gap is None:
        gap = params.get('front_group_clearance_um', params.get('fit_clearance_um'))
    force = metrics.get('force_N', excitation.get('peak_force_N'))
    unbalance = metrics.get('U_kg_m', excitation.get('U_kg_m'))
    rpm = params.get('rpm')
    return dict(variant=variant, gap_um=float(gap) if _finite(gap) else None,
                force_N=float(force) if _finite(force) else None,
                U_kg_m=float(unbalance) if _finite(unbalance) else None,
                rpm_signed=float(rpm) if _finite(rpm) else None,
                configured_turns=sim.get('retained_turns'),
                raw_sample_rate_hz=sim.get('fs_raw_hz'),
                output_sample_rate_hz=sim.get('fs_out_hz'))


def _bool_status(value):
    if value is True:
        return 'passed'
    if value is False:
        return 'failed'
    return 'pending'


def _subwindow_turns(stationarity, metrics, config):
    supplied = stationarity.get('subwindow_complete_turns')
    if supplied:
        return [int(x) for x in supplied]
    edges = stationarity.get('window_edges_s') or []
    rpm = (config.get('parameters') or {}).get('rpm')
    if len(edges) >= 2 and _finite(rpm):
        return [int(round(abs(float(rpm)) / 60 * (right-left)))
                for left, right in zip(edges[:-1], edges[1:])]
    submetrics = stationarity.get('subwindow_metrics') or []
    if submetrics and all(_finite(row.get('complete_turns')) for row in submetrics):
        return [int(row['complete_turns']) for row in submetrics]
    return []


def _phase_values(metrics):
    return {key: float(value) for key, value in metrics.items()
            if key.endswith('_max_phase_speed_relative_difference') and _finite(value)}


def _outer_control_values(metrics):
    return {key: float(value) for key, value in metrics.items()
            if key.endswith('_outer_control_self_consistent_fraction') and _finite(value)}


def _maximum_metric(metrics, suffix):
    values = [float(value) for key, value in metrics.items()
              if key.endswith(suffix) and _finite(value)]
    return max(values) if values else None


def _case_acceptance(case, group, variant, gap, path_resolution):
    folder = case.get('folder')
    metrics = case.get('metrics') or {}
    config = case.get('config') or {}
    stationarity = case.get('stationarity') or {}
    sampling = case.get('sampling') or {}
    domain = case.get('domain') or {}
    values = _configuration_values(case)
    variant = values['variant'] or variant
    gap = values['gap_um'] if values['gap_um'] is not None else float(gap)
    case_id = f'{group}.{variant or "unknown"}.{_gap_tag(gap)}'
    output = dict(case_id=case_id, group=group, variant=variant, gap_um=gap,
                  path=folder, path_resolution=path_resolution,
                  force_N=values['force_N'], U_kg_m=values['U_kg_m'],
                  rpm_signed=values['rpm_signed'],
                  complete_turns=metrics.get('complete_turns'),
                  configured_turns=values['configured_turns'],
                  E_x_um=metrics.get('E_x_um'), E_y_um=metrics.get('E_y_um'),
                  E_dir_um=metrics.get('E_dir_um'),
                  H_v_rms_x_mm_s=metrics.get('H_v_rms_x_mm_s'),
                  H_v_rms_y_mm_s=metrics.get('H_v_rms_y_mm_s'),
                  H_v_1x_x_mm_s=metrics.get('H_v_1x_x_mm_s'),
                  H_v_1x_y_mm_s=metrics.get('H_v_1x_y_mm_s'),
                  H_v_residual_x_mm_s=metrics.get('H_v_residual_x_mm_s'),
                  H_v_residual_y_mm_s=metrics.get('H_v_residual_y_mm_s'))
    if case.get('status') == 'failed':
        output.update(status='failed', numerical_core_status='failed',
                      failure=case.get('failure'),
                      gates=dict(run='failed'))
        return output
    if case.get('status') != 'available':
        output.update(status='pending', numerical_core_status='pending',
                      reason=case.get('reason', 'case output incomplete'),
                      gates=dict(run='pending'))
        return output

    turns = int(metrics.get('complete_turns', 0) or 0)
    required_turns = 480 if gap > 0 else 120
    turn_status = 'passed' if turns >= required_turns else 'pending'
    subturns = _subwindow_turns(stationarity, metrics, config)
    minimum_subwindow_turns = 120 if gap > 0 else 30
    if case.get('stationarity') is None:
        subwindows_status = 'pending'
        steady_value = None
        steady_status = 'pending'
    else:
        subwindows_status = ('passed' if len(subturns) == 4 and
                             all(n >= minimum_subwindow_turns for n in subturns)
                             else 'pending' if turns < required_turns else 'failed')
        steady_value = stationarity.get('passed', metrics.get('steady_pass'))
        steady_status = _bool_status(steady_value)
        if steady_status == 'passed' and subwindows_status != 'passed':
            steady_status = subwindows_status
    sampling_status = _bool_status(sampling.get('passed'))
    if not case.get('sampling'):
        sampling_status = 'pending'
    domain_value = domain.get('passed', metrics.get('domain_pass'))
    domain_status = (_bool_status(domain_value) if case.get('domain') is not None
                     else 'pending')

    if variant in ('B0', 'B0d'):
        local_root_status = 'not_applicable'
    elif variant == 'N1':
        local_root_status = _bool_status(metrics.get('local_root_pass'))
    elif 'local_root_pass' in metrics:
        local_root_status = _bool_status(metrics.get('local_root_pass'))
    else:
        local_root_status = 'pending'

    n3 = variant == 'N3'
    if variant in ('B0', 'B0d', 'N1'):
        outer_status = 'not_applicable'
        phase_status = 'not_applicable'
        inner_candidate_status = 'not_applicable'
    elif n3:
        outer = _outer_control_values(metrics)
        outer_status = ('pending' if not outer else
                        'passed' if all(value >= 1-1e-12 for value in outer.values())
                        else 'conditional')
        phases = _phase_values(metrics)
        threshold = float((config.get('v9_acceptance') or {}).get(
            'phase_speed_warning_fraction', .02))
        phase_max = max(phases.values()) if phases else None
        phase_status = ('pending' if phase_max is None else
                        'passed' if phase_max <= threshold else 'conditional')
        # Inner-branch candidates are independently audited in the T08 local sweep;
        # production trajectories need only retain the configured branch screen.
        inner_candidate_status = 'covered_by_local_sweep'
    else:
        outer_status = 'diagnostic_only'
        phase_status = 'diagnostic_only'
        inner_candidate_status = 'not_applicable'

    gate_statuses = dict(turns=turn_status, steady=steady_status,
                         sampling=sampling_status, local_root=local_root_status,
                         domain=domain_status)
    core_values = [value for value in gate_statuses.values()
                   if value not in ('not_applicable', 'diagnostic_only')]
    if 'failed' in core_values:
        core_status = 'failed'
    elif 'pending' in core_values:
        core_status = 'pending'
    else:
        core_status = 'numerical_pass'
    conditional_gates = dict(outer_control=outer_status, phase_speed=phase_status)
    if core_status == 'numerical_pass' and any(
            value in ('conditional', 'pending', 'pending_not_recorded')
            for value in conditional_gates.values()):
        status = 'conditional'
    else:
        status = core_status
    output.update(status=status, numerical_core_status=core_status,
        gates=gate_statuses, conditional_gates=conditional_gates,
        control_diagnostics=dict(
            outer_control_self_consistent_fraction=_outer_control_values(metrics),
            phase_speed_relative_difference=_phase_values(metrics),
            phase_speed_warning_fraction=(config.get('v9_acceptance') or {}).get(
                'phase_speed_warning_fraction', .02),
            inner_control_candidate=inner_candidate_status),
        domain=dict(passed=domain_value, physical_applicability=domain.get('physical_applicability'),
                    contact_pressure_strength=domain.get('contact_pressure_strength'),
                    checks=domain.get('checks')),
        stationarity=dict(passed=steady_value, window_count=len(subturns),
                          subwindow_complete_turns=subturns,
                          main_metric_pass=stationarity.get('main_metric_pass'),
                          contact_pass=stationarity.get('contact_pass'),
                          feature_pass=stationarity.get('feature_pass')),
        sampling=dict(passed=sampling.get('passed'), same_window=sampling.get('same_window'),
                      main_metric_pass=sampling.get('main_metric_pass'),
                      contact_pass=sampling.get('contact_pass'),
                      feature_pass=sampling.get('feature_pass'),
                      bearing_wrench_pass=sampling.get('bearing_wrench_pass'),
                      base_fs_hz=sampling.get('base_fs_hz'),
                      fine_fs_hz=(sampling.get('fine_fs_hz')
                                  if sampling.get('fine_fs_hz') is not None
                                  else sampling.get('dense_fs_hz'))))
    return output


def _case_folder_by_id(case_records, target, gap):
    gap_value = _number(gap)
    target = str(target)
    matches = []
    for record in case_records:
        if not math.isclose(float(record['gap_um']), gap_value, abs_tol=1e-10):
            continue
        if target in (record['case_id'], f"{record['group']}.{record['variant']}",
                      record['group'], record['variant']):
            matches.append(record)
    if len(matches) == 1:
        return matches[0]
    return None


def _refinement_specs(refinements, kinds=REFINEMENT_KINDS, default_target='A.N3'):
    specs = []
    if not isinstance(refinements, dict):
        return specs
    for gap_key, value in refinements.items():
        gap = _number(gap_key)
        if not isinstance(value, dict):
            continue
        if any(key in value for key in kinds) or 'target_case' in value or 'target' in value:
            target = value.get('target_case', value.get('target', default_target))
            paths = {kind: value.get(kind) for kind in kinds}
            specs.append(dict(gap_um=gap, target_case=str(target), paths=paths))
        else:
            for target, paths in value.items():
                if isinstance(paths, dict):
                    specs.append(dict(gap_um=gap, target_case=str(target),
                                      paths={kind: paths.get(kind) for kind in kinds}))
    return specs


def _local_root_details(metrics):
    geometry = {key: float(value) for key, value in metrics.items()
                if key.endswith('_max_geometry_residual_normalized') and _finite(value)}
    force_ratio = {key: float(value) for key, value in metrics.items()
                   if key.endswith('_max_force_residual_tolerance_ratio') and _finite(value)}
    failed = {key: float(value) for key, value in metrics.items()
              if key.endswith('_status_failed_count') and _finite(value)}
    complete = bool(geometry and force_ratio and failed)
    accepted = (complete and max(geometry.values()) <= 1e-9
                and max(force_ratio.values()) <= 1.0 and max(failed.values()) == 0)
    return dict(status='passed' if accepted else 'failed' if complete else 'pending',
                geometry_residual_max=geometry, geometry_limit=1e-9,
                force_residual_tolerance_ratio_max=force_ratio, force_ratio_limit=1.0,
                failed_status_count_max=failed, failed_status_limit=0.0)


def _comparison_config(root, legacy_v8=False):
    path = root / ('config/clearance_v8_config.json' if legacy_v8
                   else 'config/clearance_v9_config.json')
    config = _read_json(path)
    if not legacy_v8:
        thresholds = config['numerical_thresholds']
        # Match the explicit V9 plan limits even when a reused B0 folder carries V8 floors.
        thresholds.update(position_um=.01, velocity_mm_s=.005, contact_rate=.02,
                          relative_main=.01, relative_residual_convergence=.05,
                          relative_contact_peak=.05, relative_contact_impulse=.05)
    return config


def _metrics_comparison(base_case, fine_case, config, refinement_kind,
                        expected_variant=None, legacy_v8=False):
    base = base_case.get('metrics')
    fine = fine_case.get('metrics')
    if fine_case.get('status') == 'failed':
        return dict(status='failed', reason='independent refinement run failed',
                    failure=fine_case.get('failure'))
    if base is None or fine is None:
        return dict(status='pending', reason='metrics.json missing')
    base_values = _configuration_values(base_case)
    fine_values = _configuration_values(fine_case)
    if expected_variant:
        if base_values['variant'] is None:
            base_values['variant'] = expected_variant
        if fine_values['variant'] is None:
            fine_values['variant'] = expected_variant
    if any(base_values[key] is None or fine_values[key] is None
           for key in ('variant', 'gap_um', 'U_kg_m')):
        return dict(status='pending', reason='variant/gap/U metadata required for an independent comparison')
    mismatches = []
    if base_values['variant'] and fine_values['variant'] and base_values['variant'] != fine_values['variant']:
        mismatches.append('mechanism_variant')
    if base_values['gap_um'] is not None and fine_values['gap_um'] is not None and not math.isclose(
            base_values['gap_um'], fine_values['gap_um'], abs_tol=1e-9):
        mismatches.append('gap_um')
    if base_values['U_kg_m'] is not None and fine_values['U_kg_m'] is not None and not math.isclose(
            base_values['U_kg_m'], fine_values['U_kg_m'], rel_tol=1e-12, abs_tol=1e-15):
        mismatches.append('U_kg_m')
    if mismatches:
        return dict(status='incomparable_configuration', mismatches=mismatches,
                    base=dict(variant=base_values['variant'], gap_um=base_values['gap_um'],
                              U_kg_m=base_values['U_kg_m']),
                    fine=dict(variant=fine_values['variant'], gap_um=fine_values['gap_um'],
                              U_kg_m=fine_values['U_kg_m']))
    if int(base.get('complete_turns', 0) or 0) < 480 or int(fine.get('complete_turns', 0) or 0) < 480:
        return dict(status='pending', reason='independent comparison requires matching 480-turn final windows',
                    base_turns=base.get('complete_turns'), fine_turns=fine.get('complete_turns'))
    compared = analysis.compare_metrics(base, fine, config)
    if not compared.get('same_window'):
        return dict(status='pending', reason='analysis.compare_metrics found different measurement windows',
                    comparison=compared)
    if not legacy_v8:
        # The V9 plan assigns a 0.002 mm/s absolute floor to non-1X residual RMS.
        for key, row in compared.get('metrics', {}).items():
            if (key.startswith('H_v_') and '_residual_' in key and '_energy_' not in key
                    and _finite(row.get('base')) and _finite(row.get('fine'))):
                limit = max(.002, .05*max(abs(float(row['base'])), abs(float(row['fine']))))
                row['tolerance'] = limit
                row['passed'] = bool(float(row['difference']) <= limit)
    rows = compared.get('metrics', {})
    signal_main = ('E_x_um', 'E_y_um', 'E_dir_um')
    compared['main_metric_pass'] = bool(all(k in rows and rows[k]['passed'] for k in signal_main))
    compared['feature_pass'] = bool([row for key,row in rows.items()
        if key.startswith(('H_v_', 'H_a_'))] and all(row['passed'] for key,row in rows.items()
        if key.startswith(('H_v_', 'H_a_'))))
    compared['passed'] = bool(compared.get('same_window') and rows and all(
        row.get('passed') for row in rows.values()))
    e_row = rows.get('E_dir_um')
    if base_values['variant'] in ('B0', 'B0d'):
        local_root = dict(status='not_applicable', passed=True)
    else:
        root_base = _local_root_details(base)
        root_fine = _local_root_details(fine)
        local_root = dict(status='passed' if root_base['status'] == 'passed'
                          and root_fine['status'] == 'passed' else 'failed',
                          base=root_base, fine=root_fine,
                          passed=(root_base['status'] == 'passed'
                                  and root_fine['status'] == 'passed'))
    result_status = 'passed' if compared['passed'] and local_root['passed'] else 'failed'
    return dict(status=result_status, refinement_kind=refinement_kind,
                same_window=True, comparison=compared,
                E_dir_difference_um=(float(e_row['difference']) if e_row else None),
                local_root=local_root)


def _sampling_metric_evidence(case, metric_key):
    sampling = case.get('sampling') or {}
    row = (sampling.get('metrics') or {}).get(metric_key)
    difference = row.get('difference') if isinstance(row, dict) else None
    return dict(status=('available' if _finite(difference) and sampling.get('same_window') is True
                        else 'pending'),
                difference=float(difference) if _finite(difference) else None,
                metric_passed=row.get('passed') if isinstance(row, dict) else None,
                same_window=sampling.get('same_window'), passed=sampling.get('passed'),
                base_fs_hz=sampling.get('base_fs_hz'),
                fine_fs_hz=(sampling.get('fine_fs_hz')
                            if sampling.get('fine_fs_hz') is not None
                            else sampling.get('dense_fs_hz')))


def _sampling_evidence(case):
    evidence = _sampling_metric_evidence(case, 'E_dir_um')
    return dict(status=evidence['status'], E_dir_difference_um=evidence['difference'],
                same_window=evidence['same_window'], passed=evidence['passed'],
                base_fs_hz=evidence['base_fs_hz'], fine_fs_hz=evidence['fine_fs_hz'])


def _build_convergence(case_records, refinement_input, root, selection_parent,
                       required_kinds=REFINEMENT_KINDS, default_target='A.N3',
                       expected_variant=None, budget_scope='V9 selected case'):
    legacy_v8 = budget_scope == 'old V8 baseline'
    config = _comparison_config(root, legacy_v8=legacy_v8)
    comparisons = []
    for spec in _refinement_specs(refinement_input, required_kinds, default_target):
        target = _case_folder_by_id(case_records, spec['target_case'], spec['gap_um'])
        if target is None:
            comparisons.append(dict(target_case=spec['target_case'], gap_um=spec['gap_um'],
                status='pending', reason='target case missing or ambiguous', refinements={
                    kind: dict(status='pending', reason='target unavailable') for kind in required_kinds}))
            continue
        target_case = _load_case_json(target['path'])
        entry = dict(target_case=target['case_id'], target_path=target['path'],
                     gap_um=spec['gap_um'], refinements={},
                     expected_variant=expected_variant)
        if expected_variant and _configuration_values(target_case)['variant'] is None:
            entry['base_variant_source'] = f'{budget_scope} selection contract'
        for kind in required_kinds:
            raw_path = spec['paths'].get(kind)
            if not raw_path:
                entry['refinements'][kind] = dict(status='pending', reason='path not selected')
                continue
            path = _resolve_path(raw_path, root, selection_parent)
            refined = _load_case_json(path)
            compared = _metrics_comparison(target_case, refined, config, kind,
                                           expected_variant=expected_variant,
                                           legacy_v8=legacy_v8)
            compared.update(path=str(path), failure=refined.get('failure'))
            compared['comparison_tolerance_profile'] = 'V8' if legacy_v8 else 'V9'
            entry['refinements'][kind] = compared
        entry['output_sampling'] = _sampling_evidence(target_case)
        components = {'output_sampling': entry['output_sampling'].get('E_dir_difference_um')}
        components.update({kind: entry['refinements'][kind].get('E_dir_difference_um')
                           for kind in required_kinds})
        available = {key: float(value) for key, value in components.items() if _finite(value)}
        required_components = ('output_sampling',) + tuple(required_kinds)
        complete = all(key in available and (
            entry['output_sampling'].get('status') == 'available' if key == 'output_sampling'
            else entry['refinements'][key].get('same_window') is True)
                       for key in required_components)
        comparison_statuses = [entry['refinements'][kind].get('status') for kind in required_kinds]
        budget_passed = bool(complete and entry['output_sampling'].get('passed') is True
                             and all(status == 'passed' for status in comparison_statuses))
        convergence_absolute_um = .005 if legacy_v8 else .01
        convergence_profile = 'V8' if legacy_v8 else 'V9'
        entry['epsilon_E'] = dict(
            definition='Maximum absolute independent E_dir difference across output-sampling and the listed refinement kinds; engineering error estimate, not a confidence interval.',
            components_um=components,
            available_components_um=available,
            epsilon_num_um=max(available.values()) if available else None,
            comparison_tolerance_profile=convergence_profile,
            convergence_tolerance_absolute_um=convergence_absolute_um,
            convergence_tolerance_relative_fraction=.01,
            convergence_tolerance_definition=f'{convergence_profile} E_dir comparison pass threshold max({convergence_absolute_um:g} um, 1% of reference); this is separate from epsilon_num and the S_E denominator floor.',
            required_components=list(required_components),
            complete=complete, convergence_comparisons_passed=budget_passed)
        epsilon_h = {}
        for metric_key in H_VELOCITY_METRICS:
            sampling_h = _sampling_metric_evidence(target_case, metric_key)
            h_components = {'output_sampling': sampling_h['difference']}
            h_passes = {'output_sampling': sampling_h['metric_passed']}
            h_same_window = {'output_sampling': sampling_h['same_window'] is True}
            for kind in required_kinds:
                refinement = entry['refinements'][kind]
                compared_h = ((refinement.get('comparison') or {}).get('metrics') or {}).get(
                    metric_key, {})
                same_window = (refinement.get('same_window') is True
                               and (refinement.get('comparison') or {}).get('same_window') is True)
                difference = compared_h.get('difference') if same_window else None
                h_components[kind] = float(difference) if _finite(difference) else None
                h_passes[kind] = compared_h.get('passed') if same_window else None
                h_same_window[kind] = same_window
            h_required = ('output_sampling',) + tuple(required_kinds)
            h_available = {key: float(value) for key, value in h_components.items()
                           if _finite(value) and h_same_window.get(key) is True}
            h_complete = all(key in h_available for key in h_required)
            h_convergence_passed = bool(h_complete and all(
                h_passes.get(key) is True for key in h_required))
            epsilon_h[metric_key] = dict(
                unit='mm/s',
                definition='Maximum absolute independent difference for this H velocity metric across same-window output sampling and the listed refinement kinds; engineering error estimate, not a confidence interval.',
                components_mm_s=h_components,
                available_components_mm_s=h_available,
                component_passes=h_passes,
                same_window_by_component=h_same_window,
                epsilon_num_mm_s=max(h_available.values()) if h_complete else None,
                maximum_available_component_mm_s=max(h_available.values()) if h_available else None,
                required_components=list(h_required), complete=h_complete,
                convergence_comparisons_passed=h_convergence_passed)
        entry['epsilon_H_velocity'] = epsilon_h
        comparisons.append(entry)
    return comparisons


def _significance(case_records, convergence):
    budget_by_case = {row['target_case']: row.get('epsilon_E', {}) for row in convergence
                      if row.get('target_case')}
    by_key = {(case['group'], case['variant'], _gap_tag(case['gap_um'])): case
              for case in case_records}
    result = []
    gaps = sorted({key[2] for key in by_key if key[0] == 'A'})
    for gap in gaps:
        old = by_key.get(('A', 'B0', gap))
        new = by_key.get(('A', 'N3', gap))
        if old is None or new is None:
            result.append(dict(gap_um=float(gap), status='pending',
                               reason='matched A.B0 and A.N3 cases are both required'))
            continue
        old_u, new_u = old.get('U_kg_m'), new.get('U_kg_m')
        same_u = (_finite(old_u) and _finite(new_u) and math.isclose(
            old_u, new_u, rel_tol=1e-12, abs_tol=1e-15))
        old_e, new_e = old.get('E_dir_um'), new.get('E_dir_um')
        delta = abs(float(new_e)-float(old_e)) if _finite(new_e) and _finite(old_e) else None
        relative = (100*(float(new_e)-float(old_e))/abs(float(old_e))
                    if _finite(new_e) and _finite(old_e) and float(old_e) != 0 else None)
        old_budget = budget_by_case.get(old['case_id'], {})
        new_budget = budget_by_case.get(new['case_id'], {})
        complete = bool(old_budget.get('complete') and new_budget.get('complete'))
        epsilon_floor = SIGNIFICANCE_EPSILON_FLOOR_UM
        record = dict(gap_um=float(gap), old_case=old['case_id'], new_case=new['case_id'],
            same_U=same_u, old_U_kg_m=old_u, new_U_kg_m=new_u,
            E_old_um=old_e, E_new_um=new_e, absolute_difference_um=delta,
            signed_difference_um=(float(new_e)-float(old_e)
                                  if _finite(new_e) and _finite(old_e) else None),
            relative_change_percent=relative, epsilon_old_um=old_budget.get('epsilon_num_um'),
            epsilon_new_um=new_budget.get('epsilon_num_um'), epsilon_floor_um=epsilon_floor,
            epsilon_floor_source='Inherited predeclared V8 precision floor: config/clearance_v8_config.json health_result.epsilon_E_um=0.0001 um; same value is retained in GPTOrder/spindle_v9_tai_bearing_plan/baseline_snapshot/parameters_v8_calibrated.json. This is not the V9 E_dir convergence acceptance tolerance.',
            epsilon_num_definition='Each epsilon_num is only the maximum measured independent E_dir difference for that case; no acceptance threshold is inserted into epsilon_num.',
            epsilon_definition='S_E=|E_new-E_old|/(epsilon_num_old+epsilon_num_new+0.0001 um). The fixed 0.0001 um floor is predeclared from the inherited V8 precision baseline.',
            error_budgets_complete=complete, S_E=None,
            S_E_conservative_floor_0p01um=None,
            conservative_floor_um=CONSERVATIVE_SIGNIFICANCE_FLOOR_UM)
        if not same_u:
            record.update(status='not_comparable_different_U', S_E=None)
        elif not complete or delta is None:
            record.update(status='pending_error_budget', S_E=None)
        else:
            denominator = (float(old_budget['epsilon_num_um'])
                           + float(new_budget['epsilon_num_um']) + float(epsilon_floor))
            score = delta/denominator if denominator else None
            conservative_denominator = (float(old_budget['epsilon_num_um'])
                + float(new_budget['epsilon_num_um']) + CONSERVATIVE_SIGNIFICANCE_FLOOR_UM)
            conservative_score = delta/conservative_denominator if conservative_denominator else None
            convergence_pass = bool(old_budget.get('convergence_comparisons_passed')
                                    and new_budget.get('convergence_comparisons_passed'))
            record.update(status=('exceeds_numeric_error' if score > 5 else 'within_numeric_error')
                                  if convergence_pass else 'conditional_convergence_not_passed',
                          S_E=score, significance_threshold=5,
                          S_E_conservative_floor_0p01um=conservative_score,
                          conservative_floor_um=CONSERVATIVE_SIGNIFICANCE_FLOOR_UM,
                          conservative_floor_definition='Optional sensitivity only: add a separate 0.01 um absolute floor to the same measured epsilon_num values; it does not replace the primary V8-inherited floor.',
                          convergence_budgets_passed=convergence_pass)
        result.append(record)
    return result


def _h_velocity_significance(case_records, convergence):
    budget_by_case = {row['target_case']: row.get('epsilon_H_velocity', {})
                      for row in convergence if row.get('target_case')}
    by_key = {(case['group'], case['variant'], _gap_tag(case['gap_um'])): case
              for case in case_records}
    result = []
    gaps = sorted({key[2] for key in by_key if key[0] == 'A'})
    for gap in gaps:
        old = by_key.get(('A', 'B0', gap))
        new = by_key.get(('A', 'N3', gap))
        for metric_key in H_VELOCITY_METRICS:
            if old is None or new is None:
                result.append(dict(gap_um=float(gap), metric=metric_key, unit='mm/s',
                    status='pending', reason='matched A.B0 and A.N3 cases are both required',
                    S_y=None))
                continue
            old_u, new_u = old.get('U_kg_m'), new.get('U_kg_m')
            same_u = (_finite(old_u) and _finite(new_u) and math.isclose(
                old_u, new_u, rel_tol=1e-12, abs_tol=1e-15))
            old_value, new_value = old.get(metric_key), new.get(metric_key)
            delta = (abs(float(new_value) - float(old_value))
                     if _finite(old_value) and _finite(new_value) else None)
            relative = (100 * (float(new_value) - float(old_value)) / abs(float(old_value))
                        if _finite(old_value) and _finite(new_value)
                        and float(old_value) != 0 else None)
            old_budget = budget_by_case.get(old['case_id'], {}).get(metric_key, {})
            new_budget = budget_by_case.get(new['case_id'], {}).get(metric_key, {})
            old_complete = old_budget.get('complete') is True
            new_complete = new_budget.get('complete') is True
            complete = old_complete and new_complete
            floor = H_VELOCITY_EPSILON_FLOOR_MM_S
            record = dict(gap_um=float(gap), metric=metric_key, unit='mm/s',
                old_case=old['case_id'], new_case=new['case_id'],
                same_U=same_u, old_U_kg_m=old_u, new_U_kg_m=new_u,
                old_value_mm_s=old_value, new_value_mm_s=new_value,
                absolute_difference_mm_s=delta, relative_change_percent=relative,
                epsilon_old_mm_s=old_budget.get('epsilon_num_mm_s'),
                epsilon_new_mm_s=new_budget.get('epsilon_num_mm_s'),
                epsilon_old_complete=old_complete, epsilon_new_complete=new_complete,
                epsilon_old_components_mm_s=old_budget.get('components_mm_s'),
                epsilon_new_components_mm_s=new_budget.get('components_mm_s'),
                epsilon_old_required_components=old_budget.get('required_components'),
                epsilon_new_required_components=new_budget.get('required_components'),
                error_budgets_complete=complete,
                epsilon_floor_mm_s=floor,
                epsilon_floor_source=H_VELOCITY_EPSILON_FLOOR_SOURCE,
                epsilon_num_definition='epsilon_num is the maximum measured independent same-window difference across output sampling and the applicable h/N/root refinements; it excludes the floor and all acceptance tolerances.',
                epsilon_floor_interpretation='Fixed engineering error floor inherited from V8; it is not a statistical confidence interval or measured detectability.',
                significance_definition='S_y=|y_N3-y_B0|/(epsilon_num_B0+epsilon_num_N3+0.0001 mm/s), reported only for matching-U cases with complete same-window independent budgets.',
                S_y=None, significance_threshold=5)
            if not same_u:
                record.update(status='not_comparable_different_U', S_y=None)
            elif not complete or delta is None:
                record.update(status='pending_error_budget', S_y=None)
            else:
                denominator = (float(old_budget['epsilon_num_mm_s'])
                    + float(new_budget['epsilon_num_mm_s']) + floor)
                score = delta / denominator if denominator else None
                convergence_passed = bool(
                    old_budget.get('convergence_comparisons_passed')
                    and new_budget.get('convergence_comparisons_passed'))
                record.update(status=('exceeds_numeric_error' if score > 5
                                      else 'within_numeric_error') if convergence_passed
                                      else 'conditional_convergence_not_passed',
                    S_y=score, convergence_budgets_passed=convergence_passed)
            result.append(record)
    return result


def _read_validation_item(value, default_path, root, selection_parent):
    path = _resolve_path(value or default_path, root, selection_parent)
    if not path or not Path(path).is_file():
        return dict(status='pending', path=str(path) if path else None,
                    reason='required validation JSON missing')
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        return dict(status='pending', path=str(path), reason=f'cannot read JSON: {error}')
    return dict(status='available', path=str(path), data=data)


def _suite_summary(suite, suite_path):
    if suite['status'] != 'available':
        return dict(status='pending', path=suite.get('path'), reason=suite.get('reason'))
    data = suite['data']
    checks = data.get('checks', [])
    integration = data.get('system_integration', {})
    integration_rows = integration.get('checks') or []
    integration_ids = {str(row.get('check_id')) for row in integration_rows
                       if row.get('check_id')}
    # The top-level `checks` list contains the system checks as well in the
    # current 281-case report. Identify those by exact IDs from the nested
    # system-integration record: broad T07/T09/T10/T11/T12 prefixes also match
    # genuine local tests and undercount the local suite.
    if integration_ids:
        local_rows = [row for row in checks
                      if str(row.get('check_id', '')) not in integration_ids]
        local_count_source = 'top-level checks minus exact system_integration.check_ids'
    elif not (data.get('summary', {}).get('system_checks') or 0):
        local_rows = checks
        local_count_source = 'all top-level checks; no system checks declared'
    else:
        # Do not invent a local/system split when an older schema declares
        # system checks but does not identify them individually.
        local_rows = []
        local_count_source = 'unavailable; system checks are not individually identified'
    counts = {status: sum(row.get('status') == status for row in local_rows)
              for status in ('passed', 'conditional', 'failed', 'pending')}
    per_test_id = {}
    for row in local_rows:
        test_id = str(row.get('check_id', 'unknown'))
        status_key = str(row.get('status', 'pending'))
        per_test_id.setdefault(test_id, {})
        per_test_id[test_id][status_key] = per_test_id[test_id].get(status_key, 0) + 1
    status = ('failed' if counts['failed'] else 'pending' if counts['pending']
              else 'conditional' if counts['conditional'] else 'numerical_pass')
    original = data.get('summary', {})
    source_revision = _solver_retry_revision_summary(suite_path)
    local_split_known = local_count_source != (
        'unavailable; system checks are not individually identified')
    if not local_split_known:
        status = 'pending'
    return dict(status=status, schema=data.get('schema'), path=str(suite_path),
        scope=data.get('scope'), local_check_count=(len(local_rows) if local_split_known else None),
        local_check_count_source=local_count_source,
        local_check_status_counts=counts if local_split_known else None,
        per_test_id_status_counts=per_test_id,
        test_ids=sorted({row.get('check_id') for row in local_rows}),
        original_suite_check_count=original.get('checks_total'),
        original_suite_status_counts={key: original.get(key) for key in ('passed','conditional','failed')},
        system_integration_status=(integration.get('status')
                                   or original.get('short_system_integration_status')),
        system_integration_check_count=(integration.get('summary') or {}).get('checks_total'),
        system_integration_check_ids=sorted(integration_ids),
        machine_validated=original.get('machine_validated', False),
        source_revision=source_revision,
        note='This status covers the original local test suite only. T06 contact-loss boundary continuity is reported separately; counts do not mean every T01–T12 requirement is complete.')


def _solver_retry_revision_summary(suite_path):
    if not suite_path or Path(suite_path).parent.name != 'solver_retry_local':
        return None
    path = Path(suite_path).parent.parent / 'solver_retry_regression.json'
    if not path.is_file():
        return dict(status='pending', path=str(path), reason='solver-retry source revision record missing')
    data = _read_json(path)
    random = data.get('random_regression') or {}
    families = random.get('families') or {}
    saved = data.get('saved_boundary_samples') or {}
    formal = data.get('formal_existing_checks') or {}
    return dict(path=str(path), status='passed' if data.get('passed') is True else 'failed',
        scope=data.get('scope'), old_source=data.get('old_source'), new_source=data.get('new_source'),
        random_regression=dict(passed=random.get('passed'),
            front_state_count=(families.get('front') or {}).get('state_count'),
            front_original_successes_bitwise_unchanged=(families.get('front') or {}).get('bitwise_unchanged'),
            rear_state_count=(families.get('rear') or {}).get('state_count'),
            rear_original_successes_bitwise_unchanged=(families.get('rear') or {}).get('bitwise_unchanged')),
        saved_boundary_samples={key: saved.get(key) for key in
            ('sample_count', 'original_failures', 'new_failures', 'recovered_samples',
             'original_successes_changed', 'note')},
        formal_existing_checks=dict(path=formal.get('path'), status=formal.get('status'),
            summary=formal.get('summary')),
        note='Deterministic retry after the original Newton path fails; only that failure path changes. The unchanged-root and sampled-boundary regression records remain attached.')


def _local_control_branch_audit(suite):
    if suite['status'] != 'available':
        return dict(status='pending', path=suite.get('path'), reason=suite.get('reason'))
    rows = suite['data'].get('checks', [])
    selected = [row for row in rows if row.get('check_id') in (
        'T08_CONTROL_CLASSIFICATION', 'T08_CONTROL_SELF_CONSISTENCY',
        'T08_CAGE_PHASE_SPEED_DIAGNOSTIC')]
    if not selected:
        return dict(status='pending', reason='T08 local branch/phase evidence missing',
                    path=suite.get('path'))
    counts = {status: sum(row.get('status') == status for row in selected)
              for status in ('passed', 'conditional', 'failed', 'pending')}
    branches = set()
    classifications = {}
    candidate_consistency = {'outer': {'cases': 0, 'self_consistent': 0},
                             'inner': {'cases': 0, 'self_consistent': 0}}
    phase_differences = []
    for row in selected:
        metrics = row.get('metrics') or {}
        for branch in metrics.get('evaluated_branches', []):
            branches.add(str(branch))
        classification = metrics.get('classification')
        if classification:
            classifications[str(classification)] = classifications.get(str(classification), 0) + 1
        case_id = str(row.get('case_id', ''))
        if row.get('check_id') == 'T08_CONTROL_SELF_CONSISTENCY':
            for branch in ('outer', 'inner'):
                if case_id.endswith(':' + branch):
                    candidate_consistency[branch]['cases'] += 1
                    if metrics.get('production_flag') is True:
                        candidate_consistency[branch]['self_consistent'] += 1
        if row.get('check_id') == 'T08_CAGE_PHASE_SPEED_DIAGNOSTIC':
            value = metrics.get('reported_relative_difference', metrics.get('relative_difference'))
            if _finite(value):
                phase_differences.append(float(value))
    complete = ('outer' in branches and 'inner' in branches
                and candidate_consistency['outer']['cases'] > 0
                and candidate_consistency['inner']['cases'] > 0
                and bool(phase_differences))
    status = ('failed' if counts['failed'] else 'pending' if counts['pending'] or not complete
              else 'conditional' if counts['conditional'] else 'numerical_pass')
    return dict(status=status, path=suite.get('path'),
        scope='Independent local candidate-control and cage-phase sweep; this evidence is not re-solved at each long-trajectory RHS state.',
        check_count=len(selected), check_status_counts=counts,
        evaluated_branches=sorted(branches), classification_counts=classifications,
        candidate_branch_counts=candidate_consistency,
        phase_speed_relative_difference_max=(max(phase_differences) if phase_differences else None),
        note='Production N3 outer-control and phase-speed gates remain separate per-case checks; alternate inner candidates are supported by this local sweep and are not a required per-RHS trajectory artifact.')


def _system_integration_summary(suite):
    if suite['status'] != 'available':
        return dict(status='pending', path=suite.get('path'), reason=suite.get('reason'))
    item = suite['data'].get('system_integration') or {}
    summary = item.get('summary') or {}
    checks = item.get('checks') or []
    counts = {status: sum(row.get('status') == status for row in checks)
              for status in ('passed', 'conditional', 'failed', 'pending')}
    status = item.get('status') or summary.get('status') or 'pending'
    return dict(status=status, path=suite.get('path'), scope=item.get('scope'),
        check_count=summary.get('checks_total', len(checks)), check_status_counts=counts,
        static_equilibrium_passed=summary.get('static_equilibrium_passed'),
        trajectory_revolutions=summary.get('trajectory_revolutions'),
        machine_validated=summary.get('machine_validated', False),
        note='Previously completed static and short-trajectory integration checks; does not cover the long experiment matrix or contact-loss continuity.')


def _contact_boundary_summary(item):
    if item['status'] != 'available':
        return dict(status='pending', path=item.get('path'), reason=item.get('reason'),
                    capability='generic inner-contact-loss continuity')
    data = item['data']
    summary = data.get('summary') or {}
    cases = data.get('cases') or []
    finite_speed_pass = summary.get('finite_speed_boundary_continuity_pass')
    zero_speed_pass = summary.get('zero_speed_cases_pass')
    traces_pass = summary.get('independent_traces_all_pass')

    # This schema reports the boundary result under summary rather than a top-level
    # status/check list. Honor that explicit gate before any generic fallbacks.
    if finite_speed_pass is False or zero_speed_pass is False or traces_pass is False:
        status = 'failed'
    elif finite_speed_pass is True and zero_speed_pass is True and traces_pass is True:
        status = 'passed'
    else:
        checks = data.get('checks')
        statuses = []
        if isinstance(checks, list):
            statuses = [row.get('status') for row in checks if isinstance(row, dict)]
        elif isinstance(checks, dict):
            statuses = [row.get('passed') if isinstance(row, dict) else row
                        for row in checks.values()]
        explicit = data.get('passed')
        if explicit is None:
            explicit = data.get('pass')
        declared = data.get('status')
        if explicit is True:
            status = 'passed'
        elif explicit is False:
            status = 'failed'
        elif statuses:
            normalized = [str(value).lower() for value in statuses]
            if any(value in ('false', 'failed', 'fail', 'rejected') for value in normalized):
                status = 'failed'
            elif any(value in ('pending', 'none') for value in normalized):
                status = 'pending'
            elif any(value == 'conditional' for value in normalized):
                status = 'conditional'
            elif all(value in ('true', 'passed', 'pass', 'numerical_pass') for value in normalized):
                status = 'passed'
            else:
                status = 'pending'
        elif declared in ('pass', 'passed', 'numerical_pass'):
            status = 'passed'
        elif declared in ('failed', 'fail', 'rejected'):
            status = 'failed'
        elif declared == 'conditional':
            status = 'conditional'
        else:
            status = 'pending'

    case_summary = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        difference = case.get('loaded_minus_prescribed_limit') or {}
        case_summary.append(dict(
            family=case.get('family'), A_label=case.get('A_label'),
            rpm_magnitude=case.get('rpm_magnitude'),
            F07_minus_F10_boundary_difference=dict(
                (key, difference.get(key)) for key in
                ('Qi_N', 'Qo_N', 'Fc_N', 'Mg_Nm', 'B_m')),
            boundary_continuity_pass=case.get('boundary_continuity_pass'),
            trace_all_pass=case.get('trace_all_pass'),
            root_failure_count=case.get('root_failure_count'),
            verdict=case.get('verdict')))
    return dict(status=status, path=item['path'], schema=data.get('schema'),
        scope=data.get('scope', data.get('purpose')),
        summary=dict(case_count=summary.get('case_count', len(case_summary)),
            independent_traces_all_pass=traces_pass,
            zero_speed_cases_pass=zero_speed_pass,
            finite_speed_boundary_continuity_pass=finite_speed_pass,
            root_failure_samples=summary.get('root_failure_samples'),
            verdict=summary.get('verdict')),
        cases=case_summary,
        note='A failure here blocks generic contact-loss capability; it does not invalidate a separately passing dual-contact N3 trajectory.')


def _selection_cases(selection, root, selection_parent):
    cases = []
    mappings = selection.get('A') or {}
    for variant, gaps in mappings.items():
        if not isinstance(gaps, dict):
            continue
        for gap_key, raw_path in gaps.items():
            gap = _number(gap_key)
            declared = _resolve_path(raw_path, root, selection_parent)
            chosen, reason = _select_a_path(root, variant, gap, declared)
            if chosen is None:
                case = dict(folder=None, status='pending', reason='case path not selected')
            else:
                case = _load_case_json(chosen)
            cases.append(_case_acceptance(case, 'A', variant, gap, reason))
    b_mappings = selection.get('B') or {}
    for key, value in b_mappings.items():
        # B supports the plan's gap->path map; a nested variant->gap->path map
        # is also accepted so a B result can point directly at an A case folder.
        is_gap_key = bool(re.fullmatch(
            r'\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*(?:um|µm)?\s*',
            str(key), flags=re.IGNORECASE))
        entries = [(key, value, None)] if is_gap_key else (
            [(gap_key, raw_path, str(key)) for gap_key, raw_path in value.items()]
            if isinstance(value, dict) else [])
        for gap_key, raw_path, variant_hint in entries:
            gap = _number(gap_key)
            path = _resolve_path(raw_path, root, selection_parent)
            case = (_load_case_json(path) if path else
                    dict(status='pending', reason='case path not selected'))
            variant = _configuration_values(case).get('variant') or variant_hint or 'N3'
            cases.append(_case_acceptance(case, 'B', variant, gap,
                                          'explicit_case_selection_path'))
    return cases


def _original_failure_summary(value, root, selection_parent):
    path = _resolve_path(value, root, selection_parent)
    # `_resolve_path` normally converts run-result JSON paths into their case
    # directory. Here `failed.json` itself is the evidence being summarized,
    # so restore the file path after that general case-folder normalization.
    if value and Path(value).name == 'failed.json' and path and Path(path).is_dir():
        failure_file = Path(path) / 'failed.json'
        if failure_file.is_file():
            path = failure_file
    if not path or not Path(path).is_file():
        return dict(status='not_recorded', path=str(path) if path else None)
    data = _read_json(path)
    if isinstance(data, list):
        failures = data
        rows = failures[:5]
        return dict(status='recorded' if failures else 'no_failure_recorded',
            path=str(path), failure_count=len(failures),
            failures=[{key: row.get(key) for key in ('folder', 'error', 'error_type')}
                      for row in rows if isinstance(row, dict)])
    local = data.get('local_failure') or {}
    failure_state_path = Path(path).parent / 'failure_state.npz'
    return dict(status='recorded' if data.get('error') else 'no_failure_recorded',
        path=str(path), error_type=data.get('error_type'), error=data.get('error'),
        failure_state_saved=data.get('failure_state_saved'),
        local_failure={key: local.get(key) for key in ('t_s', 'error', 'fallback_used')},
        failure_state_path=str(failure_state_path) if failure_state_path.is_file() else None)


def _initial_state_phase_interval(metrics, config):
    parameters = config.get('parameters') or {}
    excitation = config.get('excitation') or {}
    rpm = parameters.get('rpm')
    phase = excitation.get('phase_rad')
    start = metrics.get('measurement_start_s')
    end = metrics.get('measurement_end_s')
    if not all(_finite(value) for value in (rpm, phase, start, end)):
        return dict(status='pending', measurement_start_s=start, measurement_end_s=end)
    omega = float(rpm) * 2 * math.pi / 60
    phase_start = omega * float(start) + float(phase)
    phase_end = omega * float(end) + float(phase)
    return dict(status='available', measurement_start_s=float(start),
        measurement_end_s=float(end), duration_s=float(end)-float(start),
        input_phase_start_rad=phase_start % (2*math.pi),
        input_phase_end_rad=phase_end % (2*math.pi),
        rpm_signed=float(rpm), input_phase_offset_rad=float(phase))


def _initial_state_phase_diagnostics(reference_metrics, branch_metrics):
    keys = ('H_v_1x_phase_x_rad', 'H_v_1x_phase_y_rad',
            'H_a_1x_phase_x_rad', 'H_a_1x_phase_y_rad',
            'M_1x_phase_x_rad', 'M_1x_phase_y_rad')
    rows = {}
    for key in keys:
        first, second = reference_metrics.get(key), branch_metrics.get(key)
        if not (_finite(first) and _finite(second)):
            rows[key] = dict(status='pending', reference_rad=first, branch_rad=second)
            continue
        difference = math.atan2(math.sin(float(second)-float(first)),
                                math.cos(float(second)-float(first)))
        rows[key] = dict(status='diagnostic_only', reference_rad=float(first),
            branch_rad=float(second), wrapped_difference_rad=difference,
            wrapped_difference_deg=math.degrees(difference),
            threshold=None, passed=None)
    return rows


def _initial_state_comparison(spec, root, selection_parent, config):
    reference_path = _resolve_path(spec.get('reference_path'), root, selection_parent)
    branch_path = _resolve_path(spec.get('branch_path'), root, selection_parent)
    reference = _load_case_json(reference_path) if reference_path else dict(
        status='pending', reason='reference path not selected')
    branch = _load_case_json(branch_path) if branch_path else dict(
        status='pending', reason='initial-state branch path not selected')
    ref_metrics = reference.get('metrics') or {}
    branch_metrics = branch.get('metrics') or {}
    ref_config = reference.get('config') or {}
    branch_config = branch.get('config') or {}
    ref_values = _configuration_values(reference)
    branch_values = _configuration_values(branch)
    failure = _original_failure_summary(spec.get('original_failure_evidence_path'),
                                        root, selection_parent)
    record = dict(comparison_id=spec.get('comparison_id'),
        reference_path=str(reference_path) if reference_path else None,
        branch_path=str(branch_path) if branch_path else None,
        reference_case=dict(status=reference.get('status'),
            complete_turns=ref_metrics.get('complete_turns'),
            measurement_start_s=ref_metrics.get('measurement_start_s'),
            measurement_end_s=ref_metrics.get('measurement_end_s'),
            stationarity_pass=(reference.get('stationarity') or {}).get('passed')),
        branch_case=dict(status=branch.get('status'),
            complete_turns=branch_metrics.get('complete_turns'),
            measurement_start_s=branch_metrics.get('measurement_start_s'),
            measurement_end_s=branch_metrics.get('measurement_end_s'),
            stationarity_pass=(branch.get('stationarity') or {}).get('passed')),
        original_failure_evidence=failure,
        initial_state_provenance=dict(seed_argument=spec.get('initial_seed_argument'),
            seed_file_present=bool(branch_path and (Path(branch_path)/'initial_seed.npz').is_file()),
            comparison_note='A saved initial state changes the branch start; all final-window metrics are compared as steady statistics.'),
        comparison_basis='steady_window_summary_statistics',
        absolute_window_is_numerical_refinement=False,
        same_absolute_time_window=None,
        time_window_explanation='Different absolute integration intervals are not aligned samples and are not treated as a numerical refinement. The same-window flag is retained only as a diagnostic; summary statistics are compared under existing V9 metric tolerances after both 480-turn windows pass stationarity.',
        metric_comparison=None, phase_interval=None, phase_comparison=None,
        status='pending')
    if reference.get('status') != 'available' or branch.get('status') != 'available':
        if reference.get('status') == 'failed' or branch.get('status') == 'failed':
            record['status'] = 'failed'
            record['reason'] = 'one or both runs have a saved failure and no completed metrics'
        else:
            record['reason'] = 'one or both case metrics are missing; pending without treating the run as a pass'
        return record

    ref_sim = ref_config.get('simulation') or {}
    branch_sim = branch_config.get('simulation') or {}
    numerical_keys = ('method', 'rtol', 'atol_translation_m', 'atol_velocity_m_s',
        'atol_angle_rad', 'atol_angle_rate_rad_s', 'fs_raw_hz', 'fs_out_hz',
        'max_step_s', 'retained_turns', 'subwindows', 'dense_sampling_multiplier')
    configuration = dict(
        variant_match=ref_values['variant'] == branch_values['variant'],
        gap_match=(_finite(ref_values['gap_um']) and _finite(branch_values['gap_um'])
                   and math.isclose(ref_values['gap_um'], branch_values['gap_um'], abs_tol=1e-9)),
        U_match=(_finite(ref_values['U_kg_m']) and _finite(branch_values['U_kg_m'])
                 and math.isclose(ref_values['U_kg_m'], branch_values['U_kg_m'],
                                  rel_tol=1e-12, abs_tol=1e-15)),
        force_match=(_finite(ref_values['force_N']) and _finite(branch_values['force_N'])
                     and math.isclose(ref_values['force_N'], branch_values['force_N'],
                                      rel_tol=1e-12, abs_tol=1e-12)),
        rpm_match=(_finite(ref_values['rpm_signed']) and _finite(branch_values['rpm_signed'])
                   and math.isclose(ref_values['rpm_signed'], branch_values['rpm_signed'],
                                    rel_tol=1e-12, abs_tol=1e-12)),
        physical_parameters_match=ref_config.get('parameters') == branch_config.get('parameters'),
        excitation_match=ref_config.get('excitation') == branch_config.get('excitation'),
        clearance_configuration_match=ref_config.get('clearances_um') == branch_config.get('clearances_um'),
        numerical_integration_settings_match=all(ref_sim.get(key) == branch_sim.get(key)
            for key in numerical_keys))
    configuration['passed'] = all(value is True for value in configuration.values())
    record['configuration_match'] = configuration
    ref_turns = int(ref_metrics.get('complete_turns', 0) or 0)
    branch_turns = int(branch_metrics.get('complete_turns', 0) or 0)
    turns_pass = ref_turns >= 480 and branch_turns >= 480 and ref_turns == branch_turns
    ref_steady = (reference.get('stationarity') or {}).get('passed') is True
    branch_steady = (branch.get('stationarity') or {}).get('passed') is True
    record['turns_gate'] = dict(reference=ref_turns, branch=branch_turns, passed=turns_pass)
    record['stationarity_gate'] = dict(reference=ref_steady, branch=branch_steady,
                                        passed=ref_steady and branch_steady)

    compared = analysis.compare_metrics(ref_metrics, branch_metrics, config)
    rows = compared.get('metrics') or {}
    metric_statuses = [bool(row.get('passed')) for row in rows.values()]
    metrics_pass = bool(rows and all(metric_statuses))
    record['metric_comparison'] = dict(
        same_window=compared.get('same_window') is True,
        same_window_gate_applied=False,
        status='passed' if metrics_pass else 'failed' if rows else 'pending',
        passed_metric_count=sum(metric_statuses), failed_metric_count=sum(
            not value for value in metric_statuses), metric_count=len(rows),
        metrics=rows)
    record['same_absolute_time_window'] = compared.get('same_window') is True
    record['phase_interval'] = dict(
        reference=_initial_state_phase_interval(ref_metrics, ref_config),
        branch=_initial_state_phase_interval(branch_metrics, branch_config))
    phase_diff_s = (float(branch_metrics['measurement_start_s'])
                    - float(ref_metrics['measurement_start_s']))
    rpm = ref_values.get('rpm_signed')
    record['phase_interval']['start_time_offset_s'] = phase_diff_s
    record['phase_interval']['start_offset_shaft_turns'] = (
        abs(float(rpm))/60*phase_diff_s if _finite(rpm) else None)
    record['phase_interval']['input_phase_offset_wrap_rad'] = (
        math.atan2(math.sin(2*math.pi*float(rpm)/60*phase_diff_s),
                   math.cos(2*math.pi*float(rpm)/60*phase_diff_s))
        if _finite(rpm) else None)
    record['phase_comparison'] = _initial_state_phase_diagnostics(ref_metrics, branch_metrics)

    settings_path = Path(branch_path).parent / 'actual_command.json'
    if settings_path.is_file():
        command = _read_json(settings_path)
        argv = command.get('argv') or []
        seed_index = argv.index('--seed')+1 if '--seed' in argv else None
        record['initial_state_provenance']['actual_seed_argument'] = (
            argv[seed_index] if seed_index is not None and seed_index < len(argv)
            else None)
        record['initial_state_provenance']['actual_command_path'] = str(settings_path)

    ready = configuration['passed'] and turns_pass and ref_steady and branch_steady
    if not ready:
        record['status'] = 'failed' if not configuration['passed'] or not ref_steady or not branch_steady else 'pending'
    elif not rows:
        record['status'] = 'pending'
    else:
        record['status'] = 'steady_statistics_passed' if metrics_pass else 'steady_statistics_failed'
    return record


def _initial_state_comparisons(selection, root, selection_parent):
    specs = selection.get('initial_state_comparisons', DEFAULT_INITIAL_STATE_COMPARISONS)
    if isinstance(specs, dict):
        specs = [dict(comparison_id=key, **value) for key, value in specs.items()]
    if selection_parent is None:
        selection_parent = root / DEFAULT_EXPERIMENT
    config = _comparison_config(root)
    return [_initial_state_comparison(spec, root, selection_parent, config)
            for spec in specs if isinstance(spec, dict)]


def _initial_state_csv_rows(comparisons):
    rows = []
    for pair in comparisons:
        common = dict(comparison_id=pair.get('comparison_id'), status=pair.get('status'),
            reference_path=pair.get('reference_path'), branch_path=pair.get('branch_path'),
            comparison_basis=pair.get('comparison_basis'),
            absolute_window_is_numerical_refinement=pair.get('absolute_window_is_numerical_refinement'),
            same_absolute_time_window=pair.get('same_absolute_time_window'),
            reference_start_s=(pair.get('reference_case') or {}).get('measurement_start_s'),
            reference_end_s=(pair.get('reference_case') or {}).get('measurement_end_s'),
            branch_start_s=(pair.get('branch_case') or {}).get('measurement_start_s'),
            branch_end_s=(pair.get('branch_case') or {}).get('measurement_end_s'),
            start_offset_shaft_turns=(pair.get('phase_interval') or {}).get('start_offset_shaft_turns'),
            failure_evidence_path=(pair.get('original_failure_evidence') or {}).get('path'),
            failure_error=(pair.get('original_failure_evidence') or {}).get('error'))
        metrics = (pair.get('metric_comparison') or {}).get('metrics') or {}
        for metric, value in metrics.items():
            rows.append(dict(**common, metric=metric, reference_value=value.get('base'),
                branch_value=value.get('fine'), difference=value.get('difference'),
                tolerance=value.get('tolerance'), metric_passed=value.get('passed'),
                row_kind='steady_metric'))
        for metric, value in (pair.get('phase_comparison') or {}).items():
            rows.append(dict(**common, metric=metric,
                reference_value=value.get('reference_rad'), branch_value=value.get('branch_rad'),
                difference=value.get('wrapped_difference_rad'), tolerance=None,
                metric_passed=None, row_kind='phase_diagnostic_only'))
        if not metrics and not pair.get('phase_comparison'):
            rows.append(dict(**common, metric='__pair_status__', row_kind='pair_status'))
    return rows


def assemble_acceptance(selection, root=None, output=None, selection_path=None):
    """Summarize selected V9 case JSON and write offline acceptance artifacts."""
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    selection_parent = Path(selection_path).resolve().parent if selection_path else None
    if isinstance(selection, (str, Path)):
        selection_path = Path(selection).resolve()
        selection_parent = selection_path.parent
        selection = _read_json(selection_path)
    experiment_root = root / DEFAULT_EXPERIMENT
    cases = _selection_cases(selection, root, selection_parent)
    convergence = _build_convergence(cases, selection.get('refinements'), root, selection_parent)
    old_convergence = _build_convergence(cases, selection.get('old_refinements'), root,
        selection_parent, required_kinds=OLD_REFINEMENT_KINDS, default_target='A.B0',
        expected_variant='B0', budget_scope='old V8 baseline')
    all_convergence = convergence + old_convergence
    significance = _significance(cases, all_convergence)
    h_velocity_significance = _h_velocity_significance(cases, all_convergence)
    initial_state_sensitivity = _initial_state_comparisons(
        selection, root, selection_parent)

    single_default = (DEFAULT_SINGLE_BEARING_TESTS if selection_parent is not None else
                      str(DEFAULT_EXPERIMENT / DEFAULT_SINGLE_BEARING_TESTS))
    single_path = selection.get('single_bearing_tests') or single_default
    single_item = _read_validation_item(single_path, single_default,
                                        root, selection_parent)
    single_path_resolved = single_item.get('path')
    single = _suite_summary(single_item, single_path_resolved)
    local_branch_audit = _local_control_branch_audit(single_item)
    system_integration = _system_integration_summary(single_item)
    boundary_default = (DEFAULT_CONTACT_BOUNDARY if selection_parent is not None else
                        str(DEFAULT_EXPERIMENT / DEFAULT_CONTACT_BOUNDARY))
    boundary_path = selection.get('contact_boundary') or boundary_default
    boundary_item = _read_validation_item(boundary_path, boundary_default,
                                          root, selection_parent)
    boundary = _contact_boundary_summary(boundary_item)
    for case in cases:
        if case.get('variant') == 'N3':
            case.get('control_diagnostics', {})['inner_control_candidate_local_sweep_status'] = local_branch_audit['status']
    capability = ('failed' if boundary['status'] == 'failed' or single['status'] == 'failed' else
                  'pending' if boundary['status'] in ('pending', 'conditional')
                  or single['status'] == 'pending' else 'numerical_pass')
    overall_single_bearing_status = (
        'failed_T06_contact_boundary' if boundary['status'] == 'failed' else
        'failed_original_single_bearing_suite' if single['status'] == 'failed' else
        'pending' if boundary['status'] == 'pending' or single['status'] == 'pending' else
        'conditional' if boundary['status'] == 'conditional' or single['status'] == 'conditional' else
        'numerical_pass')
    convergence_failures = sum(
        refinement.get('status') == 'failed'
        for entry in convergence for refinement in entry.get('refinements', {}).values())
    convergence_complete = bool(convergence) and all(
        entry.get('epsilon_E', {}).get('complete') is True
        and entry.get('epsilon_E', {}).get('convergence_comparisons_passed') is True
        for entry in convergence)
    convergence_status = ('failed' if convergence_failures else
                          'numerical_pass' if convergence_complete else 'pending')
    old_convergence_failures = sum(
        refinement.get('status') == 'failed'
        for entry in old_convergence for refinement in entry.get('refinements', {}).values())
    old_convergence_complete = bool(old_convergence) and all(
        entry.get('epsilon_E', {}).get('complete') is True
        and entry.get('epsilon_E', {}).get('convergence_comparisons_passed') is True
        for entry in old_convergence)
    old_convergence_status = ('failed' if old_convergence_failures else
                              'numerical_pass' if old_convergence_complete else 'pending')

    core_counts = {status: sum(case.get('numerical_core_status') == status for case in cases)
                   for status in ('numerical_pass', 'pending', 'failed')}
    conditional_count = sum(case.get('status') == 'conditional' for case in cases)
    result = dict(schema='spindle_v9_acceptance_summary_v1',
        generated_utc=datetime.now(timezone.utc).isoformat(),
        selection_file=str(selection_path) if selection_path else None,
        selection_path_policy='A prefers data_A_long same-variant/gap case with >=480 complete turns, then data_A with >=480; otherwise selected pilot is retained and marked pending. B and refinement paths are explicit.',
        scope='Offline aggregation only; no simulation, native trajectory loading, or source analysis edits.',
        acceptance_criteria=dict(
            final_positive_gap_turns=480, pilot_turns_not_final=True,
            subwindows=4, minimum_turns_per_subwindow=120,
            steady_relative_limits=dict(E_dir=.01, H_velocity_RMS=.01,
                                       non_1x_residual_RMS=.05, contact_impulse=.05),
            convergence_limits=dict(E_dir_abs_um=.01, E_dir_relative=.01,
                H_velocity_RMS_abs_mm_s=.005, H_velocity_RMS_relative=.01,
                H_non_1x_residual_abs_mm_s=.002,
                H_non_1x_residual_relative=.05, wrench_RMS_relative=.02,
                wrench_peak_relative=.05, contact_fraction_absolute=.02),
            old_B0_convergence_components=['output_sampling', 'h_half', 'n1024'],
            local_root_limits=dict(normalized_geometry_residual=1e-9,
                                   force_residual_tolerance_ratio=1.0,
                                   failed_status_count=0),
            N1_control_criterion='not_applicable',
            N3_conditional_gates=['outer-control self-consistency', 'phase-speed warning'],
            N3_inner_control_candidate_evidence='Reported from the independent T08 local branch sweep; it is not a per-RHS trajectory gate.',
            epsilon_definition='For N3 cases, epsilon_num is the maximum absolute independent difference across output sampling, h/2, root tolerance/10, and N1024. For old B0 cases, the comparison set is output sampling, h/2, and N1024 because local root tolerance is not applicable. Each is an engineering error estimate, not a statistical confidence interval.',
            epsilon_num_definition='epsilon_num contains only measured independent differences; convergence acceptance tolerances are not inserted into it.',
            significance_epsilon_floor_um=SIGNIFICANCE_EPSILON_FLOOR_UM,
            significance_epsilon_floor_source='Inherited predeclared V8 precision baseline: config/clearance_v8_config.json health_result.epsilon_E_um=0.0001 um, retained in GPTOrder/spindle_v9_tai_bearing_plan/baseline_snapshot/parameters_v8_calibrated.json.',
            significance_definition='Primary S_E=abs(E_N3-E_B0)/(epsilon_num_N3+epsilon_num_B0+0.0001 um); report only when same-U comparison and both complete independent error budgets are available. S_E>5 means the difference exceeds this numerical error estimate. Optional conservative sensitivity uses an additional 0.01 um absolute floor.',
            H_velocity_epsilon_floor_mm_s=H_VELOCITY_EPSILON_FLOOR_MM_S,
            H_velocity_epsilon_floor_source=H_VELOCITY_EPSILON_FLOOR_SOURCE,
            H_velocity_epsilon_floor_interpretation='Engineering error floor only; not a statistical confidence interval or measured detectability.',
            H_velocity_epsilon_definition='For each of the six H_v RMS, 1X and residual metrics, epsilon_num is the maximum actual same-window absolute difference across output sampling and the selected independent refinements. The floor and V9 convergence tolerances are excluded from epsilon_num.'),
        single_bearing_tests=single,
        overall_single_bearing_capability=dict(status=overall_single_bearing_status,
            original_281_test_suite_status=single['status'],
            original_local_check_count=single.get('local_check_count'),
            original_system_check_count=single.get('system_integration_check_count'),
            T06_contact_boundary_status=boundary['status'],
            note='The completed original 281 local/system checks are reported separately from T06 contact-loss boundary continuity. The 281 checks do not mean the full bearing capability passed.'),
        local_control_branch_audit=local_branch_audit,
        system_integration_tests=system_integration,
        contact_boundary=boundary,
        contact_loss_boundary=boundary,
        convergence_acceptance=dict(status=convergence_status,
            comparison_groups=len(convergence), failed_refinement_count=convergence_failures,
            complete_error_budget_groups=sum(
                entry.get('epsilon_E', {}).get('complete') is True for entry in convergence),
            note='Independent output-sampling, h/2, root-tolerance/10, and N1024 comparisons are required for a complete E_dir error budget; absent comparisons remain pending.'),
        old_B0_convergence_acceptance=dict(status=old_convergence_status,
            comparison_groups=len(old_convergence), failed_refinement_count=old_convergence_failures,
            complete_error_budget_groups=sum(
                entry.get('epsilon_E', {}).get('complete') is True for entry in old_convergence),
            required_components=['output_sampling', 'h_half', 'n1024'],
            local_root_criterion='not_applicable',
            note='Old V8 B0 error budgets require same-window sampling, h/2, and N1024 evidence. A local root-tolerance comparison is not applicable; missing full 480-turn refinements remain pending.'),
        generic_contact_loss_capability=dict(status=capability,
            blocked_by='contact_boundary' if boundary['status'] == 'failed' else None,
            note='This independent generic capability status does not gate numerical acceptance of a selected N3 run that remains in dual contact.'),
        selected_cases=dict(count=len(cases), numerical_core_status_counts=core_counts,
                            conditional_count=conditional_count, cases=cases),
        initial_state_sensitivity=initial_state_sensitivity,
        same_U_old_new_significance=significance,
        same_U_H_velocity_significance=h_velocity_significance,
        aggregate_status=('pending' if not cases else
                          'failed' if core_counts['failed'] else
                          'pending' if core_counts['pending'] else
                          'conditional' if conditional_count else 'numerical_pass'),
        aggregate_scope_note='aggregate_status reports selected-case numerical core gates only. Independent convergence and generic contact-loss capability have separate statuses; this does not claim full T01–T12 completion or real-machine validation.')

    output = Path(output) if output else experiment_root / 'validation' / 'acceptance_summary'
    output = output if output.is_absolute() else root / output
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'acceptance.json', result)
    _write_json(output / 'convergence_comparisons.json', dict(
        schema='spindle_v9_convergence_comparisons_v2', comparisons=convergence,
        old_refinements=old_convergence,
        old_new_significance=significance,
        same_U_H_velocity_significance=h_velocity_significance))
    _write_json(output / 'same_U_H_velocity_significance.json', dict(
        schema='spindle_v9_same_U_H_velocity_significance_v1',
        epsilon_floor_mm_s=H_VELOCITY_EPSILON_FLOOR_MM_S,
        epsilon_floor_source=H_VELOCITY_EPSILON_FLOOR_SOURCE,
        epsilon_floor_interpretation='Engineering error floor only; not a statistical confidence interval or measured detectability.',
        metrics=list(H_VELOCITY_METRICS), comparisons=h_velocity_significance))
    _write_json(output / 'initial_state_sensitivity.json', dict(
        schema='spindle_v9_initial_state_sensitivity_v1',
        scope='Steady-window summary-statistic comparison across distinct absolute integration windows; not same-window convergence or a numerical error budget.',
        comparison_rule='Compare complete 480-turn steady windows using the existing V9 per-metric thresholds. Preserve absolute time intervals and phase diagnostics; do not align samples or infer a convergence epsilon.',
        comparisons=initial_state_sensitivity))
    _write_main_csv(output / 'main_summary.csv', cases, significance, all_convergence)
    _write_h_velocity_csv(output / 'same_U_H_velocity_significance.csv',
                          h_velocity_significance)
    _write_initial_state_csv(output / 'initial_state_sensitivity.csv',
                             _initial_state_csv_rows(initial_state_sensitivity))
    return result


def _write_main_csv(path, cases, significance, convergence):
    comparisons = {row['gap_um']: row for row in significance if row.get('old_case')}
    budgets = {row['target_case']: row.get('epsilon_E', {}) for row in convergence
               if row.get('target_case')}
    columns = ('case_id', 'group', 'variant', 'gap_um', 'path', 'status',
        'numerical_core_status', 'complete_turns', 'configured_turns', 'force_N', 'U_kg_m',
        'E_x_um', 'E_y_um', 'E_dir_um', 'H_v_rms_x_mm_s', 'H_v_rms_y_mm_s',
        'turns_gate', 'steady_gate', 'sampling_gate', 'local_root_gate', 'domain_gate',
        'outer_control_gate', 'phase_speed_gate', 'inner_control_candidate_evidence',
        'epsilon_E_um', 'epsilon_budget_complete', 'epsilon_convergence_passed',
        'same_U_old_new_delta_um', 'same_U_old_new_relative_percent', 'same_U_old_new_S_E',
        'same_U_old_new_epsilon_floor_um', 'same_U_old_new_epsilon_old_um',
        'same_U_old_new_epsilon_new_um', 'same_U_old_new_S_E_conservative_floor_0p01um',
        'same_U_old_new_status')
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            budget = budgets.get(case['case_id'], {})
            cmp = comparisons.get(float(case['gap_um']), {})
            gates = case.get('gates', {})
            conditional = case.get('conditional_gates', {})
            writer.writerow(dict(case_id=case.get('case_id'), group=case.get('group'),
                variant=case.get('variant'), gap_um=case.get('gap_um'), path=case.get('path'),
                status=case.get('status'), numerical_core_status=case.get('numerical_core_status'),
                complete_turns=case.get('complete_turns'), configured_turns=case.get('configured_turns'),
                force_N=case.get('force_N'), U_kg_m=case.get('U_kg_m'),
                E_x_um=case.get('E_x_um'), E_y_um=case.get('E_y_um'), E_dir_um=case.get('E_dir_um'),
                H_v_rms_x_mm_s=case.get('H_v_rms_x_mm_s'), H_v_rms_y_mm_s=case.get('H_v_rms_y_mm_s'),
                turns_gate=gates.get('turns'), steady_gate=gates.get('steady'),
                sampling_gate=gates.get('sampling'), local_root_gate=gates.get('local_root'),
                domain_gate=gates.get('domain'), outer_control_gate=conditional.get('outer_control'),
                phase_speed_gate=conditional.get('phase_speed'),
                inner_control_candidate_evidence=(case.get('control_diagnostics') or {}).get(
                    'inner_control_candidate_local_sweep_status',
                    (case.get('control_diagnostics') or {}).get('inner_control_candidate')),
                epsilon_E_um=budget.get('epsilon_num_um'),
                epsilon_budget_complete=budget.get('complete'),
                epsilon_convergence_passed=budget.get('convergence_comparisons_passed'),
                same_U_old_new_delta_um=cmp.get('signed_difference_um'),
                same_U_old_new_relative_percent=cmp.get('relative_change_percent'),
                same_U_old_new_S_E=cmp.get('S_E'),
                same_U_old_new_epsilon_floor_um=cmp.get('epsilon_floor_um'),
                same_U_old_new_epsilon_old_um=cmp.get('epsilon_old_um'),
                same_U_old_new_epsilon_new_um=cmp.get('epsilon_new_um'),
                same_U_old_new_S_E_conservative_floor_0p01um=cmp.get(
                    'S_E_conservative_floor_0p01um'),
                same_U_old_new_status=cmp.get('status')))


def _write_h_velocity_csv(path, comparisons):
    columns = ('gap_um', 'metric', 'unit', 'old_case', 'new_case', 'same_U',
        'old_U_kg_m', 'new_U_kg_m', 'old_value_mm_s', 'new_value_mm_s',
        'absolute_difference_mm_s', 'relative_change_percent', 'epsilon_old_mm_s',
        'epsilon_new_mm_s', 'epsilon_old_complete', 'epsilon_new_complete',
        'epsilon_old_components_mm_s', 'epsilon_new_components_mm_s',
        'epsilon_old_required_components', 'epsilon_new_required_components',
        'epsilon_floor_mm_s', 'epsilon_floor_source', 'error_budgets_complete',
        'S_y', 'significance_threshold', 'convergence_budgets_passed', 'status')
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in comparisons:
            csv_row = {key: row.get(key) for key in columns}
            for key in ('epsilon_old_components_mm_s', 'epsilon_new_components_mm_s',
                        'epsilon_old_required_components', 'epsilon_new_required_components'):
                if csv_row[key] is not None:
                    csv_row[key] = json.dumps(csv_row[key], ensure_ascii=False, sort_keys=True)
            writer.writerow(csv_row)


def _write_initial_state_csv(path, rows):
    columns = ('comparison_id', 'status', 'comparison_basis',
        'absolute_window_is_numerical_refinement', 'same_absolute_time_window',
        'reference_path', 'branch_path', 'reference_start_s', 'reference_end_s',
        'branch_start_s', 'branch_end_s', 'start_offset_shaft_turns',
        'failure_evidence_path', 'failure_error', 'metric', 'reference_value',
        'branch_value', 'difference', 'tolerance', 'metric_passed', 'row_kind')
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path,
        default=DEFAULT_EXPERIMENT / 'case_selection.json',
        help='Explicit case_selection.json; case folders are referenced by path, not copied.')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path,
        help='Output directory; defaults to experiments/clearance_v9_tai_bearing/validation/acceptance_summary.')
    args = parser.parse_args()
    if not args.selection.is_absolute():
        selection_path = args.root / args.selection
    else:
        selection_path = args.selection
    result = assemble_acceptance(selection_path, root=args.root, output=args.output,
                                 selection_path=selection_path)
    print(json.dumps(dict(output=str(args.output or (args.root / DEFAULT_EXPERIMENT / 'validation' / 'acceptance_summary')),
                          aggregate_status=result['aggregate_status'],
                          selected_cases=result['selected_cases']['count'],
                          single_bearing_status=result['single_bearing_tests']['status'],
                          contact_boundary_status=result['contact_boundary']['status'],
                          convergence_status=result['convergence_acceptance']['status'],
                          old_B0_convergence_status=result['old_B0_convergence_acceptance']['status']),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
