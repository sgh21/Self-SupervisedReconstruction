"""Evidence-based V3 acceptance; numerical error estimates are not confidence bounds."""
import copy
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from spindle.dynamics_model import rotor_section
from utils.clearance_v3_analysis import compare_v3


HOUSING_FEATURES = tuple(f'H_v_{kind}_{axis}_mm_s'
                         for kind in ('rms', '1x', 'residual') for axis in 'xy')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _solver_identity(settings):
    return {key: value for key, value in settings.items() if key != 'accepted_steps'}


def _physical_identity(config):
    parameters = copy.deepcopy(config['parameters'])
    parameters.pop('fit_clearance_um')
    parameters.pop('N')  # Circumferential quadrature is checked with solver settings.
    return dict(parameters=parameters, excitation=config['excitation'])


def _load_case(path):
    path = Path(path)
    if path.suffix == '.npz':
        path = path.parent
    return dict(path=path, config=_read(path/'actual_config.json'),
                metrics=_read(path/'metrics.json'), steady=_read(path/'stationarity.json'),
                sampling=_read(path/'sampling_convergence.json'),
                solver=_read(path/'actual_solver_settings.json'))


def _status(value):
    return 'pass' if value else 'fail'


def _domain_evidence(case, baseline, config):
    """Declared demo tangent-domain check, not proof of real healthy fits."""
    p = config['parameters']
    with np.load(case['path']/'native.npz') as raw:
        time = raw['t_raw']
        use = (time >= case['metrics']['measurement_start_s']) & (time < case['metrics']['measurement_end_s'])
        healthy = raw['healthy_relative_m'][use]
        forces = raw['healthy_support_forces_N'][use]
        supports = []
        for index, support in enumerate(p['healthy_supports']):
            reference = rotor_section(baseline, support['z_m'])-baseline[4:6]
            maximum = float(np.linalg.norm(healthy[:, index]-reference, axis=1).max())
            limit = support['applicability_excursion_from_baseline_m']
            supports.append(dict(name=support['name'], z_m=support['z_m'],
                fit_state=support['fit_state'], baseline_relative_xy_um=(reference*1e6).tolist(),
                maximum_excursion_from_baseline_um=maximum*1e6,
                assumed_domain_limit_um=limit*1e6, passed=bool(maximum <= limit),
                mean_force_xy_N=forces[:, index].mean(axis=0).tolist(),
                peak_force_norm_N=float(np.linalg.norm(forces[:, index], axis=1).max())))
        tilt = float(np.linalg.norm(raw['q_raw_m'][use, 8:10], axis=1).max())
        active = raw['fit_active_count'][use]
        return dict(passed=bool(all(support['passed'] for support in supports)),
            scope='assumed_demo local domain only; actual healthy assembly is unvalidated',
            healthy_supports=supports, maximum_tilt_rad=tilt,
            tilt_small_angle_relative_sine_correction_estimate=tilt**2/6,
            active_direction_min=int(active.min()), active_direction_max=int(active.max()),
            direction_count_is_not_contact_area=True,
            maximum_outer_relative_radius_um=float(np.linalg.norm(raw['rO_raw_m'][use], axis=1).max()*1e6),
            maximum_target_fit_compression_um=case['metrics']['fit_max_compression_um'],
            maximum_target_bearing_force_N=case['metrics']['bearing_peak_N'])


def build_acceptance(folder, formal_paths, fine_n_paths, fine_h_paths, config):
    """Read corresponding gap cases; return acceptance and convergence JSON objects.

    Missing evidence raises instead of creating a false PASS. No files or raw
    histories are modified. All comparisons use the saved exact export window.
    """
    folder = Path(folder)
    formal = [_load_case(path) for path in formal_paths]
    fine_n = [_load_case(path) for path in fine_n_paths]
    fine_h = [_load_case(path) for path in fine_h_paths]
    if not formal or not (len(formal) == len(fine_n) == len(fine_h)):
        raise ValueError('Need an N and a time-step refinement for every formal case')
    reference_index = min(range(len(formal)), key=lambda i: formal[i]['metrics']['clearance_um'])
    reference_case = formal[reference_index]
    if reference_case['metrics']['clearance_um'] != 0:
        raise ValueError('The V3 ideal-reference domain check requires its c=0 formal case')
    with np.load(reference_case['path']/'native.npz') as reference_raw:
        baseline = reference_raw['q_equilibrium'].copy()
    physical = _physical_identity(formal[0]['config'])
    solver = _solver_identity(formal[0]['solver'])
    config_same = all(_physical_identity(case['config']) == physical for case in formal)
    solver_same = all(_solver_identity(case['solver']) == solver for case in formal)
    settings = config['simulation']
    requested = dict(method=settings['method'], rtol=settings['rtol'],
        atol=[settings['atol_translation_m']]*8+[settings['atol_angle_rad']]*2
            +[settings['atol_velocity_m_s']]*8+[settings['atol_angle_rate_rad_s']]*2,
        max_step_s=settings['max_step_s'], output_fs_hz=settings['fs_raw_hz'],
        fit_direction_count=config['parameters']['N'])
    frozen_config_matches = (_physical_identity(config) == physical
                             and all(solver[key] == value for key, value in requested.items()))
    mechanics_path = folder/'validation/mechanics_v3_validation.json'
    mechanics_pass = bool(_read(mechanics_path)['passed'])
    evidence, records = [], []
    for case, ncase, hcase in zip(formal, fine_n, fine_h):
        m = case['metrics']
        gap = m['clearance_um']
        if ncase['metrics']['clearance_um'] != gap or hcase['metrics']['clearance_um'] != gap:
            raise ValueError('Refinement case gap does not match its formal case')
        ncomparison = compare_v3(m, ncase['metrics'], config)
        hcomparison = compare_v3(m, hcase['metrics'], config)
        nsolver, hsolver = _solver_identity(ncase['solver']), _solver_identity(hcase['solver'])
        base_solver = _solver_identity(case['solver'])
        n_other = {key: value for key, value in nsolver.items() if key != 'fit_direction_count'}
        b_nother = {key: value for key, value in base_solver.items() if key != 'fit_direction_count'}
        h_other = {key: value for key, value in hsolver.items() if key != 'max_step_s'}
        b_hother = {key: value for key, value in base_solver.items() if key != 'max_step_s'}
        n_isolated = bool(n_other == b_nother and nsolver['fit_direction_count'] == 2*base_solver['fit_direction_count']
                          and _physical_identity(ncase['config']) == _physical_identity(case['config']))
        h_isolated = bool(h_other == b_hother and abs(hsolver['max_step_s']*2/base_solver['max_step_s']-1) < 1e-12
                          and _physical_identity(hcase['config']) == _physical_identity(case['config']))
        steps, fine_steps = case['solver']['accepted_steps'], hcase['solver']['accepted_steps']
        h_effective = bool(steps['fraction_ge_0p99_max_step'] > 0
                           and fine_steps['max_s'] < .75*steps['max_s'])
        steady = bool(case['steady']['passed'] and ncase['steady']['passed'] and hcase['steady']['passed'])
        sampling_pass = bool(case['sampling']['passed'] and ncase['sampling']['passed'] and hcase['sampling']['passed'])
        numerical = bool(steady and sampling_pass and ncomparison['passed'] and hcomparison['passed']
                         and n_isolated and h_isolated and h_effective)
        uncertainties = {}
        for key in hcomparison['metrics']:
            components = dict(time_step=abs(hcomparison['metrics'][key]['difference']),
                directions=abs(ncomparison['metrics'][key]['difference']),
                native_sampling=abs(case['sampling']['metrics'][key]['difference']),
                stationarity=abs(case['steady']['checks'][key]['difference']))
            uncertainties[key] = dict(estimate=sum(components.values()), components=components)
        domain = _domain_evidence(case, baseline, config)
        valid = bool(numerical and domain['passed'] and mechanics_pass and config_same and solver_same and frozen_config_matches)
        record = dict(case_id=case['path'].name, path=str(case['path']), clearance_um=gap,
            E_x_um=m['E_x_um'], E_y_um=m['E_y_um'], E_dir_um=m['E_dir_um'],
            error_estimate_um=uncertainties['E_dir_um']['estimate'],
            valid_for_range=valid, stationarity_pass=steady, numerical_pass=numerical,
            numerical_validation=_status(numerical),
            model_applicability=domain,
            housing_features={key: m[key] for key in HOUSING_FEATURES},
            uncertainty_by_metric=uncertainties)
        records.append(record)
        evidence.append(dict(case_id=record['case_id'], clearance_um=gap,
            base_path=str(case['path']), fine_N_path=str(ncase['path']), fine_h_path=str(hcase['path']),
            N_comparison=ncomparison, h_comparison=hcomparison,
            sampling_comparison=case['sampling'], fine_N_sampling_comparison=ncase['sampling'],
            fine_h_sampling_comparison=hcase['sampling'], N_changed_alone=n_isolated,
            h_changed_alone=h_isolated, h_refinement_effective=h_effective,
            base_accepted_step_statistics=steps, fine_h_accepted_step_statistics=fine_steps,
            refinement_stationarity_pass=steady, all_sampling_pass=sampling_pass,
            numerical_pass=numerical, uncertainty_by_metric=uncertainties))

    pairs = []
    for i, j in combinations(range(len(records)), 2):
        first, second = records[i], records[j]
        eligible = first['valid_for_range'] and second['valid_for_range']
        comparisons = {}
        for key in ('E_dir_um', 'E_x_um', 'E_y_um')+HOUSING_FEATURES:
            left, right = formal[i]['metrics'][key], formal[j]['metrics'][key]
            uncertainty = first['uncertainty_by_metric'][key]['estimate']+second['uncertainty_by_metric'][key]['estimate']
            difference = abs(left-right)
            comparisons[key] = dict(difference=difference, combined_uncertainty=uncertainty,
                five_times_uncertainty=5*uncertainty,
                resolved=bool(eligible and difference > 5*uncertainty))
        motion_resolved = comparisons['E_dir_um']['resolved']
        housing_resolved = any(comparisons[key]['resolved'] for key in HOUSING_FEATURES)
        pairs.append(dict(case_i=first['case_id'], case_j=second['case_id'],
            clearance_i_um=first['clearance_um'], clearance_j_um=second['clearance_um'],
            eligible=eligible, E_dir_resolved=motion_resolved, housing_resolved=housing_resolved,
            dual_output_resolved=bool(motion_resolved and housing_resolved), metrics=comparisons))

    valid = [record for record in records if record['valid_for_range']]
    valid.sort(key=lambda record: record['E_dir_um'])
    bins = []
    for low, high in [(0, 5), (5, 10), (10, 15), (15, 20)]:
        members, boundary = [], []
        for record in valid:
            value, error = record['E_dir_um'], record['error_estimate_um']
            if low <= value and (value < high or high == 20 and value <= high):
                inside = value-error >= low and (value+error < high or high == 20 and value+error <= high)
                (members if inside else boundary).append(record)
        levels = []
        for record in members:
            if all(abs(record['E_dir_um']-level['E_dir_um']) > 5*(record['error_estimate_um']+level['error_estimate_um']) for level in levels):
                levels.append(record)
        bins.append(dict(interval=f'[{low},{high}'+(']' if high == 20 else ')'),
            valid_case_count=len(members), resolved_level_count=len(levels),
            case_ids=[record['case_id'] for record in members],
            resolved_level_case_ids=[record['case_id'] for record in levels],
            boundary_uncertain_case_ids=[record['case_id'] for record in boundary]))
    baseline_value = records[reference_index]['E_dir_um']
    gap_order = sorted(records, key=lambda record: record['clearance_um'])
    adjacent = []
    for first, second in zip(gap_order[:-1], gap_order[1:]):
        adjacent.append(next(pair for pair in pairs
                             if {pair['case_i'], pair['case_j']} == {first['case_id'], second['case_id']}))
    nearest = min(valid, key=lambda record: abs(record['E_dir_um']-20)) if valid else None
    if not valid:
        model_reached = 'not_validated'
    elif any(record['E_dir_um']-record['error_estimate_um'] >= 20 for record in valid):
        model_reached = 'pass'
    elif any(record['E_dir_um']-record['error_estimate_um'] <= 20 <= record['E_dir_um']+record['error_estimate_um'] for record in valid):
        model_reached = 'boundary_uncertain'
    else:
        model_reached = 'fail'
    checks = dict(parameter_contract='pass', mechanics_implementation=_status(mechanics_pass),
        numerical_validation=_status(all(record['numerical_pass'] for record in records)),
        stationarity=_status(all(record['stationarity_pass'] for record in records)),
        time_integration_convergence=_status(all(row['h_comparison']['passed'] and row['h_changed_alone'] and row['h_refinement_effective'] for row in evidence)),
        contact_direction_convergence=_status(all(row['N_comparison']['passed'] and row['N_changed_alone'] for row in evidence)),
        native_sampling_convergence=_status(all(row['all_sampling_pass'] for row in evidence)),
        housing_acceleration_convergence=_status(all(all(row[key]['acceleration_pass'] for key in
            ('h_comparison','N_comparison','sampling_comparison','fine_N_sampling_comparison','fine_h_sampling_comparison')) for row in evidence)),
        contact_force_convergence=_status(all(all(row[key]['contact_pass'] for key in
            ('h_comparison','N_comparison','sampling_comparison','fine_N_sampling_comparison','fine_h_sampling_comparison')) for row in evidence)),
        assumed_healthy_tangent_domain=_status(all(record['model_applicability']['passed'] for record in records)),
        single_parameter_scan=_status(config_same and solver_same and frozen_config_matches),
        dual_output_effect=_status(any(pair['dual_output_resolved'] for pair in pairs)) if len(valid) >= 2 else 'not_validated',
        model_twenty_um_reached=model_reached, twenty_um_reached='not_validated',
        bin_coverage_0_20=_status(all(item['resolved_level_count'] > 0 for item in bins)) if valid else 'not_validated',
        physical_parameter_support='not_validated', real_machine_validation='not_validated')
    checks['adjacent_E_dir_severity_resolved'] = (_status(all(pair['E_dir_resolved'] for pair in adjacent))
        if adjacent and all(pair['eligible'] for pair in adjacent) else 'not_validated')
    acceptance = dict(schema_version='v3-review-1', document_kind='computed_acceptance',
        target_bearing_id='single_front_F1', model_dof=10, scan_parameter='fit_clearance_um',
        range_claim_scope=config.get('scenario_id', 'assumed_demo'),
        main_measurement_section=dict(kind=config['parameters']['measurement_kind'], z_m=config['parameters']['z_m']),
        probe_reference=config['parameters']['probe_reference'],
        runout_metric='max(mean(turn_pp_x_um),mean(turn_pp_y_um))', runout_includes_1x=True,
        frozen_config_identity=str(folder/'frozen_config.json'),
        actual_solver_settings_path=[str(case['path']/'actual_solver_settings.json') for case in formal],
        checks=checks, range=dict(healthy_reference_um=baseline_value,
            healthy_reference_kind='ideal zero-gap demo; not measured machine health',
            healthy_reference_valid=records[reference_index]['valid_for_range'],
            min_absolute_um=valid[0]['E_dir_um'] if valid else None,
            max_absolute_um=valid[-1]['E_dir_um'] if valid else None,
            minimum_case_id=valid[0]['case_id'] if valid else None,
            maximum_case_id=valid[-1]['case_id'] if valid else None,
            min_baseline_increment_um=valid[0]['E_dir_um']-baseline_value if valid else None,
            max_baseline_increment_um=valid[-1]['E_dir_um']-baseline_value if valid else None,
            nearest_20_um_case=nearest['case_id'] if nearest else None,
            distance_of_nearest_case_to_20_um=abs(nearest['E_dir_um']-20) if nearest else None,
            largest_internal_gap_um=float(max(np.diff([record['E_dir_um'] for record in valid]))) if len(valid) > 1 else None,
            absolute_zero_endpoint='not_demonstrated', continuous_coverage='not_demonstrated', bins=bins),
        representative_cases=records, pairs=pairs, adjacent_pairs=adjacent,
        housing_feature_set=list(HOUSING_FEATURES),
        uncertainty_definition='sum of absolute h, N, sampling differences and full subwindow spread; no acceptance floors added; engineering estimate, not statistical confidence or a rigorous bound',
        failed_or_unvalidated_items=[dict(item=key, status=value) for key, value in checks.items() if value != 'pass'],
        physical_limits=['assumed demo support layout and health tangent; not identified on this spindle',
                         'rigid rotor tilt plus point supports, no shaft bending or housing rotation',
                         'shaft-center motion is not surface geometric runout'],
        minimum_missing_information=['actual bearing layout and healthy fits/preloads',
            'mass/rotor inertia and support/fit compliance measurements',
            'synchronous same-condition housing/shaft/probe-reference measurements'],
        evidence_files=[str(mechanics_path)]+[str(case['path']/'native.npz') for case in formal])
    convergence = dict(cases=evidence, formal_physical_config_equal=config_same,
        formal_solver_settings_equal=solver_same, supplied_frozen_config_matches=frozen_config_matches,
        solver_identity=solver, uncertainty_definition=acceptance['uncertainty_definition'])
    return acceptance, convergence
