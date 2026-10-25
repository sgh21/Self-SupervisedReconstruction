"""V6 signal analysis and layered evidence; reuse the existing guarded V4 pipeline.

The healthy point is c=2 um at one frozen U. Error estimates are engineering
budgets, not confidence intervals. Missing independent evidence remains null.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from utils import clearance_v5_analysis as v5
from utils.clearance_v3_analysis import _harmonics
from utils.pitting_analysis import filter_vibration
from utils.runout_labels import complete_turn_runout

MAIN_KEYS = v5.MAIN_KEYS
CONTACT_KEYS = v5.CONTACT_KEYS


def keys(metrics):
    return list(dict.fromkeys(v5.metric_keys(metrics) + [k for k in metrics
        if k.startswith(('H_v_', 'H_a_', 'M_1x_', 'M_residual_', 'OH2_mean_', 'bearing_'))]))


def absolute_floor(key, config):
    if key.startswith('bearing_'):
        from utils.clearance_v8_analysis import wrench_floor
        return wrench_floor(key)
    if key.startswith(('M_', 'OH2_')):
        return None if '_phase_' in key else config['numerical_thresholds']['position_um']
    return v5.absolute_floor(key, config)


def tolerance(key, a, b, config):
    if key.startswith('bearing_'):
        fraction = .05 if '_peak_' in key else .01
        if '_contact_fraction' in key:
            return .02
        return max(absolute_floor(key, config), fraction*max(abs(a), abs(b)))
    if key.startswith(('M_', 'OH2_')):
        return max(absolute_floor(key, config), .01*max(abs(a), abs(b)))
    return v5.tolerance(key, a, b, config)


def _layers(checks):
    layers = v5._layers(checks)
    layers['weak_features_above_floor'] = {
        k: row['above_absolute_floor'] for k, row in checks.items()
        if any(s in k for s in ('_residual_', '_2x_', '_3x_', '_energy_'))}
    return layers


def compare_metrics(base, fine, config):
    checks = {}
    for key in keys(fine):
        floor = absolute_floor(key, config)
        if key not in base or floor is None:
            continue
        delta = abs(base[key]-fine[key])
        limit = tolerance(key, base[key], fine[key], config)
        checks[key] = dict(base=base[key], fine=fine[key], difference=delta,
            tolerance=limit, passed=bool(delta <= limit),
            above_absolute_floor=bool(min(abs(base[key]), abs(fine[key])) > floor))
    same = all(abs(base[k]-fine[k]) < 1e-10
               for k in ('measurement_start_s', 'measurement_end_s'))
    return dict(passed=bool(same and checks and all(x['passed'] for x in checks.values())),
                same_window=same, metrics=checks, **_layers(checks))


def analyze(result, config, start, end, stride=1):
    """V4-compatible result; infer actual dense rate before applying stride.

Main H arrays use 10-900 Hz. Wide/high-band metrics are computed independently
from the same raw state and true guards, never by filtering the main output.
"""
    local = copy.deepcopy(config)
    raw_fs = round(1/float(np.median(np.diff(result['t'][:1000]))))
    local['simulation']['fs_raw_hz'] = raw_fs
    metrics, turns, output, steady = v5.analyze(result, local, start, end, stride)
    p, s = local['parameters'], local['simulation']
    t = result['t'][::stride]
    q = result['q'][::stride]
    relative = result['measured_relative_xy_m'][::stride]
    omega, phase = p['rpm']*2*np.pi/60, local['excitation']['phase_rad']
    filtered = {}
    for lo, hi in ((10, 2048), (900, 2048)):
        proc = dict(local['processing'], band_hz=[lo, hi])
        for name, source, scale in (('v', 'v', 1000), ('a', 'acc', 1)):
            filtered[name, lo, hi] = filter_vibration(
                result[source][::stride, 4:6].T, raw_fs/stride, proc).T*scale
    tout = t[0]+np.arange(len(filtered['v', 10, 2048]))/s['fs_out_hz']

    def extra(left, right):
        values = {}
        use = (tout >= left-1e-12) & (tout < right-1e-12)
        native = (t >= left-1e-12) & (t < right-1e-12)
        for axis, center in zip('xy', (q[native, 10:12]-q[native, 4:6]).mean(axis=0)*1e6):
            values[f'OH2_mean_{axis}_um'] = float(center)
        angle = omega*t[native]+phase
        design = np.column_stack((np.ones(len(angle)), np.cos(angle), np.sin(angle)))
        xy = relative[native]*1e6
        coef = np.linalg.lstsq(design, xy, rcond=None)[0]
        for axis, amp, ph, residual in zip('xy', np.linalg.norm(coef[1:], axis=0),
                np.arctan2(-coef[2], coef[1]), np.sqrt(np.mean((xy-design@coef)**2, axis=0))):
            values[f'M_1x_{axis}_um'] = float(amp)
            values[f'M_1x_phase_{axis}_rad'] = float(ph)
            values[f'M_residual_{axis}_um'] = float(residual)
        for (name, lo, hi), signal in filtered.items():
            prefix = f'H_{name}_band_{lo}_{hi}'
            unit, energy_unit = ('mm_s', 'mm2_s2') if name == 'v' else ('m_s2', 'm2_s4')
            # The public harmonic helper removes fitted DC and 1X before residual PSD.
            values.update(_harmonics(tout[use], signal[use], omega, phase, prefix,
                                     unit, energy_unit, s['fs_out_hz']))
        return values

    metrics.update(extra(start, end))
    edges = steady['window_edges_s']
    for row, left, right in zip(steady['subwindow_metrics'], edges[:-1], edges[1:]):
        row.update(extra(left, right))
    if config.get('analysis_version') == 'v8':
        from utils.clearance_v8_analysis import add_wrench_metrics
        add_wrench_metrics(result, local, stride, metrics, steady, start, end)
    for row in [metrics]+steady['subwindow_metrics']:
        for name, unit in (('v', 'mm_s'), ('a', 'm_s2')):
            row[f'H_{name}_rms_xy_{unit}'] = float(np.hypot(
                row[f'H_{name}_rms_x_{unit}'], row[f'H_{name}_rms_y_{unit}']))
    checks = {}
    for key in keys(metrics):
        floor = absolute_floor(key, local)
        if floor is None:
            continue
        vals = np.asarray([row[key] for row in steady['subwindow_metrics']])
        limit = tolerance(key, vals.min(), vals.max(), local)
        if key.startswith(('mean_', 'OH_mean_', 'OH2_mean_')):
            limit = max(local['numerical_thresholds']['position_um'],
                        local['numerical_thresholds']['relative_main']*metrics['E_dir_um'])
        checks[key] = dict(values=vals.tolist(), difference=float(np.ptp(vals)),
            tolerance=float(limit), passed=bool(np.ptp(vals) <= limit),
            above_absolute_floor=bool(np.min(np.abs(vals)) > floor))
    expected = s['retained_turns']
    complete = expected >= 120 and expected % 4 == 0 and metrics['complete_turns'] == expected
    complete = complete and steady['subwindow_complete_turns'] == [expected//4]*4
    layers = _layers(checks)
    drift = all(row['passed'] for k, row in checks.items()
                if k.startswith(('mean_', 'OH_mean_', 'OH2_mean_')))
    steady.update(checks=checks, **layers)
    steady['main_stationarity_pass'] = bool(complete and layers['main_metric_pass'] and drift)
    steady['passed'] = bool(complete and all(row['passed'] for row in checks.values())
                            and all(row['passed'] for row in steady['band_drift_checks'].values()))
    steady['primary_vibration_band_hz'] = local['processing']['passband_hz']
    steady['auxiliary_vibration_bands_hz'] = [[10, 2048], [900, 2048]]
    metrics.update(steady_pass=steady['passed'])
    return metrics, turns, output, steady


def domain_check(time, q, config, diagnostics):
    """Small-angle/nontensile screen, without inventing a Hertz pressure limit."""
    tilt = float(np.max(np.linalg.norm(np.asarray(q)[:, 8:10], axis=1)))
    limit = config.get('constitutive_domains', {}).get('maximum_tilt_rad', .001)
    checks = dict(finite_state=bool(np.isfinite(q).all()), small_angle=bool(tilt <= limit),
                  four_bearings_reported=all(k in diagnostics for k in ('F1', 'F2', 'R1', 'R2')))
    for name, data in diagnostics.items():
        checks[name+'_no_tensile_contact'] = bool(data['Q_min_N'] >= -1e-12)
        checks[name+'_finite_contact'] = bool(all(np.isfinite(data[k])
            for k in ('delta_min_m', 'delta_max_m', 'Q_min_N', 'Q_max_N')))
    return dict(passed=bool(all(checks.values())), checks=checks,
        values=dict(maximum_tilt_rad=tilt, maximum_tilt_limit_rad=limit), bearings=diagnostics,
        physical_applicability='conditional_fixed_angle_equivalent_Hertz',
        contact_pressure_strength='not_validated_missing_groove_curvature_and_running_preload',
        note='No old 1um linear-support excursion or 2*overlap limit; delta/d and delta/overlap are diagnostics.')


def error_budget(metrics, steady, sampling, time_comparison, angular_comparison, config):
    budget = {}
    thresholds = config['numerical_thresholds']
    for key in keys(metrics):
        acceptance_floor = absolute_floor(key, config)
        if acceptance_floor is None:
            continue
        # A 0.005 um acceptance tolerance is not a measured numerical uncertainty.
        floor = thresholds.get('error_floor_position_um', 1e-4) if key.startswith(
            ('E_', 'M_', 'mean_', 'OH')) else acceptance_floor
        parts = {}
        for label, evidence in (('time', time_comparison), ('angular', angular_comparison),
                                 ('sampling', sampling)):
            row = (evidence or {}).get('metrics', {}).get(key)
            usable = evidence and evidence.get('same_window', True)
            usable = usable and evidence.get('independent_refinement_applied', True)
            parts[label] = row['difference'] if row and usable else None
        row = steady.get('checks', {}).get(key)
        parts['subwindow'] = row['difference'] if row else None
        complete = all(value is not None for value in parts.values())
        budget[key] = dict(epsilon=max(floor, sum(parts.values())) if complete else None,
            absolute_floor=floor, acceptance_floor=acceptance_floor,
            complete=complete, components=parts)
    return budget


def load_case(folder, replay=False):
    """Use existing input/solver audits; permit the declared dense output multiplier."""
    folder = Path(folder)
    case = v5.load_case(folder, replay=False)
    config = case['config']
    s = config['simulation']
    solver = v5._read(folder/'actual_solver_settings.json', [])
    expected_rate = s['fs_raw_hz']*s.get('dense_sampling_multiplier', 1)
    # The loader checks all actual tolerances/steps separately from native retention.
    aq = np.full(12, s['atol_translation_m']); aq[8:10] = s['atol_angle_rad']
    av = np.full(12, s['atol_velocity_m_s']); av[8:10] = s['atol_angle_rate_rad_s']
    p = config['parameters']
    case['solver_pass'] = bool(solver and all(row['method'] == s['method']
        and row['rtol'] == s['rtol'] and row.get('rhs_backend', 'python') == p.get('rhs_backend', 'python')
        and row['max_step_s'] == s['max_step_s'] and row['fit_direction_count'] == p['N']
        and row['output_fs_hz'] == expected_rate and np.array_equal(row['atol'], np.r_[aq, av])
        and row['accepted_steps']['max_s'] <= s['max_step_s']*(1+1e-8) for row in solver))
    sample = case['sampling']
    case['sampling_identity_pass'] = bool(sample.get('same_dense_solution')
        and sample.get('base_fs_hz') == s['fs_raw_hz']
        and sample.get('fine_fs_hz', sample.get('refined_fs_hz')) == 2*s['fs_raw_hz'])
    if replay:
        names = dict(t='t_raw', q='q_raw_m', v='v_raw_m_s', acc='a_raw_m_s2',
            measured_relative_xy_m='r_m_raw_m', fit_force_N='fit_force_N',
            fit_active_count='fit_active_count', fit2_force_N='fit2_force_N',
            fit2_active_count='fit2_active_count')
        with np.load(folder/'native.npz') as arrays:
            result = {key: arrays[value] for key, value in names.items()}
            meta = json.loads(str(arrays['metadata_json']))
        rebuilt, _, _, steady = analyze(result, config, meta['export_start_s'], meta['export_end_s'])
        case['replay_max_difference'] = max(abs(rebuilt[k]-case['metrics'][k])
            for k in keys(rebuilt) if k in case['metrics'] and '_phase_' not in k)
        case['metrics'].update(rebuilt)
        case['steady'] = steady
        # Keep the actual 65536->131072 same-dense-solution check; never replace it
        # with a 32768->65536 calculation from the retained native archive.
        with np.load(folder/'sampling_131072.npz') as dense:
            labels = complete_turn_runout(dense['t_raw'], dense['r_m_raw_m']*1e6,
                p['rpm'], meta['export_start_s'], meta['export_end_s'],
                phase_at_zero_rad=config['excitation']['phase_rad'])
            if config.get('analysis_version') == 'v8':
                from utils.clearance_v8_analysis import wrench_signals, summarize
                td = dense['t_raw']
                valid = (td >= meta['export_start_s']-1e-12) & (td < meta['export_end_s']-1e-12)
                signals = wrench_signals(td[valid], dense['q_raw_m'][valid], dense['v_raw_m_s'][valid], p)
                fine_wrenches = summarize(signals, np.ones(sum(valid), dtype=bool))
                wrench_error = max(abs(value-case['sampling']['metrics'][key]['fine'])
                    for key, value in fine_wrenches.items())
                case['sampling_wrench_replay_max_difference'] = wrench_error
                case['sampling_identity_pass'] = bool(case['sampling_identity_pass'] and wrench_error < 1e-9)
        mean = labels['peak_to_peak_um'].mean(axis=0)
        dense_pp = dict(E_x_um=float(mean[0]), E_y_um=float(mean[1]), E_dir_um=float(max(mean)))
        replay_error = 0.
        for key, value in dense_pp.items():
            saved = case['sampling']['metrics'][key]
            replay_error = max(replay_error, abs(value-saved['fine']))
            saved.update(base=rebuilt[key], fine=value, difference=abs(value-rebuilt[key]),
                passed=bool(abs(value-rebuilt[key]) <= tolerance(key, value, rebuilt[key], config)))
        case['sampling_PP_replay_max_difference'] = replay_error
        case['sampling_identity_pass'] = bool(case['sampling_identity_pass'] and replay_error < 1e-9)
    return case


def _refinement(base, fine, label):
    evidence = v5._refinement(base, fine, label)
    numerical = compare_metrics(base['metrics'], fine['metrics'], base['config'])
    evidence.update(numerical)
    evidence['passed'] = bool(numerical['passed'] and evidence['independent_refinement_applied']
                              and evidence['fine_solver_pass'])
    return evidence


def assemble(root, output=None, base_group='data', integration_group='certification/h_half',
             angular_group='certification/n1024', replay=False):
    root = Path(root); out = Path(output) if output else root/'recomputed'
    out.mkdir(parents=True, exist_ok=True)
    cases = {f.parent.relative_to(root/base_group).as_posix(): load_case(f.parent, replay)
             for f in sorted((root/base_group).rglob('metrics.json'))}
    healthy = next((case for case in cases.values() if case['metrics']['gap_um'] == 2), None)
    refined = {label: {name: load_case(root/group/name, replay) for name in cases
                      if (root/group/name/'metrics.json').exists()}
               for label, group in (('h_half', integration_group), ('N_double', angular_group))}
    records, comparisons, budgets, rows = [], {}, {}, []
    for name, case in cases.items():
        m, cfg, steady = case['metrics'], case['config'], case['steady']
        changes = v5._physical_differences(cfg, healthy['config']) if healthy else ['missing_c2_health']
        comp = {label: _refinement(case, group[name], label)
                for label, group in refined.items() if name in group}
        comparisons[name] = comp
        eps = error_budget(m, steady, case['sampling'], comp.get('h_half'), comp.get('N_double'), cfg)
        budgets[name] = eps
        common = [case['solver_pass'], case['input_pass'], case['sampling_identity_pass'],
                  not changes, case['domain'].get('passed')]
        for label, group in refined.items():
            fine = group.get(name); item = comp.get(label)
            common += [None if item is None else bool(item['independent_refinement_applied']
                and item['same_window'] and item['fine_solver_pass'] and item['fine_input_pass']),
                fine['domain'].get('passed') if fine else None,
                fine['steady'].get('main_stationarity_pass') if fine else None]

        def feature_status(key):
            checks = common + [v5._check(case['sampling'], key), v5._steady_check(steady, key)]
            for label, group in refined.items():
                checks += [v5._check(comp.get(label), key),
                           v5._steady_check(group[name]['steady'], key) if name in group else None]
            return v5._status(checks)

        statuses = {key: feature_status(key) for key in eps}
        main_valid = all(statuses[k] == 'pass' for k in MAIN_KEYS+CONTACT_KEYS)
        main_valid = bool(main_valid and steady.get('main_stationarity_pass') and eps['E_dir_um']['complete'])
        features = {k: v for k, v in statuses.items() if k.startswith(('H_v_', 'H_a_'))}
        bearing_features = {k: v for k, v in statuses.items() if k.startswith('bearing_')}
        record = dict(case=name, metrics=m, epsilon=eps, main_valid=main_valid,
            main_metric_status=v5._status([statuses[k] == 'pass' if statuses[k] != 'not_evaluated' else None for k in MAIN_KEYS]),
            contact_status=v5._status([statuses[k] == 'pass' if statuses[k] != 'not_evaluated' else None for k in CONTACT_KEYS]),
            feature_status=features, full_waveform_pass=bool(main_valid and steady.get('passed')
                and all(status == 'pass' for status in list(features.values())+list(bearing_features.values()))),
            bearing_feature_status=bearing_features,
            physics_freeze_pass=not changes, physics_changes=changes, solver_pass=case['solver_pass'],
            input_pass=case['input_pass'], sampling_identity_pass=case['sampling_identity_pass'],
            domain=case['domain'], source=case['folder'],
            sampling_PP_replay_max_difference=case.get('sampling_PP_replay_max_difference'),
            replay_max_difference=case['replay_max_difference'])
        records.append(record)
        eh = healthy['metrics']['E_dir_um'] if healthy else None
        rows.append(dict(m, case=name, epsilon_E_um=eps['E_dir_um']['epsilon'],
            delta_E_um=m['E_dir_um']-eh if eh is not None else None,
            ratio_to_health=m['E_dir_um']/eh if eh else None, main_valid=main_valid,
            main_metric_status=record['main_metric_status'], contact_status=record['contact_status'],
            full_waveform_pass=record['full_waveform_pass'], physics_freeze_pass=not changes,
            **{'epsilon_'+k: v['epsilon'] for k, v in eps.items() if k.startswith('H_v_')}))
    health = next((r for r in records if r['metrics']['gap_um'] == 2), None)
    pairs = []
    for record in records:
        if not health or record is health:
            continue
        for key in MAIN_KEYS+tuple(health['feature_status']):
            a, b = health['metrics'][key], record['metrics'][key]
            e0, ec = health['epsilon'][key]['epsilon'], record['epsilon'][key]['epsilon']
            combined = e0+ec if e0 is not None and ec is not None else None
            certified = health['main_valid'] and record['main_valid']
            if key not in MAIN_KEYS:
                certified = certified and health['feature_status'][key] == record['feature_status'][key] == 'pass'
            weak = any(part in key for part in ('_residual_', '_2x_', '_3x_', '_energy_'))
            above_floor = combined is not None and max(abs(a), abs(b)) > combined
            resolvable = bool(certified and abs(b-a) > 5*combined and (not weak or above_floor)) if combined is not None else None
            pairs.append(dict(case=record['case'], gap_um=record['metrics']['gap_um'],
                feature=key, healthy=a, degraded=b, difference=b-a,
                ratio=b/a if e0 is not None and abs(a) > 5*e0 else None,
                percent_change=100*(b-a)/abs(a) if e0 is not None and abs(a) > 5*e0 else None,
                epsilon_healthy=e0, epsilon_gap=ec, epsilon_sum=combined,
                above_numerical_floor=above_floor, certified=certified,
                numerically_resolvable=resolvable))
    valid = [r for r in records if r['main_valid']]
    maximum = max(valid, key=lambda r: r['metrics']['E_dir_um'], default=None)
    candidates = [r for r in records if r['metrics'].get('steady_pass') and r['domain'].get('passed')]
    candidate = max(candidates, key=lambda r: r['metrics']['E_dir_um'], default=None)
    near = [r['case'] for r in valid if 18 <= r['metrics']['E_dir_um']-r['epsilon']['E_dir_um']['epsilon']
            and r['metrics']['E_dir_um']+r['epsilon']['E_dir_um']['epsilon'] <= 22]
    strict = [r['case'] for r in valid if r['metrics']['E_dir_um']-r['epsilon']['E_dir_um']['epsilon'] >= 20]
    eh = health['metrics']['E_dir_um'] if health else None
    acceptance = dict(schema_version='v6', health_gap_um=2,
        frozen_U_kg_m=health['metrics']['U_kg_m'] if health else None,
        health_E_um=eh, health_epsilon_um=health['epsilon']['E_dir_um']['epsilon'] if health else None,
        health_target_numerically_certified=bool(health and health['main_valid'] and 1.9 <= eh <= 2.1
            and health['epsilon']['E_dir_um']['epsilon'] < .01),
        near20_pass=bool(near), near20_cases=near, strict20_pass=bool(strict), strict20_cases=strict,
        valid_gaps_um=sorted(r['metrics']['gap_um'] for r in valid),
        maximum_valid_E_um=maximum['metrics']['E_dir_um'] if maximum else None,
        minimum_valid_E_um=min((r['metrics']['E_dir_um'] for r in valid), default=None),
        maximum_valid_gap_um=maximum['metrics']['gap_um'] if maximum else None,
        maximum_valid_epsilon_um=maximum['epsilon']['E_dir_um']['epsilon'] if maximum else None,
        maximum_steady_in_domain_candidate_E_um=candidate['metrics']['E_dir_um'] if candidate else None,
        increase_from_health_um=maximum['metrics']['E_dir_um']-eh if maximum and eh is not None else None,
        same_U_control_pass=bool(records and all(r['physics_freeze_pass'] for r in records)),
        vibration_numerically_resolvable=any(r['numerically_resolvable'] is True and r['feature'].startswith('H_v_') for r in pairs),
        real_machine_validation='not_validated', cases=records,
        note='Main c=2 calibration is conditional, not independent validation; null uncertainty means missing refinement. Near20 uses [18,22] with engineering error bounds; strict20 uses E-epsilon>=20.')
    details = [dict(case=name, refinement=label, feature=key,
        independent_refinement_applied=item['independent_refinement_applied'],
        same_window=item['same_window'], **row) for name, groups in comparisons.items()
        for label, item in groups.items() for key, row in item['metrics'].items()]
    v5._csv(out/'gap_summary.csv', rows)
    v5._csv(out/'same_U_vibration_differences.csv', pairs)
    v5._csv(out/'convergence_summary.csv', details)
    v5._write(out/'acceptance.json', acceptance)
    v5._write(out/'convergence.json', comparisons)
    v5._write(out/'numerical_error_estimates.json', budgets)
    return dict(records=records, acceptance=acceptance, comparisons=comparisons, pairs=pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--base-group', default='data')
    parser.add_argument('--integration-group', default='certification/h_half')
    parser.add_argument('--angular-group', default='certification/n1024')
    args = parser.parse_args()
    result = assemble(args.root, args.output, base_group=args.base_group,
        integration_group=args.integration_group, angular_group=args.angular_group, replay=args.replay)
    print(json.dumps({k: v for k, v in result['acceptance'].items() if k != 'cases'}, indent=2))
