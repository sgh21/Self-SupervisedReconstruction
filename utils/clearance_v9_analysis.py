"""V9 signal, ring-wrench and chunked ball-diagnostic analysis."""
import argparse
import copy
import csv
import json
import math
import numbers
from pathlib import Path

import numpy as np

from spindle import dynamics_model as model
from spindle import tai_bearing, tai_system
from scripts.run_clearance_v3 import write_json, write_csv
from utils import clearance_v6_analysis as v6


RING_COMPONENTS = ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz')
DAMPING_COMPONENTS = ('Fx', 'Fy', 'Mx', 'My')
CHUNK_SIZE = 8192
BALL_WINDOW_TURNS = 10


def _legacy_ball_series(t, q, v, p, variant):
    """Return legacy V8 per-ball diagnostics for B0 or the B0d bridge only."""
    if variant == 'B0':
        data = model.bearing_wrenches(t, q, v, p)
        return {name: row['diagnostics'] for name, row in data.items()}
    if variant != 'B0d':
        return None
    zero = copy.deepcopy(p)
    zero['cb_total'] = 0.0
    zero['rear_bearing']['cb_total'] = 0.0
    zero['damping_policy'] = 'v8_clipped_contact'
    zero['bearing_contact_model'] = 'relative_tilt_v8'
    for key in tuple(zero):
        if key.startswith('_'):
            zero.pop(key, None)
    return {name: model.bearing_wrench_at(t, q, v, zero, name)['diagnostics']
            for name in tai_system.NAMES}


def _legacy_row_from_model(t, q, v, p):
    old = model.bearing_wrenches(t, q, v, p)
    zero = copy.deepcopy(p)
    zero['cb_total'] = 0.0
    zero['rear_bearing']['cb_total'] = 0.0
    zero['damping_policy'] = 'v8_clipped_contact'
    zero['bearing_contact_model'] = 'relative_tilt_v8'
    for key in tuple(zero):
        if key.startswith('_'):
            zero.pop(key, None)
    output = {}
    for name, row in old.items():
        force, moment, diagnostics = row['force_N'], row['moment_Nm'], row['diagnostics']
        elastic = model.bearing_wrench_at(t, q, v, zero, name)
        damping = np.concatenate((force-elastic['force_N'],
                                  moment-elastic['moment_Nm']), axis=-1)
        axial = diagnostics['axial_restoring_N']
        output[name] = dict(
            inner_restoring_6=np.concatenate((force, axial[..., None], moment,
                                               np.zeros_like(axial[..., None])), axis=-1),
            outer_restoring_6=np.concatenate((-force, -axial[..., None], -moment,
                                               np.zeros_like(axial[..., None])), axis=-1),
            force_N=force, moment_Nm=moment,
            damping_wrench4=damping,
            outer_axial_constraint_N=-axial,
            outer_tilt_constraint_Nm=-moment,
            inner_axial_constraint_N=axial,
            diagnostics=np.empty((len(t), 0, len(tai_bearing.BALL_NAMES))),
            legacy_diagnostics=diagnostics)
    return output


def _ball_count(p, name):
    """Read ball count from Tai family data or frozen V8 bearing fields."""
    family = p.get('tai_bearing', {}).get('families', {}).get(
        'front' if name.startswith('F') else 'rear')
    if family is not None:
        return int(family['balls'])
    return int(p['balls'] if name.startswith('F') else p['rear_bearing']['balls'])


def _variant(config):
    declared = config.get('mechanism_variant')
    if declared in ('B0', 'B0d', 'N1', 'N2', 'N3'):
        return declared
    p = config['parameters']
    if p.get('bearing_contact_model') == 'relative_tilt_v8':
        return 'B0d' if p.get('damping_policy') == 'common_structural_ring' else 'B0'
    return 'N3'


def _api_rows(t, q, v, p, variant):
    if variant == 'B0':
        return _legacy_row_from_model(t, q, v, p)
    rows = tai_system.diagnostics_series(t, q, v, p)
    if variant == 'B0d':
        legacy = _legacy_ball_series(t, q, v, p, variant)
        for name in tai_system.NAMES:
            rows[name]['legacy_diagnostics'] = legacy[name]
    return rows


def _array_stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(values*values))), float(np.max(np.abs(values)))


def _field_array(row, fields, legacy, nballs):
    if legacy:
        output = {}
        for key, value in row['legacy_diagnostics'].items():
            array = np.asarray(value)
            if array.ndim == 2 and array.shape[-1] == nballs:
                output[key] = array
        return output
    balls = np.asarray(row['diagnostics'], dtype=float)
    if balls.ndim != 3 or balls.shape[-1] != len(fields):
        raise ValueError(f'Unexpected V9 ball diagnostic shape {balls.shape}')
    return {field: balls[..., i] for i, field in enumerate(fields)}


def _new_ball_summary(balls, b0, control_branch):
    fields = tai_bearing.BALL_NAMES
    index = {name: i for i, name in enumerate(fields)}
    active = np.rint(balls[..., tai_bearing.ACTIVE]).astype(int)
    control = np.rint(balls[..., tai_bearing.CONTROL]).astype(int)
    status = np.rint(balls[..., tai_bearing.STATUS]).astype(int)
    margin_index = (tai_bearing.OUTER_MARGIN if control_branch == 'outer'
                    else tai_bearing.INNER_MARGIN)
    margin = balls[..., margin_index]
    q_i, q_o = balls[..., tai_bearing.QI], balls[..., tai_bearing.QO]
    di, do = balls[..., tai_bearing.DI], balls[..., tai_bearing.DO]
    g1 = balls[..., tai_bearing.G1]
    g2 = balls[..., tai_bearing.G2]
    g3 = balls[..., tai_bearing.G3]
    g4 = balls[..., tai_bearing.G4]
    force_scale = np.maximum(q_i, q_o)
    force_tolerance = np.maximum(1e-6, 1e-7*force_scale)
    force_residual = np.maximum(np.abs(g3), np.abs(g4))
    dual_count = np.sum(active == 2, axis=1)
    control_good = np.sum((margin > 0) & (active == 2), axis=1)
    result = dict(
        contact_fraction=np.mean(active > 0, axis=1),
        dual_contact_fraction=np.mean(active == 2, axis=1),
        inner_loss_fraction=np.mean(active == 1, axis=1),
        outer_control_self_consistent_fraction=np.divide(control_good, dual_count,
            out=np.full(len(active), np.nan), where=dual_count > 0),
        control_dual_checked_count=dual_count,
        control_consistent_count=control_good,
        status_success_count=np.sum(status >= 0, axis=1),
        status_retry_count=np.sum(status > 0, axis=1),
        status_failed_count=np.sum(status < 0, axis=1),
        local_failure_count=np.sum(status < 0, axis=1),
        active_dual_count=np.sum(active == 2, axis=1),
        active_outer_only_count=np.sum(active == 1, axis=1),
        active_open_count=np.sum(active == 0, axis=1),
        control_outer_count=np.sum(control == 0, axis=1),
        control_inner_count=np.sum(control == 1, axis=1),
        max_Qi_N=np.max(q_i, axis=1), max_Qo_N=np.max(q_o, axis=1),
        max_Fc_N=np.max(balls[..., tai_bearing.FC], axis=1),
        max_abs_Mg_Nm=np.max(np.abs(balls[..., tai_bearing.MG]), axis=1),
        max_p0_i_Pa=np.max(balls[..., tai_bearing.PRESSURE_I], axis=1),
        max_p0_o_Pa=np.max(balls[..., tai_bearing.PRESSURE_O], axis=1),
        max_force_residual_N=np.max(np.abs(np.stack((g3, g4), axis=-1)), axis=(1, 2)),
        max_force_residual_tolerance_ratio=np.max(force_residual/force_tolerance, axis=1),
        max_geometry_residual_normalized=np.max(np.abs(np.stack((g1, g2), axis=-1)), axis=(1, 2))/(b0*b0),
        max_phase_speed_relative_difference=np.max(np.abs(balls[..., tai_bearing.PHASE_ERROR]), axis=1),
        max_iterations=np.max(balls[..., tai_bearing.ITERATIONS], axis=1),
        min_delta_i_m=np.min(di, axis=1), max_delta_i_m=np.max(di, axis=1),
        min_delta_o_m=np.min(do, axis=1), max_delta_o_m=np.max(do, axis=1),
        mean_contact_angle_i_rad=np.mean(balls[..., tai_bearing.ANGLE_I], axis=1),
        mean_contact_angle_o_rad=np.mean(balls[..., tai_bearing.ANGLE_O], axis=1),
    )
    return result


def _legacy_ball_summary(diagnostics):
    delta = np.asarray(diagnostics['delta_m'], dtype=float)
    normal = np.asarray(diagnostics['normal_force_N'], dtype=float)
    active = (delta > 0) & (normal > 0)
    return dict(legacy_contact_fraction=np.mean(active, axis=1),
        legacy_max_normal_force_N=np.max(normal, axis=1),
        legacy_min_delta_m=np.min(delta, axis=1),
        legacy_max_delta_m=np.max(delta, axis=1))


def _ball_rows(fields, values, name, schema, count_acc):
    """Update per-ball min/max/mean accumulators without retaining the full history."""
    for field, array in values.items():
        array = np.asarray(array, dtype=float)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            continue
        # Fields are time x ball. Sum in float64 and skip undefined margins.
        finite = np.isfinite(array)
        key = (name, schema, field)
        if key not in count_acc:
            count_acc[key] = dict(min=np.full(array.shape[1], np.inf),
                max=np.full(array.shape[1], -np.inf), sum=np.zeros(array.shape[1]),
                count=np.zeros(array.shape[1], dtype=np.int64))
        agg = count_acc[key]
        for ball in range(array.shape[1]):
            selected = array[finite[:, ball], ball]
            if selected.size:
                agg['min'][ball] = min(agg['min'][ball], float(selected.min()))
                agg['max'][ball] = max(agg['max'][ball], float(selected.max()))
                agg['sum'][ball] += float(selected.sum())
                agg['count'][ball] += selected.size


def _stats_csv_rows(count_acc):
    rows = []
    for (bearing, schema, field), aggregate in sorted(count_acc.items()):
        for ball_id, count in enumerate(aggregate['count']):
            if count:
                rows.append(dict(bearing=bearing, diagnostic_schema=schema,
                    ball_id=ball_id, field=field, minimum=float(aggregate['min'][ball_id]),
                    maximum=float(aggregate['max'][ball_id]),
                    mean=float(aggregate['sum'][ball_id]/count), finite_samples=int(count)))
    return rows


def _accumulate_subwindow_metric(store, key, value, sample_count):
    """Combine chunk statistics without letting the last chunk replace a window."""
    value = float(value)
    if not np.isfinite(value):
        return
    kind = ('rms' if '_rms_' in key else 'max' if '_peak_' in key or '_max_' in key
            else 'min' if '_min_' in key else 'mean')
    row = store.setdefault(key, dict(kind=kind, total=0.0, count=0,
        extreme=(-np.inf if kind == 'max' else np.inf)))
    if kind == 'rms':
        row['total'] += value*value*sample_count
        row['count'] += sample_count
    elif kind == 'mean':
        row['total'] += value*sample_count
        row['count'] += sample_count
    elif kind == 'max':
        row['extreme'] = max(row['extreme'], value)
    else:
        row['extreme'] = min(row['extreme'], value)


def _finish_subwindow_metrics(stores):
    output, sample_counts = [], []
    for store in stores:
        row, counts = {}, []
        for key, value in store.items():
            if value['kind'] == 'rms':
                row[key] = float(np.sqrt(value['total']/value['count']))
                counts.append(value['count'])
            elif value['kind'] == 'mean':
                row[key] = float(value['total']/value['count'])
                counts.append(value['count'])
            else:
                row[key] = float(value['extreme'])
        output.append(row)
        sample_counts.append(max(counts, default=0))
    return output, sample_counts


def collect_bearing_evidence(result, config, start, end, stride=1,
                             include_wrench_timeseries=False,
                             include_ball_window=False):
    """Chunk the retained window; only the selected 10-turn ball array is kept."""
    p, sim = config['parameters'], config['simulation']
    variant = _variant(config)
    branch = p.get('tai_bearing', {}).get('control_branch', 'outer')
    t_all = np.asarray(result['t'])[::stride]
    q_all, v_all = np.asarray(result['q'])[::stride], np.asarray(result['v'])[::stride]
    keep = (t_all >= start-1e-12) & (t_all < end-1e-12)
    indices = np.flatnonzero(keep)
    t = t_all[keep]
    q, v = q_all[keep], v_all[keep]
    if not len(t):
        raise ValueError('No native samples lie in the V9 measurement window')
    omega = abs(p['rpm']*2*np.pi/60)
    ball_end = min(end, start + BALL_WINDOW_TURNS*2*np.pi/omega)
    ball_keep = (t >= start-1e-12) & (t < ball_end-1e-12)
    window_info = dict(start_s=float(start), end_s=float(ball_end),
                       requested_complete_turns=min(BALL_WINDOW_TURNS,
                           int(config['simulation']['retained_turns'])),
                       saved_samples=int(np.sum(ball_keep)),
                       sample_stride=int(stride),
                       sample_rate_hz=float(1/np.median(np.diff(t))) if len(t) > 1 else None,
                       diagnostic_schema=('tai_32_fields' if variant not in ('B0','B0d')
                                          else 'legacy_v8_bridge'))
    fields = tai_bearing.BALL_NAMES
    b0_by_name = {}
    if variant not in ('B0','B0d'):
        for name in tai_system.NAMES:
            family_name = 'front' if name.startswith('F') else 'rear'
            family = p['tai_bearing']['families'][family_name]
            b0_by_name[name] = (family['fi']+family['fo']-1)*family['D_m']
    n_sub = int(sim.get('subwindows', 4))
    edges = np.linspace(start, end, n_sub+1)
    sub_accumulators = [dict() for _ in range(n_sub)]
    time_arrays = {}
    ball_arrays = {}
    field_acc = {}
    if include_wrench_timeseries:
        time_arrays['t_raw'] = t.copy()
        for name in tai_system.NAMES:
            time_arrays[name+'_inner_restoring_6'] = np.empty((len(t), 6))
            time_arrays[name+'_outer_restoring_6'] = np.empty((len(t), 6))
            time_arrays[name+'_damping_wrench4'] = np.empty((len(t), 4))
            for key, size in (('outer_axial_constraint_N', 1),
                              ('outer_tilt_constraint_Nm', 2), ('inner_axial_constraint_N', 1)):
                time_arrays[name+'_'+key] = np.empty((len(t), size))
            for key in ('contact_fraction','dual_contact_fraction','inner_loss_fraction',
                        'outer_control_self_consistent_fraction','local_failure_count',
                        'control_dual_checked_count','control_consistent_count',
                        'status_success_count','status_retry_count','status_failed_count',
                        'active_dual_count','active_outer_only_count','active_open_count',
                        'control_outer_count','control_inner_count','max_Qi_N','max_Qo_N',
                        'max_Fc_N','max_abs_Mg_Nm','max_p0_i_Pa','max_p0_o_Pa',
                        'max_force_residual_N','max_geometry_residual_normalized',
                        'max_force_residual_tolerance_ratio',
                        'max_phase_speed_relative_difference','max_iterations',
                        'min_delta_i_m','max_delta_i_m','min_delta_o_m','max_delta_o_m',
                        'legacy_contact_fraction','legacy_max_normal_force_N',
                        'legacy_min_delta_m','legacy_max_delta_m'):
                time_arrays[name+'_'+key] = np.full(len(t), np.nan)
        if variant not in ('B0','B0d'):
            for name in tai_system.NAMES:
                nballs = _ball_count(p, name)
                ball_arrays[name+'_tai_32_fields'] = np.empty((int(ball_keep.sum()), nballs, len(fields)))
    for begin in range(0, len(t), CHUNK_SIZE):
        stop = min(len(t), begin+CHUNK_SIZE)
        sl = slice(begin, stop)
        rows = _api_rows(t[sl], q[sl], v[sl], p, variant)
        local_masks = [(t[sl] >= left-1e-12) & (t[sl] < right-1e-12)
                       for left, right in zip(edges[:-1], edges[1:])]
        for name, row in rows.items():
            inner = np.asarray(row['inner_restoring_6'])
            outer = np.asarray(row['outer_restoring_6'])
            damping = np.asarray(row['damping_wrench4'])
            diagnostics = row.get('diagnostics')
            legacy = variant in ('B0','B0d')
            if legacy:
                diag = row.get('legacy_diagnostics')
                if diag is None:
                    diag = _legacy_ball_series(t[sl], q[sl], v[sl], p, variant)[name]
                ball_summary = _legacy_ball_summary(diag)
                ball_values = _field_array(row, fields, True, _ball_count(p, name))
            else:
                ball_array = np.asarray(diagnostics, dtype=float)
                ball_summary = _new_ball_summary(ball_array, b0_by_name[name], branch)
                ball_values = {field: ball_array[..., i] for i, field in enumerate(fields)}
            for key, value in (('inner_restoring_6',inner),('outer_restoring_6',outer),
                               ('damping_wrench4',damping)):
                if include_wrench_timeseries:
                    time_arrays[name+'_'+key][sl] = value
            for key in ('outer_axial_constraint_N','outer_tilt_constraint_Nm','inner_axial_constraint_N'):
                value = row.get(key)
                if value is None:
                    continue
                arr = np.asarray(value)
                if arr.ndim == 1:
                    arr = arr[:, None]
                if include_wrench_timeseries:
                    time_arrays[name+'_'+key][sl] = arr
            if include_wrench_timeseries:
                for key, values in ball_summary.items():
                    target = name+'_'+key
                    if target in time_arrays:
                        time_arrays[target][sl] = values
            _ball_rows(fields, ball_values, name,
                       'legacy_v8_cb_zero' if legacy and variant == 'B0d'
                       else 'legacy_v8' if legacy else 'tai_32_fields', field_acc)
            if include_wrench_timeseries and include_ball_window and not legacy:
                selected = ball_keep[sl]
                if selected.any():
                    left = int(np.sum(ball_keep[:begin]))
                    right = left + int(np.sum(selected))
                    ball_arrays[name+'_tai_32_fields'][left:right] = ball_array[selected]
            if include_wrench_timeseries and include_ball_window and legacy:
                selected = ball_keep[sl]
                if selected.any():
                    left = int(np.sum(ball_keep[:begin]))
                    right = left + int(np.sum(selected))
                    for key, values in diag.items():
                        array = np.asarray(values)
                        if array.ndim != 2 or array.shape[-1] != int(
                                _ball_count(p, name)):
                            continue
                        target = name+'_legacy_'+key
                        if target not in ball_arrays:
                            ball_arrays[target] = np.empty((int(ball_keep.sum()),)+array.shape[1:])
                        ball_arrays[target][left:right] = array[selected]
            for sub_i, mask in enumerate(local_masks):
                if not mask.any():
                    continue
                metric = _summarize_ring(row, ball_summary, name, mask)
                for key, value in metric.items():
                    _accumulate_subwindow_metric(sub_accumulators[sub_i], key, value,
                                                 int(np.sum(mask)))
    sub_metrics, sub_counts = _finish_subwindow_metrics(sub_accumulators)
    metrics = {}
    keys = set.intersection(*(set(row) for row in sub_metrics)) if sub_metrics else set()
    for key in keys:
        values = np.asarray([row[key] for row in sub_metrics], dtype=float)
        weights = np.asarray(sub_counts, dtype=float)
        if '_rms_' in key:
            metrics[key] = float(np.sqrt(np.average(values*values, weights=weights)))
        elif '_peak_' in key or '_max_' in key:
            metrics[key] = float(np.max(values))
        elif '_min_' in key:
            metrics[key] = float(np.min(values))
        else:
            metrics[key] = float(np.average(values, weights=weights))
    # Recompute full-window statistics directly from preserved time arrays where available.
    if include_wrench_timeseries:
        for name in tai_system.NAMES:
            row = dict(inner_restoring_6=time_arrays[name+'_inner_restoring_6'],
                       outer_restoring_6=time_arrays[name+'_outer_restoring_6'],
                       damping_wrench4=time_arrays[name+'_damping_wrench4'])
            summaries = _summarize_ring(row, None, name, np.ones(len(t), dtype=bool))
            metrics.update(summaries)
            summary_series = {key[len(name)+1:]: value for key,value in time_arrays.items()
                if key.startswith(name+'_') and key[len(name)+1:] in _DIAG_TIME_KEYS}
            if summary_series:
                metrics.update(_diagnostic_metrics(summary_series, name, np.ones(len(t), dtype=bool)))
    return dict(t_raw=t, window_info=window_info, time_arrays=time_arrays,
        ball_arrays=ball_arrays, per_ball_rows=_stats_csv_rows(field_acc),
        metrics=metrics, subwindow_metrics=sub_metrics,
        subwindow_edges_s=edges.tolist(), source_indices=indices)


def _summarize_ring(row, ball_summary, name, mask):
    result = {}
    for side in ('inner','outer'):
        values = np.asarray(row[side+'_restoring_6'])[mask]
        for i, component in enumerate(RING_COMPONENTS):
            rms, peak = _array_stats(values[:, i])
            unit = 'N' if i < 3 else 'Nm'
            prefix = f'bearing_{name}_{side}_{component}'
            result[prefix+f'_rms_{unit}'] = rms
            result[prefix+f'_peak_{unit}'] = peak
    damping = np.asarray(row['damping_wrench4'])[mask]
    for i, component in enumerate(DAMPING_COMPONENTS):
        rms, peak = _array_stats(damping[:, i])
        unit = 'N' if i < 2 else 'Nm'
        prefix = f'bearing_{name}_damping_{component}'
        result[prefix+f'_rms_{unit}'] = rms
        result[prefix+f'_peak_{unit}'] = peak
    if ball_summary is not None:
        result.update(_diagnostic_metrics(ball_summary, name, mask))
    return result


_DIAG_TIME_KEYS = ('contact_fraction','dual_contact_fraction','inner_loss_fraction',
    'outer_control_self_consistent_fraction','local_failure_count','control_dual_checked_count',
    'control_consistent_count','status_success_count','status_retry_count','status_failed_count','active_dual_count',
    'active_outer_only_count','active_open_count','control_outer_count','control_inner_count',
    'max_Qi_N','max_Qo_N','max_Fc_N','max_abs_Mg_Nm','max_p0_i_Pa','max_p0_o_Pa',
    'max_force_residual_N','max_geometry_residual_normalized',
    'max_force_residual_tolerance_ratio',
    'max_phase_speed_relative_difference','max_iterations','min_delta_i_m','max_delta_i_m',
    'min_delta_o_m','max_delta_o_m','legacy_contact_fraction','legacy_max_normal_force_N',
    'legacy_min_delta_m','legacy_max_delta_m')


def _diagnostic_metrics(summary, name, mask):
    output = {}
    for key in _DIAG_TIME_KEYS:
        value = summary.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float)[mask]
        if not arr.size or not np.isfinite(arr).any():
            continue
        if key in ('contact_fraction','dual_contact_fraction','inner_loss_fraction',
                   'legacy_contact_fraction',
                   'outer_control_self_consistent_fraction'):
            output[f'bearing_{name}_{key}'] = float(np.nanmean(arr))
        elif key in ('local_failure_count','active_dual_count','active_outer_only_count',
                     'active_open_count','control_outer_count','control_inner_count'):
            output[f'bearing_{name}_{key}_mean'] = float(np.nanmean(arr))
        elif key.startswith('min_'):
            output[f'bearing_{name}_{key}'] = float(np.nanmin(arr))
        else:
            output[f'bearing_{name}_{key}'] = float(np.nanmax(arr))
    return output


def analyze(result, config, start, end, stride=1, bearing=None,
            include_wrench_timeseries=False, include_ball_window=False):
    """Reuse V6 signal analysis and append V9 per-bearing mechanics evidence."""
    local = copy.deepcopy(config)
    raw_fs = round(1/float(np.median(np.diff(result['t'][:1000]))))
    local['simulation']['fs_raw_hz'] = raw_fs
    p = local['parameters']
    if 'measured_relative_xy_m' in result:
        base = dict(result)
    else:
        section = model.rotor_section(result['q'], p['z_m'])
        probe = result['q'][:,4:6] if p.get('probe_reference') == 'housing' else np.zeros_like(section)
        base = dict(result, measured_relative_xy_m=section-probe)
    if 'fit_force_N' not in base and 'fit1_force_N' in base:
        base['fit_force_N'] = base['fit1_force_N']
    if 'fit2_force_N' not in base and 'fit2_N' in base:
        base['fit2_force_N'] = base['fit2_N']
    metrics, turns, output, steady = v6.analyze(base, local, start, end, stride)
    for key, row in steady.get('checks', {}).items():
        if key.startswith('H_v_') and any(term in key for term in
                ('_2x_', '_3x_', '_residual_', '_energy_')):
            values = np.asarray(row['values'], dtype=float)
            limit = max(float(local.get('v9_acceptance', {}).get(
                            'residual_velocity_absolute_mm_s', .002)),
                        .05*float(np.max(np.abs(values))))
            row.update(tolerance=limit, passed=bool(row['difference'] <= limit))
    steady['passed'] = bool(all(row['passed'] for row in steady.get('checks', {}).values())
                            and all(row['passed'] for row in steady.get('band_drift_checks', {}).values())
                            and steady.get('main_stationarity_pass', True))
    metrics['steady_pass'] = steady['passed']
    evidence = bearing if bearing is not None else collect_bearing_evidence(
        result, local, start, end, stride, include_wrench_timeseries, include_ball_window)
    metrics.update(evidence['metrics'])
    signal_pass = bool(steady.get('passed'))
    edges = steady['window_edges_s']
    bearing_rows = evidence['subwindow_metrics']
    checks = {}
    if len(bearing_rows) == len(edges)-1:
        keys = sorted(set.intersection(*(set(row) for row in bearing_rows)))
        for key in keys:
            if not ('_rms_' in key or '_peak_' in key or key.endswith(
                    ('_contact_fraction', '_dual_contact_fraction', '_inner_loss_fraction'))):
                continue
            vals = np.asarray([row[key] for row in bearing_rows], dtype=float)
            if not np.isfinite(vals).all():
                continue
            limit = tolerance(key, float(vals.min()), float(vals.max()), local)
            checks[key] = dict(values=vals.tolist(), difference=float(np.ptp(vals)),
                tolerance=float(limit), passed=bool(np.ptp(vals) <= limit))
    bearing_metrics = evidence['metrics']
    local_checks = {}
    for key, value in bearing_metrics.items():
        if key.endswith('_max_geometry_residual_normalized'):
            limit = float(local['v9_acceptance']['local_geometry_normalized'])
            passed = bool(value <= limit)
        elif key.endswith('_max_force_residual_tolerance_ratio'):
            limit = 1.0
            passed = bool(value <= limit)
        elif key.endswith('_status_failed_count'):
            limit = 0.0
            passed = bool(value <= limit)
        else:
            continue
        local_checks[key] = dict(value=float(value), tolerance=limit,
                                 passed=passed)
    branch_checks = {key: dict(value=float(value), passed=bool(value >= 1-1e-12))
        for key, value in bearing_metrics.items()
        if key.endswith('_outer_control_self_consistent_fraction') and np.isfinite(value)}
    phase_warnings = {key: dict(value=float(value), warning=bool(value >
        local['v9_acceptance']['phase_speed_warning_fraction']))
        for key, value in bearing_metrics.items()
        if key.endswith('_max_phase_speed_relative_difference')}
    bearing_pass = bool(checks and all(row['passed'] for row in checks.values()))
    tai_variant = _variant(local) not in ('B0','B0d')
    metrics['local_root_pass'] = (bool(local_checks and all(row['passed']
        for row in local_checks.values())) if tai_variant else None)
    metrics['ring_control_self_consistency_pass'] = (bool(all(row['passed']
        for row in branch_checks.values()) if branch_checks else True) if tai_variant else None)
    steady.update(signal_stationarity_pass=signal_pass,
        bearing_stationarity_pass=bearing_pass,
        bearing_checks=checks, bearing_subwindow_metrics=bearing_rows,
        bearing_local_acceptance=local_checks, ring_control_self_consistency=branch_checks,
        phase_speed_warnings=phase_warnings,
        bearing_window_info=evidence['window_info'],
        passed=bool(signal_pass and bearing_pass))
    metrics.update(signal_steady_pass=signal_pass, bearing_steady_pass=bearing_pass,
        steady_pass=bool(signal_pass and bearing_pass),
        gap_um=p['front_group_clearance_um'],
        mechanism_variant=_variant(local))
    return metrics, turns, output, steady, evidence


def absolute_floor(key, config):
    acceptance = config.get('v9_acceptance', {})
    if ('contact_fraction' in key or key.endswith('_inner_loss_fraction')
            or 'control_self_consistent_fraction' in key):
        return float(config['numerical_thresholds']['contact_rate'])
    if key.endswith('_Nm') or '_Nm_' in key:
        return float(acceptance.get('wrench_moment_absolute_Nm', 1e-7))
    if key.endswith('_N') or '_N_' in key:
        return float(acceptance.get('wrench_force_absolute_N', 1e-5))
    if 'geometry_residual' in key:
        return float(acceptance.get('local_geometry_normalized', 1e-9))
    if 'force_residual' in key:
        return float(acceptance.get('local_force_absolute_N', 1e-6))
    return 1e-8


def tolerance(key, first, second, config):
    acceptance = config.get('v9_acceptance', {})
    if ('contact_fraction' in key or key.endswith('_inner_loss_fraction')
            or 'control_self_consistent_fraction' in key):
        return float(config['numerical_thresholds']['contact_rate'])
    rel = (acceptance.get('wrench_peak_relative', .05) if '_peak_' in key or 'max_' in key
           else acceptance.get('wrench_rms_relative', .02))
    return max(absolute_floor(key, config), float(rel)*max(abs(first), abs(second)))


def compare_metrics(base, fine, config):
    signal_base = {key: value for key, value in base.items() if not key.startswith('bearing_')}
    signal_fine = {key: value for key, value in fine.items() if not key.startswith('bearing_')}
    signal = v6.compare_metrics(signal_base, signal_fine, config)
    checks = dict(signal.get('metrics', {}))
    physical_bearing = lambda key: ('_rms_' in key or '_peak_' in key or key.endswith(
        ('_contact_fraction', '_dual_contact_fraction', '_inner_loss_fraction')))
    for key in sorted(k for k in fine if k.startswith('bearing_') and physical_bearing(k)):
        if key not in base:
            continue
        delta = abs(float(base[key])-float(fine[key]))
        limit = tolerance(key, float(base[key]), float(fine[key]), config)
        checks[key] = dict(base=float(base[key]), fine=float(fine[key]), difference=delta,
            tolerance=limit, passed=bool(delta <= limit),
            above_absolute_floor=bool(min(abs(base[key]),abs(fine[key])) > absolute_floor(key,config)))
    pass_rows = [item['passed'] for item in checks.values()]
    bearing_rows = [row['passed'] for key,row in checks.items() if key.startswith('bearing_')]
    bearing_pass = bool(bearing_rows and all(bearing_rows))
    return dict(passed=bool(signal.get('same_window') and pass_rows and all(pass_rows)
                             and bearing_pass),
        same_window=signal.get('same_window',False), metrics=checks,
        main_metric_pass=signal.get('main_metric_pass'),
        contact_pass=signal.get('contact_pass'), feature_pass=signal.get('feature_pass'),
        bearing_wrench_pass=bearing_pass)


def save_bearing_evidence(folder, evidence, config):
    folder = Path(folder)
    legacy = _variant(config) in ('B0','B0d')
    if evidence['time_arrays']:
        np.savez_compressed(folder/'bearing_wrenches.npz', **evidence['time_arrays'])
    if evidence['ball_arrays']:
        np.savez_compressed(folder/'balls_first10turns.npz',
            t_raw=evidence['t_raw'][
                (evidence['t_raw'] >= evidence['window_info']['start_s']-1e-12) &
                (evidence['t_raw'] < evidence['window_info']['end_s']-1e-12)],
            **evidence['ball_arrays'])
    if evidence['per_ball_rows']:
        write_csv(folder/'per_ball_statistics.csv', evidence['per_ball_rows'])
    write_json(folder/'wrench_diagnostics.json', dict(
        metrics=evidence['metrics'], subwindow_metrics=evidence['subwindow_metrics'],
        window=evidence['window_info'],
        diagnostic_schema='legacy_v8' if legacy else 'tai_32_fields',
        state_fields=(sorted(set(key.split('_legacy_',1)[1] for key in evidence['ball_arrays']
                                 if '_legacy_' in key)) if legacy else list(tai_bearing.BALL_NAMES)),
        status_codes='0: original deterministic root; 1: same equations solved after deterministic reseed; negative: failed root. Per-ring status_retry_count is a per-time count; reported metric is its maximum over the export window.',
        wrench_convention='Tai ring wrenches are restoring quantities on the equation LHS; ideal axial/outer-tilt reaction fields are supplied by spindle.tai_system.',
        ball_retention=('Full-window legacy per-ball statistics; original V8 delta/normal-force fields for 10 complete turns.' if legacy else 'Full-window per-ball min/max/mean and per-sample per-ring extrema/status/count; all 32 fields for the first 10 complete turns.'),
        mechanism_variant=config.get('mechanism_variant')))


def load_case(folder, replay=False):
    folder = Path(folder)
    config = json.loads((folder/'actual_config.json').read_text(encoding='utf-8'))
    metrics = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    steady = json.loads((folder/'stationarity.json').read_text(encoding='utf-8'))
    with np.load(folder/'native.npz') as data:
        result = {key: data[name] for key,name in dict(t='t_raw',q='q_raw_m',v='v_raw_m_s',
            acc='a_raw_m_s2',fit_force_N='fit_force_N',fit_active_count='fit_active_count',
            fit2_force_N='fit2_force_N',fit2_active_count='fit2_active_count').items()}
        relative_key = ('measured_relative_xy_m' if 'measured_relative_xy_m' in data.files
                        else 'r_m_raw_m')
        if relative_key in data.files:
            result['measured_relative_xy_m'] = data[relative_key]
        meta = json.loads(str(data['metadata_json']))
    replay_metrics = None
    if replay:
        replay_metrics, _, _, replay_steady, _ = analyze(
            result, config, meta['export_start_s'], meta['export_end_s'],
            bearing=None, include_wrench_timeseries=False, include_ball_window=False)
        difference = max((abs(float(replay_metrics[k])-float(metrics[k]))
                          for k in replay_metrics if k in metrics and '_phase_' not in k
                          and isinstance(replay_metrics[k], numbers.Real)
                          and isinstance(metrics[k], numbers.Real)
                          and not isinstance(replay_metrics[k], bool)
                          and not isinstance(metrics[k], bool)
                          and math.isfinite(float(replay_metrics[k]))
                          and math.isfinite(float(metrics[k]))),
                         default=0.)
        return dict(folder=folder,config=config,metrics=metrics,steady=steady,
                    replay_metrics=replay_metrics,replay_max_difference=difference,
                    replay_steady=replay_steady)
    return dict(folder=folder,config=config,metrics=metrics,steady=steady)


def assemble(root, output=None, replay=False, group='data_A'):
    root = Path(root)
    out = Path(output) if output else root/'recomputed'
    out.mkdir(parents=True, exist_ok=True)
    cases = {}
    for path in sorted((root/group).rglob('metrics.json')):
        case = load_case(path.parent, replay)
        key = path.parent.relative_to(root/group).as_posix()
        cases[key] = case
    records, rows = [], []
    for name, case in cases.items():
        m = case['metrics']; cfg = case['config']; steady = case['steady']
        record = dict(case=name,metrics=m,steady=steady,
            signal_steady_pass=m.get('signal_steady_pass'),
            bearing_steady_pass=m.get('bearing_steady_pass'),
            full_steady_pass=m.get('steady_pass'),
            replay_max_difference=case.get('replay_max_difference'))
        records.append(record)
        rows.append(dict(case=name,variant=cfg.get('mechanism_variant'),
            gap_um=m.get('gap_um'),U_kg_m=cfg.get('excitation',{}).get('U_kg_m'),
            E_dir_um=m.get('E_dir_um'),signal_steady_pass=m.get('signal_steady_pass'),
            bearing_steady_pass=m.get('bearing_steady_pass'),steady_pass=m.get('steady_pass'),
            **{k:v for k,v in m.items() if k.startswith('bearing_')}))
    write_csv(out/'case_summary.csv', rows)
    write_json(out/'offline_analysis.json',dict(group=group,cases=records,
        scope='Conditional V9 dynamics; no real-machine validation implied.'))
    return dict(records=records)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--group',default='data_A')
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    result=assemble(args.root,args.output,args.replay,args.group)
    print(json.dumps(dict(group=args.group,cases=len(result['records'])),ensure_ascii=False))
