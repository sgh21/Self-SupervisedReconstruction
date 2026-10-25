"""V8 contact-wrench evidence added to the unchanged V6 signal definitions."""
import json
from pathlib import Path

import numpy as np
from spindle import dynamics_model as model
from scripts.run_clearance_v3 import write_json


def wrench_floor(key):
    if '_contact_fraction' in key:
        return .02
    return 1e-5 if key.endswith('_Nm') else 1e-4


def wrench_signals(t, q, v, p):
    """Bound contact-array memory while retaining every native wrench sample."""
    names = ('F1', 'F2', 'R1', 'R2')
    signals = {name: dict(force=np.empty((len(t), 2)), moment=np.empty((len(t), 2)),
        axial=np.empty(len(t)), ball_peak=np.empty(len(t)),
        contact=np.empty(len(t))) for name in names}
    for left in range(0, len(t), 8192):
        use = slice(left, left+8192)
        rows = model.bearing_wrenches(t[use], q[use], v[use], p)
        for name, row in rows.items():
            state = row['diagnostics']; output = signals[name]
            output['force'][use] = row['force_N']
            output['moment'][use] = row['moment_Nm']
            output['axial'][use] = state['axial_restoring_N']
            output['ball_peak'][use] = np.max(state['normal_force_N'], axis=-1)
            output['contact'][use] = np.mean(state['delta_m'] > 0, axis=-1)
    return signals


def summarize(signals, use):
    metrics = {}
    for name, row in signals.items():
        prefix = 'bearing_'+name
        for quantity, field, unit in (('radial', 'force', 'N'), ('local_moment', 'moment', 'Nm')):
            values = row[field][use]
            for axis, rms, peak in zip('xy', np.sqrt(np.mean(values**2, axis=0)),
                                      np.max(abs(values), axis=0)):
                metrics[f'{prefix}_{quantity}_rms_{axis}_{unit}'] = float(rms)
                metrics[f'{prefix}_{quantity}_peak_{axis}_{unit}'] = float(peak)
        metrics[prefix+'_axial_rms_N'] = float(np.sqrt(np.mean(row['axial'][use]**2)))
        metrics[prefix+'_ball_peak_N'] = float(row['ball_peak'][use].max())
        metrics[prefix+'_contact_fraction'] = float(row['contact'][use].mean())
    return metrics


def add_wrench_metrics(result, config, stride, metrics, steady, start, end):
    t, q, v = (result[key][::stride] for key in ('t', 'q', 'v'))
    # Guards have no role in unfiltered mechanical wrench statistics.
    keep = (t >= start-1e-12) & (t < end-1e-12)
    t, q, v = t[keep], q[keep], v[keep]
    signals = wrench_signals(t, q, v, config['parameters'])
    metrics.update(summarize(signals, np.ones(len(t), dtype=bool)))
    edges = steady['window_edges_s']
    for row, left, right in zip(steady['subwindow_metrics'], edges[:-1], edges[1:]):
        row.update(summarize(signals, (t >= left-1e-12) & (t < right-1e-12)))


def saved_wrench_report(folder):
    """Separate local/lever moments and ideal rotational/axial constraint loads."""
    folder = Path(folder)
    config = json.loads((folder/'actual_config.json').read_text(encoding='utf-8'))
    p = config['parameters']
    with np.load(folder/'native.npz') as data:
        t, q, v = (data[k] for k in ('t_raw', 'q_raw_m', 'v_raw_m_s'))
        meta = json.loads(str(data['metadata_json']))
        fit = [data['fit_force_N'], data['fit2_force_N']]
        rows = {name: dict(force=data[name+'_radial_restoring_N'],
            local=data[name+'_local_restoring_Nm'], axial=data[name+'_axial_restoring_N'])
            for name in ('F1', 'F2', 'R1', 'R2')}
    use = (t >= meta['export_start_s']) & (t < meta['export_end_s'])
    angle = p['rpm']*2*np.pi/60*t[use]+config['excitation']['phase_rad']
    design = np.column_stack((np.ones(sum(use)), np.cos(angle), np.sin(angle)))
    constraints = np.zeros((len(t), 2)); axial = np.zeros(len(t)); table = {}
    archive = {'t_raw': t}
    for i, spec in enumerate(p['bearing_layout']):
        name, z = spec['name'], spec['z_m']; row = rows[name]
        lever = row['force'] @ np.array([[0., z], [-z, 0.]])
        transmitted = fit[i] if i < 2 else row['force']
        constraints -= transmitted @ np.array([[0., z], [-z, 0.]]) + row['local']
        axial += row['axial']
        details = {}
        for label, values in (('radial_N', row['force']), ('local_Nm', row['local']), ('lever_Nm', lever)):
            selected = values[use]; coef = np.linalg.lstsq(design, selected, rcond=None)[0]
            details[label] = dict(mean=selected.mean(axis=0).tolist(),
                rms=np.sqrt(np.mean(selected**2, axis=0)).tolist(),
                peak=np.max(abs(selected), axis=0).tolist(),
                one_x_amplitude=np.linalg.norm(coef[1:], axis=0).tolist(),
                one_x_phase_rad=np.arctan2(-coef[2], coef[1]).tolist())
        details['axial_restoring_N'] = dict(mean=float(row['axial'][use].mean()),
            minimum=float(row['axial'][use].min()), maximum=float(row['axial'][use].max()))
        if i < 2:
            opened = use & (np.linalg.norm(fit[i], axis=1) == 0)
            details['local_moment_when_fit_open_peak_Nm'] = float(
                np.linalg.norm(row['local'][opened], axis=1).max()) if opened.any() else None
        archive[name+'_lever_restoring_Nm'] = lever
        archive[name+'_outer_rotation_constraint_Nm'] = -row['local']
        archive[name+'_outer_axial_constraint_N'] = -row['axial']
        table[name] = details
    archive['H_bearing_load_constraint_Nm'] = constraints
    archive['rotor_axial_constraint_N'] = axial
    np.savez_compressed(folder/'constraint_reactions.npz', **archive)
    result = dict(bearings=table,
        H_bearing_load_constraint_rms_Nm=np.sqrt(np.mean(constraints[use]**2, axis=0)).tolist(),
        H_bearing_load_constraint_peak_Nm=np.max(abs(constraints[use]), axis=0).tolist(),
        rotor_axial_constraint_peak_N=float(abs(axial[use]).max()),
        maximum_tilt_rad=float(np.linalg.norm(q[use, 8:10], axis=1).max()),
        convention='F/M are restoring quantities; actual rotor loads are -F/-M. Outer constraints are -M/-Fa. H torque is V8-16, only bearing-transmitted loads.',
        bearing_contact_fraction_definition='Mean fraction of balls with delta>0; geometric compression, not the fraction with positive force after damping clipping.',
        boundary='Outer rotation remains fixed even when radial fit is open; actual end-face restraint capacity is not validated.')
    write_json(folder/'wrench_diagnostics.json', result)
    return result
