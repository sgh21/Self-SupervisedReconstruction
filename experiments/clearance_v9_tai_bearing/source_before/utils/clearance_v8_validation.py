"""Production-code algebra and mechanics validation for the V8 spindle model.

This permanent audit calls the production bearing law, system assembly and
Numba backend directly. The small reference functions below independently
implement the documented fixed-normal equations; the plan's reference script
is not used as the production-code oracle.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import traceback

import numpy as np

from spindle import dynamics_model as model


L_SCALE = 0.03
ANGLE_DOF = np.array([8, 9])
ROT_DOF = np.array([0, 1, 8, 9])
U_OLD = 2.689135290615865e-5
PHASE_OLD = -0.3661687497264663
RNG_SEED = 20261025
J_SCALE = np.diag([1.0, 1.0, 1.0 / L_SCALE, 1.0 / L_SCALE])


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _relative_error(actual, expected, floor: float) -> float:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return float(np.max(np.abs(actual - expected) / np.maximum(floor, np.abs(expected))))


def _reference_law(t, relative_xy, velocity_xy, relative_tilt_xy,
                  relative_tilt_rate_xy, bp, orientation_sign, rho_m,
                  damping=True):
    """Independent implementation of V8-4 through V8-13."""
    alpha = float(bp['contact_angle'])
    ca, sa = np.cos(alpha), np.sin(alpha)
    balls = int(bp['balls'])
    omega = float(bp['rpm']) * (2.0 * np.pi / 60.0)
    cage = omega / 2.0 * (1.0 - float(bp['ball_diameter']) /
                          float(bp['pitch_diameter']) * ca)
    phi = (cage * np.asarray(t)[..., None] + float(bp['cage_phase'])
           + 2.0 * np.pi * np.arange(balls) / balls)
    co, si = np.cos(phi), np.sin(phi)
    h = int(orientation_sign) * float(rho_m) * sa
    b = np.stack((ca * co, ca * si, h * si, -h * co), axis=-1)
    xi = np.concatenate((np.asarray(relative_xy), np.asarray(relative_tilt_xy)), axis=-1)
    xid = np.concatenate((np.asarray(velocity_xy), np.asarray(relative_tilt_rate_xy)), axis=-1)
    delta = float(bp['normal_overlap']) + np.einsum('...ji,...i->...j', b, xi)
    vn = np.einsum('...ji,...i->...j', b, xid)
    compression = np.maximum(delta, 0.0)
    qe = float(bp['Keq']) * compression ** 1.5
    cball = float(bp['cb_total']) / (balls / 2.0 * ca ** 2) if damping else 0.0
    q = np.where(delta > 0.0, np.maximum(qe + cball * vn, 0.0), 0.0)
    wrench = np.einsum('...j,...ji->...i', q, b)
    potential = np.sum(0.4 * float(bp['Keq']) * compression ** 2.5, axis=-1)
    dissipation = np.sum((q - qe) * vn, axis=-1)
    axial = int(orientation_sign) * sa * np.sum(q, axis=-1)
    return dict(force_N=wrench[..., :2], moment_Nm=wrench[..., 2:],
                delta_m=delta, normal_force_N=q, elastic_force_N=qe,
                normal_velocity_m_s=vn, potential_energy_J=potential,
                dissipation_W=dissipation, axial_restoring_N=axial,
                projection=b)


def _clean_parameters(parameters, contact_model=None):
    p = copy.deepcopy(parameters)
    for key in tuple(p):
        if key.startswith('_'):
            p.pop(key)
    if contact_model is not None:
        p['bearing_contact_model'] = contact_model
    return p


def _same_value(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_value(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    try:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    except (TypeError, ValueError):
        return left == right


def _layout_by_name(p):
    return {item['name']: item for item in p['bearing_layout']}


def _bearing_parameters(name, p):
    return model.rear_bearing_parameters(p) if name.startswith('R') else p


def _local_inputs(name, t, q, v, p):
    item = _layout_by_name(p)[name]
    z = float(item['z_m'])
    rotor_q = np.asarray(q)[..., model.ROTOR]
    rotor_v = np.asarray(v)[..., model.ROTOR]
    relative = rotor_q @ model.section_matrix(z).T - np.asarray(q)[..., _outer_slice(item)]
    velocity = rotor_v @ model.section_matrix(z).T - np.asarray(v)[..., _outer_slice(item)]
    # All outer-ring tilt states are constrained to zero in this model.
    tilt = np.asarray(q)[..., 8:10]
    tilt_rate = np.asarray(v)[..., 8:10]
    bp = _bearing_parameters(name, p)
    return (float(t), relative, velocity, tilt, tilt_rate, bp,
            int(item['orientation_sign']), float(item['rho_m']))


def _outer_slice(item):
    source = item['outer_translation_source']
    return {'O1': slice(6, 8), 'O2': slice(10, 12), 'H': slice(4, 6)}[source]


def _call_wrench(name, t, relative_xy, velocity_xy, tilt_xy, tilt_rate_xy, p,
                 rho_override=None, sign_override=None, bp_override=None):
    item = _layout_by_name(p)[name]
    return model.bearing_wrench(
        t, np.asarray(relative_xy), np.asarray(velocity_xy),
        np.asarray(tilt_xy), np.asarray(tilt_rate_xy),
        _bearing_parameters(name, p) if bp_override is None else bp_override,
        int(item['orientation_sign']) if sign_override is None else int(sign_override),
        float(item['rho_m']) if rho_override is None else float(rho_override))


def _call_local(name, local, p, **overrides):
    t, relative, velocity, tilt, tilt_rate, _, _, _ = local
    return _call_wrench(name, t, relative, velocity, tilt, tilt_rate, p, **overrides)


def _unit_wrench(metrics, item, wrench, expected, force_floor=1.0, moment_floor=1e-3):
    force, moment = wrench[:2], wrench[2:]
    expected_force, expected_moment = expected[:2], expected[2:]
    metrics[item + '_force_relative_max'] = _relative_error(force, expected_force, force_floor)
    metrics[item + '_moment_relative_max'] = _relative_error(moment, expected_moment, moment_floor)


def _record(checks, name, function):
    try:
        details = function() or {}
        checks.append(dict(name=name, status='pass', passed=True, details=_jsonable(details)))
    except Exception as exc:  # Keep all independent findings in the result JSON.
        checks.append(dict(name=name, status='fail', passed=False,
                           error=f'{type(exc).__name__}: {exc}',
                           traceback=''.join(traceback.format_exception(exc))))


def _layout_config(base_config, layout):
    from utils.clearance_v8_parameters import layout_config
    return layout_config(base_config, layout)


def _random_case(rng, p, name, index):
    rel = rng.normal(size=2) * rng.uniform(0.5e-6, 5e-6)
    vel = rng.normal(size=2) * rng.uniform(0.001, 0.008)
    tilt = rng.normal(size=2) * rng.uniform(0.0, 0.5e-3)
    tilt_rate = rng.normal(size=2) * rng.uniform(0.0, 0.05)
    t = float(rng.uniform(0.0, 3.0))
    return t, rel, vel, tilt, tilt_rate


def _test_b01(p, rng):
    max_force, max_moment, cases = 0.0, 0.0, 0
    off_p = _clean_parameters(p, 'radial_only_v6')
    for name in ('F1', 'R1'):
        bp = _bearing_parameters(name, p)
        for i in range(40):
            t, rel, vel, tilt, tilt_rate = _random_case(rng, p, name, i)
            item = _layout_by_name(off_p)[name]
            z = float(item['z_m'])
            q, v = np.zeros(12), np.zeros(12)
            q[8:10], v[8:10] = tilt, tilt_rate
            q[:2] = rel - np.array([z*tilt[1], -z*tilt[0]])
            v[:2] = vel - np.array([z*tilt_rate[1], -z*tilt_rate[0]])
            actual = model.bearing_wrenches(t, q, v, off_p)[name]
            actual_f, actual_m = actual['force_N'], actual['moment_Nm']
            expected = model.bearing_force(t, rel, vel, bp)
            max_force = max(max_force, _relative_error(actual_f, expected, 1.0))
            max_moment = max(max_moment, float(np.max(np.abs(actual_m))))
            cases += 1
    _require(cases >= 30, 'B01 used fewer than 30 local states')
    _require(max_force <= 1e-10, f'B01 radial regression error {max_force:.3g}')
    _require(max_moment <= 1e-14,
             f'B01 radial-only switch left local moment {max_moment:.3g} N m')
    return dict(states=cases, switch='radial_only_v6', radial_relative_error=max_force,
                maximum_local_moment_Nm=max_moment)


def _test_b02(p):
    values = {}
    for name in ('F1', 'R1'):
        bp = _bearing_parameters(name, p)
        f, m, d = _call_wrench(name, 0.0, np.zeros(2), np.zeros(2),
                               np.zeros(2), np.zeros(2), p)
        expected = int(_layout_by_name(p)[name]['orientation_sign']) * float(bp['preload_reference_N'])
        axial = float(d['axial_restoring_N'])
        values[name] = dict(radial_force_N=np.asarray(f).tolist(),
                            moment_Nm=np.asarray(m).tolist(), axial_N=axial,
                            reference_axial_N=expected,
                            relative_error=abs(axial - expected) / abs(expected))
        _require(np.max(np.abs(f)) < 1e-10 and np.max(np.abs(m)) < 1e-12,
                 f'{name} zero-state force or moment is not near zero')
        _require(abs(axial - expected) / abs(expected) < 1e-9,
                 f'{name} axial preload differs from the frozen reference')
    return values


def _test_b03(p):
    details = {}
    for name in ('F1', 'R1'):
        bp = _bearing_parameters(name, p)
        rel = np.array([1.2e-7, -0.7e-7])
        t = 0.013
        f, m, _ = _call_wrench(name, t, rel, np.zeros(2), np.zeros(2),
                               np.zeros(2), p)
        ref = _reference_law(t, rel, np.zeros(2), np.zeros(2), np.zeros(2), bp,
                             _layout_by_name(p)[name]['orientation_sign'],
                             _layout_by_name(p)[name]['rho_m'], damping=False)
        err_f = _relative_error(f, ref['force_N'], 1.0)
        err_m = _relative_error(m, ref['moment_Nm'], 1e-3)
        details[name] = dict(force_relative_error=err_f,
                             moment_relative_error=err_m,
                             produced_local_moment_Nm=np.asarray(m).tolist())
        _require(err_f <= 1e-10 and err_m <= 1e-10,
                 f'{name} translation projection/sign differs from independent equation')
        _require(np.linalg.norm(m) > 1e-8,
                 f'{name} pure radial translation failed to produce the expected local moment')
    return details


def _test_b04(p):
    details = {}
    for name in ('F1', 'R1'):
        tilt = np.array([1e-8, 0.0])
        f, m, _ = _call_wrench(name, 0.0, np.zeros(2), np.zeros(2), tilt,
                               np.zeros(2), p)
        _require(np.linalg.norm(f) > 1e-3 and np.linalg.norm(m) > 1e-7,
                 f'{name} pure tilt did not produce coupled force and moment')
        details[name] = dict(force_N=np.asarray(f).tolist(), moment_Nm=np.asarray(m).tolist())
    return details


def _test_b05(p):
    # The production interface consumes relative tilt, so common inner/outer
    # rotation is removed before this function. Confirm the interface contract.
    import inspect
    params = inspect.signature(model.bearing_wrench).parameters
    _require('relative_tilt_xy' in params,
             'bearing_wrench must take relative_tilt_xy explicitly')
    common = np.array([0.11e-3, -0.07e-3])
    inner_a, outer_a = common + np.array([2e-6, -1e-6]), common
    inner_b, outer_b = np.array([2e-6, -1e-6]), np.zeros(2)
    relative_a, relative_b = inner_a - outer_a, inner_b - outer_b
    wa = _call_wrench('F1', 0.1, [1e-7, -2e-7], [0.0, 0.0], relative_a,
                      [0.0, 0.0], p)
    wb = _call_wrench('F1', 0.1, [1e-7, -2e-7], [0.0, 0.0], relative_b,
                      [0.0, 0.0], p)
    err = max(_relative_error(wa[0], wb[0], 1.0), _relative_error(wa[1], wb[1], 1e-3))
    _require(err <= 1e-12, 'Equal relative tilt produced different contact results')
    return dict(relative_tilt_a_rad=relative_a.tolist(), relative_tilt_b_rad=relative_b.tolist(),
                relative_wrench_error=err,
                note='Outer tilt is constrained to zero in the full system; common rotation cancels at the relative-angle interface.')


def _test_b06(p):
    details = {}
    for name in ('F1', 'R1'):
        item = _layout_by_name(p)[name]
        bp = _bearing_parameters(name, p)
        xi = np.zeros(4)
        # Central finite differences around the uniform loaded state.
        def tangent(sign):
            k = np.empty((4, 4))
            for j in range(4):
                step = np.zeros(4)
                step[j] = 1e-10
                xp = xi + step
                xm = xi - step
                wp = _call_wrench(name, 0.037, xp[:2], np.zeros(2), xp[2:],
                                  np.zeros(2), p, sign_override=sign)[0]
                wm = _call_wrench(name, 0.037, xm[:2], np.zeros(2), xm[2:],
                                  np.zeros(2), p, sign_override=sign)[0]
                mp = _call_wrench(name, 0.037, xp[:2], np.zeros(2), xp[2:],
                                  np.zeros(2), p, sign_override=sign)[1]
                mm = _call_wrench(name, 0.037, xm[:2], np.zeros(2), xm[2:],
                                  np.zeros(2), p, sign_override=sign)[1]
                k[:, j] = J_SCALE.T @ np.r_[wp, mp] / (2e-10) - J_SCALE.T @ np.r_[wm, mm] / (2e-10)
            return k
        k_plus, k_minus = tangent(1), tangent(-1)
        block = k_plus.copy()
        block[:2, 2:] *= -1
        block[2:, :2] *= -1
        err = float(np.max(np.abs(k_minus - block)) / np.max(np.abs(block)))
        _require(err <= 1e-6, f'{name} orientation sign did not only reverse translation-tilt coupling')
        details[name] = dict(positive_orientation_tangent_scaled=k_plus.tolist(),
                             negative_orientation_tangent_scaled=k_minus.tolist(),
                             reversal_relative_error=err,
                             rho_m=float(item['rho_m']), contact_angle_rad=float(bp['contact_angle']))
    return details


def _test_b07_b08(p):
    details = {}
    x = np.array([1.3e-7, -8e-8, 2.5e-7, -1.5e-7])
    xi = np.array([x[0], x[1], x[2] / L_SCALE, x[3] / L_SCALE])
    h_grad, h_tan = 1e-11, 2e-10
    for name in ('F1', 'R1'):
        bp = _bearing_parameters(name, p)
        item = _layout_by_name(p)[name]
        sign, rho = int(item['orientation_sign']), float(item['rho_m'])
        f, m, diag = _call_wrench(name, 0.021, xi[:2], np.zeros(2), xi[2:],
                                  np.zeros(2), p)
        w_scaled = J_SCALE.T @ np.r_[f, m]
        grad_fd, tangent_fd = np.empty(4), np.empty((4, 4))
        for j in range(4):
            step = np.zeros(4)
            step[j] = h_grad
            xp, xm = x + step, x - step
            xip = np.array([xp[0], xp[1], xp[2] / L_SCALE, xp[3] / L_SCALE])
            xim = np.array([xm[0], xm[1], xm[2] / L_SCALE, xm[3] / L_SCALE])
            fp, mp, dp = _call_wrench(name, 0.021, xip[:2], np.zeros(2), xip[2:],
                                      np.zeros(2), p)
            fm, mm, dm = _call_wrench(name, 0.021, xim[:2], np.zeros(2), xim[2:],
                                      np.zeros(2), p)
            grad_fd[j] = (float(dp['potential_energy_J']) - float(dm['potential_energy_J'])) / (2*h_grad)
            tangent_fd[:, j] = J_SCALE.T @ (np.r_[fp, mp] - np.r_[fm, mm]) / (2*h_grad)
            # A second, larger step reduces cancellation in the tangent check.
            st = np.zeros(4)
            st[j] = h_tan
            xtp, xtm = x + st, x - st
            xitp = np.array([xtp[0], xtp[1], xtp[2] / L_SCALE, xtp[3] / L_SCALE])
            xitm = np.array([xtm[0], xtm[1], xtm[2] / L_SCALE, xtm[3] / L_SCALE])
            ftp, mtp, _ = _call_wrench(name, 0.021, xitp[:2], np.zeros(2), xitp[2:], np.zeros(2), p)
            ftm, mtm, _ = _call_wrench(name, 0.021, xitm[:2], np.zeros(2), xitm[2:], np.zeros(2), p)
            tangent_fd[:, j] = J_SCALE.T @ (np.r_[ftp, mtp] - np.r_[ftm, mtm]) / (2*h_tan)
        gradient_error = float(np.linalg.norm(grad_fd - w_scaled) / max(1.0, np.linalg.norm(w_scaled)))
        analytic_ref = _reference_law(0.021, xi[:2], np.zeros(2), xi[2:], np.zeros(2),
                                      bp, sign, rho, damping=False)
        proj = analytic_ref['projection']
        delta = analytic_ref['delta_m']
        comp = np.maximum(delta, 0.0)
        k_phys = np.einsum('j,ji,jk->ik', 1.5*float(bp['Keq'])*np.sqrt(comp), proj, proj)
        k_scaled = J_SCALE.T @ k_phys @ J_SCALE
        tangent_error = float(np.linalg.norm(tangent_fd - k_scaled) / np.linalg.norm(k_scaled))
        sym_error = float(np.linalg.norm(tangent_fd - tangent_fd.T) / np.linalg.norm(tangent_fd))
        eig = np.linalg.eigvalsh((tangent_fd + tangent_fd.T) / 2)
        rank = int(np.linalg.matrix_rank(tangent_fd, tol=max(abs(eig[-1]), 1.0)*1e-10))
        _require(gradient_error <= 1e-5, f'{name} potential gradient error {gradient_error:.3g}')
        _require(tangent_error <= 1e-5, f'{name} elastic tangent error {tangent_error:.3g}')
        _require(sym_error <= 1e-8, f'{name} tangent symmetry error {sym_error:.3g}')
        _require(float(eig[0]) >= -max(float(eig[-1]), 1.0)*1e-10,
                 f'{name} tangent is not positive semidefinite')
        _require(rank == 2, f'{name} fixed-normal tangent rank was {rank}, expected 2')
        details[name] = dict(potential_gradient_relative_error=gradient_error,
                             tangent_relative_error=tangent_error,
                             tangent_symmetry_relative_error=sym_error,
                             scaled_tangent_eigenvalues_N_m=eig.tolist(),
                             tangent_rank=rank, production_energy_J=float(diag['potential_energy_J']))
    return details


def _test_b09(p, rng):
    max_error, cases = 0.0, 0
    for name in ('F1', 'R1'):
        item = _layout_by_name(p)[name]
        bp = _bearing_parameters(name, p)
        a = int(item['orientation_sign']) * float(item['rho_m']) * np.tan(float(bp['contact_angle']))
        for _ in range(48):
            rel = rng.normal(size=2) * 3e-6
            vel = rng.normal(size=2) * 0.003
            tilt = rng.normal(size=2) * 2e-4
            tilt_rate = rng.normal(size=2) * 0.02
            t, z = float(rng.uniform(0, 3)), float(rng.uniform(-0.15, 0.15))
            f, m, _ = _call_wrench(name, t, rel, vel, tilt, tilt_rate, p)
            rel_pc = np.array([rel[0] - a*tilt[1], rel[1] + a*tilt[0]])
            vel_pc = np.array([vel[0] - a*tilt_rate[1], vel[1] + a*tilt_rate[0]])
            fpc = model.bearing_force(t, rel_pc, vel_pc, bp)
            mpc = np.zeros(2)
            g1 = np.r_[f, np.array([-z*f[1] + m[0], z*f[0] + m[1]])]
            zpc = z - a
            g2 = np.r_[fpc, np.array([-zpc*fpc[1] + mpc[0], zpc*fpc[0] + mpc[1]])]
            err = _relative_error(g1, g2, np.array([1.0, 1.0, 1e-3, 1e-3]))
            max_error = max(max_error, err)
            cases += 1
    _require(cases >= 30 and max_error <= 1e-10,
             f'B09 pressure-center equivalence error {max_error:.3g} over {cases} states')
    return dict(states=cases, maximum_scaled_generalized_wrench_error=max_error)


def _pair_tilt_stiffness(p, arrangement):
    cfg = _layout_config({'parameters': p}, arrangement)
    pp = _clean_parameters(cfg['parameters'], 'relative_tilt_v8')
    layout = _layout_by_name(pp)
    details = {}
    for names, expected_char in ((('F1', 'F2'), arrangement[:2]),
                                 (('R1', 'R2'), arrangement[3:5])):
        first, second = (layout[n] for n in names)
        bp = _bearing_parameters(names[0], pp)
        alpha, rho = float(bp['contact_angle']), float(first['rho_m'])
        kr = 3.0*int(bp['balls'])/4.0*float(bp['Keq'])*np.sqrt(float(bp['normal_overlap']))*np.cos(alpha)**2
        z_mid = (float(first['z_m']) + float(second['z_m']))/2
        span = abs(float(first['z_m']) - float(second['z_m']))
        actual = 0.0
        step = 1e-9
        for item in (first, second):
            z = float(item['z_m'])
            s = int(item['orientation_sign'])
            def generalized(theta):
                # A rigid rotor tilt about the group's midpoint.
                rel = np.array([(z-z_mid)*theta, 0.0])
                f, m, _ = _call_wrench(item['name'], 0.0, rel, np.zeros(2),
                                       np.array([0.0, theta]), np.zeros(2), pp,
                                       sign_override=s)
                return (z-z_mid)*f[0] + m[1]
            actual += (generalized(step) - generalized(-step)) / (2*step)
        db_pair = expected_char == 'DB'
        sign_scale = 1.0 if db_pair else -1.0
        target = 2.0*kr*(span/2.0 + sign_scale*rho*np.tan(alpha))**2
        details[''.join(names)] = dict(arrangement=expected_char, span_m=span,
                                       kr_N_m=kr, actual_Nm_rad=actual,
                                       analytic_Nm_rad=target,
                                       relative_error=abs(actual-target)/target)
        _require(abs(actual-target)/target <= 1e-5,
                 f'{names} production pair tangent mismatch: actual={actual:.9g}, target={target:.9g}, relative={abs(actual-target)/target:.3g}')
    return details


def _test_b10(p):
    return {layout: _pair_tilt_stiffness(p, layout)
            for layout in ('DB_DB', 'DF_DB', 'DB_DF', 'DF_DF')}


def _test_b11(p, rng):
    min_diss, min_force, opened, clipped, cases = np.inf, np.inf, 0, 0, 0
    for name in ('F1', 'R1'):
        bp = _bearing_parameters(name, p)
        item = _layout_by_name(p)[name]
        for i in range(96):
            if i % 4 == 0:
                rel = rng.normal(size=2) * 1e-3
                tilt = rng.normal(size=2) * 2e-4
                vel = rng.normal(size=2) * 0.006
                tilt_rate = rng.normal(size=2) * 0.04
            elif i % 4 == 1:
                rel = np.zeros(2)
                tilt = np.zeros(2)
                vel = rng.normal(size=2) * 1.0
                tilt_rate = rng.normal(size=2) * 0.8
            else:
                rel = rng.normal(size=2) * 3e-6
                tilt = rng.normal(size=2) * 3e-4
                vel = rng.normal(size=2) * 0.006
                tilt_rate = rng.normal(size=2) * 0.04
            t = float(rng.uniform(0, 4))
            f, m, d = _call_wrench(name, t, rel, vel, tilt, tilt_rate, p)
            normal = np.asarray(d['normal_force_N'])
            delta = np.asarray(d['delta_m'])
            diss = float(np.sum((normal - np.asarray(d['elastic_force_N'])) *
                                np.asarray(d['normal_velocity_m_s'])))
            min_diss = min(min_diss, diss)
            min_force = min(min_force, float(normal.min()))
            opened += int(np.any(delta <= 0.0))
            clipped += int(np.any((delta > 0.0) &
                                  (normal == 0.0) & (np.asarray(d['elastic_force_N']) > 0.0)))
            _require(float(d['dissipation_W']) >= -1e-12,
                     f'{name} dissipative clipping produced negative power {d["dissipation_W"]:.3g} W')
            _require(np.all(normal >= 0.0) and np.all(normal[delta <= 0.0] == 0.0),
                     f'{name} produced tensile ball force')
            cases += 1
    _require(cases >= 30 and opened > 0 and clipped > 0,
             'B11 did not exercise enough states, contact opening and damping clipping')
    return dict(states=cases, opened_state_count=opened, damping_clipped_count=clipped,
                minimum_dissipation_W=min_diss, minimum_ball_force_N=min_force)


def _test_b12(p):
    p = _clean_parameters(p, 'relative_tilt_v8')
    # Positive inner/outer race contact with all outer-to-H fits detached.
    q = np.zeros(12)
    v = np.zeros(12)
    q[0], q[8], q[9] = 5e-6, 1.5e-4, -0.8e-4
    fc = model.force_components(0.019, q, v, p)
    fw = fc['bearing_wrenches']
    fit_open = max(float(np.linalg.norm(fc['fit_N'])), float(np.linalg.norm(fc['fit2_N']))) == 0.0
    moments = {name: np.asarray(fw[name]['moment_Nm']) for name in ('F1', 'F2')}
    _require(fit_open, 'Outer-to-H fits were not detached in B12 setup')
    _require(all(np.linalg.norm(value) > 1e-8 for value in moments.values()),
             'Radial fit disengagement erased a bearing local moment')
    reactions = fc['outer_rotation_constraint_moment_Nm_by_bearing']
    reaction_errors = {}
    for name, value in moments.items():
        reaction_errors[name] = _relative_error(reactions[name], -value, 1e-3)
        _require(reaction_errors[name] <= 1e-12,
                 f'{name} fixed-tilt reaction does not oppose local bearing moment')
    # A centered radial translation opens some balls and compresses others.
    # Verify each open ball carries no normal force and contributes no energy.
    opened = {}
    for name in ('F1', 'R1'):
        bp = _bearing_parameters(name, p)
        rel = np.array([-2e-3, 0.0])
        f, m, d = _call_wrench(name, 0.0, rel, np.zeros(2), np.zeros(2),
                               np.zeros(2), p)
        delta = np.asarray(d['delta_m'])
        normal = np.asarray(d['normal_force_N'])
        closed = delta <= 0.0
        expected_energy = float(np.sum(0.4*float(bp['Keq'])*np.maximum(delta, 0.0)**2.5))
        _require(np.any(closed), f'{name} B12 did not exercise an open ball')
        _require(np.all(normal[closed] == 0.0), f'{name} open ball carried normal force')
        _require(np.all(np.asarray(d['elastic_force_N'])[closed] == 0.0),
                 f'{name} open ball carried elastic load')
        energy_error = abs(float(d['potential_energy_J'])-expected_energy)
        _require(energy_error <= 1e-14,
                 f'{name} open ball contributed to potential energy')
        opened[name] = dict(open_ball_count=int(np.sum(closed)),
                            potential_energy_J=float(d['potential_energy_J']),
                            expected_active_ball_energy_J=expected_energy,
                            potential_energy_difference_J=energy_error,
                            radial_force_N=np.asarray(f).tolist(),
                            local_moment_Nm=np.asarray(m).tolist())
    return dict(fit_detached=fit_open,
                local_moment_Nm={k: v.tolist() for k, v in moments.items()},
                outer_constraint_reaction_relative_error=reaction_errors,
                fully_opened_contacts=opened)


def _test_b13(p):
    details = {}
    for name in ('F1', 'R1'):
        bp = copy.deepcopy(_bearing_parameters(name, p))
        rel, vel = np.array([2e-7, -1e-7]), np.array([0.001, -0.002])
        tilt, rate = np.array([2e-4, -1e-4]), np.array([0.02, -0.03])
        fn, mn, _ = _call_wrench(name, 0.023, rel, vel, tilt, rate, p,
                                 bp_override=dict(bp, contact_angle=0.0))
        fbase, _, _ = _call_wrench(name, 0.023, rel, vel, np.zeros(2),
                                   np.zeros(2), p,
                                   bp_override=dict(bp, contact_angle=0.0))
        # At alpha=0 the projection b has no tilt components; rho=0 also removes
        # moments. Both diagnostics are intentional model limits, not hardware.
        alpha_error = max(_relative_error(fn, fbase, 1.0),
                          _relative_error(mn, np.zeros(2), 1e-12))
        _require(alpha_error <= 1e-12,
                 f'{name} alpha=0 diagnostic retained a tilt contribution')
        details[name] = dict(alpha_zero_force_error=alpha_error,
                             diagnostic_only=True)
    return details


def _test_b14(p, rng, unbalance):
    from spindle.fast_rhs import prepare_fast_rhs
    p = _clean_parameters(p, 'relative_tilt_v8')
    fast = prepare_fast_rhs(p, unbalance)
    _require(hasattr(fast, 'internal_force'),
             'Numba RHS must expose .internal_force(t,q,v) for direct backend comparison')
    max_nf = max_nm = 0.0
    cases = 128
    for i in range(cases):
        q = rng.normal(size=12) * np.array([3e-6]*8 + [2e-4, 2e-4] + [3e-6]*2)
        v = rng.normal(size=12) * np.array([0.006]*8 + [0.04, 0.04] + [0.006]*2)
        if i < 8:
            q[:] = 0.0
            q[0] = (i - 4)*float(p['normal_overlap'])/np.cos(float(p['contact_angle']))
        t = float(rng.uniform(0.0, 4.0))
        py = model.internal_force(t, q, v, p, 0.0)
        nb = fast.internal_force(t, q, v)
        force_idx = np.r_[0:8, 10:12]
        err_n = float(np.max(np.abs(py[force_idx]-nb[force_idx]) /
                             np.maximum(1.0, np.abs(py[force_idx]))))
        err_nm = float(np.max(np.abs(py[8:10]-nb[8:10]) /
                              np.maximum(1e-3, np.abs(py[8:10]))))
        max_nf, max_nm = max(max_nf, err_n), max(max_nm, err_nm)
    _require(cases >= 100, 'B14 used fewer than 100 mixed states')
    _require(max_nf <= 1e-10 and max_nm <= 1e-10,
             f'Python/Numba normalized differences: N={max_nf:.3g}, Nm={max_nm:.3g}')
    return dict(mixed_states=cases, maximum_normalized_force_error=max_nf,
                maximum_normalized_moment_error=max_nm,
                normalization='N uses max(1 N, reference); N m uses max(1e-3 N m, reference)')


def _test_b15(p, rng):
    p = _clean_parameters(p, 'relative_tilt_v8')
    n = 36
    q = rng.normal(size=(n, 12)) * np.array([3e-6]*8 + [2e-4, 2e-4] + [3e-6]*2)
    v = rng.normal(size=(n, 12)) * np.array([0.006]*8 + [0.04, 0.04] + [0.006]*2)
    times = rng.uniform(0.0, 3.0, n)
    batch = model.bearing_wrenches(times, q, v, p)
    max_force = max_moment = max_delta = 0.0
    for name in ('F1', 'F2', 'R1', 'R2'):
        got = batch[name]
        _require(np.shape(got['force_N']) == (n, 2) and
                 np.shape(got['moment_Nm']) == (n, 2),
                 f'{name} batch force or moment shape is invalid')
        for i in range(n):
            local = _local_inputs(name, times[i], q[i], v[i], p)
            f, m, d = _call_wrench(name, times[i], local[1], local[2],
                                   local[3], local[4], p)
            max_force = max(max_force, float(np.max(np.abs(np.asarray(got['force_N'])[i]-f))))
            max_moment = max(max_moment, float(np.max(np.abs(np.asarray(got['moment_Nm'])[i]-m))))
            max_delta = max(max_delta, float(np.max(np.abs(np.asarray(got['diagnostics']['delta_m'])[i]-d['delta_m']))))
    _require(max_force <= 1e-11 and max_moment <= 1e-13 and max_delta <= 1e-15,
             f'batch/single difference: F={max_force:.3g}, M={max_moment:.3g}, delta={max_delta:.3g}')
    return dict(batch_states=n, bearings=['F1', 'F2', 'R1', 'R2'],
                max_force_difference_N=max_force, max_moment_difference_Nm=max_moment,
                max_compression_difference_m=max_delta,
                time_array_shape=np.shape(times), q_shape=np.shape(q))


def _test_system_virtual_work(p, rng):
    p = _clean_parameters(p, 'relative_tilt_v8')
    n = 48
    q = rng.normal(size=(n, 12)) * np.array([4e-6]*8 + [3e-4, 3e-4] + [4e-6]*2)
    v = rng.normal(size=(n, 12)) * np.array([0.01]*8 + [0.05, 0.05] + [0.01]*2)
    dq = rng.normal(size=(n, 12)) * np.array([2e-7]*8 + [2e-5, 2e-5] + [2e-7]*2)
    batch = model.bearing_wrenches(rng.uniform(0, 3, n), q, v, p)
    max_abs, max_rel = 0.0, 0.0
    for i in range(n):
        local_power = 0.0
        assembled = np.zeros(12)
        for name in ('F1', 'F2', 'R1', 'R2'):
            item = _layout_by_name(p)[name]
            z = float(item['z_m'])
            a = model.section_matrix(z)
            outer = _outer_slice(item)
            relative_virtual = a @ dq[i, model.ROTOR] - dq[i, outer]
            tilt_virtual = dq[i, 8:10]
            f = np.asarray(batch[name]['force_N'][i])
            m = np.asarray(batch[name]['moment_Nm'][i])
            local_power += f @ relative_virtual + m @ tilt_virtual
            assembled[model.ROTOR] += a.T @ f + np.r_[0.0, 0.0, m]
            assembled[outer] -= f
        assembled_power = float(assembled @ dq[i])
        err = abs(assembled_power-local_power)
        max_abs = max(max_abs, err)
        max_rel = max(max_rel, err/max(1e-12, abs(local_power)))
    _require(max_rel <= 1e-10 and max_abs <= 1e-12,
             f'whole-system contact virtual-work error {max_rel:.3g} relative')
    return dict(random_virtual_displacements=n,
                maximum_absolute_power_difference_W=max_abs,
                maximum_relative_power_difference=max_rel)


def _test_system_assembly(p, rng):
    p = _clean_parameters(p, 'relative_tilt_v8')
    q = rng.normal(size=12) * np.array([4e-6]*8 + [3e-4, 3e-4] + [4e-6]*2)
    v = rng.normal(size=12) * np.array([0.008]*8 + [0.04, 0.04] + [0.008]*2)
    t = 0.183
    fc = model.force_components(t, q, v, p)
    wrenches = fc['bearing_wrenches']
    expected = np.zeros(12)
    for name in ('F1', 'F2', 'R1', 'R2'):
        item = _layout_by_name(p)[name]
        a = model.section_matrix(float(item['z_m']))
        f = np.asarray(wrenches[name]['force_N'])
        m = np.asarray(wrenches[name]['moment_Nm'])
        expected[ROT_DOF] += a.T @ f + np.r_[0.0, 0.0, m]
        expected[_outer_slice(item)] -= f
    link = np.asarray(fc['link_N'])
    expected[ROT_DOF] += model.section_matrix(p['z_T']).T @ link
    expected[2:4] -= link
    expected[6:8] += np.asarray(fc['fit_N'])
    expected[10:12] += np.asarray(fc['fit2_N'])
    expected[4:6] -= np.asarray(fc['fit_N']) + np.asarray(fc['fit2_N'])
    expected[4:6] += p['kh']*q[4:6] + p['ch']*v[4:6]
    gyro = model.gyro_matrix(p) @ v
    expected += gyro
    actual = model.internal_force(t, q, v, p, 0.0)
    force_idx = np.r_[0:8, 10:12]
    force_error = float(np.max(np.abs(actual[force_idx]-expected[force_idx])))
    moment_error = float(np.max(np.abs(actual[8:10]-expected[8:10])))
    _require(force_error <= 1e-9 and moment_error <= 1e-10,
             f'assembled restoring vector differs: {force_error:.3g} N, {moment_error:.3g} Nm')
    # Constraint torque ledger from V8-16; force and local moment terms remain separate.
    torque_sum = np.zeros(2)
    for name in ('F1', 'F2'):
        item = _layout_by_name(p)[name]
        z = float(item['z_m'])
        fit = np.asarray(fc['fit_N'] if name == 'F1' else fc['fit2_N'])
        local_m = np.asarray(wrenches[name]['moment_Nm'])
        torque_sum += np.array([-z*fit[1], z*fit[0]]) + local_m
    for name in ('R1', 'R2'):
        item = _layout_by_name(p)[name]
        z = float(item['z_m'])
        f = np.asarray(wrenches[name]['force_N'])
        local_m = np.asarray(wrenches[name]['moment_Nm'])
        torque_sum += np.array([-z*f[1], z*f[0]]) + local_m
    expected_constraint = -torque_sum
    constraint = np.asarray(fc['housing_constraint_torque_Nm'])
    constraint_error = float(np.max(np.abs(constraint-expected_constraint)))
    _require(constraint_error <= 1e-10,
             f'housing constraint torque ledger error {constraint_error:.3g} N m')
    outer_reaction = fc['outer_rotation_constraint_moment_Nm_by_bearing']
    per_bearing_error = max(_relative_error(outer_reaction[name],
                              -np.asarray(wrenches[name]['moment_Nm']), 1e-3)
                            for name in ('F1', 'F2', 'R1', 'R2'))
    _require(per_bearing_error <= 1e-12,
             f'fixed outer-ring rotation reaction mismatch {per_bearing_error:.3g}')
    axial = fc['axial_constraint_reaction_N_by_bearing']
    axial_error = max(_relative_error(axial[name],
                        -float(wrenches[name]['diagnostics']['axial_restoring_N']), 1.0)
                      for name in ('F1', 'F2', 'R1', 'R2'))
    _require(axial_error <= 1e-12,
             f'fixed axial reaction mismatch {axial_error:.3g}')
    masses, gravity = model.mass_and_gravity(p)
    _require(np.shape(masses) == (12,) and np.shape(gravity) == (12,),
             'V8 mass/gravity vectors do not retain 12 DOF')
    _require(np.all(masses[:8] > 0.0) and np.all(masses[10:12] > 0.0) and
             np.allclose(masses[8:10], p['Jt']),
             'mass/inertia entries do not match the 12-DOF state contract')
    _require(np.shape(actual) == (12,) and np.all(np.isfinite(actual)),
             'internal_force must return a finite 12-vector')
    return dict(force_balance_error_N=force_error, moment_balance_error_Nm=moment_error,
                housing_constraint_torque_expected_Nm=expected_constraint.tolist(),
                housing_constraint_torque_actual_Nm=constraint.tolist(),
                housing_constraint_torque_error_Nm=constraint_error,
                outer_rotation_reaction_relative_error=per_bearing_error,
                axial_reaction_relative_error=axial_error,
                state_dof=len(actual), mass_vector_shape=list(np.shape(masses)),
                first_eight_mass_kg=masses[:8].tolist(), rotor_inertia_entries_kg_m2=masses[8:10].tolist())


def _test_gyro_and_equilibrium(p, rng):
    p = _clean_parameters(p, 'relative_tilt_v8')
    g = model.gyro_matrix(p)
    q = rng.normal(size=12) * np.array([3e-6]*8 + [2e-4, 2e-4] + [3e-6]*2)
    v = rng.normal(size=12) * np.array([0.005]*8 + [0.03, 0.03] + [0.005]*2)
    force = model.internal_force(0.071, q, v, p, 0.0)
    p_no_gyro = copy.deepcopy(p)
    p_no_gyro['Jp'] = 0.0
    no_gyro = model.internal_force(0.071, q, v, p_no_gyro, 0.0)
    gyro_err = float(np.max(np.abs(force-no_gyro-g@v)))
    _require(np.max(np.abs(g+g.T)) == 0.0 and abs(float(v@g@v)) <= 1e-12,
             'gyroscopic matrix is not skew or performs nonzero power')
    _require(gyro_err <= 1e-10,
             f'gyro contribution is missing or assembled more than once: {gyro_err:.3g}')
    qeq, equivalent_residual = model.equilibrium_continuation(p)
    masses, gravity = model.mass_and_gravity(p)
    residual = model.internal_force(0.0, qeq, np.zeros(12), p, 0.0)-gravity
    force_resid = float(np.max(np.abs(residual[np.r_[0:8, 10:12]])))
    moment_resid = float(np.max(np.abs(residual[8:10])))
    _require(force_resid <= 1e-6 and moment_resid <= 1e-7,
             f'static residual exceeds limit: {force_resid:.3g} N, {moment_resid:.3g} Nm')
    return dict(gyro_skew_max=float(np.max(np.abs(g+g.T))),
                gyro_power_W=float(v@g@v), gyro_single_assembly_error_N_or_Nm=gyro_err,
                equilibrium_max_force_residual_N=force_resid,
                equilibrium_max_moment_residual_Nm=moment_resid,
                equilibrium_scaled_residual=float(equivalent_residual),
                equilibrium_q_m_rad=qeq.tolist(), masses_kg_kgm2=masses.tolist())


def _load_snapshot(path):
    file = Path(path) / 'src' / 'spindle' / 'dynamics_model.py'
    spec = importlib.util.spec_from_file_location('clearance_v8_v6_snapshot', file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import V6 snapshot at {file}')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    return previous


def _test_v6_switch_and_snapshot(p, snapshot_root, config_root):
    previous = _load_snapshot(snapshot_root)
    base_path = Path(snapshot_root) / 'config' / 'clearance_v6_config.json'
    old_cfg = json.loads(base_path.read_text(encoding='utf-8'))
    old_p0 = copy.deepcopy(old_cfg['parameters'])
    old_p0['rhs_backend'] = 'python'
    off_p = _clean_parameters(p, 'radial_only_v6')
    off_p['rhs_backend'] = 'python'
    # The experiment mode is the only new switch; compare the old physical keys.
    metadata = {'model_status', 'calibration_status', 'bearing_contact_model',
                'bearing_layout', 'bearing_layout_name', 'bearing_layout_status',
                'bearing_layout_basis', 'bearing_position_basis',
                'front_pair_arrangement_assumption', 'rear_pair_arrangement_assumption'}
    common = [key for key in old_p0 if key in off_p and key not in metadata and not key.startswith('_')]
    changed = [key for key in common if not _same_value(old_p0[key], off_p[key])]
    _require(not changed, f'V8 config altered frozen V6 parameters: {changed}')
    rng = np.random.default_rng(RNG_SEED + 108)
    max_force, max_moment, cases = 0.0, 0.0, 0
    for i in range(64):
        q = rng.normal(size=12) * np.array([3e-6]*8 + [3e-4, 3e-4] + [3e-6]*2)
        v = rng.normal(size=12) * np.array([0.006]*8 + [0.05, 0.05] + [0.006]*2)
        t = float(rng.uniform(0.0, 2.0))
        expected = previous.internal_force(t, q, v, old_p0, 0.0)
        actual = model.internal_force(t, q, v, off_p, 0.0)
        force_idx = np.r_[0:8, 10:12]
        max_force = max(max_force, _relative_error(actual[force_idx], expected[force_idx], 1.0))
        max_moment = max(max_moment, _relative_error(actual[8:10], expected[8:10], 1e-3))
        cases += 1
    _require(cases >= 30 and max_force <= 1e-10 and max_moment <= 1e-10,
             f'radial_only_v6 snapshot regression failed: force={max_force:.3g}, moment={max_moment:.3g}')
    trajectory = {}
    for gap in (2.0, 10.0):
        old_p = previous.front_group_parameters(old_p0, gap)
        new_p = _clean_parameters(model.front_group_parameters(off_p, gap), 'radial_only_v6')
        kwargs = dict(lost_area=0.0, frequencies=[], coefficients=[], duration=0.003,
                      warmup=0.0, fs=8192, max_step=1.1260832739219432e-5,
                      rtol=2e-8, atol=np.r_[np.full(12, 1e-12), np.full(12, 1e-8)],
                      unbalance=(U_OLD, PHASE_OLD))
        old_run = previous.simulate(old_p, **kwargs)
        new_run = model.simulate(new_p, **kwargs)
        diffs = {key: float(np.max(np.abs(new_run[key]-old_run[key])))
                 for key in ('q', 'v', 'acc')}
        trajectory[str(gap)] = dict(max_difference=diffs,
                                    samples=int(len(old_run['t'])),
                                    duration_s=float(old_run['t'][-1]))
        _require(diffs['q'] <= 1e-10 and diffs['v'] <= 1e-8 and diffs['acc'] <= 1e-4,
                 f'legacy short trajectory mismatch at c={gap:g}: {diffs}')
    return dict(frozen_common_parameters=common, changed_common_parameters=changed,
                random_state_count=cases, maximum_normalized_force_error=max_force,
                maximum_normalized_moment_error=max_moment,
                random_force_backends=dict(snapshot='python; source_before internal_force called directly',
                                           updated='python; current internal_force called directly'),
                short_trajectory_by_clearance=trajectory,
                short_trajectory_backends=dict(snapshot='python', updated='python'),
                snapshot_uses_updated_fast_rhs=False,
                trajectory_scope='3 ms short regression at health and 10 um; not a steady-state run')


def run_validation(config_path, output_path, snapshot_root):
    rng = np.random.default_rng(RNG_SEED)
    config_path, output_path = Path(config_path), Path(output_path)
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    base_p = copy.deepcopy(cfg['parameters'])
    p = _clean_parameters(base_p, 'relative_tilt_v8')
    checks = []
    with tempfile.TemporaryDirectory(prefix='clearance_v8_validation_numba_') as cache:
        os.environ['NUMBA_CACHE_DIR'] = cache
        _record(checks, 'B01_radial_only_v6_regression', lambda: _test_b01(p, rng))
        _record(checks, 'B02_zero_state_preload', lambda: _test_b02(p))
        _record(checks, 'B03_pure_translation', lambda: _test_b03(p))
        _record(checks, 'B04_pure_tilt', lambda: _test_b04(p))
        _record(checks, 'B05_relative_tilt_common_rotation', lambda: _test_b05(p))
        _record(checks, 'B06_orientation_sign_reversal', lambda: _test_b06(p))
        _record(checks, 'B07_B08_potential_gradient_tangent', lambda: _test_b07_b08(p))
        _record(checks, 'B09_pressure_center_equivalence', lambda: _test_b09(p, rng))
        _record(checks, 'B10_DB_DF_pair_stiffness', lambda: _test_b10(p))
        _record(checks, 'B11_damping_dissipation_and_clipping', lambda: _test_b11(p, rng))
        _record(checks, 'B12_open_contact_and_fit_detachment', lambda: _test_b12(p))
        _record(checks, 'B13_zero_angle_or_radius_diagnostics', lambda: _test_b13(p))
        _record(checks, 'B14_python_numba_mixed_states',
                lambda: _test_b14(p, rng, (U_OLD, PHASE_OLD)))
        _record(checks, 'B15_batch_single_and_broadcast', lambda: _test_b15(p, rng))
        _record(checks, 'system_virtual_work_assembly', lambda: _test_system_virtual_work(p, rng))
        _record(checks, 'system_force_moment_reaction_assembly', lambda: _test_system_assembly(p, rng))
        _record(checks, 'system_gyro_static_equilibrium', lambda: _test_gyro_and_equilibrium(p, rng))
        _record(checks, 'radial_only_v6_snapshot_regression',
                lambda: _test_v6_switch_and_snapshot(p, snapshot_root, config_path.parent))
    passed = all(item['passed'] for item in checks)
    result = dict(
        schema='clearance_v8_production_mechanics_validation_v1',
        scope='production algebra and mechanics P1 only; no long experiments or calibration',
        config=str(config_path), source_before=str(Path(snapshot_root)),
        random_seed=RNG_SEED, production_contact_model=p.get('bearing_contact_model'),
        state_contract=dict(model_dof=p.get('model_dof'), q_dof=12, integrated_state_size=24,
                            mass_gravity_shape=list(np.shape(model.mass_and_gravity(p)[0]))),
        status='pass' if passed else 'fail', all_passed=passed,
        passed_count=sum(x['passed'] for x in checks), check_count=len(checks), checks=checks,
        boundaries=[
            'The local fixed-normal bearing law is checked independently against V8 equations.',
            'Static residual is a frozen-cage initial condition, not full rotating equilibrium.',
            'Outer-ring tilt and axial motion are constrained; computed reactions do not validate real clamp capacity.',
            'DB/DF cases are geometry sensitivity tests, not confirmation of the installed machine arrangement.',
            'The legacy short trajectories are 3 ms regression checks, not steady-state response validation.'
        ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/clearance_v8_config.json'))
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/clearance_v8_tilt_contact/validation/production_mechanics.json'))
    parser.add_argument('--source-before', type=Path,
                        default=Path('experiments/clearance_v8_tilt_contact/source_before'))
    args = parser.parse_args()
    result = run_validation(args.config, args.output, args.source_before)
    summary = {k: result[k] for k in ('status', 'passed_count', 'check_count', 'config', 'source_before')}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
