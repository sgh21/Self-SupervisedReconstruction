"""Optional compiled evaluation of the existing healthy 12-DOF equations.

Only the numerical evaluator changes. The original Python force routines remain
the reference; no fast-math, constitutive interpolation or force rescaling is used.
Numba is imported only when this backend is explicitly selected.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def internal_force_12(t, q, v, stiffness, damping, directions, constants):
    """Linear links, explicit bearing rows, and two unilateral fit sums."""
    force = stiffness @ q + damping @ v
    fit_k1, fit_d1, gap1, fit_k2, fit_d2, gap2 = constants[:6]
    bearing_stride = 13 if (len(constants)-6) % 13 == 0 else 9
    for start in range(6, len(constants), bearing_stride):
        if bearing_stride == 13:
            (z_f, endpoint, overlap, cage_speed, cage_phase, ca, sa,
             keq, cball, balls, sign, rho, contact_model) = constants[start:start+13]
        else:
            z_f, endpoint, overlap, cage_speed, cage_phase, ca, keq, cball, balls = constants[start:start+9]
            sa, sign, rho, contact_model = 0., 0., 0., 0.
        outer = int(endpoint)
        rx = q[0] + z_f*q[9] - q[outer]
        ry = q[1] - z_f*q[8] - q[outer+1]
        vx = v[0] + z_f*v[9] - v[outer]
        vy = v[1] - z_f*v[8] - v[outer+1]
        bx, by, mx, my = 0., 0., 0., 0.
        tilt_scale = sign*rho*sa if contact_model > 0.5 else 0.
        for ball in range(int(balls)):
            angle = cage_speed*t + cage_phase + ball*(2*np.pi/balls)
            nx, ny = np.cos(angle), np.sin(angle)
            tilt = tilt_scale*(q[8]*ny-q[9]*nx)
            delta = overlap + ca*(nx*rx + ny*ry) + tilt
            if delta > 0:
                normal_velocity = ca*(nx*vx+ny*vy)+tilt_scale*(v[8]*ny-v[9]*nx)
                normal = max(0., keq*delta**1.5 + cball*normal_velocity)
                bx += normal*nx
                by += normal*ny
                mx += normal*tilt_scale*ny
                my -= normal*tilt_scale*nx
        bx *= ca
        by *= ca
        force[0] += bx
        force[1] += by
        force[8] -= z_f*by
        force[9] += z_f*bx
        force[8] += mx
        force[9] += my
        force[outer] -= bx
        force[outer+1] -= by
    for outer, k, d, gap in ((6, fit_k1, fit_d1, gap1),
                             (10, fit_k2, fit_d2, gap2)):
        rx, ry = q[outer]-q[4], q[outer+1]-q[5]
        vx, vy = v[outer]-v[4], v[outer+1]-v[5]
        fx, fy = 0., 0.
        for i in range(len(directions)):
            nx, ny = directions[i, 0], directions[i, 1]
            delta = rx*nx + ry*ny - gap
            if delta > 0:
                normal = max(0., k*delta + d*(vx*nx + vy*ny))
                fx += normal*nx
                fy += normal*ny
        force[outer] += fx
        force[outer+1] += fy
        force[4] -= fx
        force[5] -= fy
    return force


@njit(cache=True)
def rhs_12(t, state, stiffness, damping, directions, constants,
           masses, gravity, drive):
    restoring = internal_force_12(t, state[:12], state[12:], stiffness,
                                  damping, directions, constants)
    amplitude, omega, phase, z_u = drive
    angle = omega*t + phase
    fx, fy = amplitude*np.cos(angle), amplitude*np.sin(angle)
    restoring[0] -= fx
    restoring[1] -= fy
    restoring[8] += z_u*fy
    restoring[9] -= z_u*fx
    result = np.empty(24)
    result[:12] = state[12:]
    result[12:] = (gravity-restoring)/masses
    return result


def prepare_fast_rhs(p, unbalance):
    """Freeze constants from the same parameter dictionary as Python simulate."""
    from .dynamics_model import (section_matrix, ROTOR, area_and_link,
                                 gyro_matrix, cage_speed, mass_and_gravity,
                                 bearing_keq, rear_bearing_parameters,
                                 bearing_contact_model, _bearing_context)
    if p.get('model_dof') != 12 or unbalance is None:
        raise ValueError('numba backend requires the 12-DOF analytic-unbalance branch')
    if any(p.get(key, 0.) != 0 for key in
           ('outer_pit_area_m2', 'inner_pit_area_m2', 'ball_deformation_amplitude_m')):
        raise ValueError('numba backend covers healthy bearing contacts and fit clearances only')
    stiffness, damping = np.zeros((12, 12)), np.zeros((12, 12))

    def pair(z, endpoint, k, c):
        difference = np.zeros((2, 12))
        difference[:, ROTOR] = section_matrix(z)
        difference[:, endpoint:endpoint+2] -= np.eye(2)
        stiffness[:] += difference.T @ k @ difference
        damping[:] += difference.T @ c @ difference

    pair(p['z_T'], 2, area_and_link(p, 0.)[1]*np.eye(2), p['cl']*np.eye(2))
    nonlinear = p.get('nonlinear_bearings_v6', False)
    if not nonlinear:
        for support in p['healthy_supports']:
            pair(support['z_m'], 4, np.asarray(support['K']), np.asarray(support['C']))
        pair(p['z_f2'], 10, np.asarray(p['Kb2']), np.asarray(p['Cb2']))
    stiffness[4:6, 4:6] += p['kh']*np.eye(2)
    damping[4:6, 4:6] += p['ch']*np.eye(2)
    damping += gyro_matrix(p)
    phi = 2*np.pi*np.arange(p['N'])/p['N']
    directions = np.column_stack((np.cos(phi), np.sin(phi)))
    constants = [4*p['Kf']/p['N'], 4*p['Df']/p['N'], p['fit_clearance_um']*1e-6,
                 4*p['Kf2']/p['N'], 4*p['Df2']/p['N'], p['fit2_clearance_um']*1e-6]

    def bearing(z, endpoint, params, orientation_sign=0., rho_m=0., contact_model=0.):
        ca = np.cos(params['contact_angle'])
        constants.extend([z, endpoint, params['normal_overlap'], cage_speed(params),
            params['cage_phase'], ca, np.sin(params['contact_angle']),
            bearing_keq(params), params['cb_total']/(params['balls']/2*ca**2),
            params['balls'], orientation_sign, rho_m, contact_model])

    contact_model = bearing_contact_model(p)
    if 'bearing_contact_model' in p:
        context = _bearing_context(p)
        model_code = 1. if contact_model == 'relative_tilt_v8' else 0.
        endpoint = {'O1': 6, 'O2': 10, 'H': 4}
        for name in ('F1', 'F2', 'R1', 'R2'):
            spec, params = context[name]
            bearing(spec['z_m'], endpoint[spec['outer_translation_source']], params,
                    spec['orientation_sign'], spec['rho_m'], model_code)
    else:
        bearing(p['z_f'], 6, p)
        if nonlinear:
            bearing(p['z_f2'], 10, p)
            rear = rear_bearing_parameters(p)
            for support in p['rear_supports']:
                bearing(support['z_m'], 4, rear)
    constants = np.asarray(constants)
    masses, gravity = mass_and_gravity(p)
    omega = p['rpm']*2*np.pi/60
    drive = np.array([unbalance[0]*omega**2, omega, unbalance[1], p['z_u']])
    args = (stiffness, damping, directions, constants, masses, gravity, drive)

    def rhs(t, state):
        return rhs_12(t, state, *args)

    # Available to the independent equivalence audit, without reassembling data.
    rhs.internal_force = lambda t, q, v: internal_force_12(t, q, v, *args[:4])
    return rhs
