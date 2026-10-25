"""V9 Tai--Cao dual-contact ball bearing, SI units and left-hand restoring loads.

F01--F17 refer to GPTOrder/spindle_v9_tai_bearing_plan.  Eliminating the two
indentations as delta_i=|[A,B]-[U,V]|-a_i, delta_o=|[U,V]|-a_o enforces F05
identically; the deterministic two-variable Newton solver then solves F08.
The independent reference solves the original four equations. No previous RHS
call, fitted V8 stiffness, or normal_overlap enters this module.

The prescribed outer-control branch remains a *conditional* physical closure;
its Table 2 criterion is reported and is never made true by switching labels.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit
from scipy.optimize import brentq, root
from scipy.special import ellipk, ellipe

# Packed immutable parameter layout; all entries are floating point for Numba.
D, DM, FI, FO, ALPHA0, EPRIME, BALL_MASS, BALL_INERTIA, A_PRE, Z, P_REF, CAGE_RATIO, K_SCALE = range(13)
PARAMETER_NAMES = ('D_m', 'Dm_m', 'fi', 'fo', 'alpha0_rad', 'Eprime_Pa',
                   'ball_mass_kg', 'ball_inertia_kg_m2', 'a_pre_m', 'balls',
                   'preload_reference_N', 'prescribed_cage_ratio', 'hertz_scale')
BALL_NAMES = ('U_m', 'V_m', 'delta_i_m', 'delta_o_m', 'alpha_i_rad',
              'alpha_o_rad', 'Qi_N', 'Qo_N', 'Fc_N', 'Mg_N_m',
              'omega_E_rad_s', 'omega_B_rad_s', 'beta_rad', 'Ki_N_m32',
              'Ko_N_m32', 'g1_m2', 'g2_m2', 'g3_N', 'g4_N', 'iterations',
              'active_code', 'status_code', 'control_code',
              'outer_control_margin', 'inner_control_margin',
              'cage_speed_relative_difference', 'ellipse_ai_m', 'ellipse_ao_m',
              'p0_i_Pa', 'p0_o_Pa', 'force_residual_scaled', 'evaluations')
U, V, DI, DO, ANGLE_I, ANGLE_O, QI, QO, FC, MG, OMEGA_E, OMEGA_B, BETA, KI, KO, G1, G2, G3, G4, ITERATIONS, ACTIVE, STATUS, CONTROL, OUTER_MARGIN, INNER_MARGIN, PHASE_ERROR, ELLIPSE_AI, ELLIPSE_AO, PRESSURE_I, PRESSURE_O, RESIDUAL, EVALUATIONS = range(32)
MODE_CODES = {'none': 0, 'centrifugal': 1, 'full': 2}
CONTROL_CODES = {'outer': 0, 'inner': 1}


@njit(cache=True)
def _elliptic_integrals(m):
    """Complete K/E by the arithmetic-geometric mean (0 <= m < 1)."""
    a, b = 1.0, math.sqrt(1.0-m)
    correction, weight = 0.5*m, 1.0
    for _ in range(30):
        c = 0.5*(a-b)
        correction += weight*c*c
        anew = .5*(a+b)
        b = math.sqrt(a*b)
        a = anew
        weight *= 2.0
        if abs(c) <= 2e-16*a:
            break
    first = math.pi/(2*a)
    return first, first*(1.0-correction)


@njit(cache=True)
def _hertz_shape(ratio):
    # NASA Table 3.1 samples through 34.869; the nominal 0.54/0.52 geometries
    # lie within that range. Near a circle use the exact limit/integrals because
    # the fitted kappa does not tend to 1. Outside the table use exact integrals.
    if 1.1 <= ratio <= 34.869:
        return (1.0339*ratio**.6360, 1.0003+.5968/ratio,
                1.5277+.6023*math.log(ratio))
    if ratio <= 1.0+1e-10:
        return 1.0, math.pi/2, math.pi/2
    lower, upper = 1.0, max(2.0, 2.0*ratio)
    for _ in range(48):
        kap = .5*(lower+upper)
        ef, ee = _elliptic_integrals(1.0-1.0/(kap*kap))
        predicted = (kap*kap*ee-ef)/(ef-ee)
        if predicted < ratio:
            lower = kap
        else:
            upper = kap
    kap = .5*(lower+upper)
    ef, ee = _elliptic_integrals(1.0-1.0/(kap*kap))
    return kap, ee, ef


@njit(cache=True)
def hertz_kernel(angle, p, inner):
    """Cao 2004 eqs.24--27 / NASA TM81691, sorted principal radii."""
    f = p[FI] if inner else p[FO]
    sign = -1.0 if inner else 1.0
    rc = 0.5*p[D]*(1.0+sign*p[D]/p[DM]*math.cos(angle))
    rg = p[D]*f/(2.0*f-1.0)
    rx, ry = min(rc, rg), max(rc, rg)
    ratio = ry/rx
    radius = rx*ry/(rx+ry)
    kap, ee, ef = _hertz_shape(ratio)
    k = math.pi*kap*p[EPRIME]/(3.0*ef)*math.sqrt(2.0*ee*radius/ef)
    return k*p[K_SCALE], kap, ee, ef, radius


@njit(cache=True)
def kinematics_kernel(ai, ao, omega, p, control):
    lam = p[D]/p[DM]
    si, ci, so, co = math.sin(ai), math.cos(ai), math.sin(ao), math.cos(ao)
    cd = math.cos(ai-ao)
    if control == 0:
        eta_e = (1.0-lam*ci)/(1.0+cd)
        beta = math.atan2(so, co+lam)
    else:
        eta_e = (cd-lam*co)/(1.0+cd)
        beta = math.atan2(si, ci-lam)
    tb, cb = math.tan(beta), math.cos(beta)
    eta_b = -1.0/(((co+tb*so)/(1.0+lam*co)
                   +(ci+tb*si)/(1.0-lam*ci))*lam*cb)
    return omega*eta_e, omega*eta_b, beta


@njit(cache=True)
def _contact_values(A, B, u, v, omega, p, mode, control):
    li, lo = math.hypot(A-u, B-v), math.hypot(u, v)
    di, do = li-(p[FI]-0.5)*p[D], lo-(p[FO]-0.5)*p[D]
    ai, ao = math.atan2(A-u, B-v), math.atan2(u, v)
    ki, _, _, _, _ = hertz_kernel(ai, p, True)
    ko, _, _, _, _ = hertz_kernel(ao, p, False)
    # Signed trial indentation is retained. An accepted dual-contact root must
    # have both >=0; the positive part here only evaluates a Newton trial.
    qi = ki*max(di, 0.0)**1.5
    qo = ko*max(do, 0.0)**1.5
    oe, ob, beta = kinematics_kernel(ai, ao, omega, p, control)
    fc = 0.5*p[BALL_MASS]*p[DM]*oe*oe if mode > 0 else 0.0
    mg = p[BALL_INERTIA]*ob*oe*math.sin(beta) if mode == 2 else 0.0
    g = mg/p[D]
    si, ci, so, co = math.sin(ai), math.cos(ai), math.sin(ao), math.cos(ao)
    r3 = qo*co-g*so-qi*ci+g*si-fc
    r4 = qo*so+g*co-qi*si-g*ci
    return r3, r4, di, do, ai, ao, qi, qo, fc, mg, oe, ob, beta, ki, ko


@njit(cache=True)
def _hertz_with_angle_derivative(sine, cosine, p, inner):
    f, sign = (p[FI], -1.0) if inner else (p[FO], 1.0)
    rc = 0.5*p[D]*(1.0+sign*p[D]/p[DM]*cosine)
    rg = p[D]*f/(2.0*f-1.0)
    drc = -0.5*sign*p[D]*p[D]/p[DM]*sine
    ratio = rg/rc
    if rc >= rg or ratio < 1.1 or ratio > 34.869:
        # The slower exact-integral branch is needed only outside the approved
        # nominal geometry; differentiate that same physical formula.
        angle = math.atan2(sine, cosine)
        kval = hertz_kernel(angle, p, inner)[0]
        plus = hertz_kernel(angle+1e-6, p, inner)[0]
        minus = hertz_kernel(angle-1e-6, p, inner)[0]
        return kval, (plus-minus)/(2e-6*kval)
    kap = 1.0339*ratio**0.6360
    ef = 1.5277+0.6023*math.log(ratio)
    ee = 1.0003+0.5968/ratio
    radius = rc*rg/(rc+rg)
    kval = math.pi*kap*p[EPRIME]/(3.0*ef)*math.sqrt(2.0*ee*radius/ef)*p[K_SCALE]
    dlogk = (-.6360/rc + .90345/(rc*ef) + .2984/(rg*ee)
             + .5*(1.0/rc-1.0/(rc+rg)))*drc
    return kval, dlogk


@njit(cache=True)
def _residual_jacobian(A, B, u, v, omega, p, mode, control):
    """Full analytic chain rule, including angle-dependent K, Fc and Mg."""
    li, lo = math.hypot(A-u, B-v), math.hypot(u, v)
    si, ci, so, co = (A-u)/li, (B-v)/li, u/lo, v/lo
    di, do = li-(p[FI]-.5)*p[D], lo-(p[FO]-.5)*p[D]
    ki, lki = _hertz_with_angle_derivative(si, ci, p, True)
    ko, lko = _hertz_with_angle_derivative(so, co, p, False)
    qi, qo = ki*max(di, 0.0)**1.5, ko*max(do, 0.0)**1.5
    dqi_dl = 1.5*ki*math.sqrt(max(di, 0.0))
    dqo_dl = 1.5*ko*math.sqrt(max(do, 0.0))
    lam = p[D]/p[DM]
    cd, sd = ci*co+si*so, si*co-ci*so
    den = 1.0+cd
    if control == 0:
        num = 1.0-lam*ci
        enum_i, enum_o = lam*si, 0.0
        bn = math.hypot(so, co+lam)
        sb, cb = so/bn, (co+lam)/bn
        db_i, db_o = 0.0, (1.0+lam*co)/(bn*bn)
    else:
        num = cd-lam*co
        enum_i, enum_o = -sd, sd+lam*so
        bn = math.hypot(si, ci-lam)
        sb, cb = si/bn, (ci-lam)/bn
        db_i, db_o = (1.0-lam*ci)/(bn*bn), 0.0
    eta_e = num/den
    de_i = (enum_i*den+num*sd)/(den*den)
    de_o = (enum_o*den-num*sd)/(den*den)
    den_o, den_i = 1.0+lam*co, 1.0-lam*ci
    cos_o, cos_i = co*cb+so*sb, ci*cb+si*sb
    sin_o, sin_i = so*cb-co*sb, si*cb-ci*sb
    spin_den = lam*(cos_o/den_o+cos_i/den_i)
    eta_b = -1.0/spin_den
    dspin_i = lam*(sin_o*db_i/den_o
                  -sin_i*(1.0-db_i)/den_i-cos_i*lam*si/(den_i*den_i))
    dspin_o = lam*(-sin_o*(1.0-db_o)/den_o+cos_o*lam*so/(den_o*den_o)
                  +sin_i*db_o/den_i)
    dbspin_i, dbspin_o = dspin_i/(spin_den*spin_den), dspin_o/(spin_den*spin_den)
    oe, ob = omega*eta_e, omega*eta_b
    fc = .5*p[BALL_MASS]*p[DM]*oe*oe if mode > 0 else 0.0
    dfc_i = p[BALL_MASS]*p[DM]*oe*omega*de_i if mode > 0 else 0.0
    dfc_o = p[BALL_MASS]*p[DM]*oe*omega*de_o if mode > 0 else 0.0
    g = p[BALL_INERTIA]*oe*ob*sb/p[D] if mode == 2 else 0.0
    dg_i = (p[BALL_INERTIA]*omega*omega/p[D]
            *((de_i*eta_b+eta_e*dbspin_i)*sb+eta_e*eta_b*cb*db_i)) if mode == 2 else 0.0
    dg_o = (p[BALL_INERTIA]*omega*omega/p[D]
            *((de_o*eta_b+eta_e*dbspin_o)*sb+eta_e*eta_b*cb*db_o)) if mode == 2 else 0.0
    r0 = qo*co-qi*ci+g*(si-so)-fc
    r1 = qo*so-qi*si+g*(co-ci)
    # Partial derivatives with respect to contact angles, holding indentation.
    r0_ai = -qi*lki*ci+qi*si+dg_i*(si-so)+g*ci-dfc_i
    r0_ao = qo*lko*co-qo*so+dg_o*(si-so)-g*co-dfc_o
    r1_ai = -qi*lki*si-qi*ci+dg_i*(co-ci)+g*si
    r1_ao = qo*lko*so+qo*co+dg_o*(co-ci)-g*so
    dai_du, dai_dv, dao_du, dao_dv = -ci/li, si/li, co/lo, -so/lo
    j00 = r0_ai*dai_du+r0_ao*dao_du+dqi_dl*si*ci+dqo_dl*so*co
    j01 = r0_ai*dai_dv+r0_ao*dao_dv+dqi_dl*ci*ci+dqo_dl*co*co
    j10 = r1_ai*dai_du+r1_ao*dao_du+dqi_dl*si*si+dqo_dl*so*so
    j11 = r1_ai*dai_dv+r1_ao*dao_dv+dqi_dl*ci*si+dqo_dl*co*so
    return r0, r1, j00, j01, j10, j11, di, do


@njit(cache=True)
def _ellipse(angle, load, p, inner):
    _, kap, ee, _, radius = hertz_kernel(angle, p, inner)
    if load <= 0.0:
        return 0.0, 0.0, ee
    a = (6.0*kap*kap*ee*load*radius/(math.pi*p[EPRIME]))**(1.0/3.0)
    b = (6.0*ee*load*radius/(math.pi*kap*p[EPRIME]))**(1.0/3.0)
    return a, 3.0*load/(2.0*math.pi*a*b), ee


@njit(cache=True)
def _finish(A, B, u, v, omega, p, mode, control, iterations, evaluations, active, status, diagnostics=True):
    result = np.zeros(32)
    values = _contact_values(A, B, u, v, omega, p, mode, control)
    for j in range(13):
        result[j+2] = values[j+2]
    # values indices 2..14 are exactly DI..KO.
    result[U], result[V] = u, v
    result[G3], result[G4] = values[0], values[1]
    li = (p[FI]-0.5)*p[D]+result[DI]
    lo = (p[FO]-0.5)*p[D]+result[DO]
    result[G1] = (A-u)**2+(B-v)**2-li*li
    result[G2] = u*u+v*v-lo*lo
    result[ITERATIONS], result[EVALUATIONS] = iterations, evaluations
    result[ACTIVE], result[STATUS], result[CONTROL] = active, status, control
    force_scale = max(p[P_REF]/(p[Z]*math.sin(p[ALPHA0])), result[FC], 1.0)
    result[RESIDUAL] = max(abs(result[G3]), abs(result[G4]))/force_scale
    if not diagnostics:
        return result
    ai_h, pi, ei = _ellipse(result[ANGLE_I], result[QI], p, True)
    ao_h, po, eo = _ellipse(result[ANGLE_O], result[QO], p, False)
    lhs_o = result[QO]*ao_h*eo
    lhs_i = result[QI]*ai_h*ei
    denom = max(lhs_i, lhs_o, 1e-30)
    cosdiff = math.cos(result[ANGLE_I]-result[ANGLE_O])
    result[OUTER_MARGIN] = (lhs_o*cosdiff-lhs_i)/denom
    result[INNER_MARGIN] = (lhs_i*cosdiff-lhs_o)/denom
    result[ELLIPSE_AI], result[ELLIPSE_AO] = ai_h, ao_h
    result[PRESSURE_I], result[PRESSURE_O] = pi, po
    cage = omega*p[CAGE_RATIO]
    result[PHASE_ERROR] = abs(result[OMEGA_E]-cage)/abs(cage) if cage != 0 else 0.0
    return result


@njit(cache=True)
def _retry_centrifugal_seed(A, B, omega, p, mode, control, tolerance_scale,
                            diagnostics, fc_open, first):
    """Retry the same equations only after the original Newton path failed.

    The finite centrifugal load sets the trial contact compressions. Their two
    groove-distance circles give a deterministic outward intersection, including
    cases where the static series load is much smaller than the centrifugal load.
    The original iteration is kept separate so its successful outputs are intact.
    """
    length = math.hypot(A, B)
    angle = math.atan2(A, B)
    ai0, ao0 = (p[FI]-.5)*p[D], (p[FO]-.5)*p[D]
    ki, ko = hertz_kernel(angle, p, True)[0], hertz_kernel(angle, p, False)[0]
    keq = (ki**(-2.0/3.0)+ko**(-2.0/3.0))**(-1.5)
    static_load = keq*max(length-ai0-ao0, 0.0)**1.5
    qseed = max(fc_open, static_load, 1e-6)
    li = ai0+(qseed/ki)**(2.0/3.0)
    lo = ao0+((qseed+fc_open)/ko)**(2.0/3.0)
    along = (length*length+lo*lo-li*li)/(2.0*length)
    height = math.sqrt(max(lo*lo-along*along, 0.0))
    side = 1.0 if A >= 0.0 else -1.0
    u = (along*A-side*height*B)/length
    v = (along*B+side*height*A)/length
    fscale = max(p[P_REF]/(p[Z]*math.sin(p[ALPHA0])), fc_open, 1.0)
    ftol = min(1e-8, 1e-9*fscale)*tolerance_scale
    val = _residual_jacobian(A, B, u, v, omega, p, mode, control)
    evaluations = int(first[EVALUATIONS])+1
    previous_iterations = int(first[ITERATIONS])
    for iteration in range(40):
        r0, r1 = val[0], val[1]
        error = max(abs(r0), abs(r1))
        if error <= ftol and val[6] >= 0.0 and val[7] >= 0.0:
            return _finish(A, B, u, v, omega, p, mode, control,
                           previous_iterations+iteration, evaluations, 2, 1, diagnostics)
        j00, j01, j10, j11 = val[2], val[3], val[4], val[5]
        determinant = j00*j11-j01*j10
        if abs(determinant) < 1e-8 or not math.isfinite(determinant):
            break
        du, dv = (-r0*j11+j01*r1)/determinant, (j10*r0-j00*r1)/determinant
        step = 1.0
        accepted = False
        for _ in range(16):
            un, vn = u+step*du, v+step*dv
            trial = _residual_jacobian(A, B, un, vn, omega, p, mode, control)
            evaluations += 1
            if max(abs(trial[0]), abs(trial[1])) < error:
                u, v = un, vn
                val = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    final = _finish(A, B, u, v, omega, p, mode, control,
                    previous_iterations+iteration+1, evaluations, 2, -2, diagnostics)
    acceptance = max(1e-6, 1e-7*fscale)*tolerance_scale
    if max(abs(final[G3]), abs(final[G4])) <= acceptance and final[DI] >= 0.0 and final[DO] >= 0.0:
        final[STATUS] = 1.0
    return final


@njit(cache=True)
def solve_ball_kernel(A, B, omega, p, mode, control, tolerance_scale=1.0, diagnostics=True, cage_factor=1.0):
    """Return BALL_NAMES array; status<0 is an explicit failed local solve.

    ACTIVE=2: dual contact; 1: outer only; 0: zero-inertia open set.
    STATUS=0: original solve succeeded; 1: deterministic reseed succeeded;
    -2: both attempts failed. Iteration/evaluation counts include both attempts.
    The outer-only kinematics use the prescribed cage speed (F10).
    """
    if mode < 0 or mode > 2 or control < 0 or control > 1:
        raise ValueError('Unknown V9 inertia/control code; no fallback')
    ai0, ao0 = (p[FI]-0.5)*p[D], (p[FO]-0.5)*p[D]
    b0 = ai0+ao0
    length = math.hypot(A, B)
    angle = math.atan2(A, B)
    ki, _, _, _, _ = hertz_kernel(angle, p, True)
    ko, _, _, _, _ = hertz_kernel(angle, p, False)
    delta = length-b0
    no_inertia = mode == 0 or omega == 0.0
    if no_inertia and delta > 0.0:
        keq = (ki**(-2.0/3.0)+ko**(-2.0/3.0))**(-1.5)
        q = keq*delta**1.5
        lo = ao0+(q/ko)**(2.0/3.0)
        return _finish(A, B, lo*math.sin(angle), lo*math.cos(angle),
                       omega, p, 0, control, 0, 1, 2, 0, diagnostics)
    # Exact outer-only candidate: Qi=0, alpha_o=0, Mg=0, Qo=Fc.
    ko_zero, _, _, _, _ = hertz_kernel(0.0, p, False)
    cage = omega*p[CAGE_RATIO]*cage_factor
    fc_open = 0.5*p[BALL_MASS]*p[DM]*cage*cage if not no_inertia else 0.0
    do_open = (fc_open/ko_zero)**(2.0/3.0)
    v_open = ao0+do_open
    inner_gap = math.hypot(A, B-v_open)-ai0
    if inner_gap <= 0.0:
        out = _finish(A, B, 0.0, v_open, omega, p, mode, control, 0, 1,
                      1 if fc_open > 0 else 0, 0, diagnostics)
        out[DI], out[DO] = inner_gap, do_open
        out[QI], out[QO], out[FC], out[MG] = 0.0, fc_open, fc_open, 0.0
        out[ANGLE_O], out[OMEGA_E], out[OMEGA_B], out[BETA] = 0.0, cage, 0.0, 0.0
        out[KI], out[KO] = hertz_kernel(out[ANGLE_I], p, True)[0], ko_zero
        out[G1] = 0.0  # inactive constraint represented by DI <= 0, not equality
        out[G2], out[G3], out[G4] = 0.0, 0.0, 0.0
        out[RESIDUAL], out[PHASE_ERROR] = 0.0, 0.0
        out[OUTER_MARGIN], out[INNER_MARGIN] = np.nan, np.nan
        out[ELLIPSE_AI], out[PRESSURE_I] = 0.0, 0.0
        if diagnostics:
            out[ELLIPSE_AO], out[PRESSURE_O], _ = _ellipse(0.0, fc_open, p, False)
        return out
    if no_inertia and delta <= 0.0:
        # Intersection of the two unloaded groove-distance disks. The ball's
        # position is indeterminate; choose the line-of-centres deterministically.
        lo = max(0.0, min(ao0, length*ao0/b0))
        out = _finish(A, B, lo*math.sin(angle), lo*math.cos(angle),
                      omega, p, 0, control, 0, 1, 0, 0, diagnostics)
        out[QI], out[QO], out[G3], out[G4], out[RESIDUAL] = 0., 0., 0., 0., 0.
        out[OUTER_MARGIN], out[INNER_MARGIN] = np.nan, np.nan
        return out
    # Centred deterministic starting point; no warm-start or history dependence.
    keq = (ki**(-2.0/3.0)+ko**(-2.0/3.0))**(-1.5)
    q = keq*max(delta, 1e-10)**1.5
    lo = ao0+(q/ko)**(2.0/3.0)
    u, v = lo*math.sin(angle), lo*math.cos(angle)
    if delta <= 0.0:
        # A loaded centrifugal solution can exist even when the two unloaded
        # groove-distance circles intersect. Start on their outward geometric
        # intersection, with finite physically scaled trial compressions.
        qseed = max(fc_open, 1e-6)
        li_seed = ai0+(qseed/ki)**(2.0/3.0)
        lo_seed = ao0+((qseed+fc_open)/ko)**(2.0/3.0)
        along = (length*length+lo_seed*lo_seed-li_seed*li_seed)/(2.0*length)
        height = math.sqrt(max(lo_seed*lo_seed-along*along, 0.0))
        u = (along*A-height*B)/length
        v = (along*B+height*A)/length
    fscale = max(p[P_REF]/(p[Z]*math.sin(p[ALPHA0])), fc_open, 1.0)
    # Solve more tightly than acceptance to limit noisy tangent/ODE evaluations.
    ftol = min(1e-8, 1e-9*fscale)*tolerance_scale
    val = _residual_jacobian(A, B, u, v, omega, p, mode, control)
    evaluations = 1
    for iteration in range(40):
        r0, r1 = val[0], val[1]
        error = max(abs(r0), abs(r1))
        if error <= ftol and val[6] >= 0.0 and val[7] >= 0.0:
            return _finish(A, B, u, v, omega, p, mode, control,
                           iteration, evaluations, 2, 0, diagnostics)
        j00, j01, j10, j11 = val[2], val[3], val[4], val[5]
        determinant = j00*j11-j01*j10
        if abs(determinant) < 1e-8 or not math.isfinite(determinant):
            break
        du, dv = (-r0*j11+j01*r1)/determinant, (j10*r0-j00*r1)/determinant
        step = 1.0
        accepted = False
        for _ in range(16):
            un, vn = u+step*du, v+step*dv
            trial = _residual_jacobian(A, B, un, vn, omega, p, mode, control)
            evaluations += 1
            if max(abs(trial[0]), abs(trial[1])) < error:
                u, v = un, vn
                val = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    final = _finish(A, B, u, v, omega, p, mode, control,
                    iteration+1, evaluations, 2, -1, diagnostics)
    acceptance = max(1e-6, 1e-7*fscale)*tolerance_scale
    if max(abs(final[G3]), abs(final[G4])) <= acceptance and final[DI] >= 0.0 and final[DO] >= 0.0:
        final[STATUS] = 0.0
    if final[STATUS] < 0:
        return _retry_centrifugal_seed(A, B, omega, p, mode, control,
                                      tolerance_scale, diagnostics, fc_open, final)
    return final


@njit(cache=True)
def bearing_wrenches_kernel(t, dx, dy, tx, ty, p, s, omega_global,
                            phase0, mode, control, tolerance_scale=1.0, phase_factor=1.0):
    """Separate GLOBAL [Fx,Fy,Fz,Mx,My,Mz] restoring wrenches and ball data."""
    inner, outer = np.zeros(6), np.zeros(6)
    balls = np.empty((int(p[Z]), 32))
    b0 = (p[FI]+p[FO]-1.0)*p[D]
    ric = p[DM]/2.0+(p[FI]-0.5)*p[D]*math.cos(p[ALPHA0])
    roc = p[DM]/2.0-(p[FO]-0.5)*p[D]*math.cos(p[ALPHA0])
    for k in range(int(p[Z])):
        phase_global = phase0+omega_global*p[CAGE_RATIO]*phase_factor*t+2.0*math.pi*k/p[Z]
        phase = s*phase_global
        sn, cs = math.sin(phase), math.cos(phase)
        # T_s maps global translations to (0,dx,s*dy), rotations to (0,tx,s*ty).
        A = b0*math.sin(p[ALPHA0])+p[A_PRE]-ric*s*ty*cs+ric*tx*sn
        B = b0*math.cos(p[ALPHA0])+dx*cs+s*dy*sn
        ball = solve_ball_kernel(A, B, s*omega_global, p, mode, control, tolerance_scale, True, phase_factor)
        balls[k] = ball
        cage = s*omega_global*p[CAGE_RATIO]*phase_factor
        balls[k, PHASE_ERROR] = abs(ball[OMEGA_E]-cage)/abs(cage) if cage != 0 else 0.0
        if ball[STATUS] < 0:
            raise ValueError('V9 local ball root failed: k,A,B,omega,g3,g4',
                             k, A, B, s*omega_global, ball[G3], ball[G4])
        g = ball[MG]/p[D]
        pi = ball[QI]*math.cos(ball[ANGLE_I])-g*math.sin(ball[ANGLE_I])
        ai = ball[QI]*math.sin(ball[ANGLE_I])+g*math.cos(ball[ANGLE_I])
        po = ball[QO]*math.cos(ball[ANGLE_O])-g*math.sin(ball[ANGLE_O])
        ao = ball[QO]*math.sin(ball[ANGLE_O])+g*math.cos(ball[ANGLE_O])
        mi = ric*ai-p[FI]*ball[MG]
        mo = roc*ao+p[FO]*ball[MG]
        inner[0] += pi*cs
        inner[1] += s*pi*sn
        inner[2] += s*ai
        inner[3] += mi*sn
        inner[4] -= s*mi*cs
        outer[0] -= po*cs
        outer[1] -= s*po*sn
        outer[2] -= s*ao
        outer[3] -= mo*sn
        outer[4] += s*mo*cs
    return inner, outer, balls


@njit(cache=True)
def bearing_wrenches_fast_kernel(t, dx, dy, tx, ty, p, s, omega_global,
                                 phase0, mode, control, tolerance_scale=1.0, phase_factor=1.0):
    """RHS form: same roots/loads; omit per-ball ellipse/control diagnostics.

    Returns inner6, outer6, dual_count, max_force_residual_N, max_iterations.
    Failed roots throw the same explicit reproducible exception as full output.
    """
    inner, outer = np.zeros(6), np.zeros(6)
    b0 = (p[FI]+p[FO]-1.0)*p[D]
    ric = p[DM]/2.0+(p[FI]-.5)*p[D]*math.cos(p[ALPHA0])
    roc = p[DM]/2.0-(p[FO]-.5)*p[D]*math.cos(p[ALPHA0])
    dual_count, max_residual, max_iterations = 0, 0.0, 0.0
    for k in range(int(p[Z])):
        phase = s*(phase0+omega_global*p[CAGE_RATIO]*phase_factor*t+2.0*math.pi*k/p[Z])
        sn, cs = math.sin(phase), math.cos(phase)
        A = b0*math.sin(p[ALPHA0])+p[A_PRE]-ric*s*ty*cs+ric*tx*sn
        B = b0*math.cos(p[ALPHA0])+dx*cs+s*dy*sn
        ball = solve_ball_kernel(A, B, s*omega_global, p, mode, control, tolerance_scale, False, phase_factor)
        if ball[STATUS] < 0:
            raise ValueError('V9 local ball root failed: k,A,B,omega,g3,g4',
                             k, A, B, s*omega_global, ball[G3], ball[G4])
        if ball[ACTIVE] == 2:
            dual_count += 1
        max_residual = max(max_residual, abs(ball[G3]), abs(ball[G4]))
        max_iterations = max(max_iterations, ball[ITERATIONS])
        g = ball[MG]/p[D]
        pi = ball[QI]*math.cos(ball[ANGLE_I])-g*math.sin(ball[ANGLE_I])
        ai = ball[QI]*math.sin(ball[ANGLE_I])+g*math.cos(ball[ANGLE_I])
        po = ball[QO]*math.cos(ball[ANGLE_O])-g*math.sin(ball[ANGLE_O])
        ao = ball[QO]*math.sin(ball[ANGLE_O])+g*math.cos(ball[ANGLE_O])
        mi, mo = ric*ai-p[FI]*ball[MG], roc*ao+p[FO]*ball[MG]
        inner[0] += pi*cs
        inner[1] += s*pi*sn
        inner[2] += s*ai
        inner[3] += mi*sn
        inner[4] -= s*mi*cs
        outer[0] -= po*cs
        outer[1] -= s*po*sn
        outer[2] -= s*ao
        outer[3] -= mo*sn
        outer[4] += s*mo*cs
    return inner, outer, dual_count, max_residual, max_iterations


def make_bearing_parameters(family, materials=None):
    """Build the fixed array and solve its rest preload once, in local +axial."""
    materials = materials or {}
    diameter = float(family['D_m'])
    pitch = float(family['Dm_m'])
    fi, fo = float(family['fi']), float(family['fo'])
    if not (0 < diameter < pitch and fi > .5 and fo > .5):
        raise ValueError('V9 requires 0<D<Dm and fi,fo>0.5')
    eb = float(materials.get('ball_E_Pa', 315e9))
    er = float(materials.get('ring_E_Pa', 210e9))
    nb = float(materials.get('ball_nu', .26))
    nr = float(materials.get('ring_nu', .30))
    density = float(materials.get('ball_density_kg_m3', 3200.))
    mass = density*math.pi*diameter**3/6
    alpha0 = math.radians(float(family['nominal_contact_angle_deg']))
    p = np.array([diameter, pitch, fi, fo, alpha0,
                  2/((1-nb*nb)/eb+(1-nr*nr)/er), mass, .1*mass*diameter**2,
                  0., float(family['balls']), float(family['preload_reference_at_rest_N']),
                  .5*(1-diameter/pitch*math.cos(alpha0)), 1.], dtype=float)
    b0 = (fi+fo-1)*diameter
    def preload(offset):
        A, B = b0*math.sin(alpha0)+offset, b0*math.cos(alpha0)
        state = solve_ball_kernel(A, B, 0., p, 0, 0, 1.)
        return p[Z]*state[QI]*math.sin(state[ANGLE_I])-p[P_REF]
    p[A_PRE] = brentq(preload, 0., diameter*.1, xtol=1e-17)
    return p


def hertz_principal_radii(radius_x_m, radius_y_m, Eprime_Pa, load_N=0.):
    """Exact elliptic-integral Hertz solution, including the circular limit.

    This diagnostic/reference handles radii outside the nominal F04 approximate
    fit.  Eprime=2 E*; exchanging the two principal radii leaves K unchanged.
    """
    rx, ry = sorted((float(radius_x_m), float(radius_y_m)))
    if rx <= 0 or Eprime_Pa <= 0 or load_N < 0:
        raise ValueError('Positive principal radii/modulus and nonnegative load required')
    ratio = ry/rx
    if ratio < 1+1e-10:
        kap, ef, ee = 1., math.pi/2, math.pi/2
    else:
        def ratio_at(kap):
            m = 1-1/(kap*kap)
            ef, ee = ellipk(m), ellipe(m)
            return (kap*kap*ee-ef)/(ef-ee)
        kap = brentq(lambda k: ratio_at(k)-ratio, 1.0000001, max(2., 2*ratio), xtol=1e-12)
        ef, ee = float(ellipk(1-1/(kap*kap))), float(ellipe(1-1/(kap*kap)))
    radius = rx*ry/(rx+ry)
    kval = math.pi*kap*Eprime_Pa/(3*ef)*math.sqrt(2*ee*radius/ef)
    a = (6*kap*kap*ee*load_N*radius/(math.pi*Eprime_Pa))**(1/3) if load_N else 0.
    b = a/kap
    return dict(K_N_m32=kval, kappa=kap, Eprime_Pa=Eprime_Pa, ellipse_E=ee,
                ellipse_F=ef, R_m=radius, semi_major_m=a, semi_minor_m=b,
                p0_Pa=3*load_N/(2*math.pi*a*b) if load_N else 0.)


def hertz_contact_properties(angle_rad, params, inner=True, load_N=0.):
    k, kap, ee, ef, radius = hertz_kernel(float(angle_rad), params, bool(inner))
    a, pressure, _ = _ellipse(float(angle_rad), float(load_N), params, bool(inner))
    b = a/kap if a > 0 else 0.
    return dict(K_N_m32=k, kappa=kap, Eprime_Pa=params[EPRIME],
                ellipse_E=ee, ellipse_F=ef, R_m=radius,
                semi_major_m=a, semi_minor_m=b, p0_Pa=pressure)


def _ball_dict(state):
    result = dict(zip(BALL_NAMES, map(float, state)))
    result['active_set'] = {0: 'open_zero_inertia', 1: 'outer_only', 2: 'dual'}[int(state[ACTIVE])]
    result['control_branch'] = {0: 'outer', 1: 'inner'}[int(state[CONTROL])]
    result['control_self_consistent'] = bool(state[OUTER_MARGIN if int(state[CONTROL]) == 0 else INNER_MARGIN] > 0)
    result['reseed_attempted'] = bool(state[STATUS] in (1, -2))
    result['residuals'] = state[G1:G4+1].copy()
    return result


def solve_ball(A, B, omega_local, params, mode='full', control='outer', tolerance_scale=1.):
    state = solve_ball_kernel(float(A), float(B), float(omega_local), params,
                              MODE_CODES[mode], CONTROL_CODES[control], float(tolerance_scale))
    if state[STATUS] < 0:
        raise RuntimeError({'message': 'V9 local root failed; no fallback',
                            'A_m': A, 'B_m': B, 'omega_local': omega_local,
                            'parameters': params.tolist(), 'mode': mode,
                            'control': control, 'last_state': _ball_dict(state)})
    return _ball_dict(state)


def solve_ball_candidates(A, B, omega_local, params, mode='full', tolerance_scale=1.):
    """Explicitly solve both Table 2 candidates; do not silently select one."""
    candidates = {}
    for control in ('outer', 'inner'):
        try:
            candidates[control] = solve_ball(A, B, omega_local, params, mode, control, tolerance_scale)
        except RuntimeError as exc:
            candidates[control] = {'failed': True, 'failure': str(exc)}
    valid = [name for name, item in candidates.items() if item.get('control_self_consistent')]
    candidates['classification'] = valid[0] if len(valid) == 1 else ('both_candidates' if valid else 'mixed_or_undetermined')
    return candidates


def solve_ball_reference(A, B, omega_local, params, mode='full', control='outer', tolerance_scale=1.):
    """Independent four-variable FD-Jacobian root of F05/F08 (SciPy)."""
    mode_code, control_code = MODE_CODES[mode], CONTROL_CODES[control]
    candidate = solve_ball_kernel(A, B, omega_local, params, mode_code, control_code, tolerance_scale)
    if candidate[ACTIVE] != 2:
        return _ball_dict(candidate)
    b0 = (params[FI]+params[FO]-1)*params[D]
    force_scale = max(params[P_REF]/(params[Z]*math.sin(params[ALPHA0])), candidate[FC], 1.)
    scales = np.array([b0, b0, 1e-6, 1e-6])
    def residual(y):
        u, v, di, do = y*scales
        li, lo = (params[FI]-.5)*params[D]+di, (params[FO]-.5)*params[D]+do
        ai, ao = math.atan2(A-u, B-v), math.atan2(u, v)
        ki = hertz_kernel(ai, params, True)[0]
        ko = hertz_kernel(ao, params, False)[0]
        qi, qo = ki*max(di, 0.)**1.5, ko*max(do, 0.)**1.5
        oe, ob, beta = kinematics_kernel(ai, ao, omega_local, params, control_code)
        fc = .5*params[BALL_MASS]*params[DM]*oe*oe if mode_code > 0 else 0.
        mg = params[BALL_INERTIA]*oe*ob*math.sin(beta) if mode_code == 2 else 0.
        g = mg/params[D]
        return np.array([((A-u)**2+(B-v)**2-li*li)/b0**2,
                         (u*u+v*v-lo*lo)/b0**2,
                         (qo*math.cos(ao)-g*math.sin(ao)-qi*math.cos(ai)+g*math.sin(ai)-fc)/force_scale,
                         (qo*math.sin(ao)+g*math.cos(ao)-qi*math.sin(ai)-g*math.cos(ai))/force_scale])
    # Perturb the deterministic candidate: this is a new four-equation solve,
    # not simply a call that reports the accelerated solution as the reference.
    y0 = candidate[:4]/scales + np.array([1e-5, -1e-5, .01, -.01])
    solved = root(residual, y0, method='hybr', options={'xtol': 1e-10*tolerance_scale})
    scaled_residual = residual(solved.x)
    norm = np.max(np.abs(scaled_residual))
    if (np.max(np.abs(scaled_residual[:2])) > 1e-9*tolerance_scale
            or np.max(np.abs(scaled_residual[2:]))*force_scale > max(1e-6, 1e-7*force_scale)*tolerance_scale):
        raise RuntimeError(f'Independent four-variable root failed: {solved.message}; residual={norm}')
    u, v, di, do = solved.x*scales
    out = _finish(A, B, u, v, omega_local, params, mode_code, control_code, 0, solved.nfev, 2, 0)
    out[DI], out[DO] = di, do
    out[QI], out[QO] = out[KI]*max(di, 0.)**1.5, out[KO]*max(do, 0.)**1.5
    out[G1:G4+1] = scaled_residual*np.array([b0*b0, b0*b0, force_scale, force_scale])
    out[RESIDUAL] = np.max(np.abs(scaled_residual[2:]))
    if di < 0 or do < 0:
        raise RuntimeError('Independent reference found an inadmissible negative-indentation dual root')
    return _ball_dict(out)


def bearing_wrenches(t, relative_xy, relative_tilt_xy, params, orientation,
                     omega_global, phase0=0., mode='full', control='outer', tolerance_scale=1., phase_factor=1.):
    inner, outer, balls = bearing_wrenches_kernel(float(t), *map(float, relative_xy),
        *map(float, relative_tilt_xy), params, float(orientation), float(omega_global),
        float(phase0), MODE_CODES[mode], CONTROL_CODES[control], float(tolerance_scale), float(phase_factor))
    return {'inner_global': inner, 'outer_global': outer, 'balls': balls,
            'ball_fields': BALL_NAMES,
            'dual_contact_fraction': float(np.mean(balls[:, ACTIVE] == 2)),
            'control_self_consistent_fraction': float(np.mean(balls[:, OUTER_MARGIN if control == 'outer' else INNER_MARGIN] > 0)),
            'max_phase_speed_relative_difference': float(np.max(balls[:, PHASE_ERROR])),
            'max_force_residual_N': float(np.max(np.abs(balls[:, G3:G4+1])))}
