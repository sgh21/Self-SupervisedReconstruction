"""V9 assembly: separate ring wrenches, fixed outer tilts, 12 structural DOFs.

All contact loads here are left-side restoring quantities. The actual contact
load on a ring is their negative. A fixed axial/rotational support therefore
applies the corresponding restoring component as its constraint reaction.
"""
import json
import numpy as np
from numba import njit

from . import tai_bearing as ball

NAMES = ('F1', 'F2', 'R1', 'R2')


def enabled(p):
    return (p.get('bearing_contact_model') == 'quasistatic_tai_v9'
            or p.get('damping_policy') == 'common_structural_ring')


@njit(cache=True)
def damping_wrench(vx, vy, tx, ty, cb, lever, active):
    out = np.zeros(6)
    if active:
        out[0] = cb*(vx-lever*ty)
        out[1] = cb*(vy+lever*tx)
        out[3] = lever*out[1]
        out[4] = -lever*out[0]
    return out


@njit(cache=True)
def legacy_elastic(t, dx, dy, tx, ty, old, sign, rho):
    overlap, speed, phase0, ca, sa, keq, count = old
    inner = np.zeros(6)
    active = False
    for k in range(int(count)):
        phi = speed*t+phase0+2*np.pi*k/count
        nx, ny = np.cos(phi), np.sin(phi)
        tilt_scale = sign*rho*sa
        delta = overlap+ca*(nx*dx+ny*dy)+tilt_scale*(tx*ny-ty*nx)
        if delta > 0:
            active = True
            normal = keq*delta**1.5
            inner[0] += ca*normal*nx
            inner[1] += ca*normal*ny
            inner[2] += sign*sa*normal
            inner[3] += tilt_scale*normal*ny
            inner[4] -= tilt_scale*normal*nx
    return inner, -inner, active


@njit(cache=True)
def ring_kernel(t, q, v, row, params, old, omega, mode, control, tolerance, use_tai):
    z, endpoint, sign, rho, cb, phase0, factor = row
    out = int(endpoint)
    dx, dy = q[0]+z*q[9]-q[out], q[1]-z*q[8]-q[out+1]
    if use_tai:
        inner, outer, states = ball.bearing_wrenches_kernel(
            t, dx, dy, q[8], q[9], params, sign, omega, phase0,
            mode, control, tolerance, factor)
        active = np.any(states[:, ball.ACTIVE] == 2)
    else:
        inner, outer, active = legacy_elastic(t, dx, dy, q[8], q[9], old, sign, rho)
        states = np.empty((0, 32))
    vx, vy = v[0]+z*v[9]-v[out], v[1]-z*v[8]-v[out+1]
    # F16 is common to B0d/N1/N2/N3, using frozen V8 reference geometry.
    lever = sign*rho*np.tan(params[ball.ALPHA0])
    damp = damping_wrench(vx, vy, v[8], v[9], cb, lever, active)
    return inner+damp, outer-damp, states, damp


@njit(cache=True)
def internal_kernel(t, q, v, stiffness, damping, directions, fit,
                    rows, params, old, omega, mode, control, tolerance, use_tai):
    force = stiffness@q+damping@v
    for j in range(4):
        z, endpoint = rows[j,0], int(rows[j,1])
        if use_tai:
            dx,dy = q[0]+z*q[9]-q[endpoint],q[1]-z*q[8]-q[endpoint+1]
            inner,outer,dual,_,_ = ball.bearing_wrenches_fast_kernel(
                t,dx,dy,q[8],q[9],params[j],rows[j,2],omega,rows[j,5],
                mode,control,tolerance,rows[j,6])
            lever = rows[j,2]*rows[j,3]*np.tan(params[j,ball.ALPHA0])
            damp = damping_wrench(v[0]+z*v[9]-v[endpoint],v[1]-z*v[8]-v[endpoint+1],
                                  v[8],v[9],rows[j,4],lever,dual>0)
            inner += damp
            outer -= damp
        else:
            inner,outer,_,_ = ring_kernel(t,q,v,rows[j],params[j],old[j],
                                         omega,mode,control,tolerance,use_tai)
        force[0:2] += inner[0:2]
        force[8] += -z*inner[1]+inner[3]
        force[9] += z*inner[0]+inner[4]
        force[endpoint:endpoint+2] += outer[0:2]
    for j in range(2):
        endpoint = 6 if j == 0 else 10
        rx, ry = q[endpoint]-q[4], q[endpoint+1]-q[5]
        vx, vy = v[endpoint]-v[4], v[endpoint+1]-v[5]
        fx, fy = 0., 0.
        k, c, gap = fit[j]
        for i in range(len(directions)):
            nx, ny = directions[i]
            delta = rx*nx+ry*ny-gap
            if delta > 0:
                normal = max(0.,k*delta+c*(vx*nx+vy*ny))
                fx += normal*nx
                fy += normal*ny
        force[endpoint] += fx
        force[endpoint+1] += fy
        force[4] -= fx
        force[5] -= fy
    return force


@njit(cache=True)
def rhs_kernel(t, state, stiffness, damping, directions, fit, rows, params,
               old, omega, mode, control, tolerance, use_tai, masses, gravity, drive):
    restoring = internal_kernel(t,state[:12],state[12:],stiffness,damping,
        directions,fit,rows,params,old,omega,mode,control,tolerance,use_tai)
    amplitude, phase, z = drive
    fx, fy = amplitude*np.cos(omega*t+phase), amplitude*np.sin(omega*t+phase)
    restoring[0] -= fx
    restoring[1] -= fy
    restoring[8] += z*fy
    restoring[9] -= z*fx
    result = np.empty(24)
    result[:12] = state[12:]
    result[12:] = (gravity-restoring)/masses
    return result


def prepare(p):
    from .dynamics_model import (section_matrix,ROTOR,area_and_link,gyro_matrix,
                                 _bearing_context,bearing_keq,cage_speed)
    # Only immutable mechanical inputs enter this derived cache; no root history.
    signature = json.dumps({k:v for k,v in p.items() if not k.startswith('_')},
                           sort_keys=True,default=lambda x: np.asarray(x).tolist())
    if p.get('_tai_signature') == signature:
        return p['_tai_args']
    if p['model_dof'] != 12 or p['bearing_layout_name'] != 'DB_DB':
        raise ValueError('V9 assembly requires 12 DOFs and DB/DB')
    if any(p.get(key,0) for key in ('outer_pit_area_m2','inner_pit_area_m2','ball_deformation_amplitude_m')):
        raise ValueError('V9 currently supports healthy internal race geometry only')
    context = _bearing_context(p)
    cfg = p['tai_bearing']
    vectors = {key:np.asarray(value['initialized_parameter_vector'],dtype=float)
               for key,value in cfg['families'].items()}
    stiffness, damping = np.zeros((12,12)), np.zeros((12,12))
    difference = np.zeros((2,12))
    difference[:,ROTOR] = section_matrix(p['z_T'])
    difference[:,2:4] -= np.eye(2)
    stiffness += area_and_link(p,0)[1]*difference.T@difference
    damping += p['cl']*difference.T@difference
    stiffness[4:6,4:6] += p['kh']*np.eye(2)
    damping[4:6,4:6] += p['ch']*np.eye(2)
    damping += gyro_matrix(p)
    phi = 2*np.pi*np.arange(p['N'])/p['N']
    directions = np.column_stack((np.cos(phi),np.sin(phi)))
    fit = np.array([[4*p['Kf']/p['N'],4*p['Df']/p['N'],p['fit_clearance_um']*1e-6],
                    [4*p['Kf2']/p['N'],4*p['Df2']/p['N'],p['fit2_clearance_um']*1e-6]])
    rows, packed, legacy = [], [], []
    for name in NAMES:
        spec,bp = context[name]
        family = 'front' if name.startswith('F') else 'rear'
        rows.append([spec['z_m'],{'O1':6,'O2':10,'H':4}[spec['outer_translation_source']],
                     spec['orientation_sign'],spec['rho_m'],
                     bp['cb_total']*cfg.get('structural_damping_factor',1.),
                     bp['cage_phase'],cfg['phase_speed_factor'][family]])
        packed.append(vectors[family])
        legacy.append([bp['normal_overlap'],cage_speed(bp),bp['cage_phase'],
            np.cos(bp['contact_angle']),np.sin(bp['contact_angle']),bearing_keq(bp),bp['balls']])
    args = (stiffness,damping,directions,fit,np.array(rows),np.array(packed),np.array(legacy),
            p['rpm']*2*np.pi/60,ball.MODE_CODES[cfg['ball_inertia_mode']],
            ball.CONTROL_CODES[cfg['control_branch']],cfg['local_tolerance_scale'],
            p['bearing_contact_model']=='quasistatic_tai_v9')
    if p.get('_derived_cache_enabled',False):
        p['_tai_args'],p['_tai_signature'] = args,signature
    return args


class LocalContactFailure(RuntimeError):
    def __init__(self,error,t,q,v):
        super().__init__(str(error))
        self.state = dict(t_s=float(t),q=np.asarray(q).tolist(),v=np.asarray(v).tolist(),
                          error=str(error),fallback_used=False)

    def __reduce__(self):
        return (type(self),(str(self),self.state['t_s'],self.state['q'],self.state['v']))


def prepare_fast_rhs(p,unbalance):
    from .dynamics_model import mass_and_gravity
    args = prepare(p)
    masses,gravity = mass_and_gravity(p)
    drive = np.array([unbalance[0]*args[7]**2,unbalance[1],p['z_u']])
    def rhs(t,state):
        try:
            return rhs_kernel(t,state,*args,masses,gravity,drive)
        except ValueError as error:
            raise LocalContactFailure(error,t,state[:12],state[12:]) from error
    rhs.internal_force = lambda t,q,v: internal_kernel(t,q,v,*args)
    return rhs


def internal_force(t,q,v,p):
    """Python assembly reference; the local contact kernel has its own SciPy oracle."""
    from .dynamics_model import section_matrix,ROTOR,fit_contact_force
    args = prepare(p)
    force = args[0]@q+args[1]@v
    wr = bearing_wrenches(t,q,v,p)
    for j,name in enumerate(NAMES):
        inner,outer = wr[name]['inner_restoring_6'],wr[name]['outer_restoring_6']
        z,endpoint = args[4][j,:2]
        force[ROTOR] += section_matrix(z).T@inner[:2]
        force[8:10] += inner[3:5]
        force[int(endpoint):int(endpoint)+2] += outer[:2]
    for endpoint,k,c,gap in [(6,p['Kf'],p['Df'],p['fit_clearance_um']),
                             (10,p['Kf2'],p['Df2'],p['fit2_clearance_um'])]:
        f,_ = fit_contact_force(q[endpoint:endpoint+2]-q[4:6],v[endpoint:endpoint+2]-v[4:6],
            dict(N=p['N'],Kf=k,Df=c,fit_clearance_um=gap))
        force[endpoint:endpoint+2] += f
        force[4:6] -= f
    return force


def equilibrium(p, initial_q=None):
    """Scaled trust-region static solve; prevent uncontrolled Newton jumps.

    The global solver is changed, not the contact equations or acceptance
    tolerance. Each call still solves the requested spin and clearance.
    """
    from scipy.optimize import least_squares
    from .dynamics_model import mass_and_gravity
    args = prepare(p)
    _,gravity = mass_and_gravity(p)
    qscale = np.full(12,1e-6)
    qscale[8:10] /= p['L_ref']
    fscale = np.ones(12)
    fscale[8:10] = p['L_ref']
    x0 = np.zeros(12) if initial_q is None else np.asarray(initial_q)/qscale
    def residual(x):
        try:
            return (internal_kernel(0.,x*qscale,np.zeros(12),*args)-gravity)/fscale
        except ValueError as error:
            raise LocalContactFailure(error,0.,x*qscale,np.zeros(12)) from error
    solution = least_squares(residual,x0,method='trf',jac='3-point',
        diff_step=1e-4,x_scale='jac',ftol=1e-12,xtol=1e-12,gtol=1e-10,max_nfev=500)
    error = float(np.linalg.norm(residual(solution.x)))
    if not np.isfinite(error) or error > 1e-6:
        raise RuntimeError(f'V9 trust-region equilibrium residual {error:.8g} N; {solution.message}')
    return solution.x*qscale,error


@njit(cache=True)
def _series(t,q,v,row,params,old,omega,mode,control,tolerance,use_tai):
    count = int(params[ball.Z]) if use_tai else 0
    inner,outer,damp = np.empty((len(t),6)),np.empty((len(t),6)),np.empty((len(t),6))
    states = np.empty((len(t),count,32))
    for i in range(len(t)):
        inner[i],outer[i],states[i],damp[i] = ring_kernel(
            t[i],q[i],v[i],row,params,old,omega,mode,control,tolerance,use_tai)
    return inner,outer,states,damp


def bearing_wrenches(t,q,v,p):
    args = prepare(p)
    q,v = np.asarray(q),np.asarray(v)
    single = q.ndim == 1
    qt,vt = np.atleast_2d(q),np.atleast_2d(v)
    tt = np.broadcast_to(t,(len(qt),)).astype(float)
    result = {}
    for j,name in enumerate(NAMES):
        try:
            inner,outer,states,damp = _series(tt,qt,vt,args[4][j],args[5][j],args[6][j],*args[7:])
        except ValueError as error:
            if single:
                raise LocalContactFailure(error,float(tt[0]),q,v) from error
            raise
        if single:
            inner,outer,states,damp = inner[0],outer[0],states[0],damp[0]
        result[name] = dict(inner_restoring_6=inner,outer_restoring_6=outer,
            force_N=inner[...,:2],moment_Nm=inner[...,3:5],diagnostics=states,
            damping_wrench4=damp[...,[0,1,3,4]],
            outer_axial_constraint_N=outer[...,2],outer_tilt_constraint_Nm=outer[...,3:5],
            inner_axial_constraint_N=inner[...,2])
    return result


diagnostics_series = bearing_wrenches


def force_components(t,q,v,p):
    from .dynamics_model import fit_contact_force,rotor_section,area_and_link
    wr = bearing_wrenches(t,q,v,p)
    fit1,_ = fit_contact_force(q[6:8]-q[4:6],v[6:8]-v[4:6],p)
    fit2,_ = fit_contact_force(q[10:12]-q[4:6],v[10:12]-v[4:6],
        dict(N=p['N'],Kf=p['Kf2'],Df=p['Df2'],fit_clearance_um=p['fit2_clearance_um']))
    # Actual housing contact moments: front fits plus rear outer contacts and
    # all fixed outer-tilt supports. Their negative is the ground reaction.
    housing_restoring_moment = np.zeros(2)
    for row,name in zip(prepare(p)[4],NAMES):
        z = row[0]
        outer = wr[name]['outer_restoring_6']
        radial = -fit1 if name=='F1' else -fit2 if name=='F2' else outer[:2]
        housing_restoring_moment += [-z*radial[1],z*radial[0]]
        housing_restoring_moment += outer[3:5]
    return dict(bearing_N=wr['F1']['force_N'],bearing2_N=wr['F2']['force_N'],
        fit_N=fit1,fit2_N=fit2,bearing_wrenches=wr,
        healthy_support_forces_N=np.stack([-wr[n]['outer_restoring_6'][:2] for n in NAMES[2:]]),
        link_N=area_and_link(p,0)[1]*(rotor_section(q,p['z_T'])-q[2:4])
            +p['cl']*(rotor_section(v,p['z_T'])-v[2:4]),
        housing_ground_N=p['kh']*q[4:6]+p['ch']*v[4:6],
        housing_constraint_torque_Nm=housing_restoring_moment,
        outer_rotation_constraint_moment_Nm_by_bearing={name:wr[name]['outer_tilt_constraint_Nm'] for name in NAMES},
        outer_axial_constraint_N_by_bearing={name:wr[name]['outer_axial_constraint_N'] for name in NAMES},
        inner_axial_constraint_N_by_bearing={name:wr[name]['inner_axial_constraint_N'] for name in NAMES},
        sign_convention='restoring on LHS; constraint reaction balances negative contact load')
