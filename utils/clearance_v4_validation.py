"""Mechanical audit and explicit applicability checks for the fixed V4 demo."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from spindle import dynamics_model as model
from utils.runout_labels import complete_turn_runout
from scripts.run_clearance_v3 import write_json,write_csv


def validate_config(config):
    p=config['parameters'];s=config['simulation']
    if p['model_dof']!=12 or len(p['healthy_supports'])!=1 or p['healthy_supports'][0]['z_m']!=-0.12:
        raise ValueError('V4 requires two front outer rings and the unchanged R1 only')
    if p['measurement_kind']!='rotor_section' or p['probe_reference']!='ground':
        raise ValueError('This V4 contract uses rotor M relative fixed ground')
    expected=model.front_group_parameters(p,p['front_group_clearance_um'])
    if any(p[k]!=expected[k] for k in ('fit_clearance_um','fit2_clearance_um')):
        raise ValueError('Derived clearances disagree with the common input')
    if s['method']!='DOP853' or s['fs_raw_hz']!=65536 or s['fs_out_hz']!=8192 or s['retained_turns']!=120 or s['subwindows']!=4:
        raise ValueError('Unsupported numerical/export settings')
    for k in ('Kb2','Cb2'):
        a=np.asarray(p[k])
        if not np.allclose(a,a.T) or np.min(np.linalg.eigvalsh(a))<0:raise ValueError(k+' must be symmetric positive semidefinite')
    if config['simulation']['guard_s']<config['processing']['required_guard_s']:
        raise ValueError('True processing guards too short')
    force=config['excitation']['U_kg_m']*(p['rpm']*2*np.pi/60)**2
    if not np.isclose(force,config['excitation']['peak_force_N'],rtol=1e-12):raise ValueError('Excitation unit mismatch')


def domain_check(t,q,config):
    p=config['parameters'];d=config['constitutive_domains']
    baseline,_=model.equilibrium_continuation(model.front_group_parameters(p,0))
    rear=p['healthy_supports'][0]
    b2=model.rotor_section(q,p['z_f2'])-q[:,10:12]
    b20=model.rotor_section(baseline,p['z_f2'])-baseline[10:12]
    r=model.rotor_section(q,rear['z_m'])-q[:,4:6]
    r0=model.rotor_section(baseline,rear['z_m'])-baseline[4:6]
    b1=model.rotor_section(q,p['z_f'])-q[:,6:8]
    angles=model.cage_speed(p)*t[:,None]+p['cage_phase']+np.arange(p['balls'])*2*np.pi/p['balls']
    approach=p['normal_overlap']+np.cos(p['contact_angle'])*(np.cos(angles)*b1[:,0,None]+np.sin(angles)*b1[:,1,None])
    values=dict(max_tilt_rad=float(np.linalg.norm(q[:,8:10],axis=1).max()),
        f2_excursion_m=float(np.linalg.norm(b2-b20,axis=1).max()),rear_excursion_m=float(np.linalg.norm(r-r0,axis=1).max()),
        f1_min_approach_m=float(approach.min()),f1_max_approach_m=float(approach.max()))
    checks=dict(finite=bool(np.isfinite(q).all()),tilt=values['max_tilt_rad']<=d['maximum_tilt_rad'],
        f2=values['f2_excursion_m']<=d['f2_internal_excursion_from_baseline_m'],
        rear=values['rear_excursion_m']<=d['rear_excursion_from_baseline_m'],
        f1=values['f1_min_approach_m']>=d['f1_min_normal_approach_m'] and values['f1_max_approach_m']<=d['f1_max_normal_approach_m'],
        gap=config['accepted_gap_range_um'][0]<=p['front_group_clearance_um']<=config['accepted_gap_range_um'][1])
    return dict(passed=all(checks.values()),checks=checks,values=values,status='assumed_demo; no machine-domain validation')


def energy(q,v,p):
    m,g=model.mass_and_gravity(p)
    value=.5*np.sum(m*v*v,axis=-1)-q@g
    value+=.5*p['kh']*np.sum(q[...,4:6]**2,axis=-1)
    link=model.rotor_section(q,p['z_T'])-q[...,2:4]
    value+=.5*model.area_and_link(p,0)[1]*np.sum(link*link,axis=-1)
    rear=p['healthy_supports'][0];r=model.rotor_section(q,rear['z_m'])-q[...,4:6]
    value+=.5*np.einsum('...i,ij,...j->...',r,np.array(rear['K']),r)
    r=model.rotor_section(q,p['z_f2'])-q[...,10:12]
    value+=.5*np.einsum('...i,ij,...j->...',r,np.array(p['Kb2']),r)
    phi=p['cage_phase']+np.arange(p['balls'])*2*np.pi/p['balls']
    r=model.rotor_section(q,p['z_f'])-q[...,6:8]
    delta=p['normal_overlap']+np.cos(p['contact_angle'])*(r@np.array([np.cos(phi),np.sin(phi)]))
    keq=(p['ki']**(-2/3)+p['ko']**(-2/3))**(-1.5)
    value+=.4*keq*np.sum(np.maximum(delta,0)**2.5,axis=-1)
    phi=np.arange(p['N'])*2*np.pi/p['N'];directions=np.array([np.cos(phi),np.sin(phi)])
    for index,k,c in ((6,p['Kf'],p['fit_clearance_um']),(10,p['Kf2'],p['fit2_clearance_um'])):
        delta=np.maximum((q[...,index:index+2]-q[...,4:6])@directions-c*1e-6,0)
        value+=.5*4*k/p['N']*np.sum(delta*delta,axis=-1)
    return value


def audit(config,folder,source_before=Path('experiments/clearance_v4_20261023/source_before')):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);validate_config(config)
    p=config['parameters'];checks={};details={}
    masses,g=model.mass_and_gravity(p)
    checks['dimension_positive_mass']=len(masses)==12 and bool(np.all(masses>0))
    total=sum(p[k] for k in ('ms','mt','mh','mO','mO2'))
    checks['mass_conservation']=abs(total-config['allocation_v4']['old_total_mass_kg'])<1e-12
    old=json.loads(Path('config/clearance_v3_config.json').read_text(encoding='utf-8'))['parameters']
    checks['inertias_unchanged']=p['Jt']==old['Jt'] and p['Jp']==old['Jp']
    checks['rear_unchanged']=p['healthy_supports'][0]==old['healthy_supports'][1]
    checks['no_old_F2_bypass']=len(p['healthy_supports'])==1 and 'rear' in p['healthy_supports'][0]['name']
    k=np.linalg.inv(np.linalg.inv(p['Kb2'])+np.eye(2)/p['Kf2'])
    mismatch=float(np.linalg.norm(k-np.array(config['allocation_v4']['old_F2_K']))/np.linalg.norm(k))
    checks['F2_condensation']=mismatch<=.001;details['condensation_relative_error']=mismatch
    rng=np.random.default_rng(4);qr=rng.normal(size=4);f=rng.normal(size=2);z=.09
    a=model.section_matrix(z)
    checks['virtual_work']=abs(f@(a@qr)-(a.T@f)@qr)<1e-12
    checks['moment_sign']=bool(np.allclose((a.T@f)[2:],[-z*f[1],z*f[0]]))
    v=rng.normal(size=12);gyro=model.gyro_matrix(p)
    checks['gyro_skew_zero_power']=bool(np.allclose(gyro+gyro.T,0) and abs(v@gyro@v)<1e-12)
    q,_=model.equilibrium_continuation(p)
    residual=model.internal_force(0,q,v,p,0)-model.internal_force(0,q,v,dict(p,Jp=0),0)
    checks['gyro_exactly_once']=bool(np.allclose(residual,gyro@v,atol=1e-8))
    for name,kf,df in (('F1',p['Kf'],p['Df']),('F2',p['Kf2'],p['Df2'])):
        fp=dict(N=p['N'],Kf=kf,Df=df,fit_clearance_um=2)
        force,diag=model.fit_contact_force(np.array([1e-6,0]),np.array([10.,0]),fp)
        checks[name+'_inside_zero']=bool(np.all(force==0))
        _,diag=model.fit_contact_force(np.array([3e-6,0]),np.array([-10.,0]),fp)
        checks[name+'_no_tension']=bool(np.all(diag['normal_force_N']>=0))
        r=np.array([1.2e-6,-.7e-6]);force,_=model.fit_contact_force(r,np.zeros(2),dict(fp,Df=0,fit_clearance_um=0))
        checks[name+'_zero_gap_elastic']=bool(np.allclose(force,kf*r,rtol=1e-12))
    qtest=np.zeros(12);qtest[10]=3e-6
    b2,fit2,_=model.front2_forces(qtest,np.zeros(12),model.front_group_parameters(p,2))
    f1,_=model.fit_contact_force(qtest[6:8],np.zeros(2),dict(p,fit_clearance_um=2))
    checks['independent_interfaces']=bool(np.linalg.norm(fit2)>0 and np.linalg.norm(f1)==0)
    rows=[];static=[]
    omega=p['rpm']*2*np.pi/60;U=config['excitation']['U_kg_m'];F=U*omega**2
    forcephasor=F*np.array([1,-1j]) # Re(-j*exp(j*omega*t)) = sin(omega*t)
    for gap in config['clearances_um']:
        local=model.front_group_parameters(p,gap);q,res=model.equilibrium_continuation(local)
        M,C,K=model.linear_matrices(local,0,q);fc=model.force_components(0,q,np.zeros(12),local)
        eigen=eigh((K+K.T)/2,M,eigvals_only=True)
        poles=np.linalg.eigvals(np.block([[np.zeros((12,12)),np.eye(12)],[-np.linalg.solve(M,K),-np.linalg.solve(M,C)]]))
        response=np.linalg.solve(K-omega**2*M+1j*omega*C,model.generalized_point_force(forcephasor,p['z_u'],12))
        measured=model.section_matrix(p['z_m'])@response[model.ROTOR]
        E=float(2e6*np.max(abs(measured)))
        row=dict(gap_um=gap,residual_equivalent_N=res,minimum_frequency_Hz=float(np.sqrt(max(0,eigen.min()))/2/np.pi),
            maximum_frequency_Hz=float(np.sqrt(eigen.max())/2/np.pi),max_pole_real_s=float(poles.real.max()),
            frozen_linear_E_um=E,force_needed_20um_N=F*20/E,
            F1_fit_N=float(np.linalg.norm(fc['fit_N'])),F2_fit_N=float(np.linalg.norm(fc['fit2_N'])),
            rear_N=float(np.linalg.norm(fc['healthy_support_forces_N'])))
        rows.append(row);static.append(dict(gap_um=gap,q=q.tolist(),forces={k:val.tolist() for k,val in fc.items()},
            domain=domain_check(np.array([0.]),q[None,:],dict(config,parameters=local))))
    checks['static_residuals']=max(r['residual_equivalent_N'] for r in rows)<=1e-6
    # Frozen V3 source must reproduce mass, restoring force and tangent exactly.
    path=Path(source_before)/'src/spindle/dynamics_model.py'
    spec=importlib.util.spec_from_file_location('v3_before',path);previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    qold,_=previous.equilibrium_continuation(old);vold=rng.normal(size=10)*1e-4
    checks['V3_force_regression']=bool(np.array_equal(previous.internal_force(.123,qold,vold,old,0),model.internal_force(.123,qold,vold,old,0)))
    checks['V3_mass_regression']=bool(np.array_equal(previous.mass_and_gravity(old)[0],model.mass_and_gravity(old)[0]))
    checks['V3_tangent_regression']=all(np.array_equal(a,b) for a,b in zip(previous.linear_matrices(old,0,qold),model.linear_matrices(old,0,qold)))
    t=np.arange(20001)/100000;xy=np.column_stack((5*np.cos(2*np.pi*100*t)+123,5*np.sin(2*np.pi*100*t)-77))
    pp=complete_turn_runout(t,xy,6000,0,.2)['peak_to_peak_um']
    checks['circle_and_static_offset']=bool(np.allclose(pp,10,atol=1e-9))
    # Stop spin to isolate passive mechanics, including gravitational potential.
    ep=dict(p,rpm=0);qe,_=model.equilibrium_continuation(ep);qi=qe.copy();qi[0]+=1e-7;qi[8]+=1e-6
    result=model.simulate(ep,0,[],[],duration=.03,warmup=0,fs=65536,max_step=config['simulation']['max_step_s'],
        state_initial=np.r_[qi,np.zeros(12)],q_initial=qe,unbalance=(0,0))
    e=energy(result['q'],result['v'],ep);excess=e-energy(qe,np.zeros(12),ep)
    maximum_gain=float(np.diff(e).max());limit=max(1e-12,float(abs(excess[0]))*1e-7)
    checks['passive_energy']=maximum_gain<=limit and excess[-1]<excess[0]
    details.update(total_mass_kg=total,force_N=F,U_g_mm=U*1e6,force_moment_peak_Nm=abs(p['z_u']*F),
        energy_initial_excess_J=float(excess[0]),energy_final_excess_J=float(excess[-1]),energy_max_step_gain_J=maximum_gain,
        energy_gain_tolerance_J=limit,rear_only_rotor_rank=int(np.linalg.matrix_rank(model.section_matrix(-.12).T@np.array(p['healthy_supports'][0]['K'])@model.section_matrix(-.12))))
    np.savez_compressed(folder/'passive_energy.npz',t=result['t'],q=result['q'],v=result['v'],energy_J=e,excess_energy_J=excess)
    ep=model.front_group_parameters(ep,5.5);qe,_=model.equilibrium_continuation(ep)
    qi=qe.copy();qi[0]+=1e-7;qi[9]+=1e-6
    gradient=[]
    for i in range(12):
        step=1e-9 if i in (8,9) else 1e-10;dq=np.zeros(12);dq[i]=step
        gradient.append((energy(qi+dq,np.zeros(12),ep)-energy(qi-dq,np.zeros(12),ep))/(2*step))
    error=np.array(gradient)-(model.internal_force(0,qi,np.zeros(12),ep,0)-model.mass_and_gravity(ep)[1])
    error[8:10]/=p['L_ref']
    checks['energy_gradient_all_12_coordinates']=max(abs(error))<1e-5
    details['energy_gradient_max_equivalent_N']=float(max(abs(error)))
    result=model.simulate(ep,0,[],[],duration=.03,warmup=0,fs=65536,max_step=config['simulation']['max_step_s'],
        state_initial=np.r_[qi,np.zeros(12)],q_initial=qe,unbalance=(0,0))
    e=energy(result['q'],result['v'],ep);excess=e-energy(qe,np.zeros(12),ep)
    checks['transition_energy']=max(np.diff(e))<=1e-12 and excess[-1]<excess[0]
    details['transition_energy_max_gain_J']=float(max(np.diff(e)))
    np.savez_compressed(folder/'passive_energy_transition.npz',t=result['t'],q=result['q'],v=result['v'],energy_J=e,excess_energy_J=excess)
    checks={k:bool(value) for k,value in checks.items()}
    write_json(folder/'mechanics.json',dict(passed=all(checks.values()),checks=checks,details=details))
    write_json(folder/'static_equilibria.json',static);write_csv(folder/'preflight.csv',rows)
    print(json.dumps(dict(passed=all(checks.values()),checks=checks,details=details)),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/clearance_v4_config.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-before',type=Path,default=Path('experiments/clearance_v4_20261023/source_before'))
    args=parser.parse_args();audit(json.loads(args.config.read_text(encoding='utf-8')),args.output,args.source_before)
