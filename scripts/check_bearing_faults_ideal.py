"""Regression, kinematics and convergence checks; no measured input is opened."""
import argparse
import json
from pathlib import Path

import numpy as np

from spindle.dynamics_model import (default_parameters, bearing_force, internal_force,
    cage_speed, ball_spin_speed, inner_race_pit_geometry, inner_race_pit_displacement,
    inner_race_passage_boundaries, ball_deformation_displacement, simulate, harmonic_response)
from scripts.generate_bearing_faults_ideal import (ROOT, cases, characteristic_frequencies,
                                                  run_case, processed, spectrum)
from utils.data_io import save_json


def relative(x,y):
    return float(np.linalg.norm(x-y)/np.linalg.norm(y))


def mechanics(out,cfg):
    entries=list(cases(cfg))
    pit=next(entry[3] for entry in entries if entry[1]=='inner')
    g=inner_race_pit_geometry(pit)
    omega=pit['rpm']*2*np.pi/60
    phi=g['center_rad']+g['span_rad']*np.array([-.6,-.5,0,.5,.6])
    drop=inner_race_pit_displacement(0,phi,pit)
    np.testing.assert_allclose(drop,[0,0,g['maximum_drop_m'],0,0],atol=1e-16,rtol=0)
    for t in (.001,.013,.1):
        np.testing.assert_allclose(inner_race_pit_displacement(t,phi+omega*t,pit),drop,atol=1e-16,rtol=0)
    # Compare actual event intervals to independently computed BPFI, including reverse rotation.
    rates=[]
    for rpm in (7000.,-7000.):
        p=dict(pit,rpm=rpm)
        ends=inner_race_passage_boundaries(p,.05)
        dwell=g['span_rad']/abs(cage_speed(p)-rpm*2*np.pi/60)
        intervals=np.diff(ends)[1:-1]
        expected=abs(rpm)/60*p['balls']/2*(1+p['ball_diameter']/p['pitch_diameter']*np.cos(p['contact_angle']))
        assert np.all(np.isclose(intervals,dwell,rtol=1e-8) | np.isclose(intervals,1/expected-dwell,rtol=1e-8))
        middles=(ends[:-1]+ends[1:])/2
        hits=[]
        for t in middles:
            angle=cage_speed(p)*t+p['cage_phase']+2*np.pi*np.arange(p['balls'])/p['balls']
            if np.sum(inner_race_pit_displacement(t,angle,p)) > .5*g['maximum_drop_m']:
                hits.append(t)
        measured=1/np.mean(np.diff(hits)[1:-1])
        assert abs(measured/expected-1)<1e-9
        rates.append(dict(rpm=rpm,event_rate_hz=measured,expected_hz=expected,dwell_s=dwell))
    ball=entries[-1][3]
    period=np.pi/ball_spin_speed(ball)
    amplitude=ball['ball_deformation_amplitude_m']
    loss=ball_deformation_displacement(0,ball)
    assert np.count_nonzero(loss)==1
    np.testing.assert_allclose(loss[0],2*amplitude,atol=1e-18)
    np.testing.assert_allclose(ball_deformation_displacement(period/2,ball),-loss,atol=1e-18)
    np.testing.assert_allclose(ball_deformation_displacement(period,ball),loss,atol=1e-18)
    # Fault forces must obey action/reaction; zero severity must restore healthy force.
    p0=default_parameters()
    q,v=np.arange(1,7)*1e-6,np.arange(1,7)*1e-4
    balance=[]
    for p in (pit,ball):
        r=internal_force(.013,q,v,p,0).reshape(3,2)
        error=np.linalg.norm(r.sum(0)-p['kh']*q[4:]-p['ch']*v[4:])
        assert error<1e-10
        balance.append(float(error))
        zero=dict(p,inner_pit_area_m2=0.,ball_deformation_amplitude_m=0.)
        np.testing.assert_array_equal(bearing_force(.013,q[:2]-q[4:],v[:2]-v[4:],zero),
                                      bearing_force(.013,q[:2]-q[4:],v[:2]-v[4:],p0))
        no_contact=dict(p,normal_overlap=-1e-3)
        np.testing.assert_array_equal(bearing_force(.01,np.zeros(2),np.ones(2),no_contact),[0,0])
    checks=dict(inner_geometry_and_rotation_passed=True,inner_passage_checks=rates,
                ball_second_order_and_signed_offset_passed=True,zero_severity_passed=True,
                no_tension_passed=True,reaction_balance_N=balance,regression=[])
    snapshot=out/'evidence/dynamics_before.py'
    if snapshot.exists():
        old={}
        exec(compile(snapshot.read_text(encoding='utf-8'),str(snapshot),'exec'),old)
        for name,p in [('healthy',p0),('outer',dict(p0,outer_pit_area_m2=.4e-6,
                            outer_pit_width_m=.001,outer_pit_depth_m=.0001,outer_pit_center_rad=1.2))]:
            args=(p,0.,np.array([120.]),np.array([[1.,-1j]]))
            before=old['simulate'](*args,duration=.012,warmup=.006,fs=32768)
            after=simulate(*args,duration=.012,warmup=.006,fs=32768)
            errors={k:relative(after[k],before[k]) for k in ('q','v','acc')}
            assert max(errors.values())<1e-10
            checks['regression'].append(dict(case=name,relative_l2=errors))
    else:
        checks['regression_note']='Historical snapshot absent; no pre-change regression claimed.'
    save_json(out/'mechanics_checks.json',checks)
    print(json.dumps(checks),flush=True)
    return checks


def convergence(out,cfg):
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    rows=summary['cases']
    healthy=rows[0]
    with np.load(out/'healthy.npz') as d:
        t=d['display_time_s']
        actual=2*np.mean(d['display_q']*np.exp(-2j*np.pi*cfg['drive']['frequency_hz']*t[:,None]),axis=0)
    target=harmonic_response(healthy['parameters'],0,[cfg['drive']['frequency_hz']],
                             cfg['drive']['amplitude_N']*np.array([[1.,-1j]]))[0]
    linear_error=relative(actual,target)
    assert linear_error<.02
    results=dict(drive_frequency_hz=cfg['drive']['frequency_hz'],
                 healthy_drive_displacement_vs_linear_relative_l2=linear_error,convergence=[],spectral_checks=[])
    outer=[r for r in rows if r['kind']=='outer']
    inner=[r for r in rows if r['kind']=='inner']
    ball=[r for r in rows if r['kind']=='ball']
    # Independent expected positions from kinematics, measured from generated output.
    for row in outer[-1:]+[inner[-1],ball[-1]]:
        fr=characteristic_frequencies(row['parameters'])
        if row['kind']=='outer':
            expected=[fr['bpfo_hz'],2*fr['bpfo_hz']]
        elif row['kind']=='inner':
            expected=[fr['bpfi_hz']+n*fr['shaft_hz'] for n in (-2,-1,1)]
        else:
            expected=[fr['twice_bsf_hz']+n*fr['cage_hz'] for n in (-1,1)]
        with np.load(out/f"{row['case']}.npz") as d:
            f,a=spectrum(d['display_acc'][:,5],cfg['simulation']['display_fs_hz'])
        spacing=f[1]-f[0]
        for target_hz in expected:
            candidates=np.flatnonzero(abs(f-target_hz)<=2*spacing)
            peak=candidates[np.argmax(a[candidates])]
            error=float(abs(f[peak]-target_hz))
            assert error<=spacing/2+1e-9
            results['spectral_checks'].append(dict(case=row['case'],expected_hz=target_hz,
                peak_bin_hz=float(f[peak]),error_hz=error,amplitude_m_s2=float(a[peak])))
    representatives={r['case']:r for r in outer[-1:]+[inner[0],inner[-1],ball[-1]]}
    for row in representatives.values():
        print('Refining '+row['case'],flush=True)
        raw=run_case(row['parameters'],cfg,duration=.0625,refinement=2)
        fine=processed(raw,cfg,source_fs=2*cfg['simulation']['native_fs_hz'],duration=.0625)
        with np.load(out/f"{row['case']}.npz") as d:
            coarse={k:d[f'display_{k}'][:len(fine[k])] for k in fine}
        errors={}
        for node,sl in [('H',slice(4,6)),('T',slice(2,4))]:
            for k in ('q','v','acc'):
                x,y=coarse[k][:,sl],fine[k][:,sl]
                if k=='q':
                    x,y=x-x.mean(0),y-y.mean(0)
                errors[f'{node}_{k}_relative_l2']=relative(x,y)
        f,a=spectrum(coarse['acc'][:,4:],cfg['simulation']['display_fs_hz'])
        _,b=spectrum(fine['acc'][:,4:],cfg['simulation']['display_fs_hz'])
        errors['H_acc_spectrum_below_6000hz_relative_l2']=relative(a[f<=6000],b[f<=6000])
        passed=all(value<(.02 if 'acc' in key else .005) for key,value in errors.items())
        item=dict(case=row['case'],compared_s=.0625,errors=errors,passed=passed)
        results['convergence'].append(item)
        np.savez_compressed(out/f"refined_{row['case']}.npz",
                            **{f'fine_{k}':v for k,v in fine.items()},**{f'coarse_{k}':v for k,v in coarse.items()})
        save_json(out/'verification.json',results)
        print(json.dumps(item),flush=True)
    results['passed']=all(r['passed'] for r in results['convergence'])
    save_json(out/'verification.json',results)
    assert results['passed'],'Convergence threshold exceeded; inspect before drawing physical conclusions.'
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--mechanics-only',action='store_true')
    args=parser.parse_args()
    config=args.results/'config.json'
    if not config.exists():
        config=ROOT/'config/bearing_faults_ideal.json'
    cfg=json.loads(config.read_text(encoding='utf-8'))
    mechanics(args.results,cfg)
    if not args.mechanics_only:
        convergence(args.results,cfg)


if __name__=='__main__':
    main()
