"""Joint 1X amplitude/structure fit with a CAD mass budget and platform Y.

The fit identifies an equivalent scenario, not a unique set of machine constants.
July 25 is evaluated only after selection using July 22--24.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.signal import firwin, freqz

from spindle.dynamics_model import (
    default_parameters, area_and_link, harmonic_response, modes_and_stability,
    equilibrium, cage_speed, ball_spin_speed,
)
from utils.data_io import save_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/platform_faults_20261022'
CONFIG = ROOT/'config/platform_faults_20261022.json'
FR = 7000/60
OMEGA = 2*np.pi*FR
G = 9.80665
PROCESS = dict(fs_hz=8192, band_hz=[10, 2048], fir_taps=4097,
               kaiser_beta=8.6, guard_s=.5)
NAMES = ['equivalent_preload_N', 'bearing_damping_time_s',
         'housing_supported_frequency_hz', 'kbend_N_m',
         'housing_branch_zeta', 'link_branch_zeta',
         'platform_static_sag_mm', 'platform_branch_zeta']
# Exploratory bounds, NOT manufacturer pad specifications or confidence intervals.
BOUNDS = np.array([[100,1900], [2.5e-6,2.5e-5], [50,200], [1e7,4e7],
                   [.01,.06], [.01,.06], [.01,5], [.01,.3]])
MASSES = dict(ms=2.5, mt=.604, mh=42.113, platform_mass=440.768)
TOTAL_MASS = sum(MASSES.values())
TAPS = firwin(PROCESS['fir_taps'], PROCESS['band_hz'], pass_zero=False,
              fs=PROCESS['fs_hz'], window=('kaiser', PROCESS['kaiser_beta']))
GAIN = float(abs(freqz(TAPS, worN=[FR], fs=PROCESS['fs_hz'])[1][0]))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def parameters(z, phase_sign=1):
    values = np.exp(np.log(BOUNDS[:,0])+np.asarray(z)*np.log(BOUNDS[:,1]/BOUNDS[:,0]))
    preload, tau, fh, kbend, zh, zl, sag, zp = values
    p = default_parameters()
    p.update(MASSES)
    spindle_mass = p['ms']+p['mt']+p['mh']
    p.update(kappa=1.09e12, kh=spindle_mass*(2*np.pi*fh)**2,
             kbend=kbend, rpm=-phase_sign*7000., outer_pit_area_m2=0.,
             inner_pit_area_m2=0., ball_deformation_amplitude_m=0.)
    keq = (p['ki']**(-2/3)+p['ko']**(-2/3))**(-1.5)
    p['normal_overlap'] = (preload/(p['balls']*keq*np.sin(p['contact_angle'])))**(2/3)
    kb = .75*p['balls']*keq*np.sqrt(p['normal_overlap'])*np.cos(p['contact_angle'])**2
    p['cb_total'] = tau*kb
    p['ch'] = 2*zh*np.sqrt(p['kh']*spindle_mass)
    _, kl = area_and_link(p, 0.)
    p['cl'] = 2*zl*np.sqrt(kl*p['ms']*p['mt']/(p['ms']+p['mt']))
    p['platform_ky'] = TOTAL_MASS*G/(sag*.001)
    p['platform_cy'] = 2*zp*np.sqrt(p['platform_ky']*TOTAL_MASS)
    p['machine_model'] = 'JD130S-24-HA40; conditional 7-DOF CAD-mass scenario'
    p['parameter_status'] = 'single-1X equivalent amplitude fit; masses and support model remain conditional'
    return p, dict(zip(NAMES, values.tolist()))


def run_weights(runs, indices):
    days = sorted({runs[i]['day'] for i in indices})
    return np.array([1/(len(days)*sum(runs[j]['day']==runs[i]['day'] for j in indices))
                     for i in indices])


def summary(predicted, target, runs, indices):
    weights = run_weights(runs, indices)
    errors = 20*np.log10(predicted[None]/target[indices])
    return dict(run_count=len(indices), rms_db=float(np.sqrt(np.sum(weights[:,None]*errors**2)/3)),
                channel_rms_db=np.sqrt(np.sum(weights[:,None]*errors**2, axis=0)).tolist(),
                target_geometric_mean=np.exp(np.sum(weights[:,None]*np.log(target[indices]),axis=0)).tolist(),
                errors_db_per_run=errors.tolist(), run_ids=[runs[i]['run_id'] for i in indices])


def fault_cases(p):
    area, _ = area_and_link(p, 0.)
    cases = [dict(case_id='healthy', fault_type='healthy', severity=0., label='Healthy',
                  lost_area=0., parameter_overrides={})]
    for fraction in (.1,.3,.5):
        cases.append(dict(case_id=f'taper_{int(fraction*100):02d}', fault_type='taper',
                          severity=fraction, label=f'Taper area loss {fraction:.0%}',
                          lost_area=area*fraction, parameter_overrides={}))
    for race in ('outer','inner'):
        for mm2 in (.2,.4,.8):
            override = {f'{race}_pit_area_m2':mm2*1e-6,
                        f'{race}_pit_width_m':.001, f'{race}_pit_depth_m':.0001,
                        f'{race}_pit_center_rad':1.5*np.pi}
            cases.append(dict(case_id=f'{race}_{mm2:.1f}'.replace('.','p'), fault_type=race,
                              severity=mm2, label=f'{race.title()} pit {mm2:g} mm2',
                              lost_area=0., parameter_overrides=override))
    for um in (.1,.2,.4):
        cases.append(dict(case_id=f'ball_{um:.1f}'.replace('.','p'), fault_type='ball',
                          severity=um, label=f'Ball waviness {um:g} um', lost_area=0.,
                          parameter_overrides=dict(ball_deformation_amplitude_m=um*1e-6,
                              ball_deformation_index=0, ball_deformation_phase_rad=0.)))
    return cases


def search(output=OUT, config_path=CONFIG, generations=200):
    output = Path(output)
    runs = read(output/'manifest.json')['runs']
    with np.load(output/'references.npz') as data:
        target = data['target_acceleration_rms']
        phasors = data['velocity_1x_phasor']
    train = [i for i,r in enumerate(runs) if r['split']=='train']
    validation = [i for i,r in enumerate(runs) if r['split']=='validation']
    rw = run_weights(runs,train)
    weights = rw[:,None]/3
    observed_relative = phasors[train,:,1]/phasors[train,:,0]
    relative_phase = observed_relative/abs(observed_relative)
    direction_scores = {s:float(np.sum(rw*np.mean(1-np.real(relative_phase*np.exp(-1j*s*np.pi/2)),axis=1)))
                        for s in (-1,1)}
    phase_sign = min(direction_scores, key=direction_scores.get)
    direction = np.array([1.,1j*phase_sign])
    history = []

    def unit_response(p):
        q = harmonic_response(p, 0., [FR], direction[None])[0,[4,5,6]]
        return -OMEGA**2*q*GAIN

    def evaluate(z, record=True):
        p, coordinates = parameters(z,phase_sign)
        response = unit_response(p)
        unit_rms = abs(response)/np.sqrt(2)
        force = float(np.exp(np.sum(weights*np.log(target[train]/unit_rms[None]))))
        predicted = force*unit_rms
        residual = (20*np.log10(predicted[None]/target[train])*np.sqrt(weights)).ravel()
        if record:
            history.append([*np.asarray(z),force,float(residual@residual)])
        return residual,p,coordinates,force,predicted,response

    def objective(z):
        r = evaluate(z)[0]
        return float(r@r)

    definition = dict(names=NAMES, bounds=BOUNDS.tolist(), coordinates='normalized logarithmic',
        objective='Date-balanced, run-balanced, channel-equal squared dB residual of three 1X RMS amplitudes.',
        excitation='Analytic positive force minimizer recomputed for EVERY structural candidate; equal XY magnitudes, quadrature.',
        direction_selection='Training XY relative phase selects +90 or -90; sensor-coordinate convention, no measured mechanical keyphase.',
        direction_scores={str(k*90):v for k,v in direction_scores.items()},
        phase='Relative channel phases are an independent adequacy diagnostic, not part of the amplitude objective.',
        bounds_status='Exploratory engineering ranges, not manufacturer specifications or identified intervals.',
        mass_fixed=True, gravity='Fixed g=9.80665; loaded equilibrium recomputed for each candidate. No acceleration-DC target.',
        selection='Within 0.02 dB RMS of minimum training loss, choose smallest distance to log-prior centre; validation is excluded.',
        reason_for_tiebreak='Single-frequency underdetermination; avoid reporting an arbitrary extreme optimizer point as unique truth.',
        near_tolerance_rms_db=.02, seeds=[11,29,47], generations=generations,
        housing_frequency='kh=(ms+mt+mh)*(2*pi*fh)^2; prior updated for CAD scenario mass.',
        platform_sag='ky=Mtotal*g/(sag_mm/1000); equivalent ground support, not measured rubber compression.')
    save_json(output/'search_definition.json',definition)
    fits = []
    for seed in definition['seeds']:
        started = time.perf_counter()
        de = differential_evolution(objective, [(0,1)]*len(NAMES), seed=seed,
            popsize=6,maxiter=generations,tol=1e-8,polish=False,x0=np.full(len(NAMES),.5))
        local = least_squares(lambda z:evaluate(z)[0],de.x,bounds=(0,1),diff_step=1e-3,
                              ftol=1e-9,xtol=1e-8,gtol=1e-7,max_nfev=120)
        best = min([de.x,local.x],key=objective)
        fits.append(dict(seed=seed,z=best.tolist(),objective_db2=objective(best),
                         seconds=time.perf_counter()-started,de_message=str(de.message),
                         local_message=str(local.message)))
        print('Search seed complete:',fits[-1],flush=True)
    # Include the prior centre explicitly; all scenarios use their own jointly fitted force.
    objective(np.full(len(NAMES),.5))
    hist = np.asarray(history)
    minimum = float(np.sqrt(hist[:,-1].min()))
    near = hist[np.sqrt(hist[:,-1])<=minimum+.02]
    selected = near[np.argmin(np.sum((near[:,:len(NAMES)]-.5)**2,axis=1)),:len(NAMES)]

    def describe(z):
        residual,p,coordinates,force,predicted,response = evaluate(z,False)
        # Common phase from one TRAINING segment only; affects no magnitude fitting.
        observed = 1j*OMEGA*phasors[train[0],0]/1000
        typical = np.exp(np.sum(rw[:,None]*np.log(target[train]),axis=0))
        phase = float(np.angle(np.vdot(force*response/typical,observed/typical)))
        q0,eqres = equilibrium(p,0.)
        return dict(z=np.asarray(z).tolist(),parameters=p,coordinates=coordinates,
            excitation=dict(frequency_hz=FR,peak_force_N=force,phase_rad=phase,
                            y_phase_sign=phase_sign,U_g_mm=force/OMEGA**2*1e6,
                            formula='Fx=A*cos(w*t+phi), Fy=-y_phase_sign*A*sin(w*t+phi)'),
            predicted_acceleration_rms_1x=predicted.tolist(),
            predicted_relative_phase_deg=np.rad2deg(np.angle(response[1:]/response[0])).tolist(),
            train_summary=summary(predicted,target,runs,train),
            validation_summary=summary(predicted,target,runs,validation),
            modes=modes_and_stability(p,0.),equilibrium_m=q0.tolist(),equilibrium_residual_N=float(eqres),
            gravity_N=TOTAL_MASS*G,platform_ground_static_deflection_mm=-q0[6]*1000,
            boundary_coordinates=[NAMES[i] for i,v in enumerate(z) if min(v,1-v)<.01])

    fitted = describe(selected)
    baseline = describe(np.full(len(NAMES),.5))
    # Phase diagnostic is useful precisely because a three-amplitude fit can hide a wrong load path.
    phase_diagnostics = {}
    response = evaluate(selected,False)[-1]
    for label,indices in [('train',train),('validation',validation)]:
        w = run_weights(runs,indices)
        channels = []
        for j in (1,2):
            measured = phasors[indices,:,j]/phasors[indices,:,0]
            measured /= abs(measured)
            mean = np.sum(w*np.mean(measured,axis=1))
            predicted = response[j]/response[0]
            difference = np.angle(measured*np.exp(-1j*np.angle(predicted)))
            channels.append(dict(channel=['ACC2','ACC1','ACC3'][j],reference='ACC2',
                measured_mean_deg=float(np.angle(mean,deg=True)),concentration=float(abs(mean)),
                predicted_deg=float(np.angle(predicted,deg=True)),
                circular_error_rms_deg=float(np.sqrt(np.sum(w*np.mean(np.rad2deg(difference)**2,axis=1))))))
        phase_diagnostics[label] = channels
    fitted.update(baseline=baseline,seeds=fits,search_definition=definition,
                  minimum_training_rms_db=minimum,phase_diagnostics=phase_diagnostics,
                  phase_reference=dict(run_id=runs[train[0]]['run_id'],segment=0),
                  processing=PROCESS)
    save_json(output/'fit.json',fitted)
    np.savez_compressed(output/'search_history.npz',evaluations=hist,
                        columns=np.asarray(NAMES+['force_N','objective_db2']))
    # Three amplitude observations place a strict ceiling of rank 3 on nine unknowns.
    jacobian = []
    for i in range(len(NAMES)):
        low,high = selected.copy(),selected.copy()
        low[i],high[i] = max(0,low[i]-.002),min(1,high[i]+.002)
        y = [20*np.log10(abs(unit_response(parameters(z,phase_sign)[0]))) for z in (low,high)]
        jacobian.append((y[1]-y[0])/(high[i]-low[i]))
    jacobian = np.column_stack([*jacobian,np.full(3,20.)])
    singular = np.linalg.svd(jacobian,compute_uv=False)
    save_json(output/'identifiability.json',dict(unknown_count=9,amplitude_observation_count=3,
        columns=NAMES+['log10_force_N'],jacobian=jacobian.tolist(),singular_values=singular.tolist(),
        derivative_coordinates='First eight columns differentiate dB amplitude against normalized log coordinates z in [0,1], not the physical units in the column labels; last column is log10(force_N).',
        rank_at_relative_1e_3=int(np.sum(singular>singular[0]*1e-3)),
        near_optimal_evaluations=len(near),evaluations=len(hist),
        near_optimal_force_N=[float(near[:,-2].min()),float(near[:,-2].max())],
        near_optimal_z_ranges=np.column_stack([near[:,:8].min(axis=0),near[:,:8].max(axis=0)]).tolist(),
        interpretation='Repeated runs estimate variability; they do not supply new excitation frequencies. Ranges are not confidence intervals.'))
    p = fitted['parameters']
    mass_budget = dict(source='experiments/cad_review_20261022/assessment.json',
        table_kg=405.008,clamps_kg=24.161,other_known_platform_attached_kg=11.599,
        platform_kg=440.768,spindle_cached_total_kg=44.613,rotor_prior_kg=2.5,
        nonrotating_spindle_residual_kg=42.113,holder_kg=.604,total_known_cad_scenario_kg=TOTAL_MASS,
        status='Saved CAD properties, not newly integrated or weighed masses; no duplicated assembly/component masses.',
        limitations=['Imported translation-stage CAD mass is missing; omitted from this known-mass scenario, not physically massless.',
            'Spindle outer-envelope mass has unspecified material; rotor/stator split remains an assumed 2.5/42.113 kg.',
            'Housing node now represents the whole nonrotating spindle lump, not a measured local effective modal mass.',
            'Source assembly cache exceeds unique component sum by 0.910 kg; not silently distributed.',
            'All counted attachments treated as rigidly table-mounted; loading mechanism assumed disengaged.',
            'Four feet and joints represented by one Y spring/damper; no rocking or local table flexure.'])
    cfg = dict(schema_version=1,description='Frozen CAD-mass single-1X calibration for controlled fault generation.',
        parameters=p,excitation=fitted['excitation'],mass_budget=mass_budget,
        processing=PROCESS,simulation=dict(native_fs_hz=65536,warmup_s=1.,duration_s=3.,
            max_step_s=1/8192,pit_steps_per_passage=4),
        cases=fault_cases(p),channel_order=['ACC2','ACC1','ACC3'],
        channel_meanings=['front housing X','front housing Y','platform Y'],
        characteristic_frequencies_hz=dict(rotation=FR,cage=abs(cage_speed(p))/(2*np.pi),
            outer_pass=abs(cage_speed(p))*p['balls']/(2*np.pi),
            inner_pass=abs(cage_speed(p)-p['rpm']*2*np.pi/60)*p['balls']/(2*np.pi),
            ball_spin=abs(ball_spin_speed(p))/(2*np.pi),ball_waviness_2x=2*abs(ball_spin_speed(p))/(2*np.pi)),
        provenance=dict(manifest=str(output/'manifest.json'),fit=str(output/'fit.json'),
            search_definition=str(output/'search_definition.json'),frozen_before_fault_generation=True),
        limitations=['Only 1X amplitudes calibrated; broadband noise, other drives and phase mismatch are not removed.',
            'Bearing geometry is a borrowed model assumption, not verified JD130S bearing geometry.',
            'Fault severity is a parameter, not a guaranteed monotonic accuracy grade.',
            'T is a lumped holder centre; relative-platform motion is not a validated ISO error-motion measurement.',
            'No real fault reference exists here; comparisons use nominally healthy real data.',
            'These 13 deterministic cases are a controlled comparison set, not an independent training dataset.'])
    save_json(Path(config_path),cfg)
    save_json(output/'frozen_config.json',cfg)
    print('Frozen config:',config_path,flush=True)
    print('Selected:',json.dumps({k:fitted[k] for k in ('excitation','train_summary','validation_summary','phase_diagnostics')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--generations',type=int,default=200)
    args = parser.parse_args()
    search(args.output,args.config,args.generations)
