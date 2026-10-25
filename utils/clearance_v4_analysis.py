"""V4 full-window metrics, using the V3 turn-boundary implementation."""
import numpy as np
from scipy.signal import periodogram
from utils.clearance_v3_analysis import _window, _comparison_keys, _tolerance
from utils.pitting_analysis import filter_vibration


def keys(metrics):
    return _comparison_keys(metrics) + [k for k in metrics if k.startswith('fit2_')]


def tolerance(key, reference, config, scale):
    original = key.replace('fit2_', 'fit_')
    value = _tolerance(original, reference, config, scale, steady=True)
    if key.startswith('H_a_'):
        value = max(config['numerical_thresholds']['acceleration_m_s2'], .01*abs(reference))
    return value


def absolute_floor(key, config):
    t = config['numerical_thresholds']
    if key.startswith('H_v_'): return t['velocity_mm_s']
    if key.startswith('H_a_'): return t['acceleration_m_s2']
    if 'contact_fraction' in key: return t['contact_rate']
    if 'impulse' in key: return t['contact_impulse_N_s']
    if key.startswith(('fit_', 'fit2_')): return t['contact_peak_N']
    return t['position_um']


def compare(base, fine, config):
    checks = {}
    for key in keys(fine):
        delta = abs(base[key]-fine[key])
        limit = tolerance(key, fine[key], config, fine['E_dir_um'])
        checks[key] = dict(difference=delta, tolerance=limit, passed=bool(delta <= limit))
    same = all(abs(base[k]-fine[k]) < 1e-10 for k in ('measurement_start_s','measurement_end_s'))
    return dict(passed=bool(same and all(x['passed'] for x in checks.values())),
                same_window=same, metrics=checks)


def analyze(result, config, start, end, stride=1):
    s, p = config['simulation'], config['parameters']
    t, q = result['t'][::stride], result['q'][::stride]
    fs = s['fs_raw_hz']/stride
    if start < t[0]+config['processing']['required_guard_s'] or end > t[-1]-config['processing']['required_guard_s']:
        raise ValueError('Insufficient true guards for FIR and antialias processing')
    processing = dict(config['processing'], band_hz=config['processing']['passband_hz'])
    hv = filter_vibration(result['v'][::stride,4:6].T, fs, processing).T*1000
    ha = filter_vibration(result['acc'][::stride,4:6].T, fs, processing).T
    tout = t[0]+np.arange(len(hv))/s['fs_out_hz']
    output = tout, hv, ha
    rel = result['measured_relative_xy_m'][::stride]
    raw1 = t,q,rel,np.linalg.norm(result['fit_force_N'][::stride],axis=1),result['fit_active_count'][::stride]
    q2 = q.copy(); q2[:,6:8] = q[:,10:12]
    raw2 = t,q2,rel,np.linalg.norm(result['fit2_force_N'][::stride],axis=1),result['fit2_active_count'][::stride]

    def window(left, right):
        m, turns = _window(raw1, output, config, left, right)
        m2, turns2 = _window(raw2, output, config, left, right)
        m.update({k.replace('fit_','fit2_'):v for k,v in m2.items() if k.startswith('fit_')})
        turns.update({k.replace('fit_','fit2_'):v for k,v in turns2.items() if k.startswith('fit_')})
        # Time-weight contact occupancy, including exact revolution boundaries.
        for number, raw in enumerate((raw1,raw2),1):
            rates=[]
            for a,b in zip(turns['start_time_s'], turns['end_time_s']):
                mask=(t>a)&(t<b); tt=np.r_[a,t[mask],b]
                occupancy=(raw[4]>0).astype(float)
                rates.append(np.trapezoid(np.interp(tt,t,occupancy),tt)/(b-a))
            prefix='fit_' if number==1 else 'fit2_'
            m[prefix+'contact_fraction']=float(np.mean(rates))
            turns[prefix+'contact_fraction']=np.array(rates)
        use=(tout>=left-1e-12)&(tout<right-1e-12)
        angle=p['rpm']*2*np.pi/60*tout[use]+config['excitation']['phase_rad']
        design=np.column_stack([np.ones(len(angle))]+[f(k*angle) for k in (1,2,3) for f in (np.cos,np.sin)])
        for signal,prefix,unit in ((hv,'H_v','mm_s'),(ha,'H_a','m_s2')):
            coef=np.linalg.lstsq(design,signal[use],rcond=None)[0]
            for k in (2,3):
                for axis,val in zip('xy',np.linalg.norm(coef[2*k-1:2*k+1],axis=0)):
                    m[f'{prefix}_{k}x_{axis}_{unit}']=float(val)
            freq,psd=periodogram(signal[use],fs=s['fs_out_hz'],window='hann',axis=0,scaling='density')
            for lo,hi in ((10,900),(900,2048)):
                energy=psd[(freq>=lo)&(freq<hi)].sum(axis=0)*(freq[1]-freq[0])
                for axis,val in zip('xy',energy): m[f'{prefix}_energy_{lo}_{hi}_{axis}']=float(val)
        return m,turns

    metrics, turns=window(start,end)
    edges=np.linspace(start,end,s['subwindows']+1)
    windows=[window(a,b)[0] for a,b in zip(edges[:-1],edges[1:])]
    checks={}
    for key in keys(metrics):
        values=np.array([m[key] for m in windows])
        limit=tolerance(key,max(abs(values)),config,metrics['E_dir_um'])
        checks[key]=dict(values=values.tolist(),difference=float(np.ptp(values)),tolerance=limit,
                         passed=bool(np.ptp(values)<=limit))
    drift={}
    for key in metrics:
        if '_energy_' not in key: continue
        values=np.array([m[key] for m in windows])
        floor=(config['numerical_thresholds']['velocity_mm_s'] if key.startswith('H_v') else config['numerical_thresholds']['acceleration_m_s2'])**2
        significant=abs(values[-1]-values[0])>max(floor,.05*max(values))
        monotonic=bool(np.all(np.diff(values)>0) or np.all(np.diff(values)<0))
        drift[key]=dict(values=values.tolist(),passed=not(significant and monotonic))
    counts=[m['complete_turns'] for m in windows]
    steady=dict(passed=bool(all(x['passed'] for x in checks.values()) and all(x['passed'] for x in drift.values()) and counts==[30]*4 and metrics['complete_turns']==120),
                window_edges_s=edges.tolist(),subwindow_metrics=windows,checks=checks,band_drift_checks=drift,
                subwindow_complete_turns=counts)
    metrics.update(gap_um=p['front_group_clearance_um'],steady_pass=steady['passed'])
    use=(tout>=start-1e-12)&(tout<end-1e-12)
    return metrics,turns,(tout[use],hv[use],ha[use]),steady
