"""Read user's existing package without modifying original data or application."""
import io
import csv
import json
import zipfile
from pathlib import Path
import h5py
import numpy as np

DEFAULT_PACKAGE=Path('D:/WorkSpace/ZMProject/SoftwareTools_PyQt/data/processed/spindle_sample_packages/20260723/run_20260723_093018_7000rpm_20min_0723_morning1.zip')


def load_velocity_record(package=DEFAULT_PACKAGE):
    """All already-processed valid stable windows; no calibration or score selection."""
    with zipfile.ZipFile(package) as z:
        manifest=json.loads(z.read('manifest.json'))
        run=manifest['runs'][0]
        meta=json.loads(z.read(run['source_metadata_file']))
        with h5py.File(io.BytesIO(z.read(run['data_file'])),'r') as h:
            g=h['windows_10s']
            starts=g['start_time_s'][:]
            valid=np.asarray(g['sample_valid'][:],dtype=bool)
            # ACC2 X, ACC1 Y; ACC3 table Y is NOT a third radial direction.
            velocity=g['velocity'][:,:2,:][:,[1,0],:]*1e-3
            fs=int(h.attrs['velocity_sample_rate_hz'])
            unit=str(h.attrs['velocity_unit'])
            temperatures=g['temperature'][:]
            temperature_valid=g['temperature_valid'][:]
        rows=list(csv.DictReader(io.StringIO(z.read(run['telemetry_file']).decode('utf-8-sig'))))
    if fs!=2048 or unit!='mm/s':raise ValueError('Review preprocessing/units before using this record')
    included=np.flatnonzero(valid)
    velocity=velocity[valid];starts=starts[valid]
    rpms=[float(r['actual_speed_rpm']) for r in rows
          if r.get('speed_ok','true').lower() in ('true','1') and r.get('actual_speed_rpm')
          and float(starts[0])<=float(r['time_s'])<=float(starts[-1]+10)]
    info=dict(package=str(package),run=run,preprocessing=manifest['preprocessing'],
              source_metadata=meta,fs_hz=fs,read_unit=unit,model_unit='m/s',
              channel_mapping=['ACC2 front housing X','ACC1 front housing Y'],
              valid_source_window_indices=included.tolist(),
              excluded_invalid_window_count=int(np.sum(~valid)),
              stable_window_count=len(starts),stable_start_s=float(starts[0]),
              stable_end_s=float(starts[-1]+10),
              temperature_ranges_C={name:[float(np.min(temperatures[:,j,:][temperature_valid[:,j,:]])),
                                          float(np.max(temperatures[:,j,:][temperature_valid[:,j,:]]))]
                                    for j,name in enumerate(['NTC1','RTD1'])},
              actual_speed_median_rpm=float(np.median(rpms)),
              holder_displacement_available=False,keyphase_available=False)
    return velocity,starts,info


def load_health(package=DEFAULT_PACKAGE):
    velocity,starts,info=load_velocity_record(package)
    train=starts<950
    validate=starts>=970
    info.update(
              train_window_indices=np.flatnonzero(train).tolist(),
              validation_window_indices=np.flatnonzero(validate).tolist(),
              train_start_s=float(starts[train][0]),train_end_s=float(starts[train][-1]+10),
              validation_start_s=float(starts[validate][0]),validation_end_s=float(starts[validate][-1]+10),
              split='nonoverlapping chronological holdout within one acquisition; 20 s gap')
    return velocity[train],velocity[validate],info


def bandlimit(signal,fs,low=10,high=900):
    """Same hard FFT analysis band as existing velocity data; last axis=time."""
    f=np.fft.rfftfreq(signal.shape[-1],1/fs)
    spectrum=np.fft.rfft(signal,axis=-1)
    spectrum[..., (f<low)|(f>high)]=0
    return np.fft.irfft(spectrum,n=signal.shape[-1],axis=-1)


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def capacitance_reference():
    """Reuse existing measured analysis; different date/rpm, NOT paired validation."""
    root=Path('D:/WorkSpace/ZMProject/LK-HDataCollector')
    table=root/'analysis/results/csh1_0920/tables/dynamic_recordings.csv'
    cache=root/'analysis/results/csh1_0920/rotation_0920_6000rpm1.npz'
    if not table.is_file() or not cache.is_file():return None,None
    with table.open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['day']=='0920' and r['file']=='6000rpm1']
    with np.load(cache) as d:
        contour=d['cycles'].mean(axis=0)
    info=dict(source_csv=str(root/'data/CSH1/0920/night/6000rpm1.csv'),
              existing_analysis_table=str(table),existing_cache=str(cache),
              sampling_hz=3906.25,unit='um',nominal_rpm=6000,date='2026-09-20',
              mapping='X=Distance2; Y=-Distance1',mechanical_keyphase=False,
              original_samples_per_turn=float(rows[0]['samples_per_turn']),
              display_interpolation_points=512,synchronous_pp_um={r['axis']:float(r['sync_pp_um']) for r in rows},
              recomputed_cached_pp_um=np.ptp(contour,axis=0).tolist(),
              quantity='probe-relative synchronous contour; surface form and center motion not separated',
              synchronized_with_vibration=False,used_to_calibrate_7000rpm_model=False)
    return info,contour
