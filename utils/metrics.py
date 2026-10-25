"""Comparison metrics shared by simulation and cross-run validation."""
import numpy as np
from scipy.signal import welch
from .data_io import bandlimit


def comparison_metrics(model_velocity,measured,fs,measured_fs):
    model_velocity=bandlimit(model_velocity.T,fs)
    model_rms=np.sqrt(np.mean(model_velocity**2,axis=-1))
    measured_rms=np.sqrt(np.mean(measured**2,axis=(0,2)))
    n=model_velocity.shape[-1]
    f,pm=welch(model_velocity,fs,nperseg=n)
    fd,pd=welch(measured,measured_fs,nperseg=int(round(n/fs*measured_fs)),axis=-1)
    pd=pd.mean(axis=0)
    keep=(f>=10)&(f<=900)
    matched=np.asarray([np.interp(f[keep],fd,x) for x in pd])
    log_errors=np.sqrt(np.mean((10*np.log10(np.maximum(pm[:,keep],1e-25))
                               -10*np.log10(np.maximum(matched,1e-25)))**2,axis=1))
    segment=measured[0,:,:int(round(n/fs*measured_fs))]
    return dict(model_rms_mm_s=(model_rms*1000).tolist(),
                measured_rms_mm_s=(measured_rms*1000).tolist(),
                relative_rms_error=(model_rms/measured_rms-1).tolist(),
                log_psd_rmse_db=log_errors.tolist(),
                first_displayed_holdout_segment_rms_mm_s=(np.sqrt(np.mean(segment**2,axis=-1))*1000).tolist(),
                exact_time_phase_match_claimed=False)

