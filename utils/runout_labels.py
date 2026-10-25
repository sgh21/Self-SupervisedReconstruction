"""Per-revolution peak-to-peak motion from the simulated position states."""
import numpy as np


def complete_turn_runout(time_s, position_um, rpm=None, start_s=None, end_s=None,
                         phase_at_zero_rad=0.0, angle_rad=None):
    """Use both interpolated phase boundaries and every native interior sample.

    ``position_um`` is [sample, coordinate], without bandpass or integration.
    Turns count in the direction of rotation, even when rpm is negative.
    An optional strictly monotone unwrapped ``angle_rad`` supports known
    variable-speed trajectories. Partial revolutions at either end are omitted.
    Labels are sampled-state
    extrema, not an analytic search of the continuous ODE dense solution.
    """
    time_s = np.asarray(time_s, dtype=float)
    position_um = np.asarray(position_um, dtype=float)
    if (time_s.ndim != 1 or len(time_s) < 2
            or position_um.shape[0] != len(time_s) or position_um.ndim != 2
            or np.any(np.diff(time_s) <= 0)):
        raise ValueError('Need increasing sample times and [sample,coordinate] positions')
    start_s = float(time_s[0] if start_s is None else start_s)
    end_s = float(time_s[-1] if end_s is None else end_s)
    if start_s < time_s[0]-1e-12 or end_s > time_s[-1]+1e-12 or end_s <= start_s:
        raise ValueError('Requested interval must lie inside the recorded states')
    if angle_rad is None:
        if rpm is None or rpm == 0:
            raise ValueError('Supply nonzero constant rpm or known monotone unwrapped angles')
        rate = abs(float(rpm))/60
        phase_cycles = np.sign(rpm)*phase_at_zero_rad/(2*np.pi)
        first = int(np.ceil(start_s*rate+phase_cycles-1e-10))
        last = int(np.floor(end_s*rate+phase_cycles+1e-10))
        edges = (np.arange(first, last+1)-phase_cycles)/rate
    else:
        angle_rad = np.asarray(angle_rad, dtype=float)
        if angle_rad.shape != time_s.shape:
            raise ValueError('Angle history must match position sample times')
        cycles = np.sign(angle_rad[-1]-angle_rad[0])*angle_rad/(2*np.pi)
        if np.any(np.diff(cycles) <= 0):
            raise ValueError('Unwrapped angle must be strictly monotone')
        first = int(np.ceil(np.interp(start_s, time_s, cycles)-1e-10))
        last = int(np.floor(np.interp(end_s, time_s, cycles)+1e-10))
        edges = np.interp(np.arange(first, last+1), cycles, time_s)
    turn_id = np.arange(first, last, dtype=np.int64)
    boundary_positions = np.column_stack([
        np.interp(edges, time_s, position_um[:, axis]) for axis in range(position_um.shape[1])
    ])
    minimum, maximum = [], []
    for index, (begin, finish) in enumerate(zip(edges[:-1], edges[1:])):
        left = np.searchsorted(time_s, begin, side='right')
        right = np.searchsorted(time_s, finish, side='left')
        boundary = boundary_positions[index:index+2]
        values = np.concatenate((boundary[:1], position_um[left:right], boundary[1:]))
        minimum.append(values.min(axis=0))
        maximum.append(values.max(axis=0))
    shape = (len(turn_id), position_um.shape[1])
    minimum = np.asarray(minimum).reshape(shape)
    maximum = np.asarray(maximum).reshape(shape)
    return dict(turn_id=turn_id, start_time_s=edges[:-1], end_time_s=edges[1:],
                min_um=minimum, max_um=maximum, peak_to_peak_um=maximum-minimum)


def turns_in_window(turns, start_s, end_s):
    """Indices of complete turns contained in one velocity window."""
    return np.flatnonzero((turns['start_time_s'] >= start_s-1e-12)
                          & (turns['end_time_s'] <= end_s+1e-12))
