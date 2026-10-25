"""Reproducible analytic checks of complete-turn labels (no mechanical fitting)."""
import argparse
from pathlib import Path
import numpy as np

from utils.data_io import save_json
from utils.runout_labels import complete_turn_runout, turns_in_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fs, rpm = 65536, -7000.0
    time = np.arange(round(.137*fs)+1)/fs+.0021
    rate = abs(rpm)/60
    phase = .37
    position = np.column_stack((4+2*np.cos(2*np.pi*rate*time+phase),
                                -5+3*np.sin(2*np.pi*rate*time+phase)))
    turns = complete_turn_runout(time, position, rpm, phase_at_zero_rad=.19)
    error = float(np.max(np.abs(turns['peak_to_peak_um']-[4, 6])))
    assert error < 1e-4
    assert np.all(turns['start_time_s'] >= time[0]-1e-12)
    assert np.all(turns['end_time_s'] <= time[-1]+1e-12)
    assert np.allclose(turns['end_time_s']-turns['start_time_s'], 1/rate)
    # Linear motion has extrema at the interpolated phase boundaries. A floor
    # or integer-samples-per-revolution implementation fails this exact check.
    linear = np.column_stack((2*time+8, -3*time-20))
    linear_turns = complete_turn_runout(time, linear, rpm, phase_at_zero_rad=.19)
    linear_error = float(np.max(np.abs(linear_turns['peak_to_peak_um']-[2/rate, 3/rate])))
    assert linear_error < 1e-12
    reverse = complete_turn_runout(time, position, -rpm, phase_at_zero_rad=-.19)
    assert np.array_equal(turns['peak_to_peak_um'], reverse['peak_to_peak_um'])
    begin, end = .02, .08
    selected = turns_in_window(turns, begin, end)
    assert np.all(turns['start_time_s'][selected] >= begin)
    assert np.all(turns['end_time_s'][selected] <= end)
    stationary = complete_turn_runout(time, np.full((len(time), 2), 8.), rpm)
    assert np.count_nonzero(stationary['peak_to_peak_um']) == 0
    short = complete_turn_runout(time[:20], position[:20], rpm)
    assert len(short['turn_id']) == 0
    variable_angle = -2*np.pi*(90*time+75*time**2)
    circle = np.column_stack((2*np.cos(variable_angle), 2*np.sin(variable_angle)))
    variable = complete_turn_runout(time, circle, angle_rad=variable_angle)
    variable_error = float(np.max(np.abs(variable['peak_to_peak_um']-4)))
    assert variable_error < 1e-4
    assert np.ptp(variable['end_time_s']-variable['start_time_s']) > 1e-4
    result = dict(passed=True, sample_rate_hz=fs, rpm=rpm,
        sinusoid_peak_to_peak_expected_um=[4, 6], sinusoid_maximum_error_um=error,
        linear_boundary_maximum_error_um=linear_error, complete_turn_count=len(turns['turn_id']),
        variable_speed_circle_maximum_error_um=variable_error,
        variable_speed_turn_count=len(variable['turn_id']),
        window_turn_count=len(selected), checks=['amplitude and physical units',
        'noninteger samples per turn', 'interpolated boundaries', 'partial-turn exclusion',
        'negative rpm direction', 'window-to-turn containment', 'stationary centre',
        'sub-turn record gives no label', 'variable-speed circular trajectory'])
    save_json(args.output/'runout_checks.json', result)
    print(result)


if __name__ == '__main__':
    main()
