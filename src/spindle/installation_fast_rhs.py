"""Installation adapter around the compiled 12/16-DOF bearing evaluator."""

import numpy as np
from numba import njit

from .fast_rhs import prepare_fast_rhs as prepare_v8_rhs


@njit(cache=True)
def append_installation(state, base_result, masses, gravity, k, c, k0, c0):
    n = len(masses)
    nb = n - 4
    result = np.empty(2 * n)
    result[:n] = state[n:]
    result[n : n + nb] = base_result[nb:]
    q = state[:n]
    v = state[n:]
    f_hc = k[0] * (q[4:6] - q[nb : nb + 2]) + c[0] * (v[4:6] - v[nb : nb + 2])
    f_cp = k[1] * (q[nb : nb + 2] - q[nb + 2 : n]) + c[1] * (v[nb : nb + 2] - v[nb + 2 : n])
    f_pg = k[2] * q[nb + 2 : n] + c[2] * v[nb + 2 : n]
    result[n + 4 : n + 6] += (k0 * q[4:6] + c0 * v[4:6] - f_hc) / masses[4:6]
    result[n + nb : n + nb + 2] = (gravity[nb : nb + 2] + f_hc - f_cp) / masses[nb : nb + 2]
    result[n + nb + 2 : 2 * n] = (gravity[nb + 2 : n] + f_cp - f_pg) / masses[nb + 2 : n]
    return result


def prepare_fast_rhs(model, unbalance=None):
    base = prepare_v8_rhs(
        model.base_parameters, model.unbalance if unbalance is None else unbalance
    )
    args = (
        model.masses,
        model.gravity,
        model.link_stiffness,
        model.link_damping,
        model.base_parameters["kh"],
        model.base_parameters["ch"],
    )
    nb, n = model.base_mechanical_dof, model.structural_dof

    def rhs(t, state):
        base_state = np.empty(2 * nb)
        base_state[:nb] = state[:nb]
        base_state[nb:] = state[n : n + nb]
        return append_installation(state, base(t, base_state), *args)

    return rhs
