"""Rigid H-C-P installation chain around the V8 bearing system.

State order: Sxy, Txy, Hxy, O1xy, theta_xy, O2xy, Cxy, Pxy.
The only grounded translational connection is P-G. Local bearing constraint
moments and the rotor gyroscopic term stay inside the base operator. The
independent rear-ring branch inserts OR1xy/OR2xy before Cxy/Pxy (20 DOF).
"""

from __future__ import annotations
from copy import deepcopy
from types import SimpleNamespace
import time
import numpy as np
from scipy.integrate import solve_ivp
from . import dynamics_model as v8


class IntegrationFailure(RuntimeError):
    """A failed solve with the accepted states that existed before termination."""

    def __init__(self, message, partial_native, diagnostics):
        super().__init__(message)
        self.partial_native = partial_native
        self.diagnostics = diagnostics


H = slice(4, 6)
C = slice(12, 14)
P = slice(14, 16)
TRANSLATION_PAIRS = (0, 2, 4, 6, 10, 12, 14)
STATE_NAMES = (
    "Sx",
    "Sy",
    "Tx",
    "Ty",
    "Hx",
    "Hy",
    "O1x",
    "O1y",
    "theta_x",
    "theta_y",
    "O2x",
    "O2y",
    "Cx",
    "Cy",
    "Px",
    "Py",
)


def assemble_mount_matrix(diagonal, base_dof=12):
    """Assemble the chain; the default preserves the legacy 16-DOF API."""
    result = np.zeros((base_dof + 4, base_dof + 4))
    for values, left, right in zip(
        diagonal, (4, base_dof, base_dof + 2), (base_dof, base_dof + 2, None)
    ):
        for axis in range(2):
            i = left + axis
            result[i, i] += values[axis]
            if right is not None:
                j = right + axis
                result[j, j] += values[axis]
                result[i, j] -= values[axis]
                result[j, i] -= values[axis]
    return result


class InstallationModel:
    """One immutable mechanical parameter instance; build anew for each candidate."""

    def __init__(self, config):
        self.config = deepcopy(config)
        self.base_parameters = deepcopy(config["base_v8"]["parameters"])
        p = self.base_parameters
        for key in tuple(p):
            if key.startswith("_"):
                p.pop(key)
        if (p["bearing_contact_model"], p["bearing_layout_name"]) != ("relative_tilt_v8", "DB_DB"):
            raise ValueError("Installation supports only relative_tilt_v8 DB/DB bearings")
        self.base_mechanical_dof = v8.base_mechanical_dof(p)
        if self.base_mechanical_dof not in (12, 16):
            raise ValueError("Installation requires the 12- or 16-DOF mechanical branch")
        self.structural_dof = self.base_mechanical_dof + 4
        self.state_names = (*v8.base_state_names(p), "Cx", "Cy", "Px", "Py")
        self.state_index = {name: index for index, name in enumerate(self.state_names)}
        self.C = slice(self.state_index["Cx"], self.state_index["Cy"] + 1)
        self.P = slice(self.state_index["Px"], self.state_index["Py"] + 1)
        self.sensor_indices = [self.state_index[name] for name in ("Hx", "Hy", "Py")]
        self.base_translation_pairs = v8.base_translation_pairs(p)
        self.translation_pairs = (*self.base_translation_pairs, self.C.start, self.P.start)
        self.units_by_state = tuple(
            ("rad" if name.startswith("theta_") else "m" for name in self.state_names)
        )
        p["_derived_cache_enabled"] = True
        phi = 2 * np.pi * np.arange(p["N"]) / p["N"]
        p["_fit_directions"] = np.column_stack((np.cos(phi), np.sin(phi)))
        self.unbalance = (config["excitation"]["U_kg_m"], config["excitation"]["phase_rad"])
        inst = config["installation"]
        e = inst["parameters"]
        mc, mp = (inst["mC_kg"], inst["mP_kg"])
        mass8, gravity8 = v8.mass_and_gravity(p)
        self.masses = np.r_[mass8, mc, mc, mp, mp]
        g = np.asarray(p["gravity_radial_m_s2"])
        self.gravity = np.r_[gravity8, mc * g, mp * g]
        self.total_moving_mass = sum((self.masses[i] for i in self.translation_pairs))
        nominal_stiffness = [[e["kHC_N_m"]] * 2, [e["kCP_N_m"]] * 2, [e["kPGx_N_m"], e["kPGy_N_m"]]]
        self.link_stiffness = np.asarray(
            inst.get("actual_link_stiffness_N_m", nominal_stiffness), dtype=float
        )
        mu = np.array([p["mh"] * mc / (p["mh"] + mc), mc * mp / (mc + mp), self.total_moving_mass])
        zeta = np.array([e["zeta_link"], e["zeta_link"], e["zeta_foot"]])
        nominal_damping = 2 * zeta[:, None] * np.sqrt(self.link_stiffness * mu[:, None])
        self.link_damping = np.asarray(
            inst.get("actual_link_damping_N_s_m", nominal_damping), dtype=float
        )
        if self.link_stiffness.shape != (3, 2) or self.link_damping.shape != (3, 2):
            raise ValueError("Actual installation K/C must contain three XY pairs")
        if np.any(self.link_stiffness <= 0) or np.any(self.link_damping < 0):
            raise ValueError("Installation stiffness must be positive and damping nonnegative")
        self.equivalent_damping_ratios = self.link_damping / (
            2 * np.sqrt(self.link_stiffness * mu[:, None])
        )
        self.mount_parameterization = (
            "actual_K_C" if "actual_link_damping_N_s_m" in inst else "stiffness_damping_ratio"
        )
        self.K_mount = assemble_mount_matrix(self.link_stiffness, self.base_mechanical_dof)
        self.C_mount = assemble_mount_matrix(self.link_damping, self.base_mechanical_dof)
        self.load_map = np.zeros((self.structural_dof, 2))
        self.load_map[v8.ROTOR] = v8.section_matrix(p["z_u"]).T
        self._base_equilibrium = None
        self._fast = None

    def link_forces(self, q, v):
        q, v = (np.asarray(q), np.asarray(v))
        C, P = (self.C, self.P)
        relative = np.stack((q[..., H] - q[..., C], q[..., C] - q[..., P], q[..., P]), axis=-2)
        velocity = np.stack((v[..., H] - v[..., C], v[..., C] - v[..., P], v[..., P]), axis=-2)
        force = relative * self.link_stiffness + velocity * self.link_damping
        return dict(
            f_HC_N=force[..., 0, :],
            f_CP_N=force[..., 1, :],
            f_PG_N=force[..., 2, :],
            relative_m=relative,
            relative_velocity_m_s=velocity,
            potential_J=0.5 * np.sum(self.link_stiffness * relative**2, axis=(-2, -1)),
            dissipated_power_W=np.sum(self.link_damping * velocity**2, axis=(-2, -1)),
        )

    def internal_force(self, t, q, v):
        p = self.base_parameters
        n = self.base_mechanical_dof
        restoring = np.zeros(self.structural_dof)
        restoring[:n] = v8.internal_force(t, q[:n], v[:n], p, 0.0)
        restoring[H] -= p["kh"] * q[H] + p["ch"] * v[H]
        return restoring + self.K_mount @ q + self.C_mount @ v

    def forcing(self, t):
        omega = self.base_parameters["rpm"] * 2 * np.pi / 60
        angle = omega * t + self.unbalance[1]
        return self.load_map @ (
            self.unbalance[0] * omega**2 * np.array([np.cos(angle), np.sin(angle)])
        )

    def rhs(self, t, state, backend="numba"):
        if backend == "numba":
            if self._fast is None:
                from .installation_fast_rhs import prepare_fast_rhs

                self._fast = prepare_fast_rhs(self)
            return self._fast(t, state)
        if backend != "python":
            raise ValueError("backend must be python or numba")
        q, v = (state[: self.structural_dof], state[self.structural_dof :])
        return np.r_[
            v, (self.gravity + self.forcing(t) - self.internal_force(t, q, v)) / self.masses
        ]

    def equilibrium(self, initial_q=None):
        """Exact common-translation construction from V8 frozen static equilibrium.

        Each original internal relative displacement and angle is retained;
        the three chain deflections are set by the actual cumulative gravity.
        The full assembled residual is checked, including rotor moments.
        """
        if self._base_equilibrium is None:
            self._base_equilibrium = v8.equilibrium_continuation(self.base_parameters)[0]
        old = self._base_equilibrium
        C, P = (self.C, self.P)
        f_hc = sum((self.gravity[i : i + 2] for i in self.base_translation_pairs), np.zeros(2))
        f_cp = f_hc + self.gravity[C]
        f_pg = f_cp + self.gravity[P]
        up = f_pg / self.link_stiffness[2]
        uc = up + f_cp / self.link_stiffness[1]
        uh = uc + f_hc / self.link_stiffness[0]
        q = np.r_[old.copy(), uc, up]
        for i in self.base_translation_pairs:
            q[i : i + 2] += uh - old[H]
        residual = self.internal_force(0.0, q, np.zeros(self.structural_dof)) - self.gravity
        if np.max(abs(np.delete(residual, [8, 9]))) > 1e-05 or np.max(abs(residual[8:10])) > 1e-06:
            raise RuntimeError(f"Installation equilibrium residual force/moment: {residual}")
        equivalent = residual.copy()
        equivalent[8:10] /= self.base_parameters["L_ref"]
        return (q, float(np.linalg.norm(equivalent)))

    def observations(self, q, v, acc=None):
        q, v = (np.asarray(q), np.asarray(v))
        C, P = (self.C, self.P)
        section = v8.rotor_section(q, self.base_parameters["z_m"])
        reference = self.config.get("probe_reference", "platform")
        references = dict(ground=np.zeros_like(section), housing=q[..., H], platform=q[..., P])
        result = dict(
            measured_section_xy_m=section,
            probe_reference_xy_m=references[reference],
            measured_relative_xy_m=section - references[reference],
            r_ground_m=section,
            r_housing_m=section - q[..., H],
            r_platform_m=section - q[..., P],
            platform_xy_m=q[..., P],
            H_velocity_m_s=v[..., H],
            C_velocity_m_s=v[..., C],
            P_velocity_m_s=v[..., P],
            sensor_velocity_m_s=v[..., self.sensor_indices],
        )
        if acc is not None:
            result.update(
                H_acceleration_m_s2=acc[..., H],
                C_acceleration_m_s2=acc[..., C],
                P_acceleration_m_s2=acc[..., P],
                sensor_acceleration_m_s2=acc[..., self.sensor_indices],
            )
        return result

    def simulate(
        self,
        duration,
        warmup=0.0,
        fs=65536,
        max_step=None,
        rtol=None,
        atol=None,
        q_initial=None,
        state_initial=None,
        t_start=0.0,
        method="DOP853",
        backend="numba",
        deadline=None,
    ):
        settings = self.config["simulation"]
        n = self.structural_dof
        nb = self.base_mechanical_dof
        max_step = settings["max_step_s"] if max_step is None else max_step
        rtol = settings["rtol"] if rtol is None else rtol
        if atol is None:
            aq = np.full(n, settings["atol_translation_m"])
            av = np.full(n, settings["atol_velocity_m_s"])
            aq[8:10] = settings["atol_angle_rad"]
            av[8:10] = settings["atol_angle_rate_rad_s"]
            atol = np.r_[aq, av]
        q0, residual = self.equilibrium() if q_initial is None else (np.asarray(q_initial), None)
        state = (
            np.r_[q0, np.zeros(n)] if state_initial is None else np.asarray(state_initial).copy()
        )
        calls = 0

        def rhs(t, value):
            nonlocal calls
            calls += 1
            if deadline is not None and calls % 256 == 0 and (time.perf_counter() > deadline):
                raise TimeoutError("V10 integration deadline reached")
            return self.rhs(t, value, backend)

        accepted = []
        accepted_times = []

        def failed_solve(solution, stage, previous_t=None, previous_states=None):
            partial_t, partial_states = solution.t, solution.y.T
            if previous_t is not None:
                partial_t = np.concatenate((previous_t, partial_t[1:]))
                partial_states = np.concatenate((previous_states, partial_states[1:]))
            steps = np.concatenate([*accepted, np.diff(solution.t)])
            partial = dict(t_s=partial_t, q=partial_states[:, :n], v=partial_states[:, n:],
                           q_equilibrium=q0, state_final=partial_states[-1],
                           state_names=np.asarray(self.state_names), sensor_indices=np.asarray(self.sensor_indices),
                           structural_dof=np.asarray(n), base_mechanical_dof=np.asarray(nb),
                           integration_accepted_steps_s=steps, absolute_start_s=np.asarray(t_start),
                           integration_accepted_times_s=np.concatenate(([t_start], *accepted_times, solution.t[1:])),
                           solver_success=np.asarray(False))
            diagnostics = dict(success=False, status=solution.status, message=str(solution.message),
                               stage=stage, last_accepted_time_s=float(solution.t[-1]),
                               retained_partial_samples=len(partial_t), nfev=calls,
                               method=method, rtol=rtol, atol=np.asarray(atol).tolist(), max_step_s=max_step,
                               sampling="accepted solver times in failed segment; earlier retained samples if available",
                               partial_is_formal_complete=False)
            return IntegrationFailure(str(solution.message), partial, diagnostics)

        def integrate(span, initial_state, stage, previous_t=None, previous_states=None, dense=False):
            try:
                solution = solve_ivp(rhs, span, initial_state, method=method, rtol=rtol,
                                     atol=atol, max_step=max_step, dense_output=dense)
            except Exception as error:
                # SciPy supplies no segment result when the RHS itself raises.
                # Preserve the last completed checkpoint, never a rejected trial state.
                checkpoint = SimpleNamespace(t=np.asarray([span[0]]), y=initial_state[:, None],
                                             status=None, message=str(error))
                failure = failed_solve(checkpoint, stage, previous_t, previous_states)
                failure.diagnostics.update(exception_type=type(error).__name__,
                    failed_segment_accepted_states_available=False,
                    sampling="last completed checkpoint and earlier retained samples only")
                raise failure from error
            if not solution.success:
                raise failed_solve(solution, stage, previous_t, previous_states)
            return solution

        if warmup > 0:
            initial = integrate((t_start, t_start + warmup), state, "warmup")
            state = initial.y[:, -1]
            accepted.append(np.diff(initial.t))
            accepted_times.append(initial.t[1:])
            del initial
        begin = t_start + warmup
        times = begin + np.arange(int(round(duration * fs)) + 1) / fs
        states = np.empty((len(times), 2 * n))
        states[0] = state
        block_samples = max(1, int(round(0.125 * fs)))
        left = 0
        segments = 0
        while left < len(times) - 1:
            right = min(len(times) - 1, left + block_samples)
            solution = integrate((times[left], times[right]), state, "retained",
                                 times[:left + 1], states[:left + 1], dense=True)
            states[left + 1 : right + 1] = solution.sol(times[left + 1 : right + 1]).T
            accepted.append(np.diff(solution.t))
            accepted_times.append(solution.t[1:])
            state = solution.y[:, -1].copy()
            left = right
            segments += 1
            del solution
        final = state
        q, v = (states[:, :n], states[:, n:])
        acc = np.asarray([self.rhs(t, value, backend)[n:] for t, value in zip(times, states)])
        steps = np.concatenate(accepted)
        stats = dict(
            count=len(steps),
            min_s=float(steps.min()),
            median_s=float(np.median(steps)),
            max_s=float(steps.max()),
            fraction_ge_0p99_max_step=float(np.mean(steps >= 0.99 * max_step)),
        )
        result = dict(
            t=times,
            q=q,
            v=v,
            acc=acc,
            q_equilibrium=q0,
            equilibrium_residual_N=residual,
            state_final=final,
            nfev=calls,
            integration_end_s=float(times[-1]),
            state_names=self.state_names,
            units_by_state=self.units_by_state,
            base_mechanical_dof=nb,
            structural_dof=n,
            sensor_indices=self.sensor_indices,
            integration_accepted_steps_s=steps,
            integration_accepted_times_s=np.concatenate(([t_start], *accepted_times)),
            solver_success=True,
            actual_solver_settings=dict(
                success=True,
                method=method,
                rtol=rtol,
                atol=np.asarray(atol).tolist(),
                max_step_s=max_step,
                output_fs_hz=fs,
                rhs_backend=backend,
                accepted_steps=stats,
                structural_dof=n,
                base_dof=nb,
                fit_direction_count=self.base_parameters["N"],
                dense_block_duration_s=0.125,
                integration_segments=segments + (warmup > 0),
                velocity_matrix_convention="V8 C+G once; replace old H-ground, append passive chain",
            ),
        )
        result.update(self.observations(q, v, acc))
        result.update(self.link_forces(q, v))
        result.update(self.contact_observations(times, q, v))
        return result

    def contact_observations(self, times, q, v):
        """Observe the unchanged contact laws on full or failed partial states."""
        nb = self.base_mechanical_dof
        result = {}
        fits = np.full((len(times), 4, 2), np.nan)
        counts = np.full((len(times), 4), -1, dtype=np.int32)
        compression = np.full((len(times), 4), np.nan)
        rolling = np.empty((len(times), 4, 2))
        relative = np.zeros((len(times), 4, 2))
        has_rear = v8.is_rear_outer_fit(self.base_parameters)
        fit_applicable = np.array([True, True, has_rear, has_rear])
        outer_pairs = (6, 10, 12, 14) if has_rear else (6, 10)
        for ring, first in enumerate(outer_pairs):
            relative[:, ring] = q[:, first : first + 2] - q[:, H]
        for left in range(0, len(times), 2048):
            use = slice(left, left + 2048)
            fits[use, 0], diag = v8.fit_contact_summary(
                relative[use, 0], v[use, 6:8] - v[use, H], self.base_parameters
            )
            _, fits[use, 1], diag2 = v8.front2_forces(
                q[use, :nb], v[use, :nb], self.base_parameters, summary=True, t=times[use]
            )
            counts[use, 0], counts[use, 1] = (diag["active_count"], diag2["active_count"])
            compression[use, 0], compression[use, 1] = (
                diag["max_compression_m"],
                diag2["max_compression_m"],
            )
            if has_rear:
                fits[use, 2:], rear_diag = v8.rear_fit_forces(
                    q[use, :nb], v[use, :nb], self.base_parameters, summary=True
                )
                for ring, name in enumerate(("R1", "R2"), start=2):
                    counts[use, ring] = rear_diag[name]["active_count"]
                    compression[use, ring] = rear_diag[name]["max_compression_m"]
            wrenches = v8.bearing_wrenches(
                times[use], q[use, :nb], v[use, :nb], self.base_parameters
            )
            for ring, name in enumerate(("F1", "F2", "R1", "R2")):
                rolling[use, ring] = wrenches[name]["force_N"]
        result.update(
            fit_force_N=fits[:, 0],
            fit_active_count=counts[:, 0],
            fit2_force_N=fits[:, 1],
            fit2_active_count=counts[:, 1],
            all_fit_force_N=fits,
            all_fit_active_count=counts,
            all_fit_max_compression_m=compression,
            all_fit_detached_flag=np.where(counts < 0, -1, compression <= 0).astype(np.int8),
            all_fit_force_free_flag=np.where(counts < 0, -1, counts == 0).astype(np.int8),
            fit_applicable=fit_applicable,
            rolling_contact_force_N=rolling,
            outer_relative_H_xy_m=relative,
        )
        if has_rear:
            result["rear_fit_force_N"] = fits[:, 2:]
        else:
            result["rear_bound_rolling_force_N"] = rolling[:, 2:]
        return result
