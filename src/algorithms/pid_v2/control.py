import copy
import typing as ty
import time

from tqdm import tqdm
import numpy as np
from scipy.integrate import odeint

from src.initials import Initials
from src.dynamics import kin_dyn
from src.utils.math import error, unhat, hat, adjoint_from_two_matrices, assert_finite


class ParamsPID:

    def __init__(self, initials: Initials, input_arrays: dict):
        self.initials = initials

        self.Kp  = initials.controller_params['Kp']   # proportional
        self.Kd  = initials.controller_params['Kd']   # derivative
        self.Kdd = initials.controller_params['Kdd']  # second-derivative
        #self.Ki       = initials.controller_params['Ki']   # integral


def _safe(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Return v if all finite, else fallback.

    Guards against transient NaN/Inf that arise when the ODE solver struggles
    (near-singular rotation matrices ↔ degenerate logm / pinv).  Using the
    last-known-good value prevents one bad step from contaminating u_pid.
    """
    if np.all(np.isfinite(v)):
        return v
    return np.array(fallback, dtype=float)


def run_pid(
        init: Initials,
        pid_par: ParamsPID,
        input_arrays: dict[str, ty.Union[np.ndarray, list[np.ndarray]]]
    ) -> dict[str, ty.Union[np.ndarray, list[np.ndarray]]]:
    output_arrays = copy.deepcopy(input_arrays)
    output_arrays.setdefault('u_raw_hist', [])
    output_arrays.setdefault('u_saturated_hist', [])

    u_min = init.u_min
    u_max = init.u_max

    I = 0
    for epoch in tqdm(range(init.num_steps - 1)):
        t1 = time.time()

        #I += output_arrays['psi_hist'][epoch] * init.dt

        # Control law: u = Kp·ψ + Kd·ψ̇ + Kdd·ψ̈
        u_pid_raw = (
            pid_par.Kp  * output_arrays['psi_hist'][epoch] +
            pid_par.Kd  * output_arrays['psi_dot_hist'][epoch] +
            pid_par.Kdd * output_arrays['psi_dot_dot_hist'][epoch] #+ 
            #pid_par.Ki * I
        )

        assert_finite("u_pid", u_pid_raw, epoch)

        u_pid = np.clip(u_pid_raw, u_min, u_max)
        u_is_saturated = ~np.isclose(u_pid, u_pid_raw, rtol=1e-10, atol=1e-12)

        output_arrays['u_raw_hist'].append(u_pid_raw)
        output_arrays['u_saturated_hist'].append(u_is_saturated)
        output_arrays['u_hist'].append(u_pid)

        # Integrate real dynamics one step
        X_xi_start = np.block([
            output_arrays['X_hist'][epoch].flatten(),
            output_arrays['xi_hist'][epoch]
        ])
        t = np.linspace(0, init.dt, 21)
        sol = odeint(
            kin_dyn, X_xi_start, t,
            args=(u_pid, init.body.Jb, init.ext_wrench),
            rtol=1e-9, atol=1e-12
        )
        X_xi_next = sol[-1, :]

        X_next  = X_xi_next[0:16].reshape(4, 4)
        xi_next = X_xi_next[16:]

        assert_finite("X_next", X_next, epoch)
        assert_finite("X_d",    output_arrays['X_d_list'][epoch + 1], epoch)

        Psi_next, psi_next = error(X_next, output_arrays['X_d_list'][epoch + 1])
        # Fallback: if logm produced NaN keep previous psi (one degenerate step)
        #psi_next = _safe(psi_next, output_arrays['psi_hist'][-1])

        # ── ψ̇  =  ξ̂ − Ψ⁻¹ ξ̂_d Ψ  ──────────────────────────────────────────
        xi_d_next     = output_arrays['xi_d_array'][epoch + 1, :]
        xi_d_next_hat = hat(xi_d_next)
        Psi_inv       = np.linalg.pinv(Psi_next)

        adj_d            = adjoint_from_two_matrices(Psi_inv, xi_d_next_hat, Psi_next)
        psi_dot_next_hat = hat(xi_next) - adj_d
        psi_dot_next     = unhat(psi_dot_next_hat)
        #psi_dot_next     = _safe(psi_dot_next, output_arrays['psi_dot_hist'][-1])

        # ── ψ̈  =  ξ̇̂ − Ψ⁻¹ ξ̇̂_d Ψ − [Ψ⁻¹ ξ̂_d Ψ,  ψ̇̂]  ─────────────────────
        # ξ̇ approximated by forward finite-difference (ξ at epoch+1 minus ξ at epoch)
        xi_prev    = output_arrays['xi_hist'][-1]
        xi_dot     = (xi_next - xi_prev) / init.dt   # ξ̇(t_{e+1}) ≈ Δξ/Δt
        xi_dot_hat = hat(xi_dot)

        xi_d_curr    = output_arrays['xi_d_array'][epoch, :]
        xi_d_dot     = (xi_d_next - xi_d_curr) / init.dt
        xi_d_dot_hat = hat(xi_d_dot)

        adj_d_dot  = adjoint_from_two_matrices(Psi_inv, xi_d_dot_hat, Psi_next)   # Ψ⁻¹ ξ̇̂_d Ψ
        adj_d2     = adjoint_from_two_matrices(Psi_inv, xi_d_next_hat, Psi_next)  # Ψ⁻¹ ξ̂_d Ψ

        commutator       = adj_d2 @ psi_dot_next_hat - psi_dot_next_hat @ adj_d2
        psi_dot_dot_next = unhat(xi_dot_hat - adj_d_dot - commutator)
        # Guard: NaN arises when Psi_next is transiently near-singular (ODE stiffness).
        # Even Kdd=0 is unsafe because 0 * NaN = NaN; clamp to zeros instead.
        #psi_dot_dot_next = _safe(psi_dot_dot_next, np.zeros(6))

        t2 = time.time()
        output_arrays['time_hist'].append(t2 - t1)

        output_arrays['X_hist'].append(X_next)
        output_arrays['Psi_hist'].append(Psi_next)
        output_arrays['xi_hist'].append(xi_next)
        output_arrays['psi_hist'].append(psi_next)
        output_arrays['psi_dot_hist'].append(psi_dot_next)
        output_arrays['psi_dot_dot_hist'].append(psi_dot_dot_next)

    return output_arrays
