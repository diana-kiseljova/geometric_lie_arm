import typing as ty
import copy
import time

from tqdm import tqdm
import numpy as np
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation
from odeintw import odeintw

from src.initials import Initials
from src.dynamics import kin_dyn_with_quat
from src.utils.math import vect2skew, adjoint, coadjoint, error, estimate_error


def inner_x_to_se3(X: np.ndarray) -> list[np.ndarray]:
    return [
        np.block([
            [Rotation.from_quat(x[:4]).as_matrix(), x[4:7].reshape(3, 1)],
            [np.zeros((1, 3)), 1]
        ])
        for x in X
    ]


def se3_to_inner_x(X_d_list: list[np.ndarray]) -> list[np.ndarray]:
    return [
        np.concatenate((
            Rotation.from_matrix(X[:3, :3]).as_quat(), X[:3, 3]
        ))
        for X in X_d_list
    ]


class ParamsLQR:

    def __init__(self, init: Initials, input_arrays: dict):
        self.Qw = init.controller_params['Qw']
        self.Rw = init.controller_params['Rw']


def run_lqr(
        init: Initials,
        lqr_par: ParamsLQR,
        input_arrays: dict[str, ty.Union[np.ndarray, list[np.ndarray]]]
    ) -> dict[str, ty.Union[np.ndarray, list[np.ndarray]]]:
    output_arrays = copy.deepcopy(input_arrays)

    T = init.T
    dt = init.dt
    N = int(T / dt) - 1
    # Controller uses the *model* inertia; physical sim uses the true one.
    Jb_model = init.body_model.Jb
    Jb_phys = init.body.Jb
    Q_w = lqr_par.Qw
    R_w = lqr_par.Rw
    u_min = init.u_min
    u_max = init.u_max

    X_d_list = input_arrays['X_d_list']
    xi_ref = np.array(input_arrays['xi_d_array'])

    # ---- Build time-varying linearized model (controller's model) ----
    At, Bt = [], []
    for k in range(N + 1):
        xi_bar = Jb_model @ xi_ref[k, :]
        G = np.zeros((6, 6))
        G[:3, :3] = vect2skew(xi_bar[:3])
        G[:3, 3:] = vect2skew(xi_bar[3:])
        G[3:, :3] = vect2skew(xi_bar[3:])
        H = np.linalg.inv(Jb_model) @ (coadjoint(xi_bar) @ Jb_model + G)

        Ac = np.block([
            [np.zeros((6, 6)), np.eye(6)],
            [H @ adjoint(xi_ref[k, :]), -adjoint(xi_ref[k, :]) + H]
        ])
        Bc = np.block([
            [np.zeros((6, 6))],
            [np.linalg.inv(Jb_model)]
        ])
        At.append(Ac)
        Bt.append(Bc)

    At = np.array(At)
    Bt = np.array(Bt)

    # ---- Solve LTV Riccati ODE backward in time ----
    t3 = np.linspace(0, T, N + 1)
    Pf = 0.001 * np.eye(12)
    dt_ric = T / N  # step used for Riccati index mapping

    def ricatti(P, t):
        # Map backward time t (in [-T, 0]) to forward index [0, N]
        idx = int(-t / dt_ric + 0.5)
        idx = min(max(N - idx, 0), N)
        dP = -(At[idx].T @ P + P @ At[idx]
               - P @ Bt[idx] @ np.linalg.pinv(R_w) @ Bt[idx].T @ P + Q_w)
        return dP


    # P_n = odeintw(ricatti, Pf, -t3)
    # K_lqr = []
    # for l in range(N + 1):
    #     K_lqr.append(-np.linalg.pinv(R_w) @ Bt[l].T @ P_n[l])
    # K_lqr = np.array(K_lqr)


    P_n = odeintw(ricatti, Pf, -t3) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECK
    P_forward = P_n[::-1]
    K_lqr = []
    for l in range(N + 1):
        K_lqr.append(-np.linalg.solve(R_w, Bt[l].T @ P_forward[l]))
    K_lqr = np.array(K_lqr)


    # ---- Forward simulation: step-by-step using src.dynamics.kin_dyn_with_quat ----
    q0_init = Rotation.from_matrix(init.R_0).as_quat()
    x0 = np.concatenate((q0_init, init.p_0, init.xi_0[:3], init.xi_0[3:]))

    X_hist = []
    Psi_hist = []
    psi_hist = []
    u_hist = []
    u_raw_hist = []
    u_saturated_hist = []
    time_hist = []

    for epoch in tqdm(range(N)):
        t1 = time.time()

        # Build SE3 matrix from current quaternion state
        R_mat = Rotation.from_quat(x0[:4]).as_matrix()
        p_curr = x0[4:7]
        w_curr = x0[7:10]
        v_curr = x0[10:13]
        xi_curr = np.hstack([w_curr, v_curr])

        X_curr = np.block([
            [R_mat, p_curr.reshape(3, 1)],
            [np.zeros((1, 3)), 1]
        ])

        # Lie algebra error at current step
        xi_ref_t = xi_ref[epoch]
        Psi_t, psi = error(X_curr, X_d_list[epoch])
        psi_dot = xi_curr - adjoint(xi_ref_t) @ psi - xi_ref_t
        xt = np.hstack([psi, psi_dot])

        # Feedforward + LQR feedback (built from controller's model)
        u_d = -coadjoint(xi_ref_t) @ Jb_model @ xi_ref_t
        u_raw = u_d + K_lqr[epoch] @ xt

        # Component-wise actuator saturation.
        # u_hist stores the actually applied constrained control.
        u = np.clip(u_raw, u_min, u_max)
        u_is_saturated = ~np.isclose(u, u_raw, rtol=1e-10, atol=1e-12)

        # Integrate one time step using shared dynamics from src.dynamics
        # (physical Jb — may differ from the controller's model Jb).
        t_int = np.linspace(0, dt, 21)
        sol = odeint(kin_dyn_with_quat, x0, t_int, args=(u, Jb_phys, init.ext_wrench), rtol=1e-9, atol=1e-12)


        # x0 = sol[-1, :]


        x0 = sol[-1, :] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECK
        x0[:4] /= np.linalg.norm(x0[:4])


        t2 = time.time()

        X_hist.append(X_curr)
        Psi_hist.append(Psi_t)
        psi_hist.append(psi)
        u_hist.append(u)
        u_raw_hist.append(u_raw)
        u_saturated_hist.append(u_is_saturated)
        time_hist.append(t2 - t1)

    # Append final state
    R_mat = Rotation.from_quat(x0[:4]).as_matrix()
    p_curr = x0[4:7]
    X_final = np.block([
        [R_mat, p_curr.reshape(3, 1)],
        [np.zeros((1, 3)), 1]
    ])
    Psi_final, psi_final = error(X_final, X_d_list[N])
    X_hist.append(X_final)
    Psi_hist.append(Psi_final)
    psi_hist.append(psi_final)

    output_arrays['X_hist'] = X_hist
    output_arrays['Psi_hist'] = Psi_hist
    output_arrays['psi_hist'] = psi_hist
    output_arrays['u_hist'] = u_hist
    output_arrays['u_raw_hist'] = u_raw_hist
    output_arrays['u_saturated_hist'] = u_saturated_hist
    output_arrays['u_min'] = u_min
    output_arrays['u_max'] = u_max
    output_arrays['time_hist'] = time_hist

    return output_arrays
