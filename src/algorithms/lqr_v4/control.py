import typing as ty
import copy
import time

import numpy as np
from scipy.integrate import odeint
from scipy.linalg import solve_discrete_are
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from src.initials import Initials
from src.dynamics import kin_dyn_with_quat


class ParamsSE3LTILQR:
    """
    Parameters for an SE(3)-error LTI LQR baseline.

    This controller is still a standard, constant, infinite-horizon LTI LQR:

        x_err[k+1] = A x_err[k] + B delta_u[k]
        delta_u[k] = -K x_err[k]

    The change relative to the quaternion/Euclidean baseline is only the
    runtime error map.

    Pose error:
        Psi = X_d^{-1} X
        psi = Log(Psi)^vee in R^6

    with ordering:
        psi = [theta, rho]

    where:
        theta is the SO(3) logarithm of R_d^T R,
        rho   is the translational part of the SE(3) logarithm.

    Velocity error:
        e_xi = xi - Ad_{Psi^{-1}} xi_d

    LTI LQR state:
        x_err = [psi, e_xi] in R^12
              = [theta, rho, e_omega, e_v]

    Important:
        The LTI model uses the local approximation psi_dot ~= e_xi.
        The exact log-coordinate derivative would include a left/right
        Jacobian factor. We intentionally keep the LTI model simple because
        this is a baseline controller, not a nonlinear geometric LQR.
    """

    def __init__(self, init: Initials, input_arrays: dict | None = None):
        self.Qw = np.asarray(init.controller_params["Qw"], dtype=float)
        self.Rw = np.asarray(init.controller_params["Rw"], dtype=float)

        # Pure LTI baseline: no feedforward by default.
        self.use_feedforward = bool(init.controller_params.get("use_feedforward", False))


# Backward-compatible alias in case the config imports the old class name.
ParamsStandardQuatLTILQR = ParamsSE3LTILQR


# =============================================================================
# Small controller-specific helpers
# =============================================================================


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion in scipy convention [qx, qy, qz, qw].

    The nonlinear simulator integrates quaternion coordinates numerically, so
    the quaternion may slightly drift from unit norm.
    """
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("Quaternion norm is zero; cannot normalize.")
    return q / norm


def hat3(a: np.ndarray) -> np.ndarray:
    """
    3D skew-symmetric matrix.

    For a = [a1, a2, a3]:

        hat(a) b = a x b
    """
    a = np.asarray(a, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ],
        dtype=float,
    )


def left_jacobian_SO3_inv(phi: np.ndarray) -> np.ndarray:
    """
    Inverse left Jacobian of SO(3).

    If R = Exp(phi), then for SE(3)

        p = J(phi) rho
        rho = J(phi)^{-1} p

    This is needed because the translational part of Log(SE(3)) is not simply
    the relative translation p when rotation error is nonzero.
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    A = hat3(phi)

    if theta < 1e-8:
        return np.eye(3) - 0.5 * A + (1.0 / 12.0) * (A @ A)

    theta2 = theta * theta
    coef = (1.0 / theta2) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) - 0.5 * A + coef * (A @ A)


def se3_inv(X: np.ndarray) -> np.ndarray:
    """
    Inverse of an SE(3) matrix.

    X = [R, p; 0, 1]
    X^{-1} = [R^T, -R^T p; 0, 1]
    """
    X = np.asarray(X, dtype=float)
    R = X[:3, :3]
    p = X[:3, 3]

    X_inv = np.eye(4, dtype=float)
    X_inv[:3, :3] = R.T
    X_inv[:3, 3] = -R.T @ p
    return X_inv


def adjoint_SE3(X: np.ndarray) -> np.ndarray:
    """
    Adjoint matrix of SE(3) for twists ordered as [omega, v].

    If X = [R, p; 0, 1], then

        Ad_X = [[R,        0],
                [hat(p) R, R]]
    """
    X = np.asarray(X, dtype=float)
    R = X[:3, :3]
    p = X[:3, 3]

    Ad = np.zeros((6, 6), dtype=float)
    Ad[:3, :3] = R
    Ad[3:6, :3] = hat3(p) @ R
    Ad[3:6, 3:6] = R
    return Ad


def se3_log(X: np.ndarray) -> np.ndarray:
    """
    SE(3) logarithm in vector form.

    Convention:
        X = [R, p; 0, 1]
        Log(X)^vee = [theta, rho]

    where:
        theta = Log(R)^vee
        rho   = J_SO3(theta)^{-1} p

    This matches the usual se(3) hat convention:

        hat([omega, v]) = [[hat(omega), v],
                           [0,          0]]
    """
    X = np.asarray(X, dtype=float)
    R = X[:3, :3]
    p = X[:3, 3]

    theta = Rotation.from_matrix(R).as_rotvec()
    rho = left_jacobian_SO3_inv(theta) @ p

    return np.concatenate((theta, rho))


def state_to_pose_matrix(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    State [q, p] -> SE(3) matrix for error computation/logging.
    """
    R = Rotation.from_quat(normalize_quat(q)).as_matrix()
    X = np.eye(4, dtype=float)
    X[:3, :3] = R
    X[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return X


def split_inertia_and_mass(Jb: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Extract rotational inertia I_b and mass m from a 6x6 body inertia matrix.

    Assumption of this baseline:

        Jb = diag(I_b, m I_3)
    """
    Jb = np.asarray(Jb, dtype=float)
    if Jb.shape != (6, 6):
        raise ValueError(f"Jb must have shape (6, 6), got {Jb.shape}")

    Ib = Jb[:3, :3]
    Mlin = Jb[3:6, 3:6]

    m = float(np.mean(np.diag(Mlin)))
    if m <= 0.0:
        raise ValueError(f"Mass extracted from Jb must be positive, got {m}")

    return Ib, m


def finite_difference(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Central finite difference for sampled reference signals.
    """
    x = np.asarray(x, dtype=float)
    return np.gradient(x, dt, axis=0, edge_order=2)


# =============================================================================
# Optional feedforward
# =============================================================================


def build_reference_feedforward_ud(
    X_d_list: list[np.ndarray],
    xi_d_array: np.ndarray,
    Jb: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Optional nominal feedforward reconstructed from the reference.

    Reference twist convention:
        xi_d = [w_d_body, v_d_body]

    Returned control convention:
        u_d = [tau_d_body, f_d_body]

    This is optional. For the pure LTI LQR baseline, use_feedforward=False.
    """
    xi_d_array = np.asarray(xi_d_array, dtype=float)
    Ib, m = split_inertia_and_mass(Jb)

    N = len(X_d_list) - 1
    if xi_d_array.shape[0] < N + 1:
        raise ValueError(
            f"xi_d_array must contain at least {N + 1} samples, "
            f"got {xi_d_array.shape[0]}"
        )

    xi_dot = finite_difference(xi_d_array, dt)
    u_d_array = np.zeros((N, 6), dtype=float)

    for k in range(N):
        w_d = xi_d_array[k, :3]
        v_d = xi_d_array[k, 3:6]
        w_dot_d = xi_dot[k, :3]
        v_dot_d = xi_dot[k, 3:6]

        # Euler rotational dynamics in the body frame.
        tau_body = Ib @ w_dot_d + np.cross(w_d, Ib @ w_d)

        # Newton translational dynamics in the body frame.
        f_body = m * (v_dot_d + np.cross(w_d, v_d))

        u_d_array[k] = np.concatenate((tau_body, f_body))

    return u_d_array


# =============================================================================
# LTI model and infinite-horizon LQR
# =============================================================================


def build_lti_lqr_model(Jb: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one constant discrete LTI error model.

    State:
        x_err = [psi, e_xi] in R^12
              = [theta, rho, e_omega, e_v]

    where:
        psi = Log(X_d^{-1} X)^vee
        e_xi = xi - Ad_{Psi^{-1}} xi_d

    Input:
        delta_u = [delta_tau_body, delta_f_body] in R^6

    Local discrete double-integrator model:

        theta[k+1] = theta[k] + dt e_omega[k]
                        + 0.5 dt^2 I_b^{-1} delta_tau[k]

        e_omega[k+1] = e_omega[k]
                        + dt I_b^{-1} delta_tau[k]

        rho[k+1] = rho[k] + dt e_v[k]
                        + 0.5 dt^2 delta_f_body[k] / m

        e_v[k+1] = e_v[k]
                        + dt delta_f_body[k] / m

    This model intentionally ignores:
        - reference-dependent terms,
        - gyroscopic coupling,
        - exact SE(3) log-coordinate Jacobians,
        - nonlinear attitude/translation coupling.

    That is exactly what makes it a simple LTI LQR baseline.
    """
    Ib, m = split_inertia_and_mass(Jb)
    Ib_inv = np.linalg.inv(Ib)

    A = np.eye(12, dtype=float)
    B = np.zeros((12, 6), dtype=float)

    # Kinematic coupling: psi_dot ~= e_xi.
    A[0:3, 6:9] = dt * np.eye(3)
    A[3:6, 9:12] = dt * np.eye(3)

    # Torque -> orientation/rate.
    B[0:3, 0:3] = 0.5 * dt**2 * Ib_inv
    B[6:9, 0:3] = dt * Ib_inv

    # Body force -> translational log part / body linear velocity error.
    B[3:6, 3:6] = 0.5 * dt**2 * np.eye(3) / m
    B[9:12, 3:6] = dt * np.eye(3) / m

    return A, B


def solve_infinite_horizon_lti_lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infinite-horizon discrete LTI LQR.

    Dynamics:
        x[k+1] = A x[k] + B u[k]

    Cost:
        J = sum_k x[k]^T Q x[k] + u[k]^T R u[k]

    DARE:
        P = A^T P A
            - A^T P B (R + B^T P B)^{-1} B^T P A
            + Q

    Gain:
        K = (R + B^T P B)^{-1} B^T P A

    Control:
        u = -K x
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    nx = A.shape[0]
    nu = B.shape[1]

    if A.shape != (nx, nx):
        raise ValueError(f"A must be square, got {A.shape}")
    if B.shape != (nx, nu):
        raise ValueError(f"B has invalid shape {B.shape}")
    if Q.shape != (nx, nx):
        raise ValueError(f"Q must have shape {(nx, nx)}, got {Q.shape}")
    if R.shape != (nu, nu):
        raise ValueError(f"R must have shape {(nu, nu)}, got {R.shape}")

    P = solve_discrete_are(A, B, Q, R)
    P = 0.5 * (P + P.T)

    S = R + B.T @ P @ B
    F = B.T @ P @ A
    K = np.linalg.solve(S, F)

    return K, P


# =============================================================================
# Runtime SE(3) error mapping
# =============================================================================


def build_error_state(
    q_curr: np.ndarray,
    p_curr: np.ndarray,
    w_curr: np.ndarray,
    v_curr_body: np.ndarray,
    X_des: np.ndarray,
    xi_des: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build both:
        1. x_err:    local LTI LQR state,
        2. pose_err: SE(3) logarithmic pose error for logging/metrics.

    Current pose:
        X = [R, p]

    Desired pose:
        X_d = [R_d, p_d]

    SE(3) pose error:
        Psi = X_d^{-1} X
        psi = Log(Psi)^vee = [theta, rho]

    Relative twist error:
        e_xi = xi - Ad_{Psi^{-1}} xi_d

    LTI LQR state:
        x_err = [psi, e_xi]
              = [theta, rho, e_omega, e_v]

    Logged pose error:
        pose_err = psi
    """
    X_curr = state_to_pose_matrix(q_curr, p_curr)
    X_des = np.asarray(X_des, dtype=float)
    xi_curr = np.concatenate(
        (
            np.asarray(w_curr, dtype=float).reshape(3),
            np.asarray(v_curr_body, dtype=float).reshape(3),
        )
    )
    xi_des = np.asarray(xi_des, dtype=float).reshape(6)

    Psi = se3_inv(X_des) @ X_curr
    psi = se3_log(Psi)

    e_xi = xi_curr - adjoint_SE3(se3_inv(Psi)) @ xi_des

    x_err = np.concatenate((psi, e_xi))
    pose_err = psi

    return x_err, pose_err


# =============================================================================
# Main controller runner
# =============================================================================


def run_lqr(
    init: Initials,
    lqr_par: ParamsSE3LTILQR,
    input_arrays: dict[str, ty.Union[np.ndarray, list[np.ndarray]]],
) -> dict[str, ty.Union[np.ndarray, list[np.ndarray]]]:
    """
    Run a standard LTI LQR tracking baseline with SE(3) logarithmic pose error.

    Main math-to-code map
    ---------------------

    SE(3) runtime error:
        Psi[k] = X_d[k]^{-1} X[k]
        psi[k] = Log(Psi[k])^vee
        e_xi[k] = xi[k] - Ad_{Psi[k]^{-1}} xi_d[k]

    LTI state:
        x_err[k] = [psi[k], e_xi[k]]

    Constant LTI model:
        x_err[k+1] = A x_err[k] + B delta_u[k]

    DARE:
        P = A^T P A
            - A^T P B (R + B^T P B)^{-1} B^T P A
            + Q

    Feedback:
        delta_u[k] = -K x_err[k]

    Control:
        if use_feedforward:
            u_body[k] = u_d[k] + delta_u[k]
        else:
            u_body[k] =          delta_u[k]

    Control convention:
        u_body = [tau_body, f_body]

    Important:
        This remains an LTI baseline. It does not use the exact nonlinear
        SE(3) error dynamics or the exact Jacobian relation between psi_dot
        and e_xi.
    """
    output_arrays = copy.deepcopy(input_arrays)

    dt = init.dt
    N = int(init.T / dt) - 1

    body_model = getattr(init, "body_model", init.body)
    Jb_model = np.asarray(body_model.Jb, dtype=float)
    Jb_phys = np.asarray(init.body.Jb, dtype=float)

    Q = np.asarray(lqr_par.Qw, dtype=float)
    R = np.asarray(lqr_par.Rw, dtype=float)

    u_min = np.asarray(init.u_min, dtype=float)
    u_max = np.asarray(init.u_max, dtype=float)

    if Q.shape != (12, 12):
        raise ValueError(f"Qw must have shape (12, 12), got {Q.shape}")
    if R.shape != (6, 6):
        raise ValueError(f"Rw must have shape (6, 6), got {R.shape}")

    X_d_list = ty.cast(list[np.ndarray], input_arrays["X_d_list"])
    xi_d_array = np.asarray(input_arrays["xi_d_array"], dtype=float)

    if len(X_d_list) < N + 1:
        raise ValueError(f"X_d_list must contain at least {N + 1} samples")
    if xi_d_array.shape[0] < N + 1:
        raise ValueError(f"xi_d_array must contain at least {N + 1} samples")

    # ----------------------------------------------------------------------
    # 1. Constant LTI model and one constant LQR gain.
    # ----------------------------------------------------------------------
    A, B = build_lti_lqr_model(Jb=Jb_model, dt=dt)
    K, P = solve_infinite_horizon_lti_lqr(A=A, B=B, Q=Q, R=R)

    # ----------------------------------------------------------------------
    # 2. Optional feedforward.
    # ----------------------------------------------------------------------
    if lqr_par.use_feedforward:
        u_d_array = build_reference_feedforward_ud(
            X_d_list=X_d_list,
            xi_d_array=xi_d_array,
            Jb=Jb_model,
            dt=dt,
        )
    else:
        u_d_array = np.zeros((N, 6), dtype=float)
    if u_d_array.shape[0] < N:
        raise ValueError(f"u_d_array must contain at least {N} samples")

    # ----------------------------------------------------------------------
    # 3. Initial physical state for nonlinear simulation.
    # ----------------------------------------------------------------------
    q0 = Rotation.from_matrix(init.R_0).as_quat()
    x = np.concatenate(
        (
            q0,
            init.p_0,
            init.xi_0[:3],
            init.xi_0[3:],
        )
    )

    X_hist = []
    q_hist = []
    p_hist = []
    x_lqr_hist = []
    pose_err_hist = []
    u_hist = []
    u_raw_hist = []
    u_d_hist = []
    u_feedback_hist = []
    u_saturated_hist = []
    time_hist = []

    # ----------------------------------------------------------------------
    # 4. Closed-loop simulation.
    # ----------------------------------------------------------------------
    for k in tqdm(range(N)):
        t_start = time.time()

        q_curr = normalize_quat(x[:4])
        p_curr = x[4:7]
        w_curr = x[7:10]
        v_curr_body = x[10:13]

        x_err, pose_err = build_error_state(
            q_curr=q_curr,
            p_curr=p_curr,
            w_curr=w_curr,
            v_curr_body=v_curr_body,
            X_des=X_d_list[k],
            xi_des=xi_d_array[k],
        )

        u_d = u_d_array[k]
        u_feedback = -K @ x_err

        # Convention: [tau_body, f_body].
        u_body_unsat = u_d + u_feedback

        u_body = np.clip(u_body_unsat, u_min, u_max)
        u_is_saturated = ~np.isclose(u_body, u_body_unsat, rtol=1e-10, atol=1e-12)

        t_int = np.linspace(0.0, dt, 21)
        sol = odeint(
            kin_dyn_with_quat,
            x,
            t_int,
            args=(u_body, Jb_phys, init.ext_wrench),
            rtol=1e-9,
            atol=1e-12,
        )

        X_curr = state_to_pose_matrix(q_curr, p_curr)

        X_hist.append(X_curr)
        q_hist.append(q_curr)
        p_hist.append(p_curr)
        x_lqr_hist.append(x_err)
        pose_err_hist.append(pose_err)

        u_hist.append(u_body)
        u_raw_hist.append(u_body_unsat)
        u_d_hist.append(u_d)
        u_feedback_hist.append(u_feedback)
        u_saturated_hist.append(u_is_saturated)
        time_hist.append(time.time() - t_start)

        x = sol[-1, :]
        x[:4] = normalize_quat(x[:4])

    # ----------------------------------------------------------------------
    # 5. Append final state and final error.
    # ----------------------------------------------------------------------
    q_final = normalize_quat(x[:4])
    p_final = x[4:7]
    w_final = x[7:10]
    v_final_body = x[10:13]

    X_final = state_to_pose_matrix(q_final, p_final)

    x_err_final, pose_err_final = build_error_state(
        q_curr=q_final,
        p_curr=p_final,
        w_curr=w_final,
        v_curr_body=v_final_body,
        X_des=X_d_list[N],
        xi_des=xi_d_array[N],
    )

    X_hist.append(X_final)
    q_hist.append(q_final)
    p_hist.append(p_final)
    x_lqr_hist.append(x_err_final)
    pose_err_hist.append(pose_err_final)

    # ----------------------------------------------------------------------
    # 6. Save outputs.
    # ----------------------------------------------------------------------
    output_arrays["X_hist"] = X_hist
    output_arrays["q_hist"] = np.asarray(q_hist)
    output_arrays["p_hist"] = np.asarray(p_hist)
    output_arrays["x_lqr_hist"] = np.asarray(x_lqr_hist)

    # SE(3) logarithmic pose error.
    output_arrays["pose_err_hist"] = np.asarray(pose_err_hist)

    # Backward compatibility with old plotting/metrics code that expected
    # `psi_hist`. Now this is really an SE(3) logarithmic pose error:
    # psi = Log(X_d^{-1} X)^vee.
    output_arrays["psi_hist"] = np.asarray(pose_err_hist)

    output_arrays["u_hist"] = np.asarray(u_hist)
    output_arrays["u_raw_hist"] = np.asarray(u_raw_hist)
    output_arrays["u_d_array"] = np.asarray(u_d_array)
    output_arrays["u_d_hist"] = np.asarray(u_d_hist)
    output_arrays["u_feedback_hist"] = np.asarray(u_feedback_hist)
    output_arrays["u_saturated_hist"] = np.asarray(u_saturated_hist)

    output_arrays["u_min"] = u_min
    output_arrays["u_max"] = u_max
    output_arrays["time_hist"] = time_hist

    # Explicit LTI objects.
    output_arrays["A_lti"] = A
    output_arrays["B_lti"] = B
    output_arrays["K_lti"] = K
    output_arrays["P_lti"] = P

    # Compatibility with old plotting/metrics code that expected sequences.
    output_arrays["A_lqr"] = np.broadcast_to(A, (N, 12, 12)).copy()
    output_arrays["B_lqr"] = np.broadcast_to(B, (N, 12, 6)).copy()
    output_arrays["K_lqr"] = np.broadcast_to(K, (N, 6, 12)).copy()
    output_arrays["P_lqr"] = np.broadcast_to(P, (N + 1, 12, 12)).copy()

    output_arrays["lqr_type"] = "LTI infinite-horizon DARE with SE(3) runtime error"
    output_arrays["use_feedforward"] = lqr_par.use_feedforward
    output_arrays["error_convention"] = {
        "pose_error": "psi = Log(X_d^{-1} X)^vee = [theta, rho]",
        "velocity_error": "e_xi = xi - Ad_{(X_d^{-1}X)^{-1}} xi_d",
        "pose_err_hist": "SE(3) logarithmic pose error psi",
        "psi_hist": "compatibility alias for pose_err_hist; now SE(3) log error",
        "x_lqr_hist": "[psi, e_xi] = [theta, rho, e_omega, e_v]",
        "u_raw_hist": "[tau_body, f_body] before saturation",
        "u_hist": "[tau_body, f_body] after saturation",
        "lti_approximation": "uses psi_dot ~= e_xi; exact log dynamics would include Jacobian factors",
    }

    return output_arrays
