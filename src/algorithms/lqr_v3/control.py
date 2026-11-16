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


class ParamsStandardQuatLTILQR:
    """
    Parameters for a standard quaternion-based LTI LQR baseline.

    Important convention
    --------------------
    This controller does NOT build the orientation error from the SE(3)
    logarithmic pose error Log(X_d^{-1} X)^vee.

    Instead, the rotational error is computed directly from the relative
    quaternion:

        q_err = q_d^{-1} ⊗ q

    and then converted to a 3D local rotation vector:

        theta = Log(q_err) in R^3

    The LTI LQR state is

        x_err = [theta, e_p, e_w, e_v] in R^12

    where

        theta = quaternion relative rotation vector
        e_p   = p - p_d
        e_w   = w - w_d
        e_v   = R v - R_d v_d

    The stored pose error `pose_err_hist` and compatibility alias `psi_hist`
    are both

        [theta_quat, e_p]

    not an SE(3) logarithmic error.
    """

    def __init__(self, init: Initials, input_arrays: dict | None = None):
        self.Qw = np.asarray(init.controller_params["Qw"], dtype=float)
        self.Rw = np.asarray(init.controller_params["Rw"], dtype=float)

        # Pure LTI baseline: no feedforward by default.
        self.use_feedforward = bool(init.controller_params.get("use_feedforward", False))


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


def matrix_to_quat_p(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    SE(3) matrix -> scipy quaternion [qx, qy, qz, qw] and position p.

    This function only extracts the desired quaternion and position from X_d.
    It does not compute an SE(3) logarithmic error.
    """
    X = np.asarray(X, dtype=float)
    q = Rotation.from_matrix(X[:3, :3]).as_quat()
    p = X[:3, 3]
    return q, p


def quat_orientation_error(q_curr: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    """
    Quaternion-based orientation error.

    Both quaternions use scipy convention:

        q = [qx, qy, qz, qw]

    Relative rotation:

        q_err = q_des^{-1} ⊗ q_curr

    Local 3D error used by LQR:

        theta = rotvec(q_err)

    For small errors this satisfies approximately:

        R ≈ R_d Exp([theta]x)

    This is an SO(3)/quaternion orientation error, not the rotational part of
    an SE(3) logarithmic pose error.
    """
    q_curr = normalize_quat(q_curr)
    q_des = normalize_quat(q_des)

    R_curr = Rotation.from_quat(q_curr)
    R_des = Rotation.from_quat(q_des)

    q_err = R_des.inv() * R_curr
    return q_err.as_rotvec()


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

    Returned raw-control convention:
        u_d = [tau_d_body, f_d_world]

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
        R_d = np.asarray(X_d_list[k], dtype=float)[:3, :3]

        w_d = xi_d_array[k, :3]
        v_d = xi_d_array[k, 3:6]
        w_dot_d = xi_dot[k, :3]
        v_dot_d = xi_dot[k, 3:6]

        # Euler rotational dynamics in the body frame.
        tau_body = Ib @ w_dot_d + np.cross(w_d, Ib @ w_d)

        # Newton translational dynamics.
        # First compute nominal body force, then express it in world coordinates
        # because the LQR translational channel is f_world.
        f_body = m * (v_dot_d + np.cross(w_d, v_d))
        f_world = R_d @ f_body

        u_d_array[k] = np.concatenate((tau_body, f_world))

    return u_d_array


# =============================================================================
# LTI model and infinite-horizon LQR
# =============================================================================


def build_lti_lqr_model(Jb: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one constant discrete LTI error model.

    State:
        x_err = [theta, e_p, e_w, e_v] in R^12

    where theta is computed from the relative quaternion, not from an SE(3)
    logarithmic pose error.

    Raw input:
        delta_u = [delta_tau_body, delta_f_world] in R^6

    Discrete double-integrator model:

        theta[k+1] = theta[k] + dt e_w[k]
                        + 0.5 dt^2 I_b^{-1} delta_tau[k]

        e_w[k+1]   = e_w[k]
                        + dt I_b^{-1} delta_tau[k]

        e_p[k+1]   = e_p[k] + dt e_v[k]
                        + 0.5 dt^2 delta_f_world[k] / m

        e_v[k+1]   = e_v[k]
                        + dt delta_f_world[k] / m

    This model intentionally ignores:
        - reference-dependent terms,
        - gyroscopic coupling,
        - attitude transport,
        - nonlinear SE(3) coupling.

    That is exactly what makes it a simple LTI LQR baseline.
    """
    Ib, m = split_inertia_and_mass(Jb)
    Ib_inv = np.linalg.inv(Ib)

    A = np.eye(12, dtype=float)
    B = np.zeros((12, 6), dtype=float)

    # Kinematic coupling.
    A[0:3, 6:9] = dt * np.eye(3)
    A[3:6, 9:12] = dt * np.eye(3)

    # Torque -> orientation/rate.
    B[0:3, 0:3] = 0.5 * dt**2 * Ib_inv
    B[6:9, 0:3] = dt * Ib_inv

    # World force -> position/world velocity.
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
# Runtime error mapping
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
        1. x_err:       local LTI LQR state,
        2. pose_err:    6D pose error for logging/metrics.

    Desired pose:
        X_des = [R_d, p_d]

    Desired quaternion:
        q_d = quat(R_d)

    Quaternion orientation error:
        q_err = q_d^{-1} ⊗ q
        theta = rotvec(q_err)

    LTI LQR state:
        x_err = [theta, p - p_d, w - w_d, R v - R_d v_d]

    Logged pose error:
        pose_err = [theta, p - p_d]

    No SE(3) logarithmic pose error is used here.
    """
    q_des, p_des = matrix_to_quat_p(X_des)

    theta = quat_orientation_error(q_curr=q_curr, q_des=q_des)

    R_curr = Rotation.from_quat(normalize_quat(q_curr)).as_matrix()
    R_des = np.asarray(X_des, dtype=float)[:3, :3]

    w_des = xi_des[:3]
    v_des_body = xi_des[3:6]

    e_p = np.asarray(p_curr, dtype=float) - p_des
    e_w = np.asarray(w_curr, dtype=float) - w_des

    v_curr_world = R_curr @ np.asarray(v_curr_body, dtype=float)
    v_des_world = R_des @ v_des_body
    e_v = v_curr_world - v_des_world

    x_err = np.concatenate((theta, e_p, e_w, e_v))
    pose_err = np.concatenate((theta, e_p))

    return x_err, pose_err


def raw_control_to_body_wrench(u_raw: np.ndarray, R_curr: np.ndarray) -> np.ndarray:
    """
    Convert controller raw input to the nonlinear dynamics input.

    Controller raw convention:
        u_raw = [tau_body, f_world]

    Dynamics-model convention:
        u_body = [tau_body, f_body]

    Important:
        This is NOT a full world-to-body wrench transform, because tau is
        already expressed in the body frame. Only the force part is rotated here.
    """
    u_raw = np.asarray(u_raw, dtype=float)

    tau_body = u_raw[:3]
    f_world = u_raw[3:6]
    f_body = R_curr.T @ f_world

    return np.concatenate((tau_body, f_body))


def state_to_pose_matrix(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    State [q, p] -> SE(3) matrix for logging/plotting.

    This function constructs a pose matrix but does not compute an SE(3)
    logarithmic error.
    """
    R = Rotation.from_quat(normalize_quat(q)).as_matrix()
    return np.block(
        [
            [R, np.asarray(p, dtype=float).reshape(3, 1)],
            [np.zeros((1, 3)), 1.0],
        ]
    )


# =============================================================================
# Main controller runner
# =============================================================================


def run_lqr(
    init: Initials,
    lqr_par: ParamsStandardQuatLTILQR,
    input_arrays: dict[str, ty.Union[np.ndarray, list[np.ndarray]]],
) -> dict[str, ty.Union[np.ndarray, list[np.ndarray]]]:
    """
    Run a standard LTI LQR tracking baseline with quaternion-based orientation
    error.

    Main math-to-code map
    ---------------------

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
            u_raw[k] = u_d[k] + delta_u[k]
        else:
            u_raw[k] =          delta_u[k]

    Error convention:
        x_err = [theta_quat, p-p_d, w-w_d, Rv-R_dv_d]

    where:
        theta_quat = rotvec(q_d^{-1} ⊗ q)

    Raw-control convention:
        u_raw = [tau_body, f_world]

    Dynamics-model input:
        u_body = [tau_body, f_body]
        f_body = R^T f_world
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

        R_curr = Rotation.from_quat(q_curr).as_matrix()

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
        u_raw = u_d + u_feedback

        # Convert raw [tau_body, f_world] -> dynamics input [tau_body, f_body].
        u_body_unsat = raw_control_to_body_wrench(u_raw, R_curr)

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
        u_raw_hist.append(u_raw)
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

    # New explicit name.
    output_arrays["pose_err_hist"] = np.asarray(pose_err_hist)

    # Backward compatibility with old plotting/metrics code that expected
    # `psi_hist`. It is no longer an SE(3) logarithmic error. It is now
    # [theta_quat, p-p_d].
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

    output_arrays["lqr_type"] = "LTI infinite-horizon DARE"
    output_arrays["use_feedforward"] = lqr_par.use_feedforward
    output_arrays["error_convention"] = {
        "orientation_error": "theta_quat = rotvec(q_d^{-1} ⊗ q)",
        "pose_err_hist": "[theta_quat, p-p_d]",
        "psi_hist": "compatibility alias for pose_err_hist; NOT SE(3) log error",
        "x_lqr_hist": "[theta_quat, p-p_d, w-w_d, Rv-R_dv_d]",
        "u_raw_hist": "[tau_body, f_world]",
        "u_hist": "[tau_body, f_body]",
    }

    return output_arrays
