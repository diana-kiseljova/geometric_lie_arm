import typing as ty
import copy
import time

import numpy as np
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from src.initials import Initials
from src.dynamics import kin_dyn_with_quat
from src.utils.math import vect2skew


class ParamsStandardQuatLQR:
    """
    Параметры quaternion-based LTV LQR.

    Мы решаем задачу tracking-LQR не для полного состояния quaternion dynamics,
    а для локального 12-мерного состояния ошибки:

        x_err = [e_theta, e_p, e_w, e_v] in R^12

    где

        e_theta = Log(R_d^T R)      — малая ошибка ориентации,
        e_p     = p - p_d           — ошибка позиции в world frame,
        e_w     = w - w_d           — ошибка угловой скорости в body frame,
        e_v     = R v - R_d v_d     — ошибка линейной скорости в world frame.

    Управление LQR имеет вид

        u = u_d - K x_err,

    где u_d — feedforward wrench, восстановленный из референса.

    Важно:
    - xi_d_array = [w_d_body, v_d_body]
    - plant принимает body wrench: [tau_body, f_body]
    - LQR внутри использует raw convention: [tau_body, f_world]
        потому что e_p и e_v выражены в world frame.
    """

    def __init__(self, init: Initials, input_arrays: dict):
        self.Qw = np.asarray(init.controller_params["Qw"], dtype=float)
        self.Rw = np.asarray(init.controller_params["Rw"], dtype=float)

        # Terminal cost for finite-horizon LTV LQR.
        self.Qfw = np.asarray(init.controller_params.get("Qfw", self.Qw), dtype=float)

        # Для простого baseline обычно лучше оставить оба члена включёнными.
        self.include_attitude_transport = init.controller_params.get(
            "include_attitude_transport", True
        )
        self.include_gyro_terms = init.controller_params.get(
            "include_gyro_terms", True
        )


# =============================================================================
# Basic geometry helpers
# =============================================================================


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """
    Нормализация quaternion.

    В симуляции quaternion интегрируется численно, поэтому из-за ошибок odeint
    его норма может немного уйти от 1. Для Rotation.from_quat желательно всегда
    подавать unit quaternion.
    """
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("Quaternion norm is zero; cannot normalize.")
    return q / norm


def matrix_to_quat_p(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    SE(3) matrix -> quaternion + position.

    X = [R p]
        [0 1]

    Возвращаем:
        q — scipy convention [qx, qy, qz, qw]
        p — position in world frame
    """
    X = np.asarray(X, dtype=float)
    q = Rotation.from_matrix(X[:3, :3]).as_quat()
    p = X[:3, 3]
    return q, p


def quat_rotvec_error(q_curr: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    """
    Ошибка ориентации через rotation vector.

    Используем правую/relative ошибку:

        R_err = R_d^T R

        e_theta = Log(R_err) in R^3

    То есть:
        e_theta = 0  <=>  R = R_d

    Для малых ошибок:
        R ≈ R_d Exp([e_theta]x)

    Это не полный Lie-group controller, но это стандартная локальная ошибка
    ориентации для quaternion/Euler-angle-free LQR.
    """
    q_curr = normalize_quat(q_curr)
    q_des = normalize_quat(q_des)

    R_curr = Rotation.from_quat(q_curr)
    R_des = Rotation.from_quat(q_des)

    R_err = R_des.inv() * R_curr
    return R_err.as_rotvec()


def split_inertia_and_mass(Jb: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Достаём rotational inertia I_b и массу m из 6x6 body inertia matrix.

    В этом baseline предполагаем простую структуру:

        Jb = diag(I_b, m I_3)

    Это ровно та модель, для которой понятный LQR выглядит как два связанных
    double-integrator блока:
        orientation/rate  -> torque,
        position/velocity -> force.

    Если translational block не совсем m I, берём среднее по диагонали и
    предупреждаем через ValueError только при явно плохой форме.
    """
    Jb = np.asarray(Jb, dtype=float)
    if Jb.shape != (6, 6):
        raise ValueError(f"Jb must have shape (6, 6), got {Jb.shape}")

    Ib = Jb[:3, :3]
    Mlin = Jb[3:6, 3:6]

    # Для обычного rigid body translational block = m I.
    m = float(np.mean(np.diag(Mlin)))
    if m <= 0.0:
        raise ValueError(f"Mass extracted from Jb must be positive, got {m}")

    return Ib, m


# =============================================================================
# Reference feedforward: build u_d from X_d and xi_d
# =============================================================================


def finite_difference(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Центральная численная производная.

    Если известна дискретная траектория x[k], то производную оцениваем как

        x_dot[k] ≈ (x[k+1] - x[k-1]) / (2 dt)

    На концах np.gradient использует односторонние формулы второго порядка.
    """
    x = np.asarray(x, dtype=float)
    return np.gradient(x, dt, axis=0, edge_order=2)


def build_reference_feedforward_ud(
    X_d_list: list[np.ndarray],
    xi_d_array: np.ndarray,
    Jb: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Восстановление feedforward управления u_d из референса X_d, xi_d.

    --------------------------------------------------------------------------
    Convention
    --------------------------------------------------------------------------
    Референсная скорость:

        xi_d = [w_d, v_d]

    где
        w_d — angular velocity in body frame,
        v_d — linear velocity in body frame.

    Plant принимает body wrench:

        u_body = [tau_body, f_body].

    Но LQR внутри использует raw управление:

        u_raw = [tau_body, f_world].

    Почему force в world frame?
        Потому что позиционная часть ошибки у нас в world coordinates:

            e_p = p - p_d,
            e_v = R v - R_d v_d.

        Поэтому удобнее, чтобы LQR translational channel был f_world.

    --------------------------------------------------------------------------
    Rotational feedforward
    --------------------------------------------------------------------------
    Для rigid body в body frame:

        I_b w_dot + w x (I_b w) = tau

    Значит номинальный момент:

        tau_d = I_b w_dot_d + w_d x (I_b w_d)

    --------------------------------------------------------------------------
    Translational feedforward
    --------------------------------------------------------------------------
    Если p_dot = R v_body, то world acceleration:

        p_ddot = R (v_dot_body + w_body x v_body)

    Ньютон:

        f_world = m p_ddot

    Значит:

        f_body  = m (v_dot_d + w_d x v_d)
        f_world = R_d f_body

    --------------------------------------------------------------------------
    Output
    --------------------------------------------------------------------------
    Возвращаем массив:

        u_d_array[k] = [tau_d_body, f_d_world] in R^6

    Его можно прямо использовать в законе:

        u_raw = u_d_array[k] - K[k] x_err[k]
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

        tau_body = Ib @ w_dot_d + np.cross(w_d, Ib @ w_d)

        f_body = m * (v_dot_d + np.cross(w_d, v_d))
        f_world = R_d @ f_body

        u_d_array[k] = np.concatenate((tau_body, f_world))

    return u_d_array


# =============================================================================
# Local LTV error model
# =============================================================================


def build_ltv_lqr_model(
    X_d_list: list[np.ndarray],
    xi_d_array: np.ndarray,
    Jb: np.ndarray,
    dt: float,
    *,
    include_attitude_transport: bool = True,
    include_gyro_terms: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Строим дискретную LTV модель ошибки:

        x_err[k+1] = A[k] x_err[k] + B[k] delta_u[k]

    где

        x_err = [e_theta, e_p, e_w, e_v] in R^12
        delta_u = [delta_tau_body, delta_f_world] in R^6

    --------------------------------------------------------------------------
    1. Orientation kinematics
    --------------------------------------------------------------------------
    Для малой ошибки ориентации

        e_theta = Log(R_d^T R)

    локальная динамика примерно:

        e_theta_dot ≈ e_w - [w_d]x e_theta

    Если убрать член -[w_d]x e_theta, получится ещё более простой double
    integrator, но на вращающихся референсах tracking будет хуже.

    --------------------------------------------------------------------------
    2. Position kinematics
    --------------------------------------------------------------------------
    Позиция и линейная скорость сравниваются в world frame:

        e_p = p - p_d
        e_v = p_dot - p_dot_d = R v - R_d v_d

    Поэтому:

        e_p_dot = e_v

    --------------------------------------------------------------------------
    3. Angular dynamics
    --------------------------------------------------------------------------
    Euler equation:

        I_b w_dot + w x I_b w = tau

    Линеаризация вокруг w_d даёт:

        e_w_dot ≈ I_b^{-1} delta_tau
                    + I_b^{-1} ( [I_b w_d]x - [w_d]x I_b ) e_w

    Второй член — gyroscopic coupling. Его можно отключить, но для LTV LQR
    лучше оставить.

    --------------------------------------------------------------------------
    4. Translational dynamics
    --------------------------------------------------------------------------
    Так как LQR force channel — это f_world, а e_v тоже в world frame:

        e_v_dot ≈ delta_f_world / m

    Это простая и понятная модель для Jb = diag(I_b, m I).

    --------------------------------------------------------------------------
    5. Discretization
    --------------------------------------------------------------------------
    Используем простую дискретизацию:

        A_d = I + dt A_c
        B_d = dt B_c

    и добавляем 0.5 dt^2 вклад управления в e_theta/e_p, как для double
    integrator:

        position_next ≈ position + dt velocity + 0.5 dt^2 acceleration
    """
    xi_d_array = np.asarray(xi_d_array, dtype=float)
    Ib, m = split_inertia_and_mass(Jb)
    Ib_inv = np.linalg.inv(Ib)

    N = len(X_d_list) - 1
    if xi_d_array.shape[0] < N + 1:
        raise ValueError(
            f"xi_d_array must contain at least {N + 1} samples, "
            f"got {xi_d_array.shape[0]}"
        )

    A_seq = np.zeros((N, 12, 12), dtype=float)
    B_seq = np.zeros((N, 12, 6), dtype=float)

    for k in range(N):
        w_d = xi_d_array[k, :3]

        Ac = np.zeros((12, 12), dtype=float)
        Bc = np.zeros((12, 6), dtype=float)

        # e_theta_dot ≈ e_w - [w_d]x e_theta
        if include_attitude_transport:
            Ac[0:3, 0:3] = -vect2skew(w_d)
        Ac[0:3, 6:9] = np.eye(3)

        # e_p_dot = e_v
        Ac[3:6, 9:12] = np.eye(3)

        # e_w_dot gyroscopic linearization
        if include_gyro_terms:
            Ac[6:9, 6:9] = Ib_inv @ (
                vect2skew(Ib @ w_d) - vect2skew(w_d) @ Ib
            )

        # e_w_dot receives torque
        Bc[6:9, 0:3] = Ib_inv

        # e_v_dot receives world force
        Bc[9:12, 3:6] = np.eye(3) / m

        # Discrete model.
        Ad = np.eye(12) + dt * Ac
        Bd = dt * Bc

        # 0.5 dt^2 acceleration contribution.
        Bd[0:3, 0:3] += 0.5 * dt**2 * Ib_inv
        Bd[3:6, 3:6] += 0.5 * dt**2 * np.eye(3) / m

        A_seq[k] = Ad
        B_seq[k] = Bd

    return A_seq, B_seq


# =============================================================================
# Finite-horizon LTV LQR
# =============================================================================


def solve_finite_horizon_ltv_lqr(
    A_seq: np.ndarray,
    B_seq: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Qf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backward Riccati recursion для discrete finite-horizon LTV LQR.

    Динамика:

        x[k+1] = A[k] x[k] + B[k] u[k]

    Cost:

        J = x[N]^T Qf x[N]
            + sum_k ( x[k]^T Q x[k] + u[k]^T R u[k] )

    Оптимальный закон:

        u[k] = -K[k] x[k]

    Рекурсия:

        S[k] = R + B[k]^T P[k+1] B[k]
        K[k] = S[k]^{-1} B[k]^T P[k+1] A[k]

        P[k] = Q + A[k]^T P[k+1] A[k]
               - A[k]^T P[k+1] B[k] K[k]

    В tracking задаче мы применяем это к отклонению от номинального управления:

        delta_u[k] = -K[k] x_err[k]

    а полный raw control:

        u_raw[k] = u_d[k] + delta_u[k]
                 = u_d[k] - K[k] x_err[k]
    """
    A_seq = np.asarray(A_seq, dtype=float)
    B_seq = np.asarray(B_seq, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    Qf = np.asarray(Qf, dtype=float)

    N = A_seq.shape[0]
    nx = A_seq.shape[1]
    nu = B_seq.shape[2]

    if A_seq.shape != (N, nx, nx):
        raise ValueError(f"A_seq has invalid shape {A_seq.shape}")
    if B_seq.shape != (N, nx, nu):
        raise ValueError(f"B_seq has invalid shape {B_seq.shape}")
    if Q.shape != (nx, nx):
        raise ValueError(f"Q must have shape {(nx, nx)}, got {Q.shape}")
    if Qf.shape != (nx, nx):
        raise ValueError(f"Qf must have shape {(nx, nx)}, got {Qf.shape}")
    if R.shape != (nu, nu):
        raise ValueError(f"R must have shape {(nu, nu)}, got {R.shape}")

    P_seq = np.zeros((N + 1, nx, nx), dtype=float)
    K_seq = np.zeros((N, nu, nx), dtype=float)

    P_seq[N] = 0.5 * (Qf + Qf.T)

    for k in range(N - 1, -1, -1):
        A = A_seq[k]
        B = B_seq[k]
        P_next = P_seq[k + 1]

        S = R + B.T @ P_next @ B
        F = B.T @ P_next @ A

        K = np.linalg.solve(S, F)

        P = Q + A.T @ P_next @ A - A.T @ P_next @ B @ K

        K_seq[k] = K
        P_seq[k] = 0.5 * (P + P.T)

    return K_seq, P_seq


# =============================================================================
# Runtime helpers
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
    Собираем 12-мерное состояние ошибки LQR.

    Current state:
        q_curr      — current quaternion [x, y, z, w]
        p_curr      — current position in world frame
        w_curr      — current angular velocity in body frame
        v_curr_body — current linear velocity in body frame

    Desired state:
        X_des  — desired SE(3) pose
        xi_des — [w_des_body, v_des_body]

    Error:

        e_theta = Log(R_d^T R)
        e_p     = p - p_d
        e_w     = w - w_d
        e_v     = R v - R_d v_d

    Возвращаем:
        x_err = [e_theta, e_p, e_w, e_v]
        psi   = [e_theta, e_p]  # удобно для старых метрик/графиков
    """
    q_des, p_des = matrix_to_quat_p(X_des)

    R_curr = Rotation.from_quat(normalize_quat(q_curr)).as_matrix()
    R_des = Rotation.from_quat(normalize_quat(q_des)).as_matrix()

    w_des = xi_des[:3]
    v_des_body = xi_des[3:6]

    e_theta = quat_rotvec_error(q_curr, q_des)
    e_p = p_curr - p_des
    e_w = w_curr - w_des

    v_curr_world = R_curr @ v_curr_body
    v_des_world = R_des @ v_des_body
    e_v = v_curr_world - v_des_world

    x_err = np.concatenate((e_theta, e_p, e_w, e_v))
    psi = np.concatenate((e_theta, e_p))

    return x_err, psi


def raw_control_to_body_wrench(u_raw: np.ndarray, R_curr: np.ndarray) -> np.ndarray:
    """
    Переводим raw LQR control в body wrench для plant.

    LQR raw convention:

        u_raw = [tau_body, f_world]

    Plant convention:

        u_body = [tau_body, f_body]

    Поэтому:

        f_body = R^T f_world

    а torque уже оставляем в body frame.
    """
    u_raw = np.asarray(u_raw, dtype=float)
    tau_body = u_raw[:3]
    f_world = u_raw[3:6]
    f_body = R_curr.T @ f_world

    return np.concatenate((tau_body, f_body))


def state_to_pose_matrix(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    [q, p] -> SE(3) matrix.
    """
    R = Rotation.from_quat(normalize_quat(q)).as_matrix()
    return np.block(
        [
            [R, np.asarray(p).reshape(3, 1)],
            [np.zeros((1, 3)), 1.0],
        ]
    )


# =============================================================================
# Main controller runner
# =============================================================================


def run_lqr(
    init: Initials,
    lqr_par: ParamsStandardQuatLQR,
    input_arrays: dict[str, ty.Union[np.ndarray, list[np.ndarray]]],
) -> dict[str, ty.Union[np.ndarray, list[np.ndarray]]]:
    """
    Запуск simple quaternion-based finite-horizon LTV LQR tracking controller.

    --------------------------------------------------------------------------
    Что делает контроллер
    --------------------------------------------------------------------------
    1. Берёт референс:
           X_d_list[k]     — desired pose,
           xi_d_array[k]   — desired body twist [w_d_body, v_d_body].

    2. Если input_arrays["u_d_array"] не передан, автоматически строит
       feedforward:

           u_d[k] = [tau_d_body, f_d_world]

       из inverse dynamics.

    3. Строит LTV модель ошибки:

           x_err[k+1] = A[k] x_err[k] + B[k] delta_u[k]

    4. Решает finite-horizon Riccati recursion и получает K[k].

    5. На каждом шаге применяет:

           u_raw[k] = u_d[k] - K[k] x_err[k]

       где

           u_raw = [tau_body, f_world].

    6. Перед plant переводит force в body frame:

           u_body = [tau_body, R^T f_world]

    --------------------------------------------------------------------------
    Почему это хороший baseline
    --------------------------------------------------------------------------
    Без u_d LQR должен сам "тащить" тело за движущимся референсом. На спиралях
    и Лиссажу это обычно даёт фазовое запаздывание и большую ошибку.

    С u_d контроллер получает номинальное управление, которое уже должно
    воспроизводить референс, а LQR только подавляет ошибку:

           u = u_d + correction.

    Это честнее для сравнения с Lie group PID / geometric PID.
    """
    output_arrays = copy.deepcopy(input_arrays)

    dt = init.dt
    N = int(init.T / dt) - 1

    body_model = getattr(init, "body_model", init.body)
    Jb_model = np.asarray(body_model.Jb, dtype=float)
    Jb_phys = np.asarray(init.body.Jb, dtype=float)

    Q = np.asarray(lqr_par.Qw, dtype=float)
    R = np.asarray(lqr_par.Rw, dtype=float)
    Qf = np.asarray(lqr_par.Qfw, dtype=float)

    u_min = np.asarray(init.u_min, dtype=float)
    u_max = np.asarray(init.u_max, dtype=float)

    if Q.shape != (12, 12):
        raise ValueError(f"Qw must have shape (12, 12), got {Q.shape}")
    if Qf.shape != (12, 12):
        raise ValueError(f"Qfw must have shape (12, 12), got {Qf.shape}")
    if R.shape != (6, 6):
        raise ValueError(f"Rw must have shape (6, 6), got {R.shape}")

    X_d_list = ty.cast(list[np.ndarray], input_arrays["X_d_list"])
    xi_d_array = np.asarray(input_arrays["xi_d_array"], dtype=float)

    if len(X_d_list) < N + 1:
        raise ValueError(f"X_d_list must contain at least {N + 1} samples")
    if xi_d_array.shape[0] < N + 1:
        raise ValueError(f"xi_d_array must contain at least {N + 1} samples")

    # ----------------------------------------------------------------------
    # 1. Feedforward u_d.
    # ----------------------------------------------------------------------
    # Preferred new name: u_d_array.
    # Backward compatibility: if old u_ff_array exists, we still accept it.
    if "u_d_array" in input_arrays:
        u_d_array = np.asarray(input_arrays["u_d_array"], dtype=float)
    elif "u_ff_array" in input_arrays:
        u_d_array = np.asarray(input_arrays["u_ff_array"], dtype=float)
    else:
        u_d_array = build_reference_feedforward_ud(
            X_d_list=X_d_list,
            xi_d_array=xi_d_array,
            Jb=Jb_model,
            dt=dt,
        )

    if u_d_array.shape[0] < N:
        raise ValueError(f"u_d_array must contain at least {N} samples")

    # ----------------------------------------------------------------------
    # 2. LTV model and Riccati gains.
    # ----------------------------------------------------------------------
    A_seq, B_seq = build_ltv_lqr_model(
        X_d_list=X_d_list,
        xi_d_array=xi_d_array,
        Jb=Jb_model,
        dt=dt,
        include_attitude_transport=lqr_par.include_attitude_transport,
        include_gyro_terms=lqr_par.include_gyro_terms,
    )

    K_seq, P_seq = solve_finite_horizon_ltv_lqr(
        A_seq=A_seq,
        B_seq=B_seq,
        Q=Q,
        R=R,
        Qf=Qf,
    )

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
    psi_hist = []
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

        x_err, psi = build_error_state(
            q_curr=q_curr,
            p_curr=p_curr,
            w_curr=w_curr,
            v_curr_body=v_curr_body,
            X_des=X_d_list[k],
            xi_des=xi_d_array[k],
        )

        # Tracking LQR:
        #   delta_u = -K x_err
        #   u_raw   = u_d + delta_u
        # Здесь u_raw = [tau_body, f_world].
        K_k = K_seq[k]
        u_d = u_d_array[k]
        u_feedback = -K_k @ x_err
        u_raw = u_d + u_feedback

        # Convert to plant convention:
        #   plant expects [tau_body, f_body].
        u_body_unsat = raw_control_to_body_wrench(u_raw, R_curr)

        u_body = np.clip(u_body_unsat, u_min, u_max)
        u_is_saturated = ~np.isclose(u_body, u_body_unsat, rtol=1e-10, atol=1e-12)

        # Nonlinear dynamics integration.
        t_int = np.linspace(0.0, dt, 21)
        sol = odeint(
            kin_dyn_with_quat,
            x,
            t_int,
            args=(u_body, Jb_phys, init.ext_wrench),
            rtol=1e-9,
            atol=1e-12,
        )

        # Store current state at interval start.
        X_curr = state_to_pose_matrix(q_curr, p_curr)

        X_hist.append(X_curr)
        q_hist.append(q_curr)
        p_hist.append(p_curr)
        x_lqr_hist.append(x_err)
        psi_hist.append(psi)

        u_hist.append(u_body)              # actual body wrench applied to plant
        u_raw_hist.append(u_raw)           # [tau_body, f_world]
        u_d_hist.append(u_d)               # feedforward [tau_body, f_world]
        u_feedback_hist.append(u_feedback) # LQR correction [tau_body, f_world]
        u_saturated_hist.append(u_is_saturated)
        time_hist.append(time.time() - t_start)

        # Advance state.
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

    x_err_final, psi_final = build_error_state(
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
    psi_hist.append(psi_final)

    # ----------------------------------------------------------------------
    # 6. Save outputs.
    # ----------------------------------------------------------------------
    output_arrays["X_hist"] = X_hist
    output_arrays["q_hist"] = np.asarray(q_hist)
    output_arrays["p_hist"] = np.asarray(p_hist)
    output_arrays["x_lqr_hist"] = np.asarray(x_lqr_hist)
    output_arrays["psi_hist"] = np.asarray(psi_hist)

    output_arrays["u_hist"] = np.asarray(u_hist)
    output_arrays["u_raw_hist"] = np.asarray(u_raw_hist)
    output_arrays["u_d_array"] = np.asarray(u_d_array)
    output_arrays["u_d_hist"] = np.asarray(u_d_hist)
    output_arrays["u_feedback_hist"] = np.asarray(u_feedback_hist)
    output_arrays["u_saturated_hist"] = np.asarray(u_saturated_hist)

    output_arrays["u_min"] = u_min
    output_arrays["u_max"] = u_max
    output_arrays["time_hist"] = time_hist

    output_arrays["A_lqr"] = A_seq
    output_arrays["B_lqr"] = B_seq
    output_arrays["K_lqr"] = K_seq
    output_arrays["P_lqr"] = P_seq
    output_arrays["Qf_lqr"] = Qf

    return output_arrays
