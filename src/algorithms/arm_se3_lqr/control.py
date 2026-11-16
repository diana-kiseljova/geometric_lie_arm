"""
SE(3)-LQR для роборуки (LTI baseline).

Состояние ошибки:

    x = [ψ; ψ̇] ∈ R^12,
        ψ   = Log(X_d^{-1} X)              ∈ R^6   (body frame)
        ψ̇  ≈ ξ − Ad_{Ψ^{-1}} ξ_d         ∈ R^6

Управление u ∈ R^6 — желаемая body-twist «псевдо-скорость».
Используется та же *локальная линеаризация*, что и в
[src/algorithms/lqr_v4/control.py](../../algorithms/lqr_v4/control.py):
double-integrator с единичной матрицей инерции (LTI baseline).

Непрерывная модель:

    ẋ = A_c · x + B_c · u
    A_c = [ [ 0, I_6 ],              B_c = [ [  0  ],
            [ 0,  0  ] ]                     [ I_6 ]  ]

Discrete-time (шаг dt):

    A = exp(A_c·dt) = [[ I, I·dt ],     B = ∫_0^dt e^{A_c τ} B_c dτ =
                       [ 0,  I  ]]            [[ 0.5·I·dt² ],
                                               [    I·dt   ]]

`K` находится из дискретного DARE один раз при `reset()` и применяется
на каждом шаге:

    u = −K · x_err
    dq_cmd = J_b^# · u
    q_cmd += dq_cmd · dt

Это **тот же LQR**, что в `lqr_v4`, только:
    - вход не общий wrench, а 6D «команда body-twist рабочего органа»;
    - выход проецируется в суставы через body-Jacobian.

Та самая архитектура:
    q, dq ─FK→ X_ee ─log→ ψ ∈R^6 ─LQR→ u ─J_b^#→ u_joint
"""
from __future__ import annotations

from dataclasses import dataclass, field
import typing as ty

import numpy as np
from scipy.linalg import solve_discrete_are

from src.arm.world import ArmWorld
from src.arm.tasks import Desired
from src.arm.se3_bridge import (
    world_to_X,
    desired_to_X,
    body_jacobian,
    body_twist,
    psi_and_psidot,
    dls_pinv,
)


def _build_discrete_lti(dt: float):
    """A, B для дискретного двойного интегратора на R^12."""
    I6 = np.eye(6)
    O6 = np.zeros((6, 6))

    A = np.block([
        [I6, dt * I6],
        [O6,     I6 ],
    ])
    B = np.block([
        [0.5 * dt * dt * I6],
        [          dt * I6 ],
    ])
    return A, B


@dataclass
class SE3LQRController:
    """
    Параметры
    ---------
    Q            : (12, 12) — стоимость состояния  diag([Q_ψ ; Q_ψ̇])
    R            : (6, 6)   — стоимость управления
    damping      : λ в DLS-псевдообратной J_b^#.
    posture_gain : усиление nullspace-задачи.
    dq_max       : предел нормы шага сустава за такт (рад).
    """
    Q: np.ndarray = field(
        default_factory=lambda: np.diag(
            # [θ_x θ_y θ_z  ρ_x ρ_y ρ_z   ω_x ω_y ω_z   v_x v_y v_z]
            [
                80.0, 80.0, 80.0, 200.0, 200.0, 200.0,
                5.0,  5.0,  5.0,  20.0,  20.0,  20.0
            ]
        )
    )
    R: np.ndarray = field(default_factory=lambda: np.eye(6) * 0.05)
    damping: float = 0.05
    posture_gain: float = 0.2
    dq_max: float = 0.03
    name: str = "arm_se3_lqr"

    def __post_init__(self) -> None:
        self._q_cmd: np.ndarray = np.zeros(7)
        self._K: np.ndarray | None = None
        self._dt_used: float | None = None

    def _ensure_K(self, dt: float) -> None:
        if self._K is not None and self._dt_used == dt:
            return
        A, B = _build_discrete_lti(dt)
        # Дискретное DARE: P = A^T P A − A^T P B (R + B^T P B)^{-1} B^T P A + Q
        P = solve_discrete_are(A, B, np.asarray(self.Q), np.asarray(self.R))
        # K такой, что u = −K · x_err  минимизирует Σ (x^T Q x + u^T R u)
        self._K = np.linalg.solve(
            np.asarray(self.R) + B.T @ P @ B,
            B.T @ P @ A,
        )
        self._dt_used = dt

    def reset(self, world: ArmWorld) -> None:
        self._q_cmd = world.q().copy()
        # K посчитаем лениво при первом step (там известен dt).

    def step(
        self,
        world: ArmWorld,
        desired: Desired,
        dt: float,
    ):
        self._ensure_K(dt)
        world.forward()

        # 1) FK -> SE(3) ошибка
        X = world_to_X(world)
        X_d = desired_to_X(desired.ee_pos, desired.ee_quat)
        J_b = body_jacobian(world)
        xi = J_b @ world.dq()
        Psi, psi, psi_dot = psi_and_psidot(X, X_d, xi, xi_d=None)

        # 2) LQR: u = −K · x,  x = [ψ; ψ̇]
        x_err = np.concatenate([psi, psi_dot])
        u_des = -self._K @ x_err                  # 6D body-frame команда

        # 3) Проекция в dq через DLS J_b^#
        J_pinv = dls_pinv(J_b, self.damping)
        dq_task = J_pinv @ u_des

        # 4) Nullspace posture
        N = np.eye(7) - J_pinv @ J_b
        dq_null = self.posture_gain * (world.q_home - self._q_cmd)

        # 5) Интеграция
        step = (dq_task + N @ dq_null) * dt
        step = np.clip(step, -self.dq_max, self.dq_max)
        self._q_cmd = np.clip(self._q_cmd + step, world.ctrl_min, world.ctrl_max)
        return self._q_cmd.copy(), float(desired.gripper)
