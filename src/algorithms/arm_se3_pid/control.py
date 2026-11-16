"""
SE(3)-PID для роборуки.

Идея — ровно та архитектура, что в README_arm.md §2.2:

    q, dq ─FK→ X_ee ─log→ ψ ∈ R^6 ─PID→ u_6D (body frame)
                                    ─J_b^#→ dq_cmd ─∫→ q_cmd ─ctrl→ MuJoCo

`PID-блок` устроен **в точности так же**, как в pid_v1
([src/algorithms/pid_v1/control.py](../../algorithms/pid_v1/control.py),
строки 41-48):

    I += ψ · dt
    u_raw = Kp · ψ + Kd · ψ̇ + Ki · I

Только теперь это применяется к ошибке *рабочего органа* (а не к
6-DOF твёрдому телу в свободном пространстве), и вместо ODE-интегратора
выход проецируется в суставы.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import typing as ty

import numpy as np

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


@dataclass
class SE3PIDController:
    """
    Параметры
    ---------
    Kp, Kd, Ki   : коэффициенты PID на 6D-ψ (можно скаляр или (6,)-вектор).
                   ВНИМАНИЕ: знаки выбраны так, что u_des = Kp·ψ + Kd·ψ̇ + Ki·I
                   должен указывать в направлении уменьшения ошибки.  ψ
                   определяется как Log(X_d^{-1} X) и **положителен** в
                   сторону цели в body-frame, поэтому Kp, Kd, Ki берутся
                   ОТРИЦАТЕЛЬНЫМИ (как в pid_v1: -np.array([…])).
    damping      : λ в DLS-псевдообратной J_b^#.
    posture_gain : усиление nullspace-задачи (приведение к home).
    dq_max       : предел нормы шага сустава за такт (рад).
    """
    Kp: ty.Union[float, np.ndarray] = field(
        default_factory=lambda: -np.array([8.0, 8.0, 8.0, 30.0, 30.0, 30.0])
    )
    Kd: ty.Union[float, np.ndarray] = field(
        default_factory=lambda: -np.array([2.0, 2.0, 2.0, 6.0, 6.0, 6.0])
    )
    Ki: ty.Union[float, np.ndarray] = field(
        default_factory=lambda: -np.array([0.5, 0.5, 0.5, 2.0, 2.0, 2.0])
    )
    damping: float = 0.05
    posture_gain: float = 0.2
    dq_max: float = 0.03
    name: str = "arm_se3_pid"

    def __post_init__(self) -> None:
        self._q_cmd: np.ndarray = np.zeros(7)
        self._integral: np.ndarray = np.zeros(6)

    def reset(self, world: ArmWorld) -> None:
        self._q_cmd = world.q().copy()
        self._integral = np.zeros(6)

    def step(
        self,
        world: ArmWorld,
        desired: Desired,
        dt: float,
    ):
        world.forward()

        # 1) FK -> SE(3) ошибка
        X = world_to_X(world)
        X_d = desired_to_X(desired.ee_pos, desired.ee_quat)
        J_b = body_jacobian(world)
        xi = J_b @ world.dq()
        Psi, psi, psi_dot = psi_and_psidot(X, X_d, xi, xi_d=None)

        # 2) PID на ψ — формула из pid_v1
        self._integral += psi * dt
        u_des = (
            np.asarray(self.Kp) * psi
            + np.asarray(self.Kd) * psi_dot
            + np.asarray(self.Ki) * self._integral
        )

        # 3) Проекция u в суставную скорость: dq = J_b^# · u (DLS)
        J_pinv = dls_pinv(J_b, self.damping)
        dq_task = J_pinv @ u_des                 # 7

        # 4) Nullspace posture (тянем к home)
        N = np.eye(7) - J_pinv @ J_b
        dq_null = self.posture_gain * (world.q_home - self._q_cmd)

        # 5) Интеграция в позиционное управление, клипы
        step = (dq_task + N @ dq_null) * dt
        step = np.clip(step, -self.dq_max, self.dq_max)
        self._q_cmd = np.clip(self._q_cmd + step, world.ctrl_min, world.ctrl_max)

        return self._q_cmd.copy(), float(desired.gripper)
