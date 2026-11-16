"""
Jacobian-IK контроллер с регуляризацией (Damped Least Squares).

На каждом шаге:
    err = desired.ee_pos (− и опц. ориентация) − текущая поза EE
    dq  = J^T (J J^T + λ² I)^{-1} (gain * err)
    q_cmd += dq    (с ограничением по `max_step` и nullspace-приведением
                    к home-конфигурации для устранения вырожденных решений)

Контроллер сам по себе ничего не знает о PID/LQR — он минимальный
обратный шаг кинематики. Подходит для задач `ReachPose`, `FollowEELine`,
`PickAndPlace`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.arm.world import ArmWorld
from src.arm.tasks import Desired
from src.arm.kinematics import dls_step, orientation_error


@dataclass
class JacobianIKController:
    """
    Параметры
    ---------
    gain_pos       : коэффициент IK по позиции
    gain_ori       : коэффициент IK по ориентации (0 — отключить)
    damping        : λ в Damped Least Squares
    posture_gain   : подтягивание к home-конфигурации в nullspace
    max_step       : предел шага по каждому суставу за такт (рад)
    """
    gain_pos: float = 0.6
    gain_ori: float = 0.4
    damping: float = 5e-2
    posture_gain: float = 0.02
    max_step: float = 0.01
    name: str = "arm_jacobian_ik"

    def __post_init__(self) -> None:
        self._q_cmd: np.ndarray = np.zeros(7)

    def reset(self, world: ArmWorld) -> None:
        self._q_cmd = world.q().copy()

    def step(
        self,
        world: ArmWorld,
        desired: Desired,
        dt: float,
    ):
        # mj_jacBody читает текущее состояние data
        world.forward()

        if self.gain_ori > 0.0:
            J = world.jacobian_arm()              # 6x7
            err = np.zeros(6)
            err[:3] = self.gain_pos * (desired.ee_pos - world.ee_pos())
            err[3:] = self.gain_ori * orientation_error(world.ee_quat(), desired.ee_quat)
            dq_task = dls_step(J, err, damping=self.damping, gain=1.0)
            # nullspace для posture
            pinvJ = J.T @ np.linalg.inv(J @ J.T + (self.damping ** 2) * np.eye(6))
            N = np.eye(7) - pinvJ @ J
        else:
            J = world.jacobian_arm_pos()          # 3x7
            err_lin = self.gain_pos * (desired.ee_pos - world.ee_pos())
            dq_task = dls_step(J, err_lin, damping=self.damping, gain=1.0)
            pinvJ = J.T @ np.linalg.inv(J @ J.T + (self.damping ** 2) * np.eye(3))
            N = np.eye(7) - pinvJ @ J

        dq_posture = self.posture_gain * (world.q_home - self._q_cmd)
        dq = dq_task + N @ dq_posture
        dq = np.clip(dq, -self.max_step, self.max_step)

        self._q_cmd = self._q_cmd + dq
        self._q_cmd = np.clip(self._q_cmd, world.ctrl_min, world.ctrl_max)
        return self._q_cmd.copy(), float(desired.gripper)
