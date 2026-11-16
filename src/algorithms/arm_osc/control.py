"""
Operational Space Control (OSC) для роборуки.

Реализация на уровне позиционных ctrl-команд (поскольку актуаторы Panda
PD-позиционные). Логика:

  1) Считаем 6D скорость EE, которая привела бы к достижению цели:
         v_des = Kp * (p_d - p) + Kd * (-v)
         w_des = Kp_r * orientation_error(q, q_d) + Kd_r * (-omega)
  2) Решаем задачу в пространстве суставов:
         dq_task = J^# * [v_des; w_des]      (с DLS-регуляризацией)
  3) Добавляем секундарную задачу в нулевом пространстве — стабилизация
     к home-конфигурации:
         dq_null = K_null * (q_home - q_cmd)
  4) Интегрируем `q_cmd += (dq_task + N dq_null) * dt`, ограничиваем по
     скорости, по диапазону управления.

В отличие от Jacobian-IK здесь PD строится во времени, поэтому KP/KD
имеют осмысленные единицы (1/s и безразмерные).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.arm.world import ArmWorld
from src.arm.tasks import Desired
from src.arm.kinematics import orientation_error


@dataclass
class OSCController:
    """
    Параметры
    ---------
    kp_pos, kd_pos : усиления PD по позиции EE (1/s, безразм.)
    kp_ori, kd_ori : усиления PD по ориентации EE
    damping        : λ в Damped Least Squares решении J # = J^T (JJ^T + λ²I)^{-1}
    k_null         : усиление nullspace-задачи (тянем q_cmd к home)
    dq_max         : предел нормы шага по суставам за такт (рад)
    """
    kp_pos: float = 6.0
    kd_pos: float = 2.0
    kp_ori: float = 4.0
    kd_ori: float = 1.0
    damping: float = 0.05
    k_null: float = 0.5
    dq_max: float = 0.02
    name: str = "arm_osc"

    def __post_init__(self) -> None:
        self._q_cmd: np.ndarray = np.zeros(7)
        self._prev_ee_pos: np.ndarray = np.zeros(3)
        self._prev_ee_quat: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])

    def reset(self, world: ArmWorld) -> None:
        self._q_cmd = world.q().copy()
        world.forward()
        self._prev_ee_pos = world.ee_pos()
        self._prev_ee_quat = world.ee_quat()

    def step(
        self,
        world: ArmWorld,
        desired: Desired,
        dt: float,
    ):
        world.forward()

        # Состояние EE
        p = world.ee_pos()
        q_ee = world.ee_quat()
        # Оценка линейной/угловой скорости через якобиан и dq
        J = world.jacobian_arm()                  # 6x7
        dq = world.dq()                           # 7
        v = J[:3] @ dq                            # 3
        w = J[3:] @ dq                            # 3

        # PD на 6D-ошибку
        e_pos = desired.ee_pos - p
        e_ori = orientation_error(q_ee, desired.ee_quat)

        v_des = self.kp_pos * e_pos + self.kd_pos * (-v)
        w_des = self.kp_ori * e_ori + self.kd_ori * (-w)
        x_dot = np.concatenate([v_des, w_des])    # 6

        # DLS pseudo-inverse: dq_task = J^T (J J^T + λ² I)^{-1} x_dot
        A = J @ J.T + (self.damping ** 2) * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(A)
        dq_task = J_pinv @ x_dot                  # 7

        # Nullspace: затягиваем к home
        N = np.eye(7) - J_pinv @ J
        dq_null = self.k_null * (world.q_home - self._q_cmd)

        dq_cmd = dq_task + N @ dq_null            # 7 (рад/с)

        # Интегрируем в позиционное управление
        step = dq_cmd * dt
        # ограничение нормы шага (мягкое — поэлементное)
        step = np.clip(step, -self.dq_max, self.dq_max)
        self._q_cmd = self._q_cmd + step
        self._q_cmd = np.clip(self._q_cmd, world.ctrl_min, world.ctrl_max)

        self._prev_ee_pos = p
        self._prev_ee_quat = q_ee
        return self._q_cmd.copy(), float(desired.gripper)
