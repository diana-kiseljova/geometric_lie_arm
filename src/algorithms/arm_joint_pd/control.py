"""
Joint-PD контроллер для роборуки.

Идея простая: задача `Task.desired()` опционально возвращает `q_target` —
целевую конфигурацию суставов. Контроллер на каждом шаге сдвигает
текущее управление `ctrl` в сторону `q_target` с ограничением скорости
(`max_step` рад/такт). Если задача не задаёт `q_target`, контроллер
держит home.

Поскольку у Panda актуаторы уже PD-позиционные (см. `panda.xml`,
`<general gainprm="..." biasprm="0 -K -D"/>`), контроллеру достаточно
выдавать желаемое `q` — низкоуровневый PD внутри MuJoCo выполняет PD по
ошибке `(ctrl - qpos)`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.arm.world import ArmWorld
from src.arm.tasks import Desired


@dataclass
class JointPDController:
    """
    Параметры
    ---------
    kp        : коэффициент пропорционального члена (на ошибку q_target − q_cmd)
    max_step  : максимальный сдвиг `ctrl` за такт, рад
    """
    kp: float = 1.0
    max_step: float = 0.02
    name: str = "arm_joint_pd"

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
        """
        На каждом шаге обновляется внутренняя команда q_cmd:

                q_cmd_next = q_cmd + clip(kp * (q_target - q_cmd),
                                        -max_step, max_step)

            где:
                q_cmd    — текущая команда, подаваемая в позиционные актуаторы;
                q_target — целевая конфигурация суставов из Desired.q_target;
                kp       — коэффициент движения к цели;
                max_step — максимальный сдвиг команды за один такт, рад.
        
        Дифференциальная часть находится ниже — внутри MuJoCo actuator.
        <general gainprm="K" biasprm="0 -K -D"/>
        tau = K(q_cmd - qpos) - D qvel

        Верхний контроллер:
        q_cmd_next = q_cmd + clip(kp * (q_target - q_cmd), ...)
        Нижний MuJoCo PD (q_cmd_next из верхнего контроллера становится ctrl для нижнего):
        tau = K(q_cmd - qpos) - D qvel

        Не реализует PD напрямую, а только выдаёт позиционную команду для уже существующего PD-актуатора!
        """
        q_target = desired.q_target if desired.q_target is not None else world.q_home
        delta = self.kp * (q_target - self._q_cmd)
        delta = np.clip(delta, -self.max_step, self.max_step)
        self._q_cmd = self._q_cmd + delta
        self._q_cmd = np.clip(self._q_cmd, world.ctrl_min, world.ctrl_max)
        return self._q_cmd.copy(), float(desired.gripper)
