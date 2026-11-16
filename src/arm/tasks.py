"""
Задачи (`Task`) для arm-контроллеров.

Сценарий движения для робота: каждый task на каждый момент времени t возвращает, куда должен стремиться робот.

`Task.desired(t, world)` возвращает целевое состояние в момент `t`:
  - `q_target`  (опц., если хотим целиться сразу в суставах)
  - `ee_pos`    — желаемая позиция EE
  - `ee_quat`   — желаемая ориентация EE (w, x, y, z)
  - `gripper`   — открытость гриппера в [0, 1] (1 — открыт)

Контроллеру не обязательно использовать все поля — `arm_joint_pd` берёт
`q_target`, `arm_jacobian_ik` и `arm_osc` — `ee_pos`/`ee_quat`.
"""
from __future__ import annotations

import typing as ty
from dataclasses import dataclass

import numpy as np

from src.arm.world import ArmWorld


# Кватернион "EE смотрит вниз" (как у Panda в home-позе): x вниз по оси Z мира.
# Это (w, x, y, z) = (0, 1, 0, 0) — поворот на pi вокруг X.
QUAT_DOWN = np.array([0.0, 1.0, 0.0, 0.0])


@dataclass
class Desired:
    """
    ee_pos — куда поставить end-effector.
    ee_quat — с какой ориентацией держать end-effector.
    gripper — открыть/закрыть захват.
    q_target — целевые углы суставов, если контроллер работает напрямую в joint-space.
    """
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    gripper: float = 1.0           # 1 — открыт, 0 — закрыт
    q_target: ty.Optional[np.ndarray] = None


class Task:
    """База: подкласс должен реализовать `desired(t, world) -> Desired`."""

    name: str = "task"

    def desired(self, t: float, world: ArmWorld) -> Desired:
        raise NotImplementedError

    def on_reset(self, world: ArmWorld) -> None:
        """Хук, вызываемый перед стартом симуляции (например, поставить куб)."""
        pass


# ---------------------------------------------------------------------- #
# 1) Просто держать home-конфигурацию
# ---------------------------------------------------------------------- #
class HoldHome(Task):
    name = "hold_home"

    def desired(self, t: float, world: ArmWorld) -> Desired:
        return Desired(
            ee_pos=world.ee_pos(),
            ee_quat=world.ee_quat(),
            gripper=1.0,
            q_target=world.q_home.copy(),
        )


# ---------------------------------------------------------------------- #
# 2) Достичь и удерживать одну позу EE
# ---------------------------------------------------------------------- #
@dataclass
class ReachPose(Task):
    pos: ty.Sequence[float] = (0.5, 0.0, 0.4)
    quat: ty.Sequence[float] = (0.0, 1.0, 0.0, 0.0)
    gripper: float = 1.0
    name: str = "reach_pose"

    def desired(self, t: float, world: ArmWorld) -> Desired:
        return Desired(
            ee_pos=np.asarray(self.pos, dtype=float),
            ee_quat=np.asarray(self.quat, dtype=float),
            gripper=self.gripper,
        )


# ---------------------------------------------------------------------- #
# 3) Двигаться по прямой EE между двумя точками с возвратом
# ---------------------------------------------------------------------- #
@dataclass
class FollowEELine(Task):
    """
    EE синусоидально ходит между `p_a` и `p_b` с периодом `period`.
    Ориентация всё время `quat`.
    """
    p_a: ty.Sequence[float] = (0.5, -0.2, 0.4)
    p_b: ty.Sequence[float] = (0.5,  0.2, 0.4)
    quat: ty.Sequence[float] = (0.0, 1.0, 0.0, 0.0)
    period: float = 6.0
    gripper: float = 1.0
    name: str = "ee_line"

    def desired(self, t: float, world: ArmWorld) -> Desired:
        a = np.asarray(self.p_a, dtype=float)
        b = np.asarray(self.p_b, dtype=float)
        # alpha колеблется в [0,1] как 0.5*(1 - cos(2 pi t / T))
        alpha = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / self.period))
        pos = (1.0 - alpha) * a + alpha * b
        return Desired(
            ee_pos=pos,
            ee_quat=np.asarray(self.quat, dtype=float),
            gripper=self.gripper,
        )


# ---------------------------------------------------------------------- #
# 4) Pick & place: ниже куба сценарий из move_to_cube.py, обобщённый
# ---------------------------------------------------------------------- #
@dataclass
class PickAndPlace(Task):
    cube_body: str = "cube"
    cube_joint: str = "cube_freejoint"
    cube_init_pos: ty.Sequence[float] = (0.60, 0.25, 0.035)
    place_xy: ty.Sequence[float] = (0.30, -0.25)

    z_approach: float = 0.45
    z_grab: float = 0.10
    z_lift: float = 0.45

    # T_approach: float = 3.0
    # T_descend: float = 3.0
    # T_close: float = 1.0
    # T_lift: float = 2.0
    # T_carry: float = 5.0
    # T_place_descend: float = 2.0
    # T_release: float = 1.0

    T_approach: float = 1.5
    T_descend: float = 1.5
    T_close: float = 0.5
    T_lift: float = 1.0
    T_carry: float = 2.5
    T_place_descend: float = 1.0
    T_release: float = 0.5

    quat: ty.Sequence[float] = (0.0, 1.0, 0.0, 0.0)
    name: str = "pick_and_place"

    def on_reset(self, world: ArmWorld) -> None:
        world.set_freejoint_pose(self.cube_joint, list(self.cube_init_pos))

        # ВАЖНО: фиксируем начальную позицию куба.
        # После захвата больше не используем world.body_pos(cube),
        # иначе цель будет "убегать" вместе с кубом.
        self._pick_pos = np.asarray(self.cube_init_pos, dtype=float)

    @staticmethod
    def _lerp(a, b, alpha):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return (1.0 - alpha) * a + alpha * b

    def desired(self, t: float, world: ArmWorld) -> Desired:
        pick_pos = np.asarray(self._pick_pos, dtype=float)
        quat = np.asarray(self.quat, dtype=float)

        place_pos = np.array([
            self.place_xy[0],
            self.place_xy[1],
            pick_pos[2],
        ])

        above_pick = pick_pos + np.array([0.0, 0.0, self.z_approach])
        grasp_pick = pick_pos + np.array([0.0, 0.0, self.z_grab])
        lift_pick = pick_pos + np.array([0.0, 0.0, self.z_lift])

        above_place = place_pos + np.array([0.0, 0.0, self.z_lift])
        grasp_place = place_pos + np.array([0.0, 0.0, self.z_grab])

        t0 = 0.0
        t1 = t0 + self.T_approach
        t2 = t1 + self.T_descend
        t3 = t2 + self.T_close
        t4 = t3 + self.T_lift
        t5 = t4 + self.T_carry
        t6 = t5 + self.T_place_descend
        t7 = t6 + self.T_release

        # 1. Навестись над кубом
        if t < t1:
            return Desired(
                ee_pos=above_pick,
                ee_quat=quat,
                gripper=1.0,
            )

        # 2. Опуститься к кубу
        if t < t2:
            alpha = (t - t1) / self.T_descend
            return Desired(
                ee_pos=self._lerp(above_pick, grasp_pick, alpha),
                ee_quat=quat,
                gripper=1.0,
            )

        # 3. Закрыть гриппер
        if t < t3:
            alpha = (t - t2) / self.T_close
            return Desired(
                ee_pos=grasp_pick,
                ee_quat=quat,
                gripper=float(np.clip(1.0 - alpha, 0.0, 1.0)),
            )

        # 4. Поднять куб
        if t < t4:
            alpha = (t - t3) / self.T_lift
            return Desired(
                ee_pos=self._lerp(grasp_pick, lift_pick, alpha),
                ee_quat=quat,
                gripper=0.0,
            )

        # 5. Перенести к месту выкладки
        if t < t5:
            alpha = (t - t4) / self.T_carry
            return Desired(
                ee_pos=self._lerp(lift_pick, above_place, alpha),
                ee_quat=quat,
                gripper=0.0,
            )

        # 6. Опустить куб в точку place
        if t < t6:
            alpha = (t - t5) / self.T_place_descend
            return Desired(
                ee_pos=self._lerp(above_place, grasp_place, alpha),
                ee_quat=quat,
                gripper=0.0,
            )

        # 7. Открыть гриппер
        if t < t7:
            alpha = (t - t6) / self.T_release
            return Desired(
                ee_pos=grasp_place,
                ee_quat=quat,
                gripper=float(np.clip(alpha, 0.0, 1.0)),
            )

        # После release держать позицию
        return Desired(
            ee_pos=grasp_place,
            ee_quat=quat,
            gripper=1.0,
        )


# ---------------------------------------------------------------------- #
# Реестр
# ---------------------------------------------------------------------- #
_TASK_REGISTRY: ty.Dict[str, ty.Type[Task]] = {
    "hold_home": HoldHome,
    "reach_pose": ReachPose,
    "ee_line": FollowEELine,
    "pick_and_place": PickAndPlace,
}


def make_task(name: str, **kwargs) -> Task:
    if name not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Available: {list(_TASK_REGISTRY)}",
        )
    return _TASK_REGISTRY[name](**kwargs)
