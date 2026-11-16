"""
Управление 7-DOF роборукой Panda в симуляции MuJoCo.

Модуль `src.arm` — это тонкая обёртка над MuJoCo (`MjModel`, `MjData`),
которая прячет boilerplate (`mj_name2id`, индексы суставов, доступ к
якобиану и т.п.) и даёт остальным частям проекта удобный API:

    world = ArmWorld(scene_xml="scene.xml")
    world.reset_home()

    ee_pos = world.ee_pos()         # 3D позиция фланца
    J      = world.jacobian_arm()   # 6x7 якобиан (lin/ang) по 7 суставам
    q, dq  = world.q(), world.dq()

    world.set_arm_ctrl(q_target)    # отправить целевые положения суставов
    world.set_gripper(open_=True)
    world.step()

Алгоритмы управления (`src.algorithms.arm_*`) принимают `ArmWorld` и
объект `Task` и на каждом шаге возвращают `ctrl[:7]` + `ctrl[7]` для
гриппера.
"""

from src.arm.world import ArmWorld
from src.arm.tasks import (
    Task,
    HoldHome,
    ReachPose,
    FollowEELine,
    PickAndPlace,
    make_task,
)
from src.arm.runner import ArmRunner

__all__ = [
    "ArmWorld",
    "Task",
    "HoldHome",
    "ReachPose",
    "FollowEELine",
    "PickAndPlace",
    "make_task",
    "ArmRunner",
]
