"""
ArmRunner — главный цикл симуляции MuJoCo для arm-контроллеров.

    runner = ArmRunner(world, controller, task, T=10.0)
    history = runner.run(open_viewer=True)

История содержит на каждом шаге: время, q, dq, ctrl_arm, ee_pos, ee_quat,
e_pos = ||p_des - p_cur||, e_ori (norm-of-axis-angle), gripper-ctrl.
"""
from __future__ import annotations

import time
import typing as ty
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from src.arm.world import ArmWorld
from src.arm.tasks import Task, Desired


class ArmController(ty.Protocol):
    """
    Интерфейс контроллера для роборуки.

    `step(world, desired, dt) -> (q_arm_target_7, gripper_open_in_0_1)`
    """
    name: str

    def step(
        self,
        world: ArmWorld,
        desired: Desired,
        dt: float,
    ) -> ty.Tuple[np.ndarray, float]: ...

    def reset(self, world: ArmWorld) -> None: ...


@dataclass
class ArmRunner:
    world: ArmWorld
    controller: ArmController
    task: Task
    T: float = 10.0
    realtime: bool = True

    def run(self, open_viewer: bool = True) -> ty.Dict[str, np.ndarray]:
        world = self.world
        world.reset_home()
        self.task.on_reset(world)
        self.controller.reset(world)

        n_steps = int(self.T / world.dt)
        hist = _HistoryBuffer(n_steps + 1)

        if open_viewer:
            world.open_viewer()

        try:
            for k in tqdm(range(n_steps), desc=f"{self.controller.name}/{self.task.name}"):
                t = world.t
                desired = self.task.desired(t, world)

                wall_t0 = time.time()
                q_target, gripper = self.controller.step(world, desired, world.dt)
                wall_dt = time.time() - wall_t0

                q_applied = world.set_arm_ctrl(q_target)
                gripper_applied = world.set_gripper_norm(gripper)

                hist.append(
                    t=t,
                    q=world.q(),
                    dq=world.dq(),
                    q_target=q_applied,
                    gripper=gripper_applied,
                    ee_pos=world.ee_pos(),
                    ee_quat=world.ee_quat(),
                    ee_pos_d=desired.ee_pos.copy(),
                    ee_quat_d=desired.ee_quat.copy(),
                    calc_time=wall_dt,
                )

                world.step()

                if open_viewer:
                    if not world.viewer_running():
                        break
                    world.viewer_sync()
                    if self.realtime:
                        time.sleep(max(0.0, world.dt - wall_dt))
        finally:
            world.close_viewer()

        return hist.as_dict()


# ---------------------------------------------------------------------- #
# История
# ---------------------------------------------------------------------- #
class _HistoryBuffer:

    KEYS: ty.Tuple[str, ...] = (
        "t", "q", "dq", "q_target", "gripper",
        "ee_pos", "ee_quat", "ee_pos_d", "ee_quat_d", "calc_time",
    )

    def __init__(self, capacity: int):
        self._lists: ty.Dict[str, list] = {k: [] for k in self.KEYS}

    def append(self, **kwargs) -> None:
        for k in self.KEYS:
            self._lists[k].append(kwargs[k])

    def as_dict(self) -> ty.Dict[str, np.ndarray]:
        out: ty.Dict[str, np.ndarray] = {}
        for k, v in self._lists.items():
            out[k] = np.array(v)
        # удобные производные
        if len(self._lists["ee_pos"]) > 0:
            out["e_pos"] = np.linalg.norm(out["ee_pos_d"] - out["ee_pos"], axis=1)
        return out
