"""
ArmWorld — обёртка над MuJoCo для управления роборукой Panda.

Хранит `MjModel`/`MjData`, индексы 7 шарниров руки и гриппера,
ID кисти `hand` и опционально дополнительные body (например, "cube"),
а также диапазоны управления.

Сценарий использования (см. `src/arm/runner.py`):

    world = ArmWorld("scene.xml")
    world.reset_home()
    while world.viewer_running():
        target = ...                      # из контроллера
        world.set_arm_ctrl(target_q)
        world.set_gripper_norm(0.0)       # 0 — закрыть, 1 — открыть
        world.step()
"""
from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


GRIPPER_OPEN_CTRL = 255.0
GRIPPER_CLOSED_CTRL = 0.0


class ArmWorld:
    """
    Доступ к 7-DOF Panda и сцене в MuJoCo.

    Параметры
    ---------
    scene_xml : путь к XML сцены (по умолчанию `scene.xml` рядом с проектом).
    ee_body   : имя body, считающегося рабочим органом (`hand` по умолчанию).
    extra_bodies : доп. имена body, чьи позиции нужно отдавать наружу
                   (например, `["cube"]`).
    """

    ARM_JOINT_NAMES: ty.Tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
    ARM_ACTUATOR_NAMES: ty.Tuple[str, ...] = tuple(f"actuator{i}" for i in range(1, 8))
    GRIPPER_ACTUATOR_NAME: str = "actuator8"
    HOME_KEY_NAME: str = "home"

    def __init__(
        self,
        scene_xml: ty.Union[str, Path] = "scene.xml",
        ee_body: str = "hand",
        extra_bodies: ty.Sequence[str] = (),
    ):
        self.scene_xml = str(Path(scene_xml).resolve())
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.data = mujoco.MjData(self.model)

        # IDs суставов руки и их адреса в qpos / qvel
        self._arm_joint_ids = np.array(
            [self._joint_id(n) for n in self.ARM_JOINT_NAMES], dtype=int,
        )
        self.qpos_arm_ids = np.array(
            [self.model.jnt_qposadr[j] for j in self._arm_joint_ids], dtype=int,
        )
        self.qvel_arm_ids = np.array(
            [self.model.jnt_dofadr[j] for j in self._arm_joint_ids], dtype=int,
        )

        # IDs актуаторов (= индексы в data.ctrl)
        self.arm_actuator_ids = np.array(
            [self._actuator_id(n) for n in self.ARM_ACTUATOR_NAMES], dtype=int,
        )
        self.gripper_actuator_id = self._actuator_id(self.GRIPPER_ACTUATOR_NAME)

        # Диапазоны управления
        self.ctrl_min = self.model.actuator_ctrlrange[self.arm_actuator_ids, 0].copy()
        self.ctrl_max = self.model.actuator_ctrlrange[self.arm_actuator_ids, 1].copy()

        # End-effector body
        self.ee_body_id = self._body_id(ee_body)
        self.ee_body_name = ee_body

        # Дополнительные body
        self.extra_body_ids: ty.Dict[str, int] = {
            name: self._body_id(name) for name in extra_bodies
        }

        # Home keyframe (опционально)
        self.home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, self.HOME_KEY_NAME,
        )

        # Запомним home-конфигурацию рукоятки (для posture-task в OSC).
        if self.home_key_id != -1:
            self.q_home = self.model.key_qpos[self.home_key_id, self.qpos_arm_ids].copy()
        else:
            self.q_home = np.zeros(7)

        self._viewer: ty.Optional[ty.Any] = None

    # ------------------------------------------------------------------ #
    # Поиск ID
    # ------------------------------------------------------------------ #
    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise ValueError(f"Joint '{name}' not found in {self.scene_xml}")
        return jid

    def _actuator_id(self, name: str) -> int:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid == -1:
            raise ValueError(f"Actuator '{name}' not found in {self.scene_xml}")
        return aid

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            raise ValueError(f"Body '{name}' not found in {self.scene_xml}")
        return bid

    # ------------------------------------------------------------------ #
    # Сброс / шаг
    # ------------------------------------------------------------------ #
    def reset_home(self) -> None:
        if self.home_key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
            self.data.ctrl[:] = self.model.key_ctrl[self.home_key_id]
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        """Пересчёт кинематики/якобианов без интеграции по времени."""
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def t(self) -> float:
        return float(self.data.time)

    # ------------------------------------------------------------------ #
    # Состояние руки
    # ------------------------------------------------------------------ #
    def q(self) -> np.ndarray:
        """Текущие 7 углов суставов руки."""
        return self.data.qpos[self.qpos_arm_ids].copy()

    def dq(self) -> np.ndarray:
        """Текущие скорости 7 суставов руки."""
        return self.data.qvel[self.qvel_arm_ids].copy()

    def ee_pos(self) -> np.ndarray:
        """Позиция EE в мировой СК."""
        return self.data.xpos[self.ee_body_id].copy()

    def ee_rotmat(self) -> np.ndarray:
        """3x3 матрица ориентации EE."""
        return self.data.xmat[self.ee_body_id].reshape(3, 3).copy()

    def ee_quat(self) -> np.ndarray:
        """Кватернион (w, x, y, z) ориентации EE."""
        return self.data.xquat[self.ee_body_id].copy()

    def body_pos(self, name: str) -> np.ndarray:
        bid = self.extra_body_ids.get(name)
        if bid is None:
            bid = self._body_id(name)
            self.extra_body_ids[name] = bid
        return self.data.xpos[bid].copy()

    def jacobian_arm(self) -> np.ndarray:
        """
        6x7 якобиан рабочего органа по 7 суставам руки.

        Верхние 3 строки — линейная скорость, нижние 3 — угловая.
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_body_id)
        J = np.vstack([jacp[:, self.qvel_arm_ids], jacr[:, self.qvel_arm_ids]])
        return J

    def jacobian_arm_pos(self) -> np.ndarray:
        """3x7 позиционный якобиан EE."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_body_id)
        return jacp[:, self.qvel_arm_ids]

    # ------------------------------------------------------------------ #
    # Управление
    # ------------------------------------------------------------------ #
    def set_arm_ctrl(self, q_target: np.ndarray) -> np.ndarray:
        """
        Записать целевые положения 7 суставов в `data.ctrl`.

        Возвращает фактически применённое (после клиппинга) управление.
        """
        q_clipped = np.clip(q_target, self.ctrl_min, self.ctrl_max)
        self.data.ctrl[self.arm_actuator_ids] = q_clipped
        return q_clipped

    def set_gripper_ctrl(self, value: float) -> float:
        """Прямое значение управления гриппером в [0, 255]."""
        gr_min = float(self.model.actuator_ctrlrange[self.gripper_actuator_id, 0])
        gr_max = float(self.model.actuator_ctrlrange[self.gripper_actuator_id, 1])
        v = float(np.clip(value, gr_min, gr_max))
        self.data.ctrl[self.gripper_actuator_id] = v
        return v

    def set_gripper_norm(self, openness: float) -> float:
        """
        Управление гриппером в нормированных координатах:
        `openness=0.0` — закрыт, `openness=1.0` — полностью открыт.
        """
        openness = float(np.clip(openness, 0.0, 1.0))
        return self.set_gripper_ctrl(
            GRIPPER_CLOSED_CTRL + openness * (GRIPPER_OPEN_CTRL - GRIPPER_CLOSED_CTRL),
        )

    # ------------------------------------------------------------------ #
    # Установка позы куба (для сцен pick-and-place)
    # ------------------------------------------------------------------ #
    def set_freejoint_pose(
        self,
        joint_name: str,
        pos: ty.Sequence[float],
        quat: ty.Sequence[float] = (1, 0, 0, 0),
    ) -> None:
        joint_id = self._joint_id(joint_name)
        qpos_adr = self.model.jnt_qposadr[joint_id]
        qvel_adr = self.model.jnt_dofadr[joint_id]
        self.data.qpos[qpos_adr:qpos_adr + 3] = np.asarray(pos, dtype=float)
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = np.asarray(quat, dtype=float)
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ #
    # Viewer
    # ------------------------------------------------------------------ #
    def open_viewer(self) -> None:
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def close_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def viewer_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def viewer_sync(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()

    def __enter__(self) -> "ArmWorld":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_viewer()
