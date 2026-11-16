"""
Реестр arm-контроллеров и их параметров.

Используется в `launch_arm.py` для создания контроллера по строке-имени.
Каждая запись — kwargs, передаваемые в конструктор соответствующего
класса контроллера.
"""
import numpy as np

from src.algorithms.arm_joint_pd.control import JointPDController
from src.algorithms.arm_jacobian_ik.control import JacobianIKController
from src.algorithms.arm_osc.control import OSCController
from src.algorithms.arm_se3_pid.control import SE3PIDController
from src.algorithms.arm_se3_lqr.control import SE3LQRController


arm_control_map = {

    # «Базовые» контроллеры, не использующие SE(3)-математику проекта.
    "arm_joint_pd": {
        "cls": JointPDController,
        "params": {
            "kp": 1.0,
            "max_step": 0.02,
        },
    },

    "arm_jacobian_ik": {
        "cls": JacobianIKController,
        "params": {
            "gain_pos": 0.6,
            "gain_ori": 0.4,
            "damping": 0.05,
            "posture_gain": 0.02,
            "max_step": 0.01,
        },
    },

    "arm_osc": {
        "cls": OSCController,
        "params": {
            "kp_pos": 6.0,
            "kd_pos": 2.0,
            "kp_ori": 4.0,
            "kd_ori": 1.0,
            "damping": 0.05,
            "k_null": 0.5,
            "dq_max": 0.02,
        },
    },

    # ------------------------------------------------------------------ #
    # SE(3)-семейство: переиспользует математику из src/utils/math.py
    # (error, hat, unhat) и формулу PID/LQR из src/algorithms/{pid,lqr}_*.
    # Архитектура:
    #     q,dq ─FK→ X_ee ─log→ ψ ─{PID|LQR}→ u_6D ─J_b^#→ dq ─∫→ q_cmd
    # ------------------------------------------------------------------ #
    "arm_se3_pid": {
        "cls": SE3PIDController,
        "params": {
            # Та же конвенция знаков, что в pid_v1: Kp/Kd/Ki ≤ 0.
            # Порядок ψ — [θ_x, θ_y, θ_z, ρ_x, ρ_y, ρ_z]
            # (угловые сначала, потом translational rho).

            "Kp": -np.array([8.0,  8.0,  8.0, 30.0, 30.0, 30.0]),
            "Kd": -np.array([2.0,  2.0,  2.0,  6.0,  6.0,  6.0]),
            "Ki": -np.array([0.5,  0.5,  0.5,  2.0,  2.0,  2.0]),
            # 'Kp': -np.array([500, 500, 500, 200, 200, 200]),
            # 'Kd': -np.array([50, 50, 50, 50, 50, 50]),
            # 'Ki': -np.array([200, 200, 200, 100, 100, 100]),

            "damping": 0.05,
            "posture_gain": 0.2,
            "dq_max": 0.03,
        },
    },

    "arm_se3_lqr": {
        "cls": SE3LQRController,
        "params": {
            "Q": np.diag([
                80.0, 80.0, 80.0, 200.0, 200.0, 200.0,    # ψ-стоимость
                5.0,  5.0,  5.0,  20.0,  20.0,  20.0,    # ψ̇-стоимость
            ]),
            "R": np.eye(6) * 0.05,
            "damping": 0.05,
            "posture_gain": 0.2,
            "dq_max": 0.03,
        },
    },
}


def make_arm_controller(name: str, overrides: dict | None = None):
    if name not in arm_control_map:
        raise ValueError(
            f"Unknown arm controller '{name}'. Available: {list(arm_control_map)}",
        )
    entry = arm_control_map[name]
    params = dict(entry["params"])
    if overrides:
        params.update(overrides)
    return entry["cls"](**params)
