"""
SE(3)-мост между состоянием роборуки в MuJoCo и алгоритмами,
работающими на ошибке EE в SE(3).

Архитектурно (см. README_arm.md, §2.2):

    q, dq ∈ R^7  ─FK→  X_ee(q) ∈ SE(3)  ─log→  psi ∈ R^6  ─J_b^#→  u_joint

Эта обёртка прячет три преобразования:

1. `world_to_X`   — берёт `mj_data.xpos[hand]` и `xquat[hand]` и
                    строит 4×4 SE(3) матрицу `X_ee`.
2. `body_jacobian`— возвращает body-frame Jacobian J_b такой, что
                    body-twist ξ = [ω_b; v_b] = J_b · dq.  Это та же
                    конвенция, что у `src/utils/math.hat`
                    (`xi = [omega, v]`) и у dynamics/kin_dyn.
3. `desired_to_X` — берёт `Desired.ee_pos, ee_quat` и строит X_d.

Дополнительно — `psi_dot_estimate`, использующий ту же формулу, что
`pid_v1.control.run_pid` (см. строки 67-68 этого файла):

    psi_dot ≈ unhat(hat(xi) − Ψ^{-1} hat(xi_d) Ψ)

Для статической задачи (ξ_d = 0) это упрощается до `psi_dot = xi`.
"""
from __future__ import annotations

import typing as ty

import numpy as np
import mujoco

from src.arm.world import ArmWorld
from src.utils.math import R_p_to_X, quat2rotm, error, unhat, hat


# ---------------------------------------------------------------------- #
# Поза EE  ->  SE(3) матрица
# ---------------------------------------------------------------------- #
def world_to_X(world: ArmWorld) -> np.ndarray:
    """X_ee = [R(q_ee), p_ee; 0, 1] из текущего состояния MuJoCo."""
    R = world.ee_rotmat()
    p = world.ee_pos()
    return R_p_to_X(R, p)


def desired_to_X(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """X_d из 3D-позиции и кватерниона (w, x, y, z) в формате MuJoCo."""
    R_d = quat2rotm(quat)
    return R_p_to_X(R_d, np.asarray(pos, dtype=float))


# ---------------------------------------------------------------------- #
# Body-frame Jacobian
# ---------------------------------------------------------------------- #
def body_jacobian(world: ArmWorld) -> np.ndarray:
    """
    Возвращает J_b ∈ R^{6×7} в конвенции `xi = [omega, v]` (body frame).

    MuJoCo's `mj_jacBody` отдаёт world-frame блоки:
        jacp · dq = dp/dt   (linear velocity in world frame)
        jacr · dq = ω_s     (angular velocity in world frame)

    Body-frame twist получается переводом обеих частей через R^T:
        ω_b = R^T · ω_s
        v_b = R^T · dp/dt

    Поэтому
        J_b = [[R^T · jacr],
               [R^T · jacp]]
    выбранный по 7 столбцам, соответствующим суставам руки.
    """
    jacp = np.zeros((3, world.model.nv))
    jacr = np.zeros((3, world.model.nv))
    mujoco.mj_jacBody(world.model, world.data, jacp, jacr, world.ee_body_id)

    Jp = jacp[:, world.qvel_arm_ids]    # 3x7  — линейная часть, world
    Jr = jacr[:, world.qvel_arm_ids]    # 3x7  — угловая часть, world

    R = world.ee_rotmat()
    Rt = R.T
    Jb = np.vstack([Rt @ Jr, Rt @ Jp])  # 6x7  — [ω_b; v_b]
    return Jb


def body_twist(world: ArmWorld) -> np.ndarray:
    """ξ = J_b · dq  ∈ R^6, конвенция [ω_b, v_b]."""
    return body_jacobian(world) @ world.dq()


# ---------------------------------------------------------------------- #
# psi и psi_dot
# ---------------------------------------------------------------------- #
def psi_and_psidot(
    X: np.ndarray,
    X_d: np.ndarray,
    xi: np.ndarray,
    xi_d: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает (Ψ, ψ, ψ̇), где:

        Ψ   = X_d^{-1} · X            ∈ SE(3),       4×4
        ψ   = unhat(log Ψ)            ∈ R^6,         [ω_b, v_b]-конвенция
        ψ̇  = unhat( hat(ξ) − Ψ^{-1} · hat(ξ_d) · Ψ )

    Если ξ_d = None или нули, формула упрощается до ψ̇ = ξ.
    """
    Psi, psi = error(X, X_d)

    if xi_d is None or not np.any(xi_d):
        psi_dot = xi.copy()
    else:
        psi_dot = unhat(hat(xi) - np.linalg.pinv(Psi) @ hat(xi_d) @ Psi)

    return Psi, psi, psi_dot


# ---------------------------------------------------------------------- #
# DLS pseudo-inverse якобиана
# ---------------------------------------------------------------------- #
def dls_pinv(J: np.ndarray, damping: float) -> np.ndarray:
    """
    DLS-псевдообратная для прямоугольного J:

        J^# = J^T · (J J^T + λ² I)^{-1}    (для широкого J, m ≤ n)
        J^# = (J^T J + λ² I)^{-1} · J^T    (для высокого J, m > n)
    """
    m, n = J.shape
    if m <= n:
        return J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(m))
    return np.linalg.inv(J.T @ J + (damping ** 2) * np.eye(n)) @ J.T
