"""
Кинематические утилиты для роборуки.

Реализованы:
- `dls_step` — итерация Damped Least Squares IK по позиции EE
- `pose_error` — ошибка по позиции + угол-ось по ориентации
- кватернионная арифметика (xyzw <-> wxyz, log, mul, conj)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------- #
# Кватернионы (формат MuJoCo: w, x, y, z)
# ---------------------------------------------------------------------- #
def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_log_vec(q: np.ndarray) -> np.ndarray:
    """
    Возвращает 3D углово-векторное представление поворота кватерниона
    (axis * angle).  Подходит как ошибка ориентации в R^3.
    """
    w = float(np.clip(q[0], -1.0, 1.0))
    v = q[1:4]
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(s, w)
    # нормируем угол в [-pi, pi]
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return angle * (v / s)


def orientation_error(q_cur: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    """
    Векторная ошибка ориентации: log( q_des * conj(q_cur) ) выраженная как
    вектор axis*angle.  Возвращает вектор в R^3, который при PD ↦
    угловой скорости приводит EE к целевой ориентации.
    """
    q_err = quat_mul(q_des, quat_conj(q_cur))
    # Чтобы избежать "длинного пути": если w<0, инвертируем знак.
    if q_err[0] < 0.0:
        q_err = -q_err
    return quat_log_vec(q_err)


# ---------------------------------------------------------------------- #
# Damped Least Squares IK по позиции EE
# ---------------------------------------------------------------------- #
def dls_step(
    J: np.ndarray,
    error: np.ndarray,
    damping: float = 1e-2,
    gain: float = 1.0,
) -> np.ndarray:
    r"""
    Шаг IK с регуляризацией:

        dq = J^T (J J^T + lambda^2 I)^{-1} (gain * error)

    `J` может быть 3xN (по позиции) или 6xN (поза). `error` — соответствующей
    размерности (в R^3 или R^6).
    """
    m = J.shape[0]
    A = J @ J.T + (damping ** 2) * np.eye(m)
    return J.T @ np.linalg.solve(A, gain * error)


# ---------------------------------------------------------------------- #
# Удобные обёртки: ошибка позы и pose-DLS
# ---------------------------------------------------------------------- #
def pose_error(
    pos_cur: np.ndarray,
    quat_cur: np.ndarray,
    pos_des: np.ndarray,
    quat_des: np.ndarray,
) -> np.ndarray:
    """6D-вектор ошибки [linear_xyz, axis*angle_xyz]."""
    err = np.zeros(6)
    err[:3] = pos_des - pos_cur
    err[3:] = orientation_error(quat_cur, quat_des)
    return err
