import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.math import coadjoint, hat, world_wrench_to_body


def kin_dyn(
        X_xi: np.ndarray,
        t: float,
        u: np.ndarray,
        Jb: np.ndarray,
        ext_wrench: np.ndarray = None
    ) -> np.ndarray:
    """
    SE(3) kinematics + Newton-Euler dynamics.

    State vector (22 elements):
        X_xi[0:16]  — SE(3) matrix X (4x4) flattened, body-to-world transform
        X_xi[16:22] — body-frame twist xi = [omega, v]

    ext_wrench : optional (6,) constant wrench [Tx, Ty, Tz, Fx, Fy, Fz] in world
        frame. Converted to body frame and added to the dynamics.
        Build from launch.yaml fields external_torque and external_force
        via Initials.ext_wrench.
    """
    assert len(X_xi) == 22

    X_flat = X_xi[0:16]
    xi = X_xi[16:22]
    X = X_flat.reshape(4, 4)

    # Kinematics
    X_dot = X @ hat(xi)

    # External world-frame wrench → body frame
    u_ext = np.zeros(6)
    if ext_wrench is not None:
        R_mat = X[:3, :3]   # rotation body → world
        u_ext = world_wrench_to_body(np.asarray(ext_wrench, dtype=float), R_mat)

    # Dynamics
    xi_dot = np.linalg.pinv(Jb, rcond=1e-12) @ (u + u_ext + coadjoint(xi) @ Jb @ xi)

    return np.block([X_dot.flatten(), xi_dot])


def kin_dyn_with_quat(
        x: np.ndarray,
        t: float,
        u: np.ndarray,
        Jb: np.ndarray,
        ext_wrench: np.ndarray = None
    ) -> np.ndarray:
    """
    Quaternion-based kinematics + Newton-Euler dynamics.

    State vector (13 elements):
        x[0:4]   — unit quaternion q = [qx, qy, qz, qw] (scipy convention)
        x[4:7]   — position p in world frame
        x[7:10]  — body angular velocity omega
        x[10:13] — body linear velocity v

    ext_wrench : optional (6,) constant wrench [Tx, Ty, Tz, Fx, Fy, Fz] in world
                 frame. Converted to body frame and added to the dynamics.
                 Build from launch.yaml fields external_torque and external_force
                 via Initials.ext_wrench.
    """
    assert len(x) == 13

    q = x[0:4]
    p = x[4:7]
    w = x[7:10]
    v = x[10:13]

    R_mat = R.from_quat(q).as_matrix()   # rotation body → world

    Omega = np.array([
        [   0,  w[2], -w[1],  w[0]],
        [-w[2],    0,  w[0],  w[1]],
        [ w[1], -w[0],   0,   w[2]],
        [-w[0], -w[1], -w[2],   0 ]
    ])

    # Quaternion kinematics
    dQuat = 0.5 * Omega @ q

    # Position kinematics: p_dot = R @ v_body
    dx = np.concatenate([dQuat, R_mat @ v])

    # External world-frame wrench → body frame
    u_ext = np.zeros(6)
    if ext_wrench is not None:
        u_ext = world_wrench_to_body(np.asarray(ext_wrench, dtype=float), R_mat)

    # Newton-Euler dynamics
    xi = np.concatenate([w, v])
    d2x = np.linalg.pinv(Jb, rcond=1e-12) @ (coadjoint(xi) @ Jb @ xi + u + u_ext)

    return np.concatenate([dx, d2x])
