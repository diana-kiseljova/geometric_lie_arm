import typing as ty
import copy
import time

from tqdm import tqdm
import numpy as np
from scipy import sparse
from scipy.integrate import odeint
from scipy.linalg import logm, solve_discrete_are, block_diag
from scipy.spatial.transform import Rotation
from osqp import OSQP

from src.initials import Initials
from src.utils.math import *
from src.dynamics import kin_dyn_with_quat


def eMPCCost(
        Q: np.ndarray, 
        R: np.ndarray, 
        P: np.ndarray, 
        xid: np.ndarray,
        Nh: int,
        Nx: int,
        Nu: int
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    """
    # Initialize M and q
    M = np.zeros((Q.shape[0], Q.shape[1]))
    q = np.zeros(((Nh+1) * Nx + Nh * Nu, 1))

    for k in range(Nh - 1):
        C = np.eye(12)
        C[6:12, 0:6] = -adjoint(xid[k, :])
        M = block_diag(M, C.T @ Q @ C)
        b = np.zeros((12, 1))
        b[6:12] = xid[k, :].reshape(-1, 1)
        b = C.T @ Q @ b
        q[k * Nx:(k + 1) * Nx] = -b.reshape(-1, 1)
    
    # Final step in the loop
    k = Nh - 1
    C = np.eye(12)
    C[6:12, 0:6] = -adjoint(xid[k, :])
    b = np.zeros((12, 1))
    b[6:12] = xid[k, :].reshape(-1, 1)
    b = C.T @ P @ b
    M = block_diag(M, C.T @ P @ C)
    q[k * Nx:(k + 1) * Nx] = -b.reshape(-1, 1)

    # Adding the R terms to the diagonal of M
    for k in range(Nh):
        M = block_diag(M, R)
    
    return M, q


def eMPCConstraints(
        xid: np.ndarray, 
        p0: np.ndarray,
        Jb: np.ndarray, 
        dt: float, 
        Nx: int, 
        Nu: int, 
        Nh: int,
        umax: np.ndarray,
        umin: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    A = np.zeros((Nx * (Nh + 1) + Nu * Nh, Nh * (Nx + Nu) + Nx))
    Noff = Nx * (Nh + 1)
    bmin = np.zeros(A.shape[0])

    for k in range(Nh):
        xi_bar = Jb @ xid[k,:]
        # xi_bar = I @ x0[7:13] original
        G = np.zeros((6, 6))
        G[:3, :3] = vect2skew(xi_bar[:3])
        G[:3, 3:6] = vect2skew(xi_bar[3:6])
        G[3:6, :3] = vect2skew(xi_bar[3:6])

        # H = np.linalg.inv(I) @ (coadjoint(x0[7:13]) @ I + G) # original
        H = np.linalg.inv(Jb) @ (coadjoint(xi_bar) @ Jb + G)

        Ac = np.block([[-adjoint(xid[k, :]), -np.eye(6)], [np.zeros((6, 6)), H]])
        Bc = np.block([[np.zeros((6, 6))], [np.linalg.inv(Jb)]])

        # b = -np.linalg.inv(I) @ G @ x0[7:13]
        b = -np.linalg.pinv(Jb) @ G @ xi_bar

        hc = np.concatenate([xid[k, :], b])

        Ad = np.eye(Nx) + Ac * dt
        Bd = dt * Bc
        hd = dt * hc

        A[k * Nx:(k + 1) * Nx, k * Nx:(k + 1) * Nx] = -Ad
        A[k * Nx:(k + 1) * Nx, (k + 1) * Nx:(k + 2) * Nx] = np.eye(Nx)
        A[k * Nx:(k + 1) * Nx, Noff + k * Nu:Noff + (k + 1) * Nu] = -Bd
        bmin[k * Nx:(k + 1) * Nx] = hd

    A[Nh * Nx:(Nh + 1) * Nx, :Nx] = np.eye(Nx)

    for k in range(Nh):
        A[Noff + k * Nu:Noff + (k + 1) * Nu, Noff + k * Nu:Noff + (k + 1) * Nu] = np.eye(Nu)

    bmin[Nh * Nx:(Nh + 1) * Nx] = p0
    bmax = bmin.copy()
    bmax[(Nh + 1) * Nx:] = np.tile(umax, Nh)
    bmin[(Nh + 1) * Nx:] = np.tile(umin, Nh)

    return A, bmin, bmax


def eMPC(
        
        x0: np.ndarray,
        
        xd: np.ndarray,
        xid: np.ndarray,

        Jb: np.ndarray,
        dt: float,
        Nx: int,
        Nu: int,
        Nh: int,

        umax: np.ndarray,
        umin: np.ndarray,

        P: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray

    ):
    """
    x0 - состояние системы в начале окна MPC (q, p, w, v)
    xd - массив 
    xid
    """
    # Convert quaternions to rotation matrices
    qq_d = np.concatenate([xd[0:1],xd[1:4]])
    qq_0 = np.concatenate([x0[0:1],x0[1:4]])

    Xd = np.block([
        [Rotation.from_quat(qq_d).as_matrix(), xd[4:7].reshape(3, 1)],
        [np.zeros((1, 3)), 1]
    ])

    X00 = np.block([
        [Rotation.from_quat(qq_0).as_matrix(), x0[4:7].reshape(3, 1)],
        [np.zeros((1, 3)), 1]
    ])

    # Compute the inverse of the left error
    X000 = np.linalg.pinv(X00) @ Xd

    X0 = logm(X000)

    # Error vector p0
    p0 = np.concatenate(([X0[2, 1], X0[0, 2], X0[1, 0]], X0[:3, 3], x0[7:]))

    # Compute constraints
    A, bmin, bmax = eMPCConstraints(
        xid, 
        p0,
        Jb, 
        dt, 
        Nx, 
        Nu, 
        Nh,
        umax,
        umin
    )

    # Discrete-time Algebraic Riccati Equation (DARE) solver
    P = solve_discrete_are(
        -A[:Nx, :Nx],
        -A[:Nx, Nx * (Nh + 1):(Nx * (Nh + 1)) + Nu],
        Q,
        R
    )

    # Compute cost matrices
    M, q = eMPCCost(Q, R, P, xid, Nh, Nx, Nu)

    M = sparse.csc_matrix(M)
    A = sparse.csc_matrix(A)

    # Setup and solve the QP problem using OSQP
    solver = OSQP()
    solver.setup(
        P=M, 
        q=q, 
        A=A, 
        l=bmin, 
        u=bmax, 
        verbose=False
    )
    result = solver.solve()

    # Extract control input
    u = result.x[(Nh+1)*Nx: (Nh+1)*Nx + Nu]

    return u


def inner_x_to_se3(X: np.ndarray) -> list[np.ndarray]:
    return [
        np.block(
            [
                [Rotation.from_quat(x[:4]).as_matrix(), x[4:7].reshape(3, 1)],
                [np.zeros((1, 3)), 1]
            ]
        )
        for x in X]


def se3_to_inner_x(X_d_list: list[np.ndarray]) -> list[np.ndarray]:
    return [
        np.concatenate((
            Rotation.from_matrix(X[:3, :3]).as_quat(), X[:3, 3] # Convert rotation matrix to quaternion
        ))
        for X in X_d_list]



class ParamsMPC:

    def __init__(self, init: Initials, input_arrays: dict):

        self.N = init.controller_params['N'] # horizon

        # To be computed:
        # self.x_0 = np.block([init.psi_0, init.xi_0])         # начальный вектор состояния
        # self.x_hist = [self.x_0]

        self.Q = init.controller_params['Q']
        self.R = init.controller_params['R']
        self.P = init.controller_params['P']


def run_mpc(
        init: Initials, 
        mpc_par: ParamsMPC, 
        input_arrays: dict[str, ty.Union[np.ndarray, list[np.ndarray]]]
    ) -> dict[str, ty.Union[np.ndarray, list[np.ndarray]]]:
    output_arrays = copy.deepcopy(input_arrays)

    dt = init.dt
    #T = init.T
    #Nsim = int(self.T / self.dt) #int(np.ceil(T / dt))

    Nh = mpc_par.N
    Nx = 12
    Nu = 6

    #xmax = np.inf * np.ones(12)  # state constraints
    #xmin = -np.inf * np.ones(12)  # state constraints

    # Controller's internal model (may differ from physical Jb).
    Jb_model = init.body_model.Jb
    Jb_phys = init.body.Jb

    # Wrench actuator limits resolved centrally in Initials (single source
    # of truth shared with PID / LQR). OSQP accepts ±inf for unbounded
    # channels, so the same code path works with or without limits.
    umin = init.u_min
    umax = init.u_max

    # Reference trajectories
    X_d_list = input_arrays['X_d_list']
    X_ref = se3_to_inner_x(X_d_list)
    X_ref = np.array(X_ref)
    xi_d_array = input_arrays['xi_d_array']

    q0 = Rotation.from_matrix(init.R_0).as_quat()
    p0 = init.p_0
    w0 = init.xi_0[:3]
    v0 = init.xi_0[3:]

    P = mpc_par.P
    Q = mpc_par.Q
    R = mpc_par.R

    # Initialize logging variables
    x0 = np.concatenate((q0, p0, w0, v0))
    X = [x0]
    U = []

    Xerr = []
    XXerr = []

    for i in tqdm(range(init.num_steps - Nh - 1)):
        t1 = time.time()

        X_ref_rt = X_ref[i, :]
        xi_ref_rt = xi_d_array[i:i + Nh, :]

        u = eMPC(
            x0=x0, 
            xd=X_ref_rt, 
            xid=xi_ref_rt,
            Jb=Jb_model,
            dt=dt,
            Nx=Nx,
            Nu=Nu,
            Nh=Nh, 
            umax=umax, 
            umin=umin, 
            P=P, 
            Q=Q, 
            R=R
        )
        #output_arrays['u_hist'].append(u) TODO !!!

        t = np.linspace(0, dt, 21)
        sol = odeint(kin_dyn_with_quat, x0, t, args=(u, Jb_phys, init.ext_wrench), rtol=1e-9, atol=1e-12)

        x0 = sol[-1, :]

        XX0 = np.block([
            [Rotation.from_quat(x0[:4]).as_matrix(), x0[4:7].reshape(3, 1)],
            [np.zeros((1, 3)), 1]
        ])

        XXd = np.block([
            [Rotation.from_quat(X_ref_rt[:4]).as_matrix(), X_ref_rt[4:7].reshape(3, 1)],
            [np.zeros((1, 3)), 1]
        ])

        Xerr.append(logm(np.linalg.pinv(XX0) @ XXd))
        XXerr.append(np.linalg.pinv(XX0) @ XXd)

        X.append(x0)
        U.append(u)

        t2 = time.time()
        output_arrays['time_hist'].append(t2-t1)

    X = np.array(X)
    U = np.array(U)
    Xerr = np.array(Xerr)
    XXerr = np.array(XXerr)

    X_ref = X_ref[:-Nh]

    X_hist = inner_x_to_se3(X)
    X_d_list = inner_x_to_se3(X_ref)

    #psi_hist = inner_x_to_se3(Xerr)
    Psi_hist, psi_hist = estimate_error(X_hist, X_d_list)

    output_arrays['X_hist'] = X_hist
    output_arrays['Psi_hist'] = Psi_hist
    output_arrays['psi_hist'] = psi_hist
    output_arrays['u_hist'] = U

    # print(len(X_hist))
    # print(len(X_d_list))
    # print(len(psi_hist))
    
    return output_arrays
