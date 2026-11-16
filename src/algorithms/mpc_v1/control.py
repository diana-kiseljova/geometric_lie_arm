import typing as ty
import copy
import time

from tqdm import tqdm
import numpy as np
import cvxpy as cp
from scipy import linalg as la
from scipy.integrate import odeint

from src.initials import Initials
from src.dynamics import kin_dyn
from src.utils.math import error
from src.algorithms.mpc_v1.linear import get_A_B_h, get_C_d


class ParamsMPC:

    def __init__(self, init: Initials, input_arrays: dict):

        self.N = init.controller_params['N'] # horizon

        # To be computed:
        self.x_0 = np.block([init.psi_0, init.xi_0])         # начальный вектор состояния
        self.x_hist = [self.x_0]

        #TODO: Добавить Уравнение Риккати
        # Матрицы для функции стоимости
        # self.Q = np.diag([1, 1, 1, 10, 10, 10, 1, 1, 1, 1, 1, 1]) * 1        # вес состояния       * 1
        # self.R = np.eye(6) * 1e-5                                            # вес управления      * 1e-5
        # self.P = np.diag([1, 1, 1, 10, 10, 10, 1, 1, 1, 1, 1, 1]) * 10       # конечная стоимость  * 10

        # self.Q = np.eye(12)
        # self.R = np.eye(6) * 1e-5
        # self.P = np.eye(12)

        self.Q = init.controller_params['Q']
        self.R = init.controller_params['R']
        self.P = init.controller_params['P']

        # Wrench actuator limits (length-6 arrays of finite / ±inf).
        self.u_min = init.u_min
        self.u_max = init.u_max

        # Переменные
        self.u = cp.Variable((self.N, 6))
        self.x = cp.Variable((self.N + 1, 12))
        self.y = cp.Variable((self.N + 1, 12))

        # MPC constants and parameters:
        self.x_op_param = cp.Parameter((12)) # operating point
        self.x_op_param.value = (self.x_hist[0]).copy()

        _, B_k, _ = get_A_B_h(self.x_hist[0], input_arrays['xi_d_array'][0, :], init.body_model.Jb, init.dt)
        self.A_param = cp.Parameter((12, 12))
        self.B_const = cp.Constant(B_k)
        self.h_param = cp.Parameter((12))

        self.C_param = cp.Parameter((12,12))
        self.d_param = cp.Parameter((12))

        self.Q_const = cp.Constant(self.Q)
        self.R_const = cp.Constant(self.R)
        self.P_const = cp.Constant(self.P) # Сделать изменяемой с уравнением Рикатти


def create_mpc_problem(
        mpc_par: ParamsMPC
    ):

    cost = 0
    constraints = [mpc_par.x[0, :] == mpc_par.x_op_param]

    # Box constraints u_min <= u_k <= u_max are appended only on channels
    # with finite limits. cvxpy does not accept ±inf in constraints, so we
    # skip those entries; an unbounded channel just gets no constraint.
    u_min = mpc_par.u_min
    u_max = mpc_par.u_max
    finite_lo = np.isfinite(u_min)
    finite_hi = np.isfinite(u_max)

    for k in range(mpc_par.N):
        constraints += [mpc_par.y[k, :] == (mpc_par.C_param @ mpc_par.x[k, :] - mpc_par.d_param)]
        constraints += [mpc_par.x[k + 1, :] == (mpc_par.A_param @ mpc_par.x[k, :] + mpc_par.B_const @ mpc_par.u[k, :] + mpc_par.h_param)]

        if np.any(finite_hi):
            idx = np.where(finite_hi)[0]
            constraints += [mpc_par.u[k, idx] <= u_max[idx]]
        if np.any(finite_lo):
            idx = np.where(finite_lo)[0]
            constraints += [mpc_par.u[k, idx] >= u_min[idx]]

        cost = cost + cp.quad_form(mpc_par.y[k, :], mpc_par.Q_const) + cp.quad_form(mpc_par.u[k, :], mpc_par.R_const)

    cost = cost + cp.quad_form((mpc_par.C_param @ mpc_par.x[mpc_par.N, :] - mpc_par.d_param), mpc_par.P_const)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    return problem


def run_mpc(
        init: Initials, 
        mpc_par: ParamsMPC, 
        input_arrays: dict[str, ty.Union[np.ndarray, list[np.ndarray]]],
        problem
    ) -> dict[str, ty.Union[np.ndarray, list[np.ndarray]]]:
    
    output_arrays = copy.deepcopy(input_arrays)
    #P_current = mpc_par.P
    #problem = create_mpc_problem(mpc_par)

    # Многократный запуск MPC:
    for epoch in tqdm(range(init.num_steps - mpc_par.N)):
        t1 = time.time()

        mpc_par.x_op_param.value = mpc_par.x_hist[epoch]

        A_k, _, h_k = get_A_B_h(mpc_par.x_hist[epoch], input_arrays['xi_d_array'][epoch, :], init.body_model.Jb, init.dt)
        mpc_par.A_param.value = A_k
        mpc_par.h_param.value = h_k

        C_t, d_t = get_C_d(input_arrays['xi_d_array'][epoch, :])
        mpc_par.C_param.value = C_t
        mpc_par.d_param.value = d_t

        # Решение задачи оптимизации
        problem.solve(solver=cp.OSQP, verbose=False)

        # Получаем результаты
        u_optimal = mpc_par.u.value

        u_mpc = u_optimal[0]
        output_arrays['u_hist'].append(u_mpc)

        # Applying the control to the real dynamics:
        X_xi_start = np.block([output_arrays['X_hist'][epoch].flatten(), output_arrays['xi_hist'][epoch]])
        t = np.linspace(0, init.dt, 21) # почему?
        sol = odeint(kin_dyn, X_xi_start, t, args=(u_mpc, init.body.Jb, init.ext_wrench), rtol=1e-9, atol=1e-12)
        X_xi_next = sol[-1,:]

        # Determing the values for the next step:
        X_next = X_xi_next[0:16].reshape(4,4)
        xi_next = X_xi_next[16::]
        Psi_next, psi_next = error(X_next, output_arrays['X_d_list'][epoch+1])
        x_next = np.hstack([psi_next, xi_next])

        t2 = time.time()
        output_arrays['time_hist'].append(t2-t1)

        # Adding the values to the history
        output_arrays['X_hist'].append(X_next)
        output_arrays['Psi_hist'].append(Psi_next)
        output_arrays['xi_hist'].append(xi_next)
        output_arrays['psi_hist'].append(psi_next)
        mpc_par.x_hist.append(x_next)

    return output_arrays