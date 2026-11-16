import copy
import typing as ty
import time

from tqdm import tqdm
import numpy as np
from scipy.integrate import odeint

from src.initials import Initials
from src.dynamics import kin_dyn
from src.utils.math import error, unhat, hat


class ParamsPID:

    def __init__(self, initials: Initials, input_arrays: dict):
        self.initials = initials

        self.Kp = initials.controller_params['Kp']
        self.Kd = initials.controller_params['Kd']
        self.Ki = initials.controller_params['Ki']



def run_pid(
        init: Initials, 
        pid_par: ParamsPID, 
        input_arrays: dict[str, ty.Union[np.ndarray, list[np.ndarray]]]
    ) -> dict[str, ty.Union[np.ndarray, list[np.ndarray]]]:
    output_arrays = copy.deepcopy(input_arrays)
    output_arrays.setdefault('u_raw_hist', [])
    output_arrays.setdefault('u_saturated_hist', [])

    u_min = init.u_min
    u_max = init.u_max

    I = 0
    for k in tqdm(range(init.num_steps-1)):
        t1 = time.time()

        I += output_arrays['psi_hist'][k] * init.dt

        u_pid_raw = (
            pid_par.Kp * output_arrays['psi_hist'][k]
            + pid_par.Kd * output_arrays['psi_dot_hist'][k]
            + pid_par.Ki * I
        )

        u_pid = np.clip(u_pid_raw, u_min, u_max)
        u_is_saturated = ~np.isclose(u_pid, u_pid_raw, rtol=1e-10, atol=1e-12)

        output_arrays['u_raw_hist'].append(u_pid_raw)
        output_arrays['u_saturated_hist'].append(u_is_saturated)
        output_arrays['u_hist'].append(u_pid)

        # Applying the control to the real dynamics:
        X_xi_start = np.block([output_arrays['X_hist'][k].flatten(), output_arrays['xi_hist'][k]])
        t = np.linspace(0, init.dt, 21)
        sol = odeint(kin_dyn, X_xi_start, t, args=(u_pid, init.body.Jb, init.ext_wrench), rtol=1e-9, atol=1e-12)
        X_xi_next = sol[-1,:]

        # Determing the values for the next step:
        X_next = X_xi_next[0:16].reshape(4,4)
        xi_next = X_xi_next[16::]
        Psi_next, psi_next = error(X_next, output_arrays['X_d_list'][k+1])
        
        #x_next = np.hstack([psi_next, xi_next])
        psi_dot_next = unhat(hat(xi_next) - np.linalg.pinv(Psi_next) @ hat(output_arrays['xi_d_array'][k+1,:]) @ Psi_next) # ???

        t2 = time.time()
        output_arrays['time_hist'].append(t2-t1)

        # Adding the values to the history
        output_arrays['X_hist'].append(X_next)
        output_arrays['Psi_hist'].append(Psi_next)
        output_arrays['xi_hist'].append(xi_next)
        output_arrays['psi_hist'].append(psi_next)
        output_arrays['psi_dot_hist'].append(psi_dot_next)

    return output_arrays
