import numpy as np


# -----------------------------------------------------------------------------
# Wrench actuator limits (body frame, u = [tau_xyz, f_xyz])
# -----------------------------------------------------------------------------
# Every controller below honours the same set of optional keys:
#
#   tau_max   : scalar | 3-vector  — symmetric torque limit  (|tau_i| <= ...)
#   force_max : scalar | 3-vector  — symmetric force limit   (|f_i|   <= ...)
#   u_min     : scalar | 6-vector  — direct lower bound (overrides tau/force)
#   u_max     : scalar | 6-vector  — direct upper bound (overrides tau/force)
#
# Resolution is performed once in ``src/initials.py`` (Initials.u_min /
# Initials.u_max). Per-controller keys take precedence over the global YAML
# keys (``launch.yaml`` -> ``tau_max`` / ``force_max`` / ``u_min`` / ``u_max``).
#
# Channels left unspecified become ±inf, i.e. no constraint, so leaving the
# block empty turns saturation off everywhere.
# -----------------------------------------------------------------------------


control_map = {

    'MPC_v1': {
        'N': 20,
        'Q': np.diag([1, 1, 1, 10, 10, 10, 1, 1, 1, 1, 1, 1]) * 1,
        'R': np.eye(6) * 1e-5,
        'P': np.diag([1, 1, 1, 10, 10, 10, 1, 1, 1, 1, 1, 1]) * 10,

        # "tau_max": 3.0,
        # "force_max": 15.0,
    },

    'MPC_v2': {
        'N': 20,
        'Q': np.diag([1, 1, 1, 10, 10, 10, 1, 1, 1, 1, 1, 1]) * 1,
        'R': np.eye(6) * 1e-5,
        'P': np.diag([1, 1, 1, 10, 10, 10, 1, 1, 1, 1, 1, 1]) * 10,

        # "tau_max": 3.0,
        # "force_max": 15.0,
    },

    'PID_v1': {

        # 'Kp': -np.array([500,500, 500, 200, 200, 200]),
        # 'Kd': -np.array([50, 50, 50, 50, 50, 50]),
        # 'Ki': -np.array([0.2, 0.2, 0.2, 100, 100, 100])

        'Kp': -np.array([500, 500, 500, 200, 200, 200]),
        'Kd': -np.array([50, 50, 50, 50, 50, 50]),
        'Ki': -np.array([200, 200, 200, 100, 100, 100]),

        # "tau_max": 3.0,
        # "force_max": 15.0,
    },

    'PID_v2': {
        # Same proportional / derivative gains as PID_v1 (tested, stable).
        # Kdd adds second-derivative (acceleration) feedback.
        # Rule: keep |Kdd| << J_b to avoid amplifying numerical-diff noise.
        # J_b = diag([1,2,3, 1,1,1])  →  Kdd = -0.5 adds +0.5 to effective inertia.
        # Kdd=0: PD mode (safe base).  Increase carefully — large |Kdd| amplifies
        # control during the initial transient (ψ̈ feeds back through the EOM).
        'Kp':  -np.array([500, 500, 500, 200, 200, 200]),
        'Kd':  -np.array([ 50,  50,  50,  50,  50,  50]),
        'Kdd': -np.array([ 0.1,  0.05,  0.05,  0.05,  0.05,  0.05]),
        #'Ki':  -np.array([200, 200, 200, 100, 100, 100])

        # "tau_max": 3.0,
        # "force_max": 15.0,
    },

    'LQR_v1': {
        'Qw': 10000*np.diag([25,25,25,25,25,25,0.5,0.5,0.5,0.5,0.5,0.5]),
        'Rw': 10*np.eye(6),

        # "tau_max": 3.0,
        # "force_max": 15.0,
    },

    'LQR_v2': {

        # 'Qw': np.diag([
        #     100.0, 100.0, 100.0,   # orientation error
        #     100.0, 100.0, 100.0,   # position error
        #     10.0, 10.0, 10.0,      # angular velocity error
        #     10.0, 10.0, 10.0,      # linear velocity error
        # ]),
        # 'Rw': np.diag([
        #     1.0, 1.0, 1.0,         # torque penalty
        #     1.0, 1.0, 1.0,         # force penalty
        # ]),

        # 'Qw': np.diag([
        #     100.0, 100.0, 100.0,      # orientation error
        #     1000.0, 1000.0, 1000.0,   # position error
        #     20.0, 20.0, 20.0,         # angular velocity error
        #     50.0, 50.0, 50.0,         # linear velocity error
        # ]),
        # 'Rw': np.diag([
        #     1.0, 1.0, 1.0,            # torque penalty
        #     0.3, 0.3, 0.3,            # force penalty
        # ]),

        # 'Qw': np.diag([
        #     100.0, 100.0, 100.0,
        #     5000.0, 5000.0, 5000.0,
        #     30.0, 30.0, 30.0,
        #     300.0, 300.0, 300.0,
        # ]),
        # 'Rw': np.diag([
        #     1.0, 1.0, 1.0,
        #     0.1, 0.1, 0.1,
        # ]),

        'Qw': np.diag([
            100.0, 100.0, 100.0,        # orientation error
            10000.0, 10000.0, 10000.0,  # position error
            30.0, 30.0, 30.0,           # angular velocity error
            300.0, 300.0, 300.0,        # linear velocity error
        ]),
        'Rw': np.diag([
            1.0, 1.0, 1.0,              # torque penalty
            0.05, 0.05, 0.05,           # force penalty
        ]),

        # "tau_max": 3.0,
        # "force_max": 15.0,
    },

    'LQR_v3': {

        # 'Qw': np.diag([
        #     100.0, 100.0, 100.0,   # orientation error
        #     100.0, 100.0, 100.0,   # position error
        #     10.0, 10.0, 10.0,      # angular velocity error
        #     10.0, 10.0, 10.0,      # linear velocity error
        # ]),
        # 'Rw': np.diag([
        #     1.0, 1.0, 1.0,         # torque penalty
        #     1.0, 1.0, 1.0,         # force penalty
        # ]),

        'Qw': np.diag([
            100.0, 100.0, 100.0,        # orientation error
            10000.0, 10000.0, 10000.0,  # position error
            30.0, 30.0, 30.0,           # angular velocity error
            300.0, 300.0, 300.0,        # linear velocity error
        ]),
        'Rw': np.diag([
            1.0, 1.0, 1.0,              # torque penalty
            0.05, 0.05, 0.05,           # force penalty
        ]),

        # "tau_max": 3.0,
        # "force_max": 15.0,
    },


    'LQR_v4': {

        'Qw': np.diag([

            # 100, 100, 100,
            # 100, 100, 100,
            # 10, 10, 10,
            # 10, 10, 10,

            100.0, 100.0, 100.0,        # orientation error
            10000.0, 10000.0, 10000.0,  # position error
            30.0, 30.0, 30.0,           # angular velocity error
            300.0, 300.0, 300.0,        # linear velocity error
        ]),

        'Rw': np.diag([
            # 1, 1, 1,
            # 1, 1, 1,

            1.0, 1.0, 1.0,              # torque penalty
            0.05, 0.05, 0.05,           # force penalty
        ]),

        'Vw': 10 * np.diag([
            # 100, 100, 100,
            # 100, 100, 100,
            # 10, 10, 10,
            # 10, 10, 10,

            100.0, 100.0, 100.0,        # orientation error
            10000.0, 10000.0, 10000.0,  # position error
            30.0, 30.0, 30.0,           # angular velocity error
            300.0, 300.0, 300.0,        # linear velocity error
        ]),

        # 'ilqr_iterations': 10,
        # 'line_search_alphas': [1.0, 0.5, 0.25, 0.1],
        # 'ilqr_reg': 1e-8,
        # 'fd_eps_state': 1e-5,
        # 'fd_eps_control': 1e-5,
        # 'use_feedforward': False,

        'ilqr_iterations': 2,
        'line_search_alphas': [1.0],
        'ilqr_reg': 1e-5,
        'fd_eps_state': 1e-4,
        'fd_eps_control': 1e-4,
        'ode_points': 2,
        'use_feedforward': True,
        'initialize_nominal_with_goal_u': True,
        'apply_saturation': True,
        'controller_knows_external_wrench': False,
        'recompute_final_gains': False,
        'execute_final_feedback': True,

        # "tau_max": 3.0,
        # "force_max": 15.0,
    },

}