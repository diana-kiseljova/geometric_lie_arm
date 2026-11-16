"""
SE(3)-PID на ошибке EE.  Использует ту же математику ψ из pid_v1
(src/utils/math.error), но не интегрирует ξ̇ через ODE, а проектирует
выход PID в суставную скорость через J_b^#.
"""
from src.algorithms.arm_se3_pid.control import SE3PIDController

__all__ = ["SE3PIDController"]
