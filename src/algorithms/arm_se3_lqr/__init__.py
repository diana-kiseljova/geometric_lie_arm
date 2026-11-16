"""
LTI LQR на ошибке EE в SE(3) — портирует баг-фри идею lqr_v3/lqr_v4 на
руку.  Линеаризация: x = [ψ, ψ̇] ∈ R^12, A = [[0,I],[0,0]], B = [[0],[I]].
"""
from src.algorithms.arm_se3_lqr.control import SE3LQRController

__all__ = ["SE3LQRController"]
