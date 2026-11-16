"""
Joint-PD контроллер: ведёт текущее q к целевому q_target из задачи.
"""
from src.algorithms.arm_joint_pd.control import JointPDController

__all__ = ["JointPDController"]
