import typing as ty
from src.utils.math import *


def get_A_B_h(
        x_t: np.ndarray, 
        xi_d: np.ndarray, 
        Jb: np.ndarray, 
        delta_t
    ) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    assert len(xi_d) == 6
    assert len(x_t) == 12

    # Извлечение I_b и массы из J_b
    Ib = Jb[:3, :3]
    m = Jb[3, 3]
    Jb_inv = np.linalg.pinv(Jb)

    # Распаковка вектора состояния:
    xi_op = x_t[6:12].copy()            # operating point
    #xi_op = xi_d.copy()

    w_op = xi_op[0:3]
    v_op = xi_op[3:6]
    vx_op = vect2skew(v_op)

    ad_d = adjoint(xi_d)
    ad_t = adjoint(xi_op)
    coad_t = coadjoint(xi_op)

    Ibw_hat = vect2skew(Ib @ w_op)
    mv_hat  = m * vx_op
    Block_t   = np.block([[Ibw_hat, mv_hat], [mv_hat, np.zeros([3,3])]])
    H_t =  Jb_inv @ coad_t @ Jb + Jb_inv @ Block_t
    b_t = -Jb_inv @ Block_t @ xi_op

    ############################################ Почему у нас разные знаки?
    B_t = np.block(
        [
            [np.zeros([6,6])],
            [Jb_inv]
        ]
    )

    A_t = np.block(
        [
            [-ad_d, np.eye(6)],
            [np.zeros([6,6]), H_t]
        ]
    )

    h_t = np.block([-xi_d, b_t])

    A_k = A_t * delta_t + np.eye(12, 12)
    B_k = B_t * delta_t
    h_k = h_t * delta_t

    return A_k, B_k, h_k


# def get_A_B_h(
#         x_t: np.ndarray, 
#         xi_d: np.ndarray, 
#         Jb: np.ndarray
#     ) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     """
#     assert len(xi_d) == 6
#     assert len(x_t) == 12

#     # Извлечение I_b и массы из J_b
#     Ib = Jb[:3, :3]
#     m = Jb[3, 3]
#     Jb_inv = np.linalg.pinv(Jb)

#     # Распаковка вектора состояния:
#     xi_op = x_t[6:12].copy()            # operating point
#     #xi_op = xi_d.copy()

#     w_op = xi_op[0:3]
#     v_op = xi_op[3:6]
#     vx_op = vect2skew(v_op)

#     ad_d = adjoint(xi_d)
#     ad_t = adjoint(xi_op)
#     coad_t = np.transpose(ad_t.copy()) ###################### поменять, проверить

#     Ibw_hat = vect2skew(Ib @ w_op)
#     mv_hat  = m * vx_op
#     Block_t   = np.block([[Ibw_hat, mv_hat], [mv_hat, np.zeros([3,3])]])
#     H_t =  Jb_inv @ coad_t @ Jb + Jb_inv @ Block_t
#     b_t = -Jb_inv @ Block_t @ xi_op

#     ############################################ Почему у нас разные знаки?
#     B_t = np.block([[np.zeros([6,6])],
#                     [Jb_inv]])

#     A_t = np.block([[-ad_d, np.eye(6)],
#                     [np.zeros([6,6]), H_t]])

#     h_t = np.block([-xi_d, b_t])

#     # A_t = np.block([[-ad_d, -np.eye(6)],
#     #                 [np.zeros([6,6]), H_t]])
#     # h_t = np.block([xi_d, b_t])

#     return A_t, B_t, h_t


def get_C_d(xi_d):

    ad_d = adjoint(xi_d)                                # 6x6

    C_t = np.block([ [ np.eye(6), np.zeros([6,6])],     # 12x12
                    [-ad_d,      np.eye(6)]])

    d_t = np.block([np.zeros(6), xi_d])                 # 1x12

    return C_t, d_t


# def get_C_d(xi_d: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
#     """
#     """
#     assert len(xi_d) == 6

#     ad_d = adjoint(xi_d)                                # 6x6

#     C_t = np.block([                                    # 12x12
#         [ np.eye(6), np.zeros([6,6])],     
#         [-ad_d,      np.eye(6)]
#     ])

#     d_t = np.block([np.zeros(6), xi_d])                 # 1x12

#     return C_t, d_t