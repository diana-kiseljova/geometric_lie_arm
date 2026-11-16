import numpy as np
from scipy.linalg import logm


def R_p_to_X(R: np.ndarray, p: np.ndarray):
  """
  R - матрица поворота.
  p - вектор смещения.
  Возвращает матрицу преобразования X.
  """
  assert R.shape == (3, 3)

  X = np.block([
    [R, p.reshape(-1, 1)],
    [0, 0, 0, 1]
  ])
  return X


def vect2skew(vect: np.ndarray) -> np.ndarray:
  """
  Превращает вектор (3,) в кососимметричную матрицу 3x3.
  """
  assert vect.shape == (3,)
  matrix = np.array(
    [ 
      [0,-vect[2],vect[1]],
      [vect[2], 0, -vect[0]],
      [-vect[1], vect[0], 0]
    ], 
    dtype=np.float64
  )
  return matrix


def skew2vect(matrix: np.ndarray) -> np.ndarray:
  """
  Превращает кососимметричную матрицу 3x3 в вектор (3,).
  """
  vect = np.array(
    [matrix[2,1], matrix[0,2], matrix[1, 0]], 
    dtype=np.float64
  )
  return vect


def adjoint(xi: np.ndarray) -> np.ndarray:
    """
    xi = [ω1, ω2, ω3, v1, v2, v3]
    Возвращает "маленький" adjoint.
    """
    assert xi.ndim == 1
    assert len(xi) == 6

    w = xi[0:3]
    v = xi[3:6]

    vx = vect2skew(v)
    wx = vect2skew(w)

    ad = np.block([ 
      [wx,  np.zeros([3, 3])],
      [vx,                wx]
    ])

    return ad


def coadjoint(xi: np.ndarray) -> np.ndarray:
    """
    xi = [ω1, ω2, ω3, v1, v2, v3]
    Возврашает adjoint ^ T .
    """
    assert xi.ndim == 1
    assert len(xi) == 6

    ad = adjoint(xi)
    coad = np.transpose(ad)

    return coad


def adjoint_mat(X: np.ndarray) -> np.ndarray:
    """
    Возврашает 6x6 Adjoint матрицу для группы SE(3).
    X - 4x4 матрица преобразования (поворот и смещение) для группы SE(3).
    """
    R = X[:3, :3]
    p = X[:3, 3]
    Adx = np.vstack((
        np.hstack((R, np.zeros((3, 3)))),
        np.hstack((vect2skew(p) @ R, R))
    ))
    return Adx


def hat(xi: np.ndarray) -> np.ndarray:
  """
  Make element of algebra Lie se(3) from vector (isomorphism).
  """
  assert xi.ndim == 1
  assert len(xi) == 6

  w = xi[0:3]
  v = xi[3:6]
  wx = vect2skew(w)

  xi_hat = np.block([
    [wx, v.reshape(-1, 1)],
    [np.zeros((1, 4))]
  ])

  return xi_hat


def unhat(xi_hat: np.ndarray) -> np.ndarray:
  """
  From vector to the element of algebra Lie se(3).
  """
  assert xi_hat.shape == (4, 4)

  wx = xi_hat[0:3, 0:3]

  # проверяем кососимметричность: TODO
  #skew_ok = np.allclose(wx, -wx.T, atol=1e-9)
  #assert skew_ok == True

  v = xi_hat[0:3,3]
  w = skew2vect(wx)

  xi = np.block([w, v])

  return xi


def error(
    X: np.ndarray, 
    X_d: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
  """
  X - преобразование (поворот и смещение) для группы SE(3).
  X_d - преобразование (поворот и смещение) для группы SE(3) для референсной траектории.

  Ошибка = X_d^-1 X
  ошибка на алгебре = unhat(Ошибка)

  Возвращает ошибку и ошибку на алгебре.
  """
  assert X.shape == (4, 4)
  assert X_d.shape == (4, 4)

  X_d_inv = np.linalg.pinv(X_d)
  Psi = X_d_inv @ X

  psi_hat = logm(Psi)
  psi = unhat(psi_hat)

  # logm may return complex values when Ψ has near-negative eigenvalues (rotation
  # error near 180°). Take the real part so that downstream computations stay real.
  #psi = np.real(psi)

  return Psi, psi


def rotm2quat(R: np.ndarray) -> np.ndarray:
  """
  This function converts a rotation matrix R to a quaternion
  """
  trace = np.trace(R)
  if trace > 0:
    s = 0.5 / np.sqrt(trace + 1.0)
    qw = 0.25 / s
    qx = (R[2, 1] - R[1, 2]) * s
    qy = (R[0, 2] - R[2, 0]) * s
    qz = (R[1, 0] - R[0, 1]) * s
  else:
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
      s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
      qw = (R[2, 1] - R[1, 2]) / s
      qx = 0.25 * s
      qy = (R[0, 1] + R[1, 0]) / s
      qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
      s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
      qw = (R[0, 2] - R[2, 0]) / s
      qx = (R[0, 1] + R[1, 0]) / s
      qy = 0.25 * s
      qz = (R[1, 2] + R[2, 1]) / s
    else:
      s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
      qw = (R[1, 0] - R[0, 1]) / s
      qx = (R[0, 2] + R[2, 0]) / s
      qy = (R[1, 2] + R[2, 1]) / s
      qz = 0.25 * s
  return np.array([qw, qx, qy, qz])


def quat2rotm(quat: list) -> np.ndarray:
  """
  Converts a quaternion to a rotation matrix.

  Parameters:
  quat (array-like): A quaternion represented as a 4-element array [w, x, y, z].

  Returns:
  numpy.ndarray: A 3x3 rotation matrix.
  """
  w, x, y, z = quat
  rotm = np.array([
    [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
    [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
    [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
  ])
  return rotm


def world_wrench_to_body(ext_wrench_world: np.ndarray, R: np.ndarray) -> np.ndarray:
  """
  Converts a 6D wrench [tau_world, f_world] from world frame to body frame.

  Both components are rotated by R^T — no moment-arm cross-product because the
  wrench is expressed about the CoM (= body-frame origin).

  ext_wrench_world : (6,) [Tx, Ty, Tz, Fx, Fy, Fz] in world frame (N·m / N)
  R                : (3,3) rotation matrix body → world
  Returns          : (6,) [tau_body, f_body] in body frame
  """
  assert ext_wrench_world.shape == (6,)
  assert R.shape == (3, 3)
  result = np.zeros(6)
  result[:3] = R.T @ ext_wrench_world[:3]   # torque
  result[3:] = R.T @ ext_wrench_world[3:]   # force
  return result


def estimate_error(
    X_hist: list[np.ndarray],
    X_d_list: list[np.ndarray]
  ) -> tuple[list[np.ndarray], list[np.ndarray]]:
  """
  """
  Psi_hist, psi_hist = [], []
  for M_tr, M_pr in zip(X_d_list, X_hist):
    _Psi_hist = np.linalg.pinv(M_tr) @ M_pr
    Psi_hist.append(_Psi_hist)
    _psi_hist = unhat(logm(_Psi_hist))
    psi_hist.append(_psi_hist)
  
  return Psi_hist, psi_hist


def adjoint_from_two_matrices(M1: np.ndarray, vec: np.ndarray, M2: np.ndarray) -> np.ndarray:
  """
  Computes M1 @ vec @ M2.
  Used to express the adjoint action in 4×4 matrix form:
    Ad_{Ψ⁻¹}(ξ̂) = Ψ⁻¹ ξ̂ Ψ
  """
  return M1 @ vec @ M2


def assert_finite(name, A, epoch=None):
  if not np.all(np.isfinite(A)):
    msg = f"{name} has NaN/Inf"
    if epoch is not None:
        msg += f" at epoch {epoch}"
    print(msg)
    print(A)
    raise ValueError(msg)