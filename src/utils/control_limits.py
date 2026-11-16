"""Single source of truth for wrench actuator limits.

The body-frame wrench applied to the rigid body is

    u = [tau_x, tau_y, tau_z, f_x, f_y, f_z]

so any saturation / box-constraint must live in R^6 with units of
[N·m, N·m, N·m, N, N, N].

Two equivalent ways to specify limits are supported:

    1. Symmetric form (the usual case)::

           tau_max  : scalar or 3-vector  -> |tau_i|  <= tau_max[i]
           force_max: scalar or 3-vector  -> |f_i|    <= force_max[i]

    2. Direct form (allows asymmetry)::

           u_min : scalar or 6-vector
           u_max : scalar or 6-vector

If both forms are supplied the direct form wins. Channels left ``None``
become ``±inf`` (meaning *no constraint*) so downstream code never has to
do another ``is None`` check.
"""

from __future__ import annotations

import typing as ty

import numpy as np


__all__ = [
    "limit_to_array",
    "make_symmetric_wrench_limits",
    "resolve_wrench_limits",
    "has_finite_limits",
]


LimitLike = ty.Optional[ty.Union[float, int, ty.Sequence[float], np.ndarray]]


def limit_to_array(
        value: LimitLike,
        default: float,
        size: int,
    ) -> np.ndarray:
    """Convert a scalar / vector limit specification into a length-``size`` array.

    Parameters
    ----------
    value : None | scalar | sequence of length ``size``
        If ``None`` the result is a constant array filled with ``default``.
        If a scalar it is broadcast to length ``size``.
        Otherwise it must already have shape ``(size,)``.
    default : float
        Filler value used when ``value is None``. Typically ``+inf`` for
        upper bounds and ``-inf`` for lower bounds.
    size : int
        Expected output length (3 for tau-only / force-only blocks,
        6 for the full wrench).
    """
    if value is None:
        return np.full(size, default, dtype=float)

    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(size, float(arr), dtype=float)

    if arr.shape != (size,):
        raise ValueError(
            f"Control limit must be scalar or a vector with shape ({size},), "
            f"got {arr.shape}."
        )
    return arr


def make_symmetric_wrench_limits(
        tau_max: LimitLike,
        force_max: LimitLike,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric (u_min, u_max) from tau_max / force_max.

    Channels left as ``None`` become ``±inf``. Returned arrays always have
    shape ``(6,)``.
    """
    if tau_max is None:
        tau_hi = np.full(3, np.inf, dtype=float)
    else:
        tau_hi = limit_to_array(tau_max, np.inf, size=3)

    if force_max is None:
        force_hi = np.full(3, np.inf, dtype=float)
    else:
        force_hi = limit_to_array(force_max, np.inf, size=3)

    u_hi = np.hstack([tau_hi, force_hi])
    return -u_hi, u_hi


def resolve_wrench_limits(
        tau_max: LimitLike = None,
        force_max: LimitLike = None,
        u_min: LimitLike = None,
        u_max: LimitLike = None,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Single entry point used by ``Initials`` to compute (u_min, u_max).

    Resolution rules:
      * if ``u_min`` or ``u_max`` is set, the direct form wins (channels left
        ``None`` become ``±inf``) — useful for asymmetric actuators;
      * otherwise fall back to the symmetric ``tau_max`` / ``force_max`` form;
      * if everything is ``None`` the result is ``(-inf, +inf)`` — i.e. no
        constraint at all.
    """
    if u_min is not None or u_max is not None:
        lo = limit_to_array(u_min, -np.inf, size=6)
        hi = limit_to_array(u_max, np.inf, size=6)
    else:
        lo, hi = make_symmetric_wrench_limits(tau_max, force_max)

    if np.any(lo > hi):
        raise ValueError(
            f"Invalid wrench limits: u_min={lo} is not <= u_max={hi} element-wise."
        )

    return lo, hi


def has_finite_limits(u_min: np.ndarray, u_max: np.ndarray) -> bool:
    """True iff at least one channel of (u_min, u_max) is finite."""
    return bool(np.any(np.isfinite(u_min)) or np.any(np.isfinite(u_max)))
