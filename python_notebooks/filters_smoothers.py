"""
Modules for filters and smoothers

Zheng Zhao 2021
"""
import jax.numpy as jnp
import jax
import scipy
import numpy as np
from jax import jit, lax, jacfwd
from functools import partial
from typing import Callable, Tuple


def ekfs(f_Q: Callable, H: jnp.ndarray, R: float,
         m0: jnp.ndarray, P0: jnp.ndarray,
         dts: jnp.ndarray, ys: jnp.ndarray):
         
    def jac_disc(u, dt):
        return jacfwd(lambda zz, ddt: f_Q(zz, ddt)[0], argnums=0)(u, dt)

    def scan_ekf(carry, elem):
        mf, Pf, mp, Pp, _ = carry
        dt, y = elem

        # Prediction step
        jac_f = jac_disc(mf, dt)
        f, Q = f_Q(mf, dt)
        mp = f
        Pp = jac_f @ Pf @ jac_f.T + Q

        # Update step
        S = H @ Pp @ H.T + R
        K = Pp @ H.T / S
        mf = mp + K @ (y - H @ mp)
        Pf = Pp - K @ K.T * S
        return (mf, Pf, mp, Pp, jac_f), (mf, Pf, mp, Pp, jac_f)

    def scan_eks(carry, elem):
        ms, Ps = carry
        mf, Pf, mp, Pp, jac_f = elem

        c, low = jax.scipy.linalg.cho_factor(Pp)
        G = Pf @ jax.scipy.linalg.cho_solve((c, low), jac_f).T
        ms = mf + G @ (ms - mp)
        Ps = Pf + G @ (Ps - Pp) @ G.T
        return (ms, Ps), (ms, Ps)

    _, filtering_results = lax.scan(scan_ekf, (m0, P0, m0, P0, P0), (dts, ys))
    mfs, Pfs, mps, Pps, jac_fs = filtering_results
    _, smoothing_results = lax.scan(scan_eks, (mfs[-1], Pfs[-1]),
                                    (mfs[:-1], Pfs[:-1], mps[1:], Pps[1:], jac_fs[1:]), reverse=True)
    mss = jnp.concatenate([smoothing_results[0], mfs[-1, None]], axis=0)
    Pss = jnp.concatenate([smoothing_results[1], Pfs[-1, None]], axis=0)
    return filtering_results[:2], (mss, Pss)


def kf_rts(F: np.ndarray, Q: np.ndarray,
           H: np.ndarray, R: float,
           y: np.ndarray,
           m0: np.ndarray, p0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple enough Kalman filter and RTS smoother.

    x_k = F x_{k-1} + q_{k-1},
    y_k = H x_k + r_k,

    Parameters
    ----------
    F : np.ndarray
        State transition.
    Q : np.ndarray
        State covariance.
    H : np.ndarray
        Measurement matrix.
    R : float
        Measurement noise variance.
    y : np.ndarray
        Measurements.
    m0, P0 : np.ndarray
        Initial mean and cov.

    Returns
    -------
    ms, ps : np.ndarray
        Smoothing posterior mean and covariances.
    """
    dim_x = m0.size
    num_y = y.size

    mm = np.zeros(shape=(num_y, dim_x))
    pp = np.zeros(shape=(num_y, dim_x, dim_x))

    mm_pred = mm.copy()
    pp_pred = pp.copy()

    m = m0
    p = p0

    # Filtering pass
    for k in range(num_y):
        # Pred
        m = F @ m
        p = F @ p @ F.T + Q
        mm_pred[k] = m
        pp_pred[k] = p

        # Update
        S = H @ p @ H.T + R
        K = p @ H.T / S
        m = m + K @ (y[k] - H @ m)
        p = p - K @ S @ K.T

        # Save
        mm[k] = m
        pp[k] = p

    # Smoothing pass
    ms = mm.copy()
    ps = pp.copy()
    for k in range(num_y - 2, -1, -1):
        (c, low) = scipy.linalg.cho_factor(pp_pred[k + 1])
        G = pp[k] @ scipy.linalg.cho_solve((c, low), F).T
        ms[k] = mm[k] + G @ (ms[k + 1] - mm_pred[k + 1])
        ps[k] = pp[k] + G @ (ps[k + 1] - pp_pred[k + 1]) @ G.T

    return mm, pp, ms, ps


@partial(jit, static_argnums=(0,))
def simulate_sample_normal_inc(m_and_cov: Callable,
                               x0: jnp.ndarray,
                               dts: jnp.ndarray, dws: jnp.ndarray) -> jnp.ndarray:
    r"""Simulate a sample from SDE solution by Gaussian increment type of discretisation.

    Formally, a process X, which solves an SDE, at time t_k is given by

    X(t_k) \approx f(X(t_{k-1})) + q(X(t_{k-1})),   q(X(t_{k-1})) ~ N(0, Q(X(t_{k-1}))).

    Parameters
    ----------
    m_and_cov : Callable
        A function that returns f and Q.
    x0 : jnp.ndarray
        Initial value.
    dts : jnp.ndarray
        Time intervals.
    dws : jnp.ndarray
        Increments of Wiener process.

    Returns
    -------
    jnp.ndarray
        A trajectory of X(t) at the times specified by dts.
    """

    def scan_body(carry, elem):
        x = carry
        dt, dw = elem

        m, cov = m_and_cov(x, dt)
        chol = jnp.linalg.cholesky(cov)
        x = m + chol @ dw
        return x, x

    _, sample = lax.scan(scan_body, x0, (dts, dws))
    return sample


@partial(jit, static_argnums=(0,))
def simulate_sample_normal_inc_diag(m_and_cov: Callable,
                                    x0: jnp.ndarray,
                                    dts: jnp.ndarray, dws: jnp.ndarray) -> jnp.ndarray:
    """The same as with simulate_sample_normal_inc, except that this function assumes Q is diagonal.
    """

    def scan_body(carry, elem):
        x = carry
        dt, dw = elem

        m, cov = m_and_cov(x, dt)
        chol = jnp.sqrt(cov)
        x = m + chol @ dw
        return x, x

    _, sample = lax.scan(scan_body, x0, (dts, dws))
    return sample
