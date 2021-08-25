"""
Unit test filters and smoothers.

EKFS should return the same results as KFS on linear models.

x = F x + q

Zheng Zhao 2021
"""
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from jax.config import config
from filters_smoothers import ekfs, kf_rts

config.update("jax_enable_x64", True)

np.random.seed(666)

F = 0.1 * np.eye(2)
Q = 0.01 * np.eye(2)
R = 0.01
H = np.array([[1., 1.]])
m0 = np.zeros((2,))
P0 = 0.1 * np.eye(2)

F_jnp = jnp.array(F)
Q_jnp = jnp.array(Q)
H_jnp = jnp.array(H)
m0_jnp = jnp.array(m0)
P0_jnp = jnp.array(P0)

# Simulate
num_measurements = 1000
xx = np.zeros((num_measurements, 2))
yy = np.zeros((num_measurements,))
x = m0.copy()
for i in range(num_measurements):
    x = F @ x + np.sqrt(Q) @ np.random.randn(2)
    y = H @ x + np.sqrt(R) * np.random.randn()
    xx[i] = x
    yy[i] = y

# KFS
mf, pf, ms, ps = kf_rts(F, Q, H, R, yy, m0, P0)


# EKFS
def f_Q(u, dt):
    return F_jnp @ u, Q_jnp


ekf_results, eks_results = ekfs(f_Q, H_jnp, R, m0_jnp, P0_jnp,
                                jnp.array(yy), jnp.array(yy))

npt.assert_allclose(mf, ekf_results[0])
npt.assert_allclose(pf, ekf_results[1])

npt.assert_allclose(ms, eks_results[0])
npt.assert_allclose(ps, eks_results[1])
