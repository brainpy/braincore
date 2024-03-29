import jax.numpy as jnp
import numpy as np
from scipy.special import exprel

import braincore as bc
from braincore import math


def test_exprel():
  np.printoptions(precision=30)

  with bc.environ.context(precision=64):
    # Test with float64 input
    x = jnp.array([0.0, 1.0, 10.0, 100.0, 717.0, 718.0], dtype=jnp.float64)
    # expected = jnp.array([1.0, 1.718281828459045, 2.2025466e+03, jnp.inf, jnp.inf, jnp.inf])
    # print(math.exprel(x), exprel(np.asarray(x)))
    assert jnp.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-6)

  with bc.environ.context(precision=32):
    # Test with float32 input
    x = jnp.array([0.0, 1.0, 10.0, 100.0], dtype=jnp.float32)
    # expected = jnp.array([1.0, 1.7182817, 2.2025466e+03, jnp.inf])
    # print(math.exprel(x), exprel(np.asarray(x)))
    assert jnp.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-6)

  # Test with float16 input
  x = jnp.array([0.0, 1.0, 10.0], dtype=jnp.float16)
  # expected = jnp.array([1.0, 1.71875, 2.2025466e+03])
  # print(math.exprel(x), exprel(np.asarray(x)))
  assert jnp.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-3)

  # Test with int input
  x = jnp.array([0, 1, 10])
  # expected = jnp.array([1.0, 1.718281828459045, 2.20254658e+03])
  # print(math.exprel(x), exprel(np.asarray(x)))
  assert jnp.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-6)

  with bc.environ.context(precision=64):
    # Test with negative input
    x = jnp.array([-1.0, -10.0, -100.0], dtype=jnp.float64)
    # expected = jnp.array([0.63212055, 0.09999546, 0.01 ])
    # print(math.exprel(x), exprel(np.asarray(x)))
    assert jnp.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-6)
