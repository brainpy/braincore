import jax.numpy as jnp

from . import environ

__all__ = [
  'get_dtype',
  'exprel',
]


def get_dtype(a):
  """
  Get the dtype of a.
  """
  if hasattr(a, 'dtype'):
    return a.dtype
  else:
    if isinstance(a, bool):
      return bool
    elif isinstance(a, int):
      return environ.ditype()
    elif isinstance(a, float):
      return environ.dftype()
    elif isinstance(a, complex):
      return environ.dctype()
    else:
      raise ValueError(f'Can not get dtype of {a}.')


def exprel(x):
  """
  Relative error exponential, ``(exp(x) - 1)/x``.

  When ``x`` is near zero, ``exp(x)`` is near 1, so the numerical calculation of ``exp(x) - 1`` can
  suffer from catastrophic loss of precision. ``exprel(x)`` is implemented to avoid the loss of
  precision that occurs when ``x`` is near zero.

  Args:
    x: ndarray. Input array. ``x`` must contain real numbers.

  Returns:
    ``(exp(x) - 1)/x``, computed element-wise.
  """

  # following the implementation of exprel from scipy.special
  x = jnp.asarray(x)
  dtype = x.dtype

  # Adjust the tolerance based on the dtype of x
  if dtype == jnp.float64:
    small_threshold = 1e-16
    big_threshold = 717
  elif dtype == jnp.float32:
    small_threshold = 1e-8
    big_threshold = 100
  elif dtype == jnp.float16:
    small_threshold = 1e-4
    big_threshold = 10
  else:
    small_threshold = 1e-4
    big_threshold = 10

  small = jnp.abs(x) < small_threshold
  big = x > big_threshold
  origin = jnp.expm1(x) / x
  return jnp.where(small,
                   jnp.asarray(1.0, origin.dtype),
                   jnp.where(big, jnp.inf, origin))
