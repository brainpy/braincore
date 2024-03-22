from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

__all__ = ['trapezoid', "romb"]


def _difftrapn(function, interval, numtraps):
  """
  Perform part of the trapezoidal rule to integrate a function.
  Assume that we had called difftrap with all lower powers-of-2
  starting with 1.  Calling difftrap only returns the summation
  of the new ordinates.  It does _not_ multiply by the width
  of the trapezoids.  This must be performed by the caller.
      'function' is the function to evaluate (must accept vector arguments).
      'interval' is a sequence with lower and upper limits
                 of integration.
      'numtraps' is the number of trapezoids to use (must be a
                 power-of-2).
  """
  numtosum = numtraps // 2
  h = (1.0 * interval[1] - 1.0 * interval[0]) / numtosum
  lox = interval[0] + 0.5 * h
  points = lox + h * jnp.arange(0, numtosum)
  s = jnp.sum(function(points))
  return s


def _romberg_diff(b, c, k):
  """
  Compute the differences for the Romberg quadrature corrections.
  See Forman Acton's "Real Computing Made Real," p 143.
  """
  tmp = 4.0 ** k
  return (tmp * c - b) / (tmp - 1.0)


@partial(jit, static_argnames=('axis',))
def trapezoid(y: jax.typing.ArrayLike,
              x: jax.typing.ArrayLike = None,
              dx: jax.typing.ArrayLike = 1.0,
              axis: int = -1) -> jax.Array:
  r"""
  Integrate along the given axis using the composite trapezoidal rule.

  If `x` is provided, the integration happens in sequence along its
  elements - they are not sorted.

  Integrate `y` (`x`) along each 1d slice on the given axis, compute
  :math:`\int y(x) dx`.
  When `x` is specified, this integrates along the parametric curve,
  computing :math:`\int_t y(t) dt =
  \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`.

  Parameters
  ----------
  y : array_like
      Input array to integrate.
  x : array_like, optional
      The sample points corresponding to the `y` values. If `x` is None,
      the sample points are assumed to be evenly spaced `dx` apart. The
      default is None.
  dx : scalar, optional
      The spacing between sample points when `x` is None. The default is 1.
  axis : int, optional
      The axis along which to integrate.

  Returns
  -------
  trapezoid : float or ndarray
      Definite integral of `y` = n-dimensional array as approximated along
      a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,
      then the result is a float. If `n` is greater than 1, then the result
      is an `n`-1 dimensional array.

  See Also
  --------
  cumulative_trapezoid, simpson, romb

  Notes
  -----
  Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
  will be taken from `y` array, by default x-axis distances between
  points will be 1.0, alternatively they can be provided with `x` array
  or with `dx` scalar.  Return value will be equal to combined area under
  the red lines.

  References
  ----------
  .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule

  .. [2] Illustration image:
         https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

  Examples
  --------
  Use the trapezoidal rule on evenly spaced points:

  >>> import numpy as np
  >>> from braincore import integrate
  >>> integrate.trapezoid([1, 2, 3])
  4.0

  The spacing between sample points can be selected by either the
  ``x`` or ``dx`` arguments:

  >>> integrate.trapezoid([1, 2, 3], x=[4, 6, 8])
  8.0
  >>> integrate.trapezoid([1, 2, 3], dx=2)
  8.0

  Using a decreasing ``x`` corresponds to integrating in reverse:

  >>> integrate.trapezoid([1, 2, 3], x=[8, 6, 4])
  -8.0

  More generally ``x`` is used to integrate along a parametric curve. We can
  estimate the integral :math:`\int_0^1 x^2 = 1/3` using:

  >>> x = np.linspace(0, 1, num=50)
  >>> y = x**2
  >>> integrate.trapezoid(y, x)
  0.33340274885464394

  Or estimate the area of a circle, noting we repeat the sample which closes
  the curve:

  >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)
  >>> integrate.trapezoid(np.cos(theta), x=np.sin(theta))
  3.141571941375841

  ``trapezoid`` can be applied along a specified axis to do multiple
  computations in one call:

  >>> a = np.arange(6).reshape(2, 3)
  >>> a
  array([[0, 1, 2],
         [3, 4, 5]])
  >>> integrate.trapezoid(a, axis=0)
  array([1.5, 2.5, 3.5])
  >>> integrate.trapezoid(a, axis=1)
  array([2.,  8.])
  """
  if x is None:
    y_arr = jnp.asarray(y)
    dx_array = jnp.asarray(dx)
  else:
    y_arr = jnp.asarray(y)
    x_arr = jnp.asarray(x)
    if x_arr.ndim == 1:
      dx_array = jnp.diff(x_arr)
    else:
      dx_array = jnp.moveaxis(jnp.diff(x_arr, axis=axis), axis, -1)
  y_arr = jnp.moveaxis(y_arr, axis, -1)
  return 0.5 * (dx_array * (y_arr[..., 1:] + y_arr[..., :-1])).sum(-1)


def _difftrap1(function, interval):
  """
  Perform part of the trapezoidal rule to integrate a function.
  Assume that we had called difftrap with all lower powers-of-2
  starting with 1.  Calling difftrap only returns the summation
  of the new ordinates.  It does _not_ multiply by the width
  of the trapezoids.  This must be performed by the caller.
      'function' is the function to evaluate (must accept vector arguments).
      'interval' is a sequence with lower and upper limits
                 of integration.
      'numtraps' is the number of trapezoids to use (must be a
                 power-of-2).
  """
  return 0.5 * (function(interval[0]) + function(interval[1]))


def romb(function, a, b, args=(), divmax=6, return_error=False):
  """
  Romberg integration of a callable function or method.
  Returns the integral of `function` (a function of one variable)
  over the interval (`a`, `b`).
  If `show` is 1, the triangular array of the intermediate results
  will be printed.  If `vec_func` is True (default is False), then
  `function` is assumed to support vector arguments.

  Parameters
  ----------
  function : callable
      Function to be integrated.
  a : float
      Lower limit of integration.
  b : float
      Upper limit of integration.

  Returns
  -------
  results  : float
      Result of the integration.

  Other Parameters
  ----------------
  args : tuple, optional
      Extra arguments to pass to function. Each element of `args` will
      be passed as a single argument to `func`. Default is to pass no
      extra arguments.
  divmax : int, optional
      Maximum order of extrapolation. Default is 10.

  References
  ----------
  .. [1] 'Romberg's method' http://en.wikipedia.org/wiki/Romberg%27s_method

  Examples
  --------
  Integrate a gaussian from 0 to 1 and compare to the error function.
  >>> from braincore import integrate
  >>> from scipy.special import erf
  >>> gaussian = lambda x: 1/np.sqrt(np.pi) * np.exp(-x**2)
  >>> result = integrate.romberg(gaussian, 0, 1, show=True)
  Romberg integration of <function vfunc at ...> from [0, 1]
  ::
     Steps  StepSize  Results
         1  1.000000  0.385872
         2  0.500000  0.412631  0.421551
         4  0.250000  0.419184  0.421368  0.421356
         8  0.125000  0.420810  0.421352  0.421350  0.421350
        16  0.062500  0.421215  0.421350  0.421350  0.421350  0.421350
        32  0.031250  0.421317  0.421350  0.421350  0.421350  0.421350  0.421350
  The final result is 0.421350396475 after 33 function evaluations.
  >>> print("%g %g" % (2*result, erf(1)))
  0.842701 0.842701
  """
  vfunc = jit(lambda x: function(x, *args))

  interval = [a, b]
  intrange = b - a
  ordsum = _difftrap1(vfunc, interval)
  result = intrange * ordsum
  state = jnp.repeat(jnp.atleast_1d(result), divmax + 1, axis=-1)
  err = jnp.inf

  def scan_fn(carry, y):
    x, k = carry
    x = _romberg_diff(y, x, k + 1)
    return (x, k + 1), x

  for i in range(1, divmax + 1):
    n = 2 ** i
    ordsum = ordsum + _difftrapn(vfunc, interval, n)

    x = intrange * ordsum / n
    _, new_state = jax.lax.scan(scan_fn, (x, 0), state[:-1])

    new_state = jnp.concatenate([jnp.atleast_1d(x), new_state])

    err = jnp.abs(state[i - 1] - new_state[i])
    state = new_state

  if return_error:
    return state[i], err
  else:
    return state[i]

