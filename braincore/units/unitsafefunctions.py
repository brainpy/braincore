"""
Unit-aware replacements for jax numpy functions.
"""

from functools import wraps

import jax.numpy as jnp

from .base import (
  DIMENSIONLESS,
  Quantity,
  check_units,
  fail_for_dimension_mismatch,
  is_dimensionless,
  wrap_function_dimensionless,
  wrap_function_keep_dimensions,
  wrap_function_remove_dimensions,
)

__all__ = [
  "log",
  "log10",
  "exp",
  "expm1",
  "log1p",
  "exprel",
  "sin",
  "cos",
  "tan",
  "arcsin",
  "arccos",
  "arctan",
  "sinh",
  "cosh",
  "tanh",
  "arcsinh",
  "arccosh",
  "arctanh",
  "diagonal",
  "ravel",
  "trace",
  "dot",
  "where",
  "ones_like",
  "zeros_like",
  "arange",
  "linspace",
  "ptp",
]


def where(condition, *args, **kwds):  # pylint: disable=C0111
  condition = jnp.asarray(condition)
  if len(args) == 0:
    # nothing to do
    return jnp.where(condition, *args, **kwds)
  elif len(args) == 2:
    # check that x and y have the same dimensions
    fail_for_dimension_mismatch(
      args[0], args[1], "x and y need to have the same dimensions"
    )
    new_args = []
    for arg in args:
      if isinstance(arg, Quantity):
        new_args.append(arg.value)
    if is_dimensionless(args[0]):
      if len(new_args) == 2:
        return jnp.where(condition, *new_args, **kwds)
      else:
        return jnp.where(condition, *args, **kwds)
    else:
      # as both arguments have the same unit, just use the first one's
      dimensionless_args = [jnp.asarray(arg.value) if isinstance(arg, Quantity) else jnp.asarray(arg) for arg in args]
      return Quantity.with_units(
        jnp.where(condition, *dimensionless_args), args[0].unit
      )
  else:
    # illegal number of arguments
    if len(args) == 1:
      raise ValueError("where() takes 2 or 3 positional arguments but 1 was given")
    elif len(args) > 2:
      raise TypeError("where() takes 2 or 3 positional arguments but {} were given".format(len(args)))


where.__doc__ = jnp.where.__doc__
where._do_not_run_doctests = True

# Functions that work on dimensionless Arrays only
sin = wrap_function_dimensionless(jnp.sin)
sinh = wrap_function_dimensionless(jnp.sinh)
arcsin = wrap_function_dimensionless(jnp.arcsin)
arcsinh = wrap_function_dimensionless(jnp.arcsinh)
cos = wrap_function_dimensionless(jnp.cos)
cosh = wrap_function_dimensionless(jnp.cosh)
arccos = wrap_function_dimensionless(jnp.arccos)
arccosh = wrap_function_dimensionless(jnp.arccosh)
tan = wrap_function_dimensionless(jnp.tan)
tanh = wrap_function_dimensionless(jnp.tanh)
arctan = wrap_function_dimensionless(jnp.arctan)
arctanh = wrap_function_dimensionless(jnp.arctanh)

log = wrap_function_dimensionless(jnp.log)
log10 = wrap_function_dimensionless(jnp.log10)
exp = wrap_function_dimensionless(jnp.exp)
expm1 = wrap_function_dimensionless(jnp.expm1)
log1p = wrap_function_dimensionless(jnp.log1p)

ptp = wrap_function_keep_dimensions(jnp.ptp)


@check_units(x=1, result=1)
def exprel(x):
  x = jnp.asarray(x)
  if issubclass(x.dtype.type, jnp.integer):
    result = jnp.empty_like(x, dtype=jnp.float64)
  else:
    result = jnp.empty_like(x)
  # Following the implementation of exprel from scipy.special
  if x.shape == ():
    if jnp.abs(x) < 1e-16:
      return 1.0
    elif x > 717:
      return jnp.inf
    else:
      return jnp.expm1(x) / x
  else:
    small = jnp.abs(x) < 1e-16
    big = x > 717
    in_between = jnp.logical_not(small | big)
    result[small] = 1.0
    result[big] = jnp.inf
    result[in_between] = jnp.expm1(x[in_between]) / x[in_between]
    return result


ones_like = wrap_function_remove_dimensions(jnp.ones_like)
zeros_like = wrap_function_remove_dimensions(jnp.zeros_like)


def wrap_function_to_method(func):
  """
  Wraps a function so that it calls the corresponding method on the
  Arrays object (if called with a Arrays object as the first
  argument). All other arguments are left untouched.
  """

  @wraps(func)
  def f(x, *args, **kwds):  # pylint: disable=C0111
    if isinstance(x, Quantity):
      return getattr(x, func.__name__)(*args, **kwds)
    else:
      # no need to wrap anything
      return func(x, *args, **kwds)

  f.__doc__ = func.__doc__
  f.__name__ = func.__name__
  f._do_not_run_doctests = True
  return f


@wraps(jnp.arange)
def arange(*args, **kwargs):
  # arange has a bit of a complicated argument structure unfortunately
  # we leave the actual checking of the number of arguments to numpy, though

  # default values
  start = kwargs.pop("start", 0)
  step = kwargs.pop("step", 1)
  stop = kwargs.pop("stop", None)
  if len(args) == 1:
    if stop is not None:
      raise TypeError("Duplicate definition of 'stop'")
    stop = args[0]
  elif len(args) == 2:
    if start != 0:
      raise TypeError("Duplicate definition of 'start'")
    if stop is not None:
      raise TypeError("Duplicate definition of 'stop'")
    start, stop = args
  elif len(args) == 3:
    if start != 0:
      raise TypeError("Duplicate definition of 'start'")
    if stop is not None:
      raise TypeError("Duplicate definition of 'stop'")
    if step != 1:
      raise TypeError("Duplicate definition of 'step'")
    start, stop, step = args
  elif len(args) > 3:
    raise TypeError("Need between 1 and 3 non-keyword arguments")
  if stop is None:
    raise TypeError("Missing stop argument.")
  fail_for_dimension_mismatch(
    start,
    stop,
    error_message=(
      "Start value {start} and stop value {stop} have to have the same units."
    ),
    start=start,
    stop=stop,
  )
  fail_for_dimension_mismatch(
    stop,
    step,
    error_message=(
      "Stop value {stop} and step value {step} have to have the same units."
    ),
    stop=stop,
    step=step,
  )
  dim = getattr(stop, "dim", DIMENSIONLESS)
  # start is a position-only argument in numpy 2.0
  # https://numpy.org/devdocs/release/2.0.0-notes.html#arange-s-start-argument-is-positional-only
  # TODO: check whether this is still the case in the final release
  if start == 0:
    return Quantity(
      jnp.arange(
        stop=jnp.asarray(stop),
        step=jnp.asarray(step),
        **kwargs,
      ),
      dim=dim,
      copy=False,
    )
  else:
    return Quantity(
      jnp.arange(
        jnp.asarray(start),
        stop=jnp.asarray(stop),
        step=jnp.asarray(step),
        **kwargs,
      ),
      dim=dim,
      copy=False,
    )


arange._do_not_run_doctests = True


@wraps(jnp.linspace)
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
  fail_for_dimension_mismatch(
    start,
    stop,
    error_message=(
      "Start value {start} and stop value {stop} have to have the same units."
    ),
    start=start,
    stop=stop,
  )
  dim = getattr(start, "dim", DIMENSIONLESS)
  result = jnp.linspace(
    jnp.asarray(start),
    jnp.asarray(stop),
    num=num,
    endpoint=endpoint,
    retstep=retstep,
    dtype=dtype,
  )
  return Quantity(result, dim=dim, copy=False)


linspace._do_not_run_doctests = True

# these functions discard subclass info -- maybe a bug in numpy?
ravel = wrap_function_to_method(jnp.ravel)
diagonal = wrap_function_to_method(jnp.diagonal)
trace = wrap_function_to_method(jnp.trace)
dot = wrap_function_to_method(jnp.dot)

# This is a very minor detail: setting the __module__ attribute allows the
# automatic reference doc generation mechanism to attribute the functions to
# this module. Maybe also helpful for IDEs and other code introspection tools.
sin.__module__ = 'braincore.units'
sinh.__module__ = 'braincore.units'
arcsin.__module__ = 'braincore.units'
arcsinh.__module__ = 'braincore.units'
cos.__module__ = 'braincore.units'
cosh.__module__ = 'braincore.units'
arccos.__module__ = 'braincore.units'
arccosh.__module__ = 'braincore.units'
tan.__module__ = 'braincore.units'
tanh.__module__ = 'braincore.units'
arctan.__module__ = 'braincore.units'
arctanh.__module__ = 'braincore.units'

log.__module__ = 'braincore.units'
exp.__module__ = 'braincore.units'
ravel.__module__ = 'braincore.units'
diagonal.__module__ = 'braincore.units'
trace.__module__ = 'braincore.units'
dot.__module__ = 'braincore.units'
arange.__module__ = 'braincore.units'
linspace.__module__ = 'braincore.units'
