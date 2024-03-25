import inspect
from functools import partial, wraps

import jax
from jax import numpy as jnp
from jax._src.api import _vjp
from jax.api_util import argnums_partial
from jax.extend import linear_util

from braincore._common import set_module_as

__all__ = [
  'vector_grad',
]


def _isgeneratorfunction(fun):
  # re-implemented here because of https://bugs.python.org/issue33261
  while inspect.ismethod(fun):
    fun = fun.__func__
  while isinstance(fun, partial):
    fun = fun.func
  return inspect.isfunction(fun) and bool(fun.__code__.co_flags & inspect.CO_GENERATOR)


def _check_callable(fun):
  # In Python 3.10+, the only thing stopping us from supporting staticmethods
  # is that we can't take weak references to them, which the C++ JIT requires.
  if isinstance(fun, staticmethod):
    raise TypeError(f"staticmethod arguments are not supported, got {fun}")
  if not callable(fun):
    raise TypeError(f"Expected a callable value, got {fun}")
  if _isgeneratorfunction(fun):
    raise TypeError(f"Expected a function, got a generator function: {fun}")


@set_module_as('braincore')
def vector_grad(func, argnums=0, return_value: bool = False, has_aux: bool = False):
  """
   Compute the gradient of a vector with respect to the input.
   """
  _check_callable(func)

  @wraps(func)
  def grad_fun(*args, **kwargs):
    f = linear_util.wrap_init(func, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args, require_static_args_hashable=False)
    if has_aux:
      y, vjp_fn, aux = _vjp(f_partial, *dyn_args, has_aux=True)
    else:
      y, vjp_fn = _vjp(f_partial, *dyn_args, has_aux=False)
    leaves, tree = jax.tree.flatten(y)
    tangents = jax.tree.unflatten(tree, [jnp.ones(l.shape, dtype=l.dtype) for l in leaves])
    grads = vjp_fn(tangents)
    if isinstance(argnums, int):
      grads = grads[0]
    if has_aux:
      return (grads, y, aux) if return_value else (grads, aux)
    else:
      return (grads, y) if return_value else grads

  return grad_fun

