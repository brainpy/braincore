from functools import wraps, partial
from typing import Callable, Union

import jax
from jax import numpy as jnp
from jax.core import Primitive, ShapedArray
from jax.experimental.host_callback import id_tap
from jax.interpreters import batching, mlir, xla
from jax.lax import cond

from braincore._common import set_module_as

__all__ = [
  'jit_error',
]


@set_module_as('braincore.transform')
def remove_vmap(x, op='any'):
  if op == 'any':
    return _any_without_vmap(x)
  elif op == 'all':
    return _all_without_vmap(x)
  else:
    raise ValueError(f'Do not support type: {op}')


_any_no_vmap_prim = Primitive('any_no_vmap')


def _any_without_vmap(x):
  return _any_no_vmap_prim.bind(x)


def _any_without_vmap_imp(x):
  return jnp.any(x)


def _any_without_vmap_abs(x):
  return ShapedArray(shape=(), dtype=jnp.bool_)


def _any_without_vmap_batch(x, batch_axes):
  (x,) = x
  return _any_without_vmap(x), batching.not_mapped


_any_no_vmap_prim.def_impl(_any_without_vmap_imp)
_any_no_vmap_prim.def_abstract_eval(_any_without_vmap_abs)
batching.primitive_batchers[_any_no_vmap_prim] = _any_without_vmap_batch
if hasattr(xla, "lower_fun"):
  xla.register_translation(_any_no_vmap_prim,
                           xla.lower_fun(_any_without_vmap_imp, multiple_results=False, new_style=True))
mlir.register_lowering(_any_no_vmap_prim, mlir.lower_fun(_any_without_vmap_imp, multiple_results=False))

_all_no_vmap_prim = Primitive('all_no_vmap')


def _all_without_vmap(x):
  return _all_no_vmap_prim.bind(x)


def _all_without_vmap_imp(x):
  return jnp.all(x)


def _all_without_vmap_abs(x):
  return ShapedArray(shape=(), dtype=jnp.bool_)


def _all_without_vmap_batch(x, batch_axes):
  (x,) = x
  return _all_without_vmap(x), batching.not_mapped


_all_no_vmap_prim.def_impl(_all_without_vmap_imp)
_all_no_vmap_prim.def_abstract_eval(_all_without_vmap_abs)
batching.primitive_batchers[_all_no_vmap_prim] = _all_without_vmap_batch
if hasattr(xla, "lower_fun"):
  xla.register_translation(_all_no_vmap_prim,
                           xla.lower_fun(_all_without_vmap_imp, multiple_results=False, new_style=True))
mlir.register_lowering(_all_no_vmap_prim, mlir.lower_fun(_all_without_vmap_imp, multiple_results=False))


def _err_jit_true_branch(err_fun, x):
  id_tap(err_fun, x)
  return


def _err_jit_false_branch(x):
  return


def _cond(err_fun, pred, err_arg):
  @wraps(err_fun)
  def true_err_fun(arg, transforms):
    err_fun(arg)

  cond(pred,
       partial(_err_jit_true_branch, true_err_fun),
       _err_jit_false_branch,
       err_arg)


def _error_msg(msg):
  raise ValueError(msg)


@set_module_as('braincore.transform')
def jit_error(pred, err_fun: Union[Callable, str], err_arg=None, scope: str = 'any'):
  """Check errors in a jit function.

  >>> def error(arg):
  >>>    raise ValueError(f'error {arg}')
  >>> x = jax.random.uniform(jax.random.PRNGKey(0), (10,))
  >>> jit_error(x.sum() < 5., error, err_arg=x)

  Parameters
  ----------
  pred: bool, Array
    The boolean prediction.
  err_fun: callable
    The error function, which raise errors.
  err_arg: any
    The arguments which passed into `err_f`.
  scope: str
    The scope of the error message. Can be None, 'all' or 'any'.
  """
  if isinstance(err_fun, str):
    err_fun = partial(_error_msg, err_fun)
    assert err_arg is None, "err_arg should be None when err_fun is a string."
  if scope is None:
    pred = pred
  elif scope == 'all':
    pred = remove_vmap(pred, 'all')
  elif scope == 'any':
    pred = remove_vmap(pred, 'any')
  else:
    raise ValueError(f"Unknown scope: {scope}")
  _cond(err_fun, pred, err_arg)
