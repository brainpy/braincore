import inspect
from functools import partial, wraps
from typing import Any, Sequence, Union

import jax.tree
from jax import numpy as jnp
from jax._src.api import _vjp
from jax.api_util import argnums_partial
from jax.extend import linear_util
from jax.tree_util import (tree_flatten, tree_unflatten)
from braincore._common import set_module_as

__all__ = [
  'vector_grad',
  'ifelse',
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
    leaves, tree = tree_flatten(y)
    tangents = tree_unflatten(tree, [jnp.ones(l.shape, dtype=l.dtype) for l in leaves])
    grads = vjp_fn(tangents)
    if isinstance(argnums, int):
      grads = grads[0]
    if has_aux:
      return (grads, y, aux) if return_value else (grads, aux)
    else:
      return (grads, y) if return_value else grads

  return grad_fun



def _warp_data(data):
  def new_f(*args, **kwargs):
    return data

  return new_f


def _check_f(f):
  if callable(f):
    return f
  else:
    return _warp_data(f)


@set_module_as('braincore')
def ifelse(
    conditions: Union[bool, Sequence[bool]],
    branches: Sequence[Any],
    operands: Any = None,
    show_code: bool = False,
):
  """
  ``If-else`` control flows looks like native Pythonic programming.

  Examples
  --------

  >>> import braincore as bc
  >>> def f(a):
  >>>    return bc.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
  >>>                     branches=[lambda: 1,
  >>>                               lambda: 2,
  >>>                               lambda: 3,
  >>>                               lambda: 4,
  >>>                               lambda: 5])
  >>> f(1)
  4
  >>> # or, it can be expressed as:
  >>> def f(a):
  >>>   return bc.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
  >>>                    branches=[1, 2, 3, 4, 5])
  >>> f(3)
  3

  Parameters
  ----------
  conditions: bool, sequence of bool
    The boolean conditions.
  branches: Any
    The branches, at least has two elements. Elements can be functions,
    arrays, or numbers. The number of ``branches`` and ``conditions`` has
    the relationship of `len(branches) == len(conditions) + 1`.
    Each branch should receive one arguement for ``operands``.
  operands: optional, Any
    The operands for each branch.
  show_code: bool
    Whether show the formatted code.

  Returns
  -------
  res: Any
    The results of the control flow.
  """
  # checking
  if not isinstance(conditions, (tuple, list)):
    conditions = [conditions]
  if not isinstance(conditions, (tuple, list)):
    raise ValueError(f'"conditions" must be a tuple/list of boolean values. '
                     f'But we got {type(conditions)}: {conditions}')
  if not isinstance(branches, (tuple, list)):
    raise ValueError(f'"branches" must be a tuple/list. '
                     f'But we got {type(branches)}.')
  branches = [_check_f(b) for b in branches]
  if len(branches) != len(conditions) + 1:
    raise ValueError(f'The numbers of branches and conditions do not match. '
                     f'Got len(conditions)={len(conditions)} and len(branches)={len(branches)}. '
                     f'We expect len(conditions) + 1 == len(branches). ')
  if operands is None:
    operands = tuple()
  if not isinstance(operands, (list, tuple)):
    operands = [operands]

  # format new codes
  if len(conditions) == 1:
    return jax.lax.cond(conditions[0],
                        branches[0],
                        branches[1],
                        *operands)
  else:
    code_scope = {'conditions': conditions, 'branches': branches}
    codes = ['def f(*operands):',
             f'  f0 = branches[{len(conditions)}]']
    num_cond = len(conditions) - 1
    code_scope['_cond'] = jax.lax.cond
    for i in range(len(conditions) - 1):
      codes.append(f'  f{i + 1} = lambda *r: _cond(conditions[{num_cond - i}], branches[{num_cond - i}], f{i}, *r)')
    codes.append(f'  return _cond(conditions[0], branches[0], f{len(conditions) - 1}, *operands)')
    codes = '\n'.join(codes)
    if show_code:
      print(codes)
    exec(compile(codes.strip(), '', 'exec'), code_scope)
    f = code_scope['f']
    return f(*operands)
