from typing import Any, Sequence, Union

import jax.tree

from braincore._common import set_module_as

__all__ = [
  'ifelse',
]


def jit(

):
  pass



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
