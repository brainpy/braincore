"""
This module implements how to create a JAX Jaxpr from a given function by considering the states that are read and
written by the function. The states are collected by the function and returned as a StateManager instance. The
StateManager instance can be used to manage the states in the JAX program. The module provides a function called
`make_jaxpr` that wraps a given function and returns a JAX Jaxpr and a StateManager instance. The function can also
return the shape, dtype, and named shape of the output of the function.

"""

import functools
import operator
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Callable, Tuple, Union

import jax
from jax.interpreters import partial_eval as pe
from jax.util import wraps

from braincore._state import State, StateTrace

PyTree = Any

__all__ = ["StatefulFunForJaxpr", "make_jaxpr", ]


def _assign_states(states, state_vals):
  assert len(states) == len(state_vals), 'State length mismatch.'
  for st, val in zip(states, state_vals):
    st.value = val


def _ensure_index_tuple(x: Any) -> tuple[int, ...]:
  """Convert x to a tuple of indices."""
  x = jax.core.concrete_or_error(None, x, "expected a static index or sequence of indices.")
  try:
    return (operator.index(x),)
  except TypeError:
    return tuple(jax.util.safe_map(operator.index, x))


class StatefulFunForJaxpr(object):
  """
  A wrapper class for a function that collects the states that are read and written by the function. The states are
  collected by the function and returned as a StateManager instance. The StateManager instance can be used to
  manage the states in the JAX program. The class provides a function called `states` that returns the states
  that are read and written by the function. The class provides a function called `to_state_manager` that returns
  a StateManager instance that contains the states that are read and written by the function. The class provides
  a function called `__call__` that wraps the function and returns the states that are read and written by the
  function and the output of the function.


  Args:
    fun: The function whose ``jaxpr`` is to be computed. Its positional
      arguments and return value should be arrays, scalars, or standard Python
      containers (tuple/list/dict) thereof.
    static_argnums: See the :py:func:`jax.jit` docstring.
    axis_env: Optional, a sequence of pairs where the first element is an axis
        name and the second element is a positive integer representing the size of
        the mapped axis with that name. This parameter is useful when lowering
        functions that involve parallel communication collectives, and it
        specifies the axis name/size environment that would be set up by
        applications of :py:func:`jax.pmap`.
    abstracted_axes: Optional, a pytree with the same structure as the input
        arguments to ``fun``. The leaves of the pytree can be either None or a
        dict with axis names as keys and integers as values. If the leaf is None,
        then the corresponding axis is not abstracted. If the leaf is a dict, then
        the corresponding axis is abstracted, and the dict specifies the axis name
        and size. The abstracted axes are used to infer the input type of the
        function. If None, then all axes are abstracted.
  """

  def __init__(
      self,
      fun: Callable,
      static_argnums: int | Iterable[int] = (),
      axis_env: Sequence[tuple[Hashable, int]] | None = None,
      abstracted_axes: Any | None = None,
      state_returns: Union[str, Tuple[str, ...]] = ('read', 'write')
  ):
    self.fun = fun
    self._static_argnums = _ensure_index_tuple(tuple() if static_argnums is None else static_argnums)
    self._static_args_last = ()
    self._axis_env = axis_env
    self._abstracted_axes = abstracted_axes
    self._jaxpr: jax.core.ClosedJaxpr = None
    self._out_shapes: PyTree = None
    self._jaxpr_out_tree = None
    self._trace: StateTrace = StateTrace()
    self.state_returns = tuple(state_returns) if isinstance(state_returns, (tuple, list)) else (state_returns,)

  def __repr__(self) -> str:
    return (f"{self.__class__.__name__}({self.fun}, "
            f"static_argnums={self._static_argnums}, "
            f"axis_env={self._axis_env}, "
            f"abstracted_axes={self._abstracted_axes})")

  @property
  def jaxpr(self):
    if self._jaxpr is None:
      raise ValueError(f"before accessing the jaxpr, please call "
                       f"'{StatefulFunForJaxpr.__name__}.make_jaxpr()' function.")
    return self._jaxpr

  @property
  def out_shapes(self):
    if self._out_shapes is None:
      raise ValueError(f"before accessing the output shapes, please call "
                       f"'{StatefulFunForJaxpr.__name__}.make_jaxpr()' function.")
    return self._out_shapes

  @property
  def out_treedef(self):
    if self._jaxpr_out_tree is None:
      raise ValueError(f"before accessing the output tree, please call "
                       f"'{StatefulFunForJaxpr.__name__}.make_jaxpr()' function.")
    return self._jaxpr_out_tree

  @property
  def states_to_read(self):
    return tuple([st for st, ty in zip(self._trace.states, self._trace.types) if ty == 'read'])

  @property
  def states_to_write(self):
    return tuple([st for st, ty in zip(self._trace.states, self._trace.types) if ty == 'write'])

  @property
  def states(self):
    return tuple(self._trace.states)

  @property
  def state_values(self):
    return self._trace.collect_values('read', 'write')

  def compile_and_get_states_by_static_args(self, static_args, *args, **kwargs):
    """
    Get the states that are read and written by the function.

    Args:
      static_args: The static arguments to the function.
      *args: The arguments to the function.
      **kwargs: The keyword arguments to the function.

    Returns:
      The states that are read and written by the function.
    """
    pass

  def _init_trace_and_newarg(self) -> None:
    self._trace: StateTrace = StateTrace()
    main = jax.core.thread_local_state.trace_state.trace_stack.stack[-1]
    frame = main.jaxpr_stack[-1]
    trace = pe.DynamicJaxprTrace(main, jax.core.cur_sublevel())
    self._trace.set_new_arg(functools.partial(_new_arg, frame, trace))

  def _wrapped_fun_to_eval(self, *args, **kwargs) -> Tuple[Any, Tuple[State, ...]]:
    """
    Wrap the function and return the states that are read and written by the function and the output of the function.

    Args:
      *args: The arguments to the function.
      **kwargs: The keyword arguments to the function.

    Returns:
      A tuple of the states that are read and written by the function and the output of the function.
    """
    self._init_trace_and_newarg()
    with self._trace:
      out = self.fun(*args, **kwargs)
      state_values = self._trace.collect_values('read', 'write')
    self._trace.recovery_original_values()
    return out, state_values

  def make_jaxpr(self, *args, **kwargs):
    """Creates a function that produces its jaxpr given example args.

    Args:
      *args: The arguments to the function.
      **kwargs: The keyword arguments to the function.

    Returns:
      A ``ClosedJaxpr`` representation of ``fun`` on those arguments. If the
      argument ``return_shape`` is ``True``, then the returned function instead
      returns a pair where the first element is the ``ClosedJaxpr``
      representation of ``fun`` and the second element is a pytree representing
      the structure, shape, dtypes, and named shapes of the output of ``fun``.

    """
    # parameters
    if self._static_argnums:
      dyn_argnums = [i for i in range(len(args)) if i not in self._static_argnums]
      self._static_args_last = tuple(args[i] for i in range(len(args)) if i in dyn_argnums)

    # jaxpr
    jaxpr, (out_shapes, state_shapes) = jax.make_jaxpr(self._wrapped_fun_to_eval,
                                                       static_argnums=self._static_argnums,
                                                       axis_env=self._axis_env,
                                                       return_shape=True,
                                                       abstracted_axes=self._abstracted_axes)(*args, **kwargs)

    # returns
    self._jaxpr_out_tree = jax.tree.structure((out_shapes, state_shapes))
    self._out_shapes = out_shapes
    self._jaxpr = jaxpr
    return self

  def jaxpr_call(self, state_vals, *args, **kwargs) -> Any:
    """
    Call the function at the JAX Jaxpr level.

    Args:
      *args: The arguments to the function.
      **kwargs: The keyword arguments to the function.

    Returns:
      State values and the function output.
    """
    # parameters
    _static_args_last = ()
    if self._static_argnums:
      _static_args_last = tuple(args[i] for i in range(len(args)) if i in self._static_argnums)
      args = tuple(args[i] for i in range(len(args)) if i not in self._static_argnums)
    if self._static_args_last != _static_args_last:
      raise ValueError(f"the function is called with different static arguments. "
                       f"Expected: {self._static_args_last}, got: {_static_args_last}")
    args = jax.tree.flatten((args, kwargs, state_vals))[0]
    # calling the function
    jaxpr_outs = jax.core.eval_jaxpr(self.jaxpr.jaxpr, self.jaxpr.consts, *args)
    out, new_state_vals = self.out_treedef.unflatten(jaxpr_outs)
    assert len(new_state_vals) == len(state_vals)
    return new_state_vals, out

  def jaxpr_call_without_states(self, *args, **kwargs) -> Any:
    """
    Call the function at the JAX Jaxpr level.

    Args:
      *args: The arguments to the function.
      **kwargs: The keyword arguments to the function.

    Returns:
      The output of the function.
    """
    state_vals, out = self.jaxpr_call([st.value for st in self.states], *args, **kwargs)
    for st, val in zip(self.states, state_vals):
      st.value = val
    return out


def make_jaxpr(
    fun: Callable,
    static_argnums: int | Iterable[int] = (),
    axis_env: Sequence[tuple[Hashable, int]] | None = None,
    return_shape: bool = False,
    abstracted_axes: Any | None = None,
    state_returns: Union[str, Tuple[str, ...]] = ('read', 'write')
) -> Callable[..., (Tuple[jax.core.ClosedJaxpr, Tuple[State, ...]] |
                    Tuple[jax.core.ClosedJaxpr, Tuple[State, ...], PyTree])]:
  """Creates a function that produces its jaxpr given example args.

  Args:
    fun: The function whose ``jaxpr`` is to be computed. Its positional
      arguments and return value should be arrays, scalars, or standard Python
      containers (tuple/list/dict) thereof.
    static_argnums: See the :py:func:`jax.jit` docstring.
    axis_env: Optional, a sequence of pairs where the first element is an axis
      name and the second element is a positive integer representing the size of
      the mapped axis with that name. This parameter is useful when lowering
      functions that involve parallel communication collectives, and it
      specifies the axis name/size environment that would be set up by
      applications of :py:func:`jax.pmap`.
    return_shape: Optional boolean, defaults to ``False``. If ``True``, the
      wrapped function returns a pair where the first element is the XLA
      computation and the second element is a pytree with the same structure as
      the output of ``fun`` and where the leaves are objects with ``shape``,
      ``dtype``, and ``named_shape`` attributes representing the corresponding
      types of the output leaves.
    abstracted_axes: Optional, a pytree with the same structure as the input
      arguments to ``fun``. The leaves of the pytree can be either None or a
      dict with axis names as keys and integers as values. If the leaf is None,
      then the corresponding axis is not abstracted. If the leaf is a dict, then
      the corresponding axis is abstracted, and the dict specifies the axis name
      and size. The abstracted axes are used to infer the input type of the
      function. If None, then all axes are abstracted.
    state_returns: Optional, a string or a tuple of strings. The default is
      ``('read', 'write')``. The strings specify the categories of states to be
      returned by the wrapped function. The categories are ``'read'`` and
      ``'write'``. If the category is ``'read'``, then the wrapped function
      returns the states that are read by the function. If the category is
      ``'write'``, then the wrapped function returns the states that are written
      by the function. If the category is ``'read'`` and ``'write'``, then the
      wrapped function returns both the read and write states.


  Returns:
    A wrapped version of ``fun`` that when applied to example arguments returns
    a ``ClosedJaxpr`` representation of ``fun`` on those arguments. If the
    argument ``return_shape`` is ``True``, then the returned function instead
    returns a pair where the first element is the ``ClosedJaxpr``
    representation of ``fun`` and the second element is a pytree representing
    the structure, shape, dtypes, and named shapes of the output of ``fun``.

  A ``jaxpr`` is JAX's intermediate representation for program traces. The
  ``jaxpr`` language is based on the simply-typed first-order lambda calculus
  with let-bindings. :py:func:`make_jaxpr` adapts a function to return its
  ``jaxpr``, which we can inspect to understand what JAX is doing internally.
  The ``jaxpr`` returned is a trace of ``fun`` abstracted to
  :py:class:`ShapedArray` level. Other levels of abstraction exist internally.

  We do not describe the semantics of the ``jaxpr`` language in detail here, but
  instead give a few examples.

  >>> import jax
  >>> import braincore as bc
  >>>
  >>> def f(x): return jax.numpy.sin(jax.numpy.cos(x))
  >>> print(f(3.0))
  -0.83602
  >>> jaxpr, states = bc.transform.make_jaxpr(f)(3.0)
  >>> jaxpr
  { lambda ; a:f32[]. let b:f32[] = cos a; c:f32[] = sin b in (c,) }
  >>> jaxpr, states = bc.transform.make_jaxpr(jax.grad(f))(3.0)
  >>> jaxpr
  { lambda ; a:f32[]. let
      b:f32[] = cos a
      c:f32[] = sin a
      _:f32[] = sin b
      d:f32[] = cos b
      e:f32[] = mul 1.0 d
      f:f32[] = neg e
      g:f32[] = mul f c
    in (g,) }
  """

  stateful_fun = StatefulFunForJaxpr(fun, static_argnums, axis_env, abstracted_axes, state_returns)

  @wraps(fun)
  def make_jaxpr_f(*args, **kwargs):
    stateful_fun.make_jaxpr(*args, **kwargs)
    if return_shape:
      return stateful_fun.jaxpr, stateful_fun.states, stateful_fun.out_shapes
    else:
      return stateful_fun.jaxpr, stateful_fun.states

  # wrapped jaxpr builder function
  make_jaxpr_f.__module__ = "braincore.transform"
  if hasattr(fun, "__qualname__"):
    make_jaxpr_f.__qualname__ = f"make_jaxpr({fun.__qualname__})"
  if hasattr(fun, "__name__"):
    make_jaxpr_f.__name__ = f"make_jaxpr({fun.__name__})"
  return make_jaxpr_f
