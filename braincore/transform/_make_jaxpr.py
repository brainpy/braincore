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
from typing import Any, Callable, Tuple, Union, Dict

import jax
from jax._src import source_info_util
from jax.interpreters import partial_eval as pe
from jax.util import wraps

from braincore._state import State, StateTrace

PyTree = Any

__all__ = ["StatefulFunction", "make_jaxpr", ]


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


# modified from jax.interpreters.partial_eval.DynamicJaxprTrace.new_arg()
def _new_arg(frame, trace, aval):
  tracer = pe.DynamicJaxprTracer(trace, aval, source_info_util.current())
  frame.tracers.append(tracer)
  frame.tracer_to_var[id(tracer)] = var = frame.newvar(aval)
  frame.invars.append(var)
  return tracer


class StatefulFunction(object):
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
    state_returns: Optional, a string or a tuple of strings. The default is
        ``('read', 'write')``. The strings specify the categories of states to be
        returned by the wrapped function. The categories are ``'read'`` and
        ``'write'``. If the category is ``'read'``, then the wrapped function
        returns the states that are read by the function. If the category is
        ``'write'``, then the wrapped function returns the states that are written
        by the function. If the category is ``'read'`` and ``'write'``, then the
        wrapped function returns both the read and write states.

  """

  def __init__(
      self,
      fun: Callable,
      static_argnums: int | Iterable[int] = (),
      axis_env: Sequence[tuple[Hashable, int]] | None = None,
      abstracted_axes: Any | None = None,
      state_returns: Union[str, Tuple[str, ...]] = ('read', 'write')
  ):
    # explicit parameters
    self.fun = fun
    self.static_argnums = _ensure_index_tuple(tuple() if static_argnums is None else static_argnums)
    self.axis_env = axis_env
    self.abstracted_axes = abstracted_axes
    self.state_returns = tuple(state_returns) if isinstance(state_returns, (tuple, list)) else (state_returns,)

    # implicit parameters
    self._jaxpr: Dict[Any, jax.core.ClosedJaxpr] = dict()
    self._out_shapes: Dict[Any, PyTree] = dict()
    self._jaxpr_out_tree: Dict[Any, PyTree] = dict()
    self._state_trace: Dict[Any, StateTrace] = dict()

  def __repr__(self) -> str:
    return (f"{self.__class__.__name__}({self.fun}, "
            f"static_argnums={self.static_argnums}, "
            f"axis_env={self.axis_env}, "
            f"abstracted_axes={self.abstracted_axes}, "
            f"state_returns={self.state_returns})")

  def _check_static_args(self, static_args: tuple):
    assert len(static_args) == len(self.static_argnums), 'Static arguments length mismatch.'

  def get_jaxpr(self, *static_args) -> jax.core.ClosedJaxpr:
    """
    Read the JAX Jaxpr representation of the function.

    Args:
      static_args: The static arguments to the function.

    Returns:
      The JAX Jaxpr representation of the function.
    """
    self._check_static_args(static_args)
    if static_args not in self._jaxpr:
      raise ValueError(f"the function is not called with the static arguments: {static_args}")
    return self._jaxpr[static_args]

  def get_out_shapes(self, *static_args) -> PyTree:
    """
    Read the output shapes of the function.

    Args:
      *static_args: The static arguments to the function.

    Returns:
      The output shapes of the function.
    """
    self._check_static_args(static_args)
    if static_args not in self._out_shapes:
      raise ValueError(f"the function is not called with the static arguments: {static_args}")
    return self._out_shapes[static_args]

  def get_out_treedef(self, *static_args) -> PyTree:
    """
    Read the output tree of the function.

    Args:
      *static_args: The static arguments to the function.

    Returns:
      The output tree of the function.
    """
    self._check_static_args(static_args)
    if static_args not in self._jaxpr_out_tree:
      raise ValueError(f"the function is not called with the static arguments: {static_args}")
    return self._jaxpr_out_tree[static_args]

  def get_states(self, *static_args) -> Tuple[State, ...]:
    """
    Read the states that are read and written by the function.

    Args:
      *static_args: The static arguments to the function.

    Returns:
      The states that are read and written by the function.
    """
    self._check_static_args(static_args)
    if static_args not in self._state_trace:
      raise ValueError(f"the function is not called with the static arguments: {static_args}")
    return tuple(self._state_trace[static_args].states)

  def get_read_states(self, *static_args) -> Tuple[State, ...]:
    """
    Read the states that are read by the function.

    Args:
      *static_args: The static arguments to the function.

    Returns:
      The states that are read by the function.
    """
    _state_trace = self._state_trace[static_args]
    return tuple([st for st, ty in zip(_state_trace.states, _state_trace.types) if ty == 'read'])

  def get_write_states(self, *static_args) -> Tuple[State, ...]:
    """
    Read the states that are written by the function.

    Args:
      *static_args: The static arguments to the function.

    Returns:
      The states that are written by the function.
    """
    state_trace = self._state_trace[static_args]
    return tuple([st for st, ty in zip(state_trace.states, state_trace.types) if ty == 'write'])

  def get_static_args(self, *args):
    """
    Get the static arguments from the arguments.

    Args:
      *args: The arguments to the function.

    Returns:
      The static arguments.
    """
    return tuple(args[i] for i in self.static_argnums)

  def compile_and_get_states_by_static_args(self, *args, **kwargs) -> Tuple[State, ...]:
    """
    Get the states that are read and written by the function.

    Args:
      *args: The arguments to the function.
      **kwargs: The keyword arguments to the function.

    Returns:
      The states that are read and written by the function.
    """
    static_args = self.get_static_args(*args)
    if static_args not in self._state_trace:
      self.make_jaxpr(*args, **kwargs)
    return self.get_states(*static_args)

  def _init_trace_and_newarg(self) -> StateTrace:
    state_trace: StateTrace = StateTrace()
    main = jax.core.thread_local_state.trace_state.trace_stack.stack[-1]
    frame = main.jaxpr_stack[-1]
    trace = pe.DynamicJaxprTrace(main, jax.core.cur_sublevel())
    state_trace.set_new_arg(functools.partial(_new_arg, frame, trace))
    return state_trace

  def _wrapped_fun_to_eval(self, *args, **kwargs) -> Tuple[Any, Tuple[State, ...]]:
    """
    Wrap the function and return the states that are read and written by the function and the output of the function.

    Args:
      *args: The arguments to the function.
      **kwargs: The keyword arguments to the function.

    Returns:
      A tuple of the states that are read and written by the function and the output of the function.
    """
    static_args = self.get_static_args(*args)
    # state trace
    _state_trace = self._init_trace_and_newarg()
    self._state_trace[static_args] = _state_trace
    with _state_trace:
      out = self.fun(*args, **kwargs)
      state_values = _state_trace.collect_values('read', 'write')
    _state_trace.recovery_original_values()
    return out, state_values

  def make_jaxpr(self, *args, **kwargs):
    """Creates a function that produces its jaxpr given example args.

    A ``ClosedJaxpr`` representation of ``fun`` on those arguments. If the
    argument ``return_shape`` is ``True``, then the returned function instead
    returns a pair where the first element is the ``ClosedJaxpr``
    representation of ``fun`` and the second element is a pytree representing
    the structure, shape, dtypes, and named shapes of the output of ``fun``.

    Args:
      *args: The arguments to the function.
      **kwargs: The keyword arguments to the function.
    """

    # static args
    static_args = self.get_static_args(*args)

    if static_args not in self._state_trace:
      try:
        # jaxpr
        jaxpr, (out_shapes, state_shapes) = jax.make_jaxpr(
          self._wrapped_fun_to_eval,
          static_argnums=self.static_argnums,
          axis_env=self.axis_env,
          return_shape=True,
          abstracted_axes=self.abstracted_axes
        )(*args, **kwargs)

        # returns
        self._jaxpr_out_tree[static_args] = jax.tree.structure((out_shapes, state_shapes))
        self._out_shapes[static_args] = out_shapes
        self._jaxpr[static_args] = jaxpr
      except Exception as e:
        self._state_trace.pop(static_args)
        raise e

    return self

  def jaxpr_call(self, state_vals, *args, **kwargs) -> Any:
    """
    Call the function at the JAX Jaxpr level.

    Args:
      state_vals: The state values.
      *args: The arguments to the function.
      **kwargs: The keyword arguments to the function.

    Returns:
      State values and the function output.
    """
    # state checking
    _static_args = self.get_static_args(*args)
    states = self.get_states(*_static_args)
    assert len(state_vals) == len(states), 'State length mismatch.'

    # parameters
    args = tuple(args[i] for i in range(len(args)) if i not in self.static_argnums)
    args = jax.tree.flatten((args, kwargs, state_vals))[0]

    # calling the function
    closed_jaxpr = self.get_jaxpr(*_static_args)
    out_treedef = self.get_out_treedef(*_static_args)
    jaxpr_outs = jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)

    # output processing
    out, new_state_vals = out_treedef.unflatten(jaxpr_outs)
    assert len(new_state_vals) == len(state_vals), 'State length mismatch.'
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
    _static_args = self.get_static_args(*args)
    states = self.get_states(*_static_args)
    state_vals, out = self.jaxpr_call([st.value for st in states], *args, **kwargs)
    for st, val in zip(states, state_vals):
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

  stateful_fun = StatefulFunction(fun, static_argnums, axis_env, abstracted_axes, state_returns)

  @wraps(fun)
  def make_jaxpr_f(*args, **kwargs):
    stateful_fun.make_jaxpr(*args, **kwargs)
    static_args = tuple(args[i] for i in stateful_fun.static_argnums)
    if return_shape:
      return (stateful_fun.get_jaxpr(*static_args),
              stateful_fun.get_states(*static_args),
              stateful_fun.get_out_shapes(*static_args))
    else:
      return (stateful_fun.get_jaxpr(*static_args),
              stateful_fun.get_states(*static_args))

  # wrapped jaxpr builder function
  make_jaxpr_f.__module__ = "braincore.transform"
  if hasattr(fun, "__qualname__"):
    make_jaxpr_f.__qualname__ = f"make_jaxpr({fun.__qualname__})"
  if hasattr(fun, "__name__"):
    make_jaxpr_f.__name__ = f"make_jaxpr({fun.__name__})"
  return make_jaxpr_f
