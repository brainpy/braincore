"""
This module implements how to create a JAX Jaxpr from a given function by considering the states that are read and
written by the function. The states are collected by the function and returned as a StateManager instance. The
StateManager instance can be used to manage the states in the JAX program. The module provides a function called
`make_jaxpr` that wraps a given function and returns a JAX Jaxpr and a StateManager instance. The function can also
return the shape, dtype, and named shape of the output of the function.

"""

import operator
from collections.abc import Hashable, Iterable, Sequence
from contextlib import ExitStack
from typing import Any, Callable, Tuple, Union

import jax
from jax._src import source_info_util
from jax._src.core import DBIdx
from jax._src.linear_util import annotate
from jax._src.traceback_util import api_boundary
from jax.api_util import flatten_fun, argnums_partial, shaped_abstractify
from jax.extend.linear_util import wrap_init, WrappedFun
from jax.interpreters import partial_eval as pe
from jax.util import unzip2, wraps

from braincore._state import StateTrace, StateManager

PyTree = Any

__all__ = ["make_jaxpr"]


def _ensure_index_tuple(x: Any) -> tuple[int, ...]:
  """Convert x to a tuple of indices."""
  x = jax.core.concrete_or_error(None, x, "expected a static index or sequence of indices.")
  try:
    return (operator.index(x),)
  except TypeError:
    return tuple(map(operator.index, x))


def _broadcast_prefix(prefix_tree: Any, full_tree: Any, is_leaf: Callable[[Any], bool] | None = None) -> list[Any]:
  # If prefix_tree is not a tree prefix of full_tree, this code can raise a
  # ValueError; use prefix_errors to find disagreements and raise more precise
  # error messages.
  result = []
  num_leaves = lambda t: jax.tree.structure(t).num_leaves
  add_leaves = lambda x, subtree: result.extend([x] * num_leaves(subtree))
  jax.tree.map(add_leaves, prefix_tree, full_tree, is_leaf=is_leaf)
  return result


def _flat_axes_specs(abstracted_axes, *args, **kwargs) -> list[pe.AbstractedAxesSpec]:
  if kwargs: raise NotImplementedError

  def ax_leaf(l):
    return (isinstance(l, dict) and jax.tree_util.all_leaves(l.values()) or
            isinstance(l, tuple) and jax.tree_util.all_leaves(l, lambda x: x is None))

  return _broadcast_prefix(abstracted_axes, args, ax_leaf)


def _abstractify(args, kwargs, abstracted_axes):
  flat_args, in_tree = jax.tree.flatten((args, kwargs))
  if abstracted_axes is None:
    return map(shaped_abstractify, flat_args), in_tree, [True] * len(flat_args)
  else:
    axes_specs = _flat_axes_specs(abstracted_axes, *args, **kwargs)
    in_type = pe.infer_lambda_input_type(axes_specs, flat_args)
    in_avals, keep_inputs = unzip2(in_type)
    return in_avals, in_tree, keep_inputs


class WrappedFunctionToCall(object):
  def __init__(self, fun: Callable, return_categories: Tuple[str, ...] = ('read', 'write')):
    self.fun = fun
    self.state_trace: StateTrace = None
    self.return_categories = return_categories

  def init_trace_newarg(self, trace_new_arg):
    self.state_trace = StateTrace(trace_new_arg)

  def __call__(self, *args, **kwargs) -> Tuple[Any, Tuple]:
    out = self.fun(*args)
    assert self.state_trace is not None, "before calling the function, please assign its state trace instance."
    return out, self.state_trace.collect_values(*self.return_categories)

  def to_state_manager(self) -> StateManager:
    manager = StateManager()
    if 'read' in self.return_categories:
      for state in self.state_trace.reads:
        manager[id(state)] = state
    if 'write' in self.return_categories:
      for state in self.state_trace.writes:
        manager[id(state)] = state
    return manager

  def __repr__(self):
    return f"WrappedFunctionToCall({self.fun})"


def make_jaxpr(
    fun: Callable,
    static_argnums: int | Iterable[int] = (),
    axis_env: Sequence[tuple[Hashable, int]] | None = None,
    return_shape: bool = False,
    abstracted_axes: Any | None = None,
    state_returns: Union[str, Tuple[str, ...]] = ('read', 'write')
) -> Callable[..., (Tuple[jax.core.ClosedJaxpr, StateManager] | Tuple[jax.core.ClosedJaxpr, StateManager, PyTree])]:
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
  if isinstance(state_returns, str):
    state_returns = (state_returns,)
  if not all(cat in ('read', 'write') for cat in state_returns):
    raise ValueError(f"Expected `state_returns` to be 'read', 'write', or a tuple of them, got {state_returns}")
  assert callable(fun), "Expected `fun` to be a callable, got {}".format(fun)
  fun_as_wrapped_obj = WrappedFunctionToCall(fun, return_categories=state_returns)
  static_argnums = _ensure_index_tuple(static_argnums)

  @wraps(fun)
  @api_boundary
  def make_jaxpr_f(*args, **kwargs):
    f = wrap_init(fun_as_wrapped_obj)
    if static_argnums:
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      f, args = argnums_partial(f, dyn_argnums, args)
    in_avals, in_tree, keep_inputs = _abstractify(args, kwargs, abstracted_axes)
    in_type = tuple(zip(in_avals, keep_inputs))
    f, out_tree = flatten_fun(f, in_tree)
    f = annotate(f, in_type)
    debug_info = pe.debug_info(fun_as_wrapped_obj, in_tree, out_tree, True, 'make_jaxpr')
    with ExitStack() as stack:
      for axis_name, size in axis_env or []:
        stack.enter_context(jax.core.extend_axis_env(axis_name, size, None))
      jaxpr, out_type, consts = trace_to_jaxpr_dynamic2(f, debug_info=debug_info)
    closed_jaxpr = jax.core.ClosedJaxpr(jaxpr, consts)
    state_stack = fun_as_wrapped_obj.to_state_manager()
    fun_as_wrapped_obj.state_trace.recovery_original_values()

    if return_shape:
      out_avals, _ = unzip2(out_type)
      out_shapes_flat = [jax.ShapeDtypeStruct(a.shape, a.dtype, a.named_shape) for a in out_avals]
      return closed_jaxpr, state_stack, jax.tree.unflatten(out_tree(), out_shapes_flat)
    else:
      return closed_jaxpr, state_stack

  # wrapped jaxpr builder function
  make_jaxpr_f.__module__ = "braincore.transform"
  if hasattr(fun, "__qualname__"):
    make_jaxpr_f.__qualname__ = f"make_jaxpr({fun.__qualname__})"
  if hasattr(fun, "__name__"):
    make_jaxpr_f.__name__ = f"make_jaxpr({fun.__name__})"
  return make_jaxpr_f


@jax.profiler.annotate_function
def trace_to_jaxpr_dynamic2(
    fun: WrappedFun,
    debug_info: pe.DebugInfo | None = None
) -> tuple[jax.core.Jaxpr, jax.core.OutputType, list[Any]]:
  with jax.core.new_main(pe.DynamicJaxprTrace, dynamic=True) as main:  # type: ignore
    main.jaxpr_stack = ()  # type: ignore
    jaxpr, out_type, consts = trace_to_subjaxpr_dynamic2(fun, main, debug_info)
    del main, fun
  return jaxpr, out_type, consts


def trace_to_subjaxpr_dynamic2(
    fun: WrappedFun,
    main: jax.core.MainTrace,
    debug_info: pe.DebugInfo | None = None
) -> tuple[jax.core.Jaxpr, jax.core.OutputType, list[Any]]:
  in_avals, keep_inputs = unzip2(fun.in_type)
  frame = pe.JaxprStackFrame()
  frame.debug_info = debug_info
  with pe.extend_jaxpr_stack(main, frame), source_info_util.reset_name_stack():
    trace = pe.DynamicJaxprTrace(main, jax.core.cur_sublevel())
    in_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
    in_tracers = [t for t, keep in zip(in_tracers, keep_inputs) if keep]
    fun.f.init_trace_newarg(trace.new_arg)
    with fun.f.state_trace:  # collect reads and writes
      ans = fun.call_wrapped(*in_tracers)
    out_tracers = map(trace.full_raise, ans)
    jaxpr, out_type, consts = frame.to_jaxpr2(out_tracers)
    del fun, main, trace, frame, out_tracers, ans
  return jaxpr, out_type, consts


def _input_type_to_tracers(
    new_arg: Callable[[jax.core.AbstractValue], jax.core.Tracer],
    in_avals: Sequence[jax.core.AbstractValue]
) -> Sequence[jax.core.Tracer]:
  # Create input Tracers given input AbstractValues, each of which can contain
  # DeBruijn indices which refer to positions in the input argument list. That
  # is, each element `a` of `in_avals` can have DBIdx instances in its shape,
  # which must refer to positions left of `a`'s.
  in_tracers: list[jax.core.Tracer] = []

  def _substitute_tracers_in_aval(a: jax.core.AbstractValue) -> jax.core.AbstractValue:
    if isinstance(a, jax.core.DShapedArray) and any(type(d) is DBIdx for d in a.shape):
      shape = [in_tracers[d.val] if type(d) is DBIdx else d for d in a.shape]  # type: ignore
      return a.update(shape=tuple(shape))
    return a

  for a in in_avals:
    in_tracers.append(new_arg(_substitute_tracers_in_aval(a)))
  return in_tracers

# if __name__ == '__main__':
#
#   import jax.numpy as jnp
#
#   jaxpr, states = make_jaxpr(jnp.sin)(3.0)
#   print(jaxpr)
#   print(states)
#
#   st1 = State(jnp.ones(10))
#
#   def fa(x):
#     st1.value = x + st1.value
#
#
#   jaxpr, states = make_jaxpr(fa)(jnp.zeros(1))
#   print(jaxpr)
#   print(states)
#
#
#   def ffa(x):
#     jaxpr, states = make_jaxpr(fa)(x)
#     return 1.
#
#
#   jaxpr, states = make_jaxpr(ffa)(jnp.zeros(1))
#   print(jaxpr)
#   print(states)
