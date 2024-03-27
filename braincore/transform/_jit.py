from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence
from typing import (Any, Callable)

import jax
from jax._src import sharding_impls
from jax.lib import xla_client as xc

from ._make_jaxpr import StatefulFunForJaxpr, _ensure_index_tuple, _assign_states

__all__ = ['jit']


def jit(
  fun: Callable,
  in_shardings=sharding_impls.UNSPECIFIED,
  out_shardings=sharding_impls.UNSPECIFIED,
  static_argnums: int | Sequence[int] | None = None,
  donate_argnums: int | Sequence[int] | None = None,
  donate_argnames: str | Iterable[str] | None = None,
  keep_unused: bool = False,
  device: xc.Device | None = None,
  backend: str | None = None,
  inline: bool = False,
  abstracted_axes: Any | None = None,
  **kwargs
):
  """
  Sets up ``fun`` for just-in-time compilation with XLA.

  Does not support setting ``static_argnames`` as in ``jax.jit()``.


  Args:
    fun: Function to be jitted.
    in_shardings: Pytree of structure matching that of arguments to ``fun``,
      with all actual arguments replaced by resource assignment specifications.
      It is also valid to specify a pytree prefix (e.g. one value in place of a
      whole subtree), in which case the leaves get broadcast to all values in
      that subtree.

      The ``in_shardings`` argument is optional. JAX will infer the shardings
      from the input :py:class:`jax.Array`'s and defaults to replicating the input
      if the sharding cannot be inferred.

      The valid resource assignment specifications are:
        - :py:class:`XLACompatibleSharding`, which will decide how the value
            will be partitioned. With this, using a mesh context manager is not
            required.
        - :py:obj:`None`, will give JAX the freedom to choose whatever sharding
          it wants.
          For in_shardings, JAX will mark is as replicated but this behavior
          can change in the future.
          For out_shardings, we will rely on the XLA GSPMD partitioner to
          determine the output shardings.

      The size of every dimension has to be a multiple of the total number of
      resources assigned to it. This is similar to pjit's in_shardings.
    out_shardings: Like ``in_shardings``, but specifies resource
      assignment for function outputs. This is similar to pjit's
      out_shardings.

      The ``out_shardings`` argument is optional. If not specified, :py:func:`jax.jit`
      will use GSPMD's sharding propagation to figure out what the sharding of the
      output(s) should be.
    static_argnums: An optional int or collection of ints that specify which
      positional arguments to treat as static (compile-time constant).
      Operations that only depend on static arguments will be constant-folded in
      Python (during tracing), and so the corresponding argument values can be
      any Python object.

      Static arguments should be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and immutable. Calling the jitted function
      with different values for these constants will trigger recompilation.
      Arguments that are not arrays or containers thereof must be marked as
      static.

      If neither ``static_argnums`` nor ``static_argnames`` is provided, no
      arguments are treated as static. If ``static_argnums`` is not provided but
      ``static_argnames`` is, or vice versa, JAX uses
      :code:`inspect.signature(fun)` to find any positional arguments that
      correspond to ``static_argnames``
      (or vice versa). If both ``static_argnums`` and ``static_argnames`` are
      provided, ``inspect.signature`` is not used, and only actual
      parameters listed in either ``static_argnums`` or ``static_argnames`` will
      be treated as static.
    donate_argnums: Specify which positional argument buffers are "donated" to
      the computation. It is safe to donate argument buffers if you no longer
      need them once the computation has finished. In some cases XLA can make
      use of donated buffers to reduce the amount of memory needed to perform a
      computation, for example recycling one of your input buffers to store a
      result. You should not reuse buffers that you donate to a computation, JAX
      will raise an error if you try to. By default, no argument buffers are
      donated.

      If neither ``donate_argnums`` nor ``donate_argnames`` is provided, no
      arguments are donated. If ``donate_argnums`` is not provided but
      ``donate_argnames`` is, or vice versa, JAX uses
      :code:`inspect.signature(fun)` to find any positional arguments that
      correspond to ``donate_argnames``
      (or vice versa). If both ``donate_argnums`` and ``donate_argnames`` are
      provided, ``inspect.signature`` is not used, and only actual
      parameters listed in either ``donate_argnums`` or ``donate_argnames`` will
      be donated.

      For more details on buffer donation see the
      `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.
    donate_argnames: An optional string or collection of strings specifying
      which named arguments are donated to the computation. See the
      comment on ``donate_argnums`` for details. If not
      provided but ``donate_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    keep_unused: If `False` (the default), arguments that JAX determines to be
      unused by `fun` *may* be dropped from resulting compiled XLA executables.
      Such arguments will not be transferred to the device nor provided to the
      underlying executable. If `True`, unused arguments will not be pruned.
    device: This is an experimental feature and the API is likely to change.
      Optional, the Device the jitted function will run on. (Available devices
      can be retrieved via :py:func:`jax.devices`.) The default is inherited
      from XLA's DeviceAssignment logic and is usually to use
      ``jax.devices()[0]``.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.
    inline: Specify whether this function should be inlined into enclosing
      jaxprs (rather than being represented as an application of the xla_call
      primitive with its own subjaxpr). Default False.
    abstracted_axes:

  Returns:
    A wrapped version of ``fun``, set up for just-in-time compilation.

  """
  if static_argnums is None:
    static_argnums = tuple()
  static_argnums = _ensure_index_tuple(static_argnums)
  fun = StatefulFunForJaxpr(fun, static_argnums=static_argnums, abstracted_axes=abstracted_axes)
  jit_fun = jax.jit(fun.jaxpr_call,
                    static_argnums=tuple(i + 1 for i in static_argnums),
                    donate_argnums=donate_argnums,
                    donate_argnames=donate_argnames,
                    keep_unused=keep_unused,
                    device=device,
                    backend=backend,
                    inline=inline,
                    in_shardings=in_shardings,
                    out_shardings=out_shardings,
                    abstracted_axes=abstracted_axes,
                    **kwargs)

  @functools.wraps(fun.fun)
  def fun_to_jit(*args, **kwargs):
    static_args = tuple(args[i] for i in static_argnums)
    states = fun.compile_and_get_states_by_static_args(static_args)
    state_vals, outs = jit_fun([st.value for st in states], *args, **kwargs)
    _assign_states(states, state_vals)
    return outs

  return fun_to_jit

