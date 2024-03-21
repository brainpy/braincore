# -*- coding: utf-8 -*-


import contextlib
import functools
import os
import re
from collections import defaultdict
from typing import Union

import numpy as np
from jax import config, devices, numpy as jnp
from jax._src.typing import DTypeLike

from ._utils import MemScaling, IdMemScaling
from .mixin import Mode

__all__ = [
  'set', 'context', 'get', 'all',
  'set_host_device_count', 'set_platform', 'set_gpu_preallocation',
  'get_host_device_count', 'get_platform', 'get_dt', 'get_mode', 'get_mem_scaling', 'get_precision',
  'tolerance',
  'dftype', 'ditype', 'dutype', 'dctype',
]

_environment_defaults = dict()
_environment_contexts = defaultdict(list)


@contextlib.contextmanager
def context(**kwargs):
  r"""
  Context-manager that sets a computing environment for brain dynamics computation.

  In BrainPy, there are several basic computation settings when constructing models,
  including ``mode`` for controlling model computing behavior, ``dt`` for numerical
  integration, ``int_`` for integer precision, and ``float_`` for floating precision.
  :py:class:`~.environment`` provides a context for model construction and
  computation. In this temporal environment, models are constructed with the given
  ``mode``, ``dt``, ``int_``, etc., environment settings.

  For instance::

  >>> import braincore.core as bc
  >>> with bc.environ.context(dt=0.1) as env:
  ...     dt = bc.environ.get('dt')
  ...     print(env)

  """
  if 'platform' in kwargs:
    raise ValueError('Cannot set platform in environment context. '
                     'Please use set_platform() or set() for the global setting.')
  if 'host_device_count' in kwargs:
    raise ValueError('Cannot set host_device_count in environment context. '
                     'Please use set_host_device_count() or set() for the global setting.')
  if 'gpu_preallocation' in kwargs:
    raise ValueError('Cannot set gpu_preallocation in environment context. '
                     'Please use set_gpu_preallocation() or set() for the global setting.')
  if 'precision' in kwargs:
    last_precision = get_precision()
    _set_jax_precision(kwargs['precision'])

  try:
    # update the current environment
    for k, v in kwargs.items():
      _environment_contexts[k].append(v)
    # yield the current all environment information
    yield all()
  finally:
    for k, v in kwargs.items():
      _environment_contexts[k].pop()
    if 'precision' in kwargs:
      _set_jax_precision(last_precision)


def get(key: str):
  """
  Get one of the default computation environment.

  Returns
  -------
  item: Any
    The default computation environment.
  """
  if key == 'platform':
    return get_platform()
  if key == 'host_device_count':
    return get_host_device_count()

  if key in _environment_contexts:
    if len(_environment_contexts[key]) > 1:
      return _environment_contexts[key][-1]
  if key in _environment_defaults:
    return _environment_defaults[key]
  raise KeyError(f'No such environmental key: {key}')


def all() -> dict:
  """
  Get all the current default computation environment.
  """
  r = dict()
  for k, v in _environment_contexts.items():
    if v:
      r[k] = v[-1]
  for k, v in _environment_defaults.items():
    if k not in r:
      r[k] = v
  return r


def get_dt():
  """Get the numerical integrator precision.

  Returns
  -------
  dt : float
      Numerical integration precision.
  """
  return get('dt')


def get_mode() -> Mode:
  """Get the default computing mode.

  References
  ----------
  mode: Mode
    The default computing mode.
  """
  return get('mode')


def get_mem_scaling() -> MemScaling:
  """Get the default computing membrane_scaling.

  Returns
  -------
  membrane_scaling: MemScaling
    The default computing membrane_scaling.
  """
  return get('mem_scaling')


def get_platform() -> str:
  """Get the computing platform.

  Returns
  -------
  platform: str
    Either 'cpu', 'gpu' or 'tpu'.
  """
  return devices()[0].platform


def get_host_device_count():
  """
  Get the number of host devices.

  Returns
  -------
  n: int
    The number of host devices.
  """
  xla_flags = os.getenv("XLA_FLAGS", "")
  match = re.search(r"--xla_force_host_platform_device_count=(\d+)", xla_flags)
  return int(match.group(1)) if match else 1


def get_precision() -> int:
  """
  Get the default precision.

  Returns
  -------
  precision: int
    The default precision.
  """
  return get('precision')


def set(
    platform: str = None,
    host_device_count: int = None,
    gpu_preallocation: Union[float, bool] = None,
    mem_scaling: MemScaling = None,
    precision: int = None,
    mode: Mode = None,
    **kwargs
):
  """
  Set the global default computation environment.
  """
  if platform is not None:
    set_platform(platform)
  if host_device_count is not None:
    set_host_device_count(host_device_count)
  if gpu_preallocation is not None:
    set_gpu_preallocation(gpu_preallocation)
  if mem_scaling is not None:
    assert isinstance(mem_scaling, MemScaling), 'mem_scaling must be a MemScaling instance.'
    kwargs['mem_scaling'] = mem_scaling
  if precision is not None:
    _set_jax_precision(precision)
    kwargs['precision'] = precision
  if mode is not None:
    assert isinstance(kwargs['mode'], Mode), 'mode must be a Mode instance.'
    kwargs['mode'] = mode
  _environment_defaults.update(kwargs)


def set_host_device_count(n):
  """
  By default, XLA considers all CPU cores as one device. This utility tells XLA
  that there are `n` host (CPU) devices available to use. As a consequence, this
  allows parallel mapping in JAX :func:`jax.pmap` to work in CPU platform.

  .. note:: This utility only takes effect at the beginning of your program.
      Under the hood, this sets the environment variable
      `XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]`, where
      `[num_device]` is the desired number of CPU devices `n`.

  .. warning:: Our understanding of the side effects of using the
      `xla_force_host_platform_device_count` flag in XLA is incomplete. If you
      observe some strange phenomenon when using this utility, please let us
      know through our issue or forum page. More information is available in this
      `JAX issue <https://github.com/google/jax/issues/1408>`_.

  :param int n: number of devices to use.
  """
  xla_flags = os.getenv("XLA_FLAGS", "")
  xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", xla_flags).split()
  os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count={}".format(n)] + xla_flags)


def set_platform(platform: str):
  """
  Changes platform to CPU, GPU, or TPU. This utility only takes
  effect at the beginning of your program.
  """
  assert platform in ['cpu', 'gpu', 'tpu']
  config.update("jax_platform_name", platform)


def _set_jax_precision(precision: int):
  """
  Set the default precision.

  Args:
    precision: int. The default precision.
  """
  assert precision in [64, 32, 16, 8], f'Precision must be in [64, 32, 16, 8]. But got {precision}.'
  if precision == 64:
    config.update("jax_enable_x64", True)
  else:
    config.update("jax_enable_x64", False)


def _disable_gpu_memory_preallocation(release_memory: bool = True):
  """Disable pre-allocating the GPU memory.

  This disables the preallocation behavior. JAX will instead allocate GPU memory as needed,
  potentially decreasing the overall memory usage. However, this behavior is more prone to
  GPU memory fragmentation, meaning a JAX program that uses most of the available GPU memory
  may OOM with preallocation disabled.

  Args:
    release_memory: bool. Whether we release memory during the computation.
  """
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  if release_memory:
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


def _enable_gpu_memory_preallocation():
  """Disable pre-allocating the GPU memory."""
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
  os.environ.pop('XLA_PYTHON_CLIENT_ALLOCATOR', None)


def set_gpu_preallocation(mode: Union[float, bool]):
  """GPU memory allocation.

  If preallocation is enabled, this makes JAX preallocate ``percent`` of the total GPU memory,
  instead of the default 75%. Lowering the amount preallocated can fix OOMs that occur when the JAX program starts.
  """
  if mode is False:
    _disable_gpu_memory_preallocation()
    return
  if mode is True:
    _enable_gpu_memory_preallocation()
    return
  assert isinstance(mode, float) and 0. <= mode < 1., f'GPU memory preallocation must be in [0., 1.]. But got {mode}.'
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(mode)


@functools.lru_cache()
def _get_uint(precision: int):
  if precision == 64:
    return np.uint64
  elif precision == 32:
    return np.uint32
  elif precision == 16:
    return np.uint16
  elif precision == 8:
    return np.uint8
  else:
    raise ValueError(f'Unsupported precision: {precision}')


@functools.lru_cache()
def _get_int(precision: int):
  if precision == 64:
    return np.int64
  elif precision == 32:
    return np.int32
  elif precision == 16:
    return np.int16
  elif precision == 8:
    return np.int8
  else:
    raise ValueError(f'Unsupported precision: {precision}')


@functools.lru_cache()
def _get_float(precision: int):
  if precision == 64:
    return np.float64
  elif precision == 32:
    return np.float32
  elif precision == 16:
    return jnp.bfloat16
    # return np.float16
  else:
    raise ValueError(f'Unsupported precision: {precision}')


@functools.lru_cache()
def _get_complex(precision: int):
  if precision == 64:
    return np.complex128
  elif precision == 32:
    return np.complex64
  elif precision == 16:
    return np.complex32
  else:
    raise ValueError(f'Unsupported precision: {precision}')


def dftype() -> DTypeLike:
  """
  Default floating data type.
  """
  return _get_float(get_precision())


def ditype() -> DTypeLike:
  """
  Default integer data type.
  """
  return _get_int(get_precision())


def dutype() -> DTypeLike:
  """
  Default unsigned integer data type.
  """
  return _get_uint(get_precision())


def dctype() -> DTypeLike:
  """
  Default complex data type.
  """
  return _get_complex(get_precision())


def tolerance():
  if get_precision() == 64:
    return jnp.array(1e-12, dtype=np.float64)
  elif get_precision() == 32:
    return jnp.array(1e-5, dtype=np.float32)
  else:
    return jnp.array(1e-2, dtype=np.float16)


set(dt=0.1, precision=32, mode=Mode(), mem_scaling=IdMemScaling(), platform='cpu')
