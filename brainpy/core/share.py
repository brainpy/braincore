# -*- coding: utf-8 -*-


import contextlib
from typing import Any

__all__ = [
  'clear', 'set', 'get', 'context',
]


# Default, there are several shared arguments in the global context.
I = 'i'  # the index of the current computation.
T = 't'  # the current time of the current computation.
JIT_ERROR_CHECK = 'jit_error_check'  # whether to record the current computation.
FIT = 'fit'  # whether to fit the model.

_context = dict()
_NOT_PROVIDE = object()


def clear() -> None:
  """
  Clear all shared data in this computation context.
  """
  _context.clear()


def set(**kwargs) -> None:
  """
  Save shared arguments in the global context.
  """
  for k, v in kwargs.items():
    _context[k] = v


def get(key: str, default: Any = _NOT_PROVIDE) -> Any:
  """
  Get shared arguments in the global context.

  Args:
    key: str
      The key of the shared argument.
    default: Any
      The default value if the key is not found.
  """
  if key not in _context:
    if default is _NOT_PROVIDE:
      raise KeyError(f"Key {key} not found in the context. You can set it by `share.set({key}=value)`.")
    return default
  return _context[key]


@contextlib.contextmanager
def context(**kwargs):
  """
  A context manager to set shared arguments in the global context.
  """
  old_conflict = dict()
  for k, v in kwargs.items():
    # Save the old shared arguments.
    if k in _context:
      old_conflict[k] = _context[k]
    # Set the new shared arguments.
    _context[k] = v
  try:
    yield
  finally:
    # Remove the current shared arguments.
    for k, v in kwargs.items():
      if k in _context:
        del _context[k]
    # Restore the old shared arguments.
    set(**old_conflict)
