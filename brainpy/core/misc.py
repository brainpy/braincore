from brainpy import environ

__all__ = [
  'get_dtype'
]


def get_dtype(a):
  """
  Get the dtype of a.
  """
  if hasattr(a, 'dtype'):
    return a.dtype
  else:
    if isinstance(a, bool):
      return bool
    elif isinstance(a, int):
      return environ.ditype()
    elif isinstance(a, float):
      return environ.dftype()
    elif isinstance(a, complex):
      return environ.dctype()
    else:
      raise ValueError(f'Can not get dtype of {a}.')
