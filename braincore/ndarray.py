import collections
import itertools
import numbers
import operator
from typing import Any, Union, Optional, Sequence, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.dtypes import canonicalize_dtype
from jax.tree_util import register_pytree_node_class
from brainpy.core.units import (get_or_create_dimension,UFUNCS_PRESERVE_DIMENSIONS,
                                UFUNCS_CHANGE_DIMENSIONS, UFUNCS_MATCHING_DIMENSIONS,
                                UFUNCS_COMPARISONS, UFUNCS_LOGICAL, UFUNCS_DIMENSIONLESS,
                                UFUNCS_DIMENSIONLESS_TWOARGS, UFUNCS_INTEGERS, )

__all__ = [
  'Array',
  'ndarray',
]

numpy_func_return = 'bp_array'
_all_slice = slice(None, None, None)
unit_checking = True


def _as_jax_array_(obj):
  return obj.value if isinstance(obj, Array) else obj


def _check_input_array(array):
  if isinstance(array, Array):
    return array
  elif isinstance(array, np.ndarray):
    return Array(value=jnp.asarray(array))
  else:
    return array


def _return(a):
  if numpy_func_return == 'bp_array' and isinstance(a, jax.Array) and a.ndim > 0:
    return Array(value=a.value, unit=a.unit, copy=False)
  return a


def _check_out(out):
  if not isinstance(out, Array):
    raise TypeError(f'out must be an instance of brainpy Array. But got {type(out)}')


def _get_dtype(v):
  if hasattr(v, 'dtype'):
    dtype = v.dtype
  else:
    dtype = canonicalize_dtype(type(v))
  return dtype


def _short_str(arr):
  """
  Return a short string representation of an array, suitable for use in
  error messages.
  """
  arr = np.asanyarray(arr)
  old_printoptions = jnp.get_printoptions()
  jnp.set_printoptions(edgeitems=2, threshold=5)
  arr_string = str(arr)
  jnp.set_printoptions(**old_printoptions)
  return arr_string


def get_unit_for_display(d):
  """
  Return a string representation of an appropriate unscaled unit or ``'1'``
  for a dimensionless quantity.

  Parameters
  ----------
  d : `Dimension` or int
      The dimension to find a unit for.

  Returns
  -------
  s : str
      A string representation of the respective unit or the string ``'1'``.
  """
  if (isinstance(d, int) and d == 1) or d is DIMENSIONLESS:
    return "1"
  else:
    return str(get_unit(d))

# SI dimensions (see table at the top of the file) and various descriptions,
# each description maps to an index i, and the power of each dimension
# is stored in the variable dims[i]
_di = {
  "Length": 0,
  "length": 0,
  "metre": 0,
  "metres": 0,
  "meter": 0,
  "meters": 0,
  "m": 0,
  "Mass": 1,
  "mass": 1,
  "kilogram": 1,
  "kilograms": 1,
  "kg": 1,
  "Time": 2,
  "time": 2,
  "second": 2,
  "seconds": 2,
  "s": 2,
  "Electric Current": 3,
  "electric current": 3,
  "Current": 3,
  "current": 3,
  "ampere": 3,
  "amperes": 3,
  "A": 3,
  "Temperature": 4,
  "temperature": 4,
  "kelvin": 4,
  "kelvins": 4,
  "K": 4,
  "Quantity of Substance": 5,
  "Quantity of substance": 5,
  "quantity of substance": 5,
  "Substance": 5,
  "substance": 5,
  "mole": 5,
  "moles": 5,
  "mol": 5,
  "Luminosity": 6,
  "luminosity": 6,
  "candle": 6,
  "candles": 6,
  "cd": 6,
}

_ilabel = ["m", "kg", "s", "A", "K", "mol", "cd"]

# The same labels with the names used for constructing them in Python code
_iclass_label = ["metre", "kilogram", "second", "amp", "kelvin", "mole", "candle"]


class Dimension:
  """
  Stores the indices of the 7 basic SI unit dimension (length, mass, etc.).

  Provides a subset of arithmetic operations appropriate to dimensions:
  multiplication, division and powers, and equality testing.

  Parameters
  ----------
  dims : sequence of `float`
      The dimension indices of the 7 basic SI unit dimensions.

  Notes
  -----
  Users shouldn't use this class directly, it is used internally in Quantity
  and Unit. Even internally, never use ``Dimension(...)`` to create a new
  instance, use `get_or_create_dimension` instead. This function makes
  sure that only one Dimension instance exists for every combination of
  indices, allowing for a very fast dimensionality check with ``is``.
  """

  __slots__ = ["_dims"]

  __array_priority__ = 1000

  #### INITIALISATION ####

  def __init__(self, dims):
    self._dims = dims

  #### METHODS ####
  def get_dimension(self, d):
    """
    Return a specific dimension.

    Parameters
    ----------
    d : `str`
        A string identifying the SI basic unit dimension. Can be either a
        description like "length" or a basic unit like "m" or "metre".

    Returns
    -------
    dim : `float`
        The dimensionality of the dimension `d`.
    """
    return self._dims[_di[d]]

  @property
  def is_dimensionless(self):
    """
    Whether this Dimension is dimensionless.

    Notes
    -----
    Normally, instead one should check dimension for being identical to
    `DIMENSIONLESS`.
    """
    return all([x == 0 for x in self._dims])

  @property
  def dim(self):
    """
    Returns the `Dimension` object itself. This can be useful, because it
    allows to check for the dimension of an object by checking its ``dim``
    attribute -- this will return a `Dimension` object for `Quantity`,
    `Unit` and `Dimension`.
    """
    return self

  #### REPRESENTATION ####
  def _str_representation(self, python_code=False):
    """
    String representation in basic SI units, or ``"1"`` for dimensionless.
    Use ``python_code=False`` for display purposes and ``True`` for valid
    Python code.
    """

    if python_code:
      power_operator = " ** "
    else:
      power_operator = "^"

    parts = []
    for i in range(len(self._dims)):
      if self._dims[i]:
        if python_code:
          s = _iclass_label[i]
        else:
          s = _ilabel[i]
        if self._dims[i] != 1:
          s += power_operator + str(self._dims[i])
        parts.append(s)
    if python_code:
      s = " * ".join(parts)
      if not len(s):
        return f"{self.__class__.__name__}()"
    else:
      s = " ".join(parts)
      if not len(s):
        return "1"
    return s.strip()

  def __repr__(self):
    return self._str_representation(python_code=True)

  def __str__(self):
    return self._str_representation(python_code=False)

  #### ARITHMETIC ####
  # Note that none of the dimension arithmetic objects do sanity checking
  # on their inputs, although most will throw an exception if you pass the
  # wrong sort of input
  def __mul__(self, value):
    return get_or_create_dimension([x + y for x, y in zip(self._dims, value._dims)])

  def __div__(self, value):
    return get_or_create_dimension([x - y for x, y in zip(self._dims, value._dims)])

  def __truediv__(self, value):
    return self.__div__(value)

  def __pow__(self, value):
    value = np.array(value, copy=False)
    if value.size > 1:
      raise TypeError("Too many exponents")
    return get_or_create_dimension([x * value for x in self._dims])

  def __imul__(self, value):
    raise TypeError("Dimension object is immutable")

  def __idiv__(self, value):
    raise TypeError("Dimension object is immutable")

  def __itruediv__(self, value):
    raise TypeError("Dimension object is immutable")

  def __ipow__(self, value):
    raise TypeError("Dimension object is immutable")

  #### COMPARISON ####
  def __eq__(self, value):
    try:
      return np.allclose(self._dims, value._dims)
    except AttributeError:
      # Only compare equal to another Dimensions object
      return False

  def __ne__(self, value):
    return not self.__eq__(value)

  def __hash__(self):
    return hash(self._dims)

  #### MAKE DIMENSION PICKABLE ####
  def __getstate__(self):
    return self._dims

  def __setstate__(self, state):
    self._dims = state

  def __reduce__(self):
    # Make sure that unpickling Dimension objects does not bypass the singleton system
    return (get_or_create_dimension, (self._dims,))

  ### Dimension objects are singletons and deepcopy is therefore not necessary
  def __deepcopy__(self, memodict):
    return self

def get_or_create_dimension(*args, **kwds):
  """
  Create a new Dimension object or get a reference to an existing one.
  This function takes care of only creating new objects if they were not
  created before and otherwise returning a reference to an existing object.
  This allows to compare dimensions very efficiently using ``is``.

  Parameters
  ----------
  args : sequence of `float`
      A sequence with the indices of the 7 elements of an SI dimension.
  kwds : keyword arguments
      a sequence of ``keyword=value`` pairs where the keywords are the names of
      the SI dimensions, or the standard unit.

  Examples
  --------
  The following are all definitions of the dimensions of force

  >>> from brainpy.math.units import *
  >>> get_or_create_dimension(length=1, mass=1, time=-2)
  metre * kilogram * second ** -2
  >>> get_or_create_dimension(m=1, kg=1, s=-2)
  metre * kilogram * second ** -2
  >>> get_or_create_dimension([1, 1, -2, 0, 0, 0, 0])
  metre * kilogram * second ** -2

  Notes
  -----
  The 7 units are (in order):

  Length, Mass, Time, Electric Current, Temperature,
  Quantity of Substance, Luminosity

  and can be referred to either by these names or their SI unit names,
  e.g. length, metre, and m all refer to the same thing here.
  """
  if len(args):
    # initialisation by list
    dims = args[0]
    try:
      if len(dims) != 7:
        raise TypeError()
    except TypeError:
      raise TypeError("Need a sequence of exactly 7 items")
  else:
    # initialisation by keywords
    dims = [0, 0, 0, 0, 0, 0, 0]
    for k in kwds:
      # _di stores the index of the dimension with name 'k'
      dims[_di[k]] = kwds[k]

  dims = tuple(dims)

  # check whether this Dimension object has already been created
  if dims in _dimensions:
    return _dimensions[dims]
  else:
    new_dim = Dimension(dims)
    _dimensions[dims] = new_dim
    return new_dim

DIMENSIONLESS = Dimension((0, 0, 0, 0, 0, 0, 0))
_dimensions = {(0, 0, 0, 0, 0, 0, 0): DIMENSIONLESS}

class DimensionMismatchError(Exception):
  """
  Exception class for attempted operations with inconsistent dimensions.

  For example, ``3*mvolt + 2*amp`` raises this exception. The purpose of this
  class is to help catch errors based on incorrect units. The exception will
  print a representation of the dimensions of the two inconsistent objects
  that were operated on.

  Parameters
  ----------
  description : ``str``
      A description of the type of operation being performed, e.g. Addition,
      Multiplication, etc.
  dims : `Dimension`
      The physical dimensions of the objects involved in the operation, any
      number of them is possible
  """

  def __init__(self, description, *dims):
    # Call the base class constructor to make Exception pickable, see:
    # http://bugs.python.org/issue1692335
    Exception.__init__(self, description, *dims)
    self.dims = dims
    self.desc = description

  def __repr__(self):
    dims_repr = [repr(dim) for dim in self.dims]
    return f"{self.__class__.__name__}({self.desc!r}, {', '.join(dims_repr)})"

  def __str__(self):
    s = self.desc
    if len(self.dims) == 0:
      pass
    elif len(self.dims) == 1:
      s += f" (unit is {get_unit_for_display(self.dims[0])}"
    elif len(self.dims) == 2:
      d1, d2 = self.dims
      s += (
        f" (units are {get_unit_for_display(d1)} and {get_unit_for_display(d2)}"
      )
    else:
      s += (
        " (units are"
        f" {' '.join([f'({get_unit_for_display(d)})' for d in self.dims])}"
      )
    if len(self.dims):
      s += ")."
    return s


def get_dimensions(obj):
  """
  Return the dimensions of any object that has them.

  Slightly more general than `Quantity.dimensions` because it will
  return `DIMENSIONLESS` if the object is of number type but not a `Quantity`
  (e.g. a `float` or `int`).

  Parameters
  ----------
  obj : `object`
      The object to check.

  Returns
  -------
  dim : `Dimension`
      The physical dimensions of the `obj`.
  """
  try:
    return obj.unit
  except AttributeError:
    # The following is not very pretty, but it will avoid the costly
    # isinstance check for the common types
    if type(obj) in [
      int,
      float,
      np.int32,
      np.int64,
      np.float32,
      np.float64,
      np.ndarray,
    ] or isinstance(obj, (numbers.Number, jnp.number, jnp.ndarray)):
      return DIMENSIONLESS
    try:
      return Array(obj).unit
    except TypeError:
      raise TypeError(f"Object of type {type(obj)} does not have dimensions")

def fail_for_dimension_mismatch(
    obj1, obj2=None, error_message=None, **error_quantities
):
  """
  Compare the dimensions of two objects.

  Parameters
  ----------
  obj1, obj2 : {array-like, `Quantity`}
      The object to compare. If `obj2` is ``None``, assume it to be
      dimensionless
  error_message : str, optional
      An error message that is used in the DimensionMismatchError
  error_quantities : dict mapping str to `Quantity`, optional
      Quantities in this dictionary will be converted using the `_short_str`
      helper method and inserted into the ``error_message`` (which should
      have placeholders with the corresponding names). The reason for doing
      this in a somewhat complicated way instead of directly including all the
      details in ``error_messsage`` is that converting large quantity arrays
      to strings can be rather costly and we don't want to do it if no error
      occured.

  Returns
  -------
  dim1, dim2 : `Dimension`, `Dimension`
      The physical dimensions of the two arguments (so that later code does
      not need to get the dimensions again).

  Raises
  ------
  DimensionMismatchError
      If the dimensions of `obj1` and `obj2` do not match (or, if `obj2` is
      ``None``, in case `obj1` is not dimensionsless).

  Notes
  -----
  Implements special checking for ``0``, treating it as having "any
  dimensions".
  """
  if not unit_checking:
    return None, None

  dim1 = get_dimensions(obj1)
  if obj2 is None:
    dim2 = DIMENSIONLESS
  else:
    dim2 = get_dimensions(obj2)

  if dim1 is not dim2 and not (dim1 is None or dim2 is None):
    # Special treatment for "0":
    # if it is not a Quantity, it has "any dimension".
    # This allows expressions like 3*mV + 0 to pass (useful in cases where
    # zero is treated as the neutral element, e.g. in the Python sum
    # builtin) or comparisons like 3 * mV == 0 to return False instead of
    # failing # with a DimensionMismatchError. Note that 3*mV == 0*second
    # is not allowed, though.
    if (dim1 is DIMENSIONLESS and np.all(obj1 == 0)) or (
        dim2 is DIMENSIONLESS and np.all(obj2 == 0)
    ):
      return dim1, dim2

    # We do another check here, this should allow Brian1 units to pass as
    # having the same dimensions as a Brian2 unit
    if dim1 == dim2:
      return dim1, dim2

    if error_message is None:
      error_message = "Dimension mismatch"
    else:
      error_quantities = {
        name: _short_str(q) for name, q in error_quantities.items()
      }
      error_message = error_message.format(**error_quantities)
    # If we are comparing an object to a specific unit, we don't want to
    # restate this unit (it is probably mentioned in the text already)
    if obj2 is None or isinstance(obj2, (Dimension, Unit)):
      raise DimensionMismatchError(error_message, dim1)
    else:
      raise DimensionMismatchError(error_message, dim1, dim2)
  else:
    return dim1, dim2

def get_unit(obj):
  """
  Return the unit dimensions of the object.

  Parameters
  ----------
  obj : `object`
      The object to get the unit dimensions from.

  Returns
  -------
  dim : `Dimension`
      The unit dimensions of the object.
  """

  if isinstance(obj, Array):
    return obj.unit
  else:
    return DIMENSIONLESS


def array_with_units(floatval, units):
  """
  Create a new `Array` with the given dimensions. Calls
  `get_or_create_dimension` with the dimension tuple of the `dims`
  argument to make sure that unpickling (which calls this function) does not
  accidentally create new Dimension objects which should instead refer to
  existing ones.

  Parameters
  ----------
  floatval : `float`
      The floating point value of the quantity.
  units: `Dimension`
      The unit dimensions of the quantity.

  Returns
  -------
  array : `Array`
      The new `Array` object.

  Examples
  --------
  >>> from brainpy.math.units import *
  >>> array_with_units(0.001, volt.unit)
  1. * mvolt
  """
  return Array(floatval, get_or_create_dimension(units._dims))


@register_pytree_node_class
class Array(object):
  # value: jax.Array, np.ndarray, or number, custom type, pytree
  # unit: Unit, 1, None
  __slots__ = ('_value', '_unit')

  def __init__(self, value, dtype: Any = None, unit=DIMENSIONLESS, copy=False):
    # array value
    if isinstance(value, Array):
      value = value._value
    elif isinstance(value, (tuple, list, np.ndarray)):
      value = jnp.asarray(value, dtype=dtype, copy=copy)
    if dtype is not None:
      value = jnp.asarray(value, dtype=dtype, copy=copy)
    self._value = value

    # unit
    self._unit = unit

  @property
  def value(self):
    # return the value
    return self._value

  @value.setter
  def value(self, value):
    self_value = self._check_tracer()

    if isinstance(value, Array):
      value = value.value
    elif isinstance(value, np.ndarray):
      value = jnp.asarray(value)
    elif isinstance(value, jax.Array):
      pass
    else:
      value = jnp.asarray(value)
    # check
    if value.shape != self_value.shape:
      raise ValueError(f"The shape of the original data is {self_value.shape}, "
                       f"while we got {value.shape}.")
    if value.dtype != self_value.dtype:
      raise ValueError(f"The dtype of the original data is {self_value.dtype}, "
                       f"while we got {value.dtype}.")
    self._value = value

  @property
  def unit(self):
    """
    The physical unit dimensions of this Array
    """
    return self._unit

  @unit.setter
  def unit(self, unit):
    self._unit = unit

  ### UNITS ###

  @staticmethod
  def with_dimensions(value, *args, **keywords):
    """
    Create a `Array` object with dim.

    Parameters
    ----------
    value : {array_like, number}
        The value of the dimension
    args : {`Dimension`, sequence of float}
        Either a single argument (a `Dimension`) or a sequence of 7 values.
    keywords
        Keywords defining the dim, see `Dimension` for details.

    Returns
    -------
    q : `Quantity`
        A `Quantity` object with the given dim

    Examples
    --------
    All of these define an equivalent `Quantity` object:

    >>> from brainpy.math.units import *
    >>> Array.with_dimensions(2, get_or_create_dimension(length=1))
    2. * metre
    >>> Array.with_dimensions(2, length=1)
    2. * metre
    >>> 2 * metre
    2. * metre
    """
    if len(args) and isinstance(args[0], Dimension):
      dimensions = args[0]
    else:
      dimensions = get_or_create_dimension(*args, **keywords)
    return Array(value, unit=dimensions)

  is_dimensionless = property(
    lambda self: self._unit.is_dimensionless,
    doc="Wehther the array is dimensionless"
  )

  def has_same_unit(self, other):
    """
    Whether this Array has the same unit dimensions as another Array

    Parameters
    ----------
    other : Array
        The other Array to compare with

    Returns
    -------
    bool
        Whether the two Arrays have the same unit dimensions
    """
    other_unit = get_unit(other)
    return (self.unit is other_unit) or (self.unit == other_unit)

  def in_unit(self, u, precision=None, python_code=False):
    """
    Represent the Array in a given unit.

    Parameters
    ----------
    u : `Array`
        The unit in which to show the quantity.
    precision : `int`, optional
        The number of digits of precision (in the given unit)
        If no value is given, numpy's `get_printoptions` is used.
    python_code : `bool`, optional
        Whether to return a string that can be used as python code.
        If True, the string will be formatted as a python expression.
        If False, the string will be formatted as a human-readable string.
    Returns
    -------
    s : `str`
        The string representation of the Array in the given unit.

    Examples
    --------
    >>> x = 25.123456 * mV
    >>> x.in_unit(volt)
    '0.02512346 V'
    >>> x.in_unit(volt, 3)
    '0.025 V'
    >>> x.in_unit(mV, 3)
    '25.123 mV'
    """
    fail_for_dimension_mismatch(self.unit, u, 'Non-matching unit for method "in_unit"')

    value = jnp.array(self.value / u, copy=False)

    if value.shape == ():
      s = jnp.array_str(jnp.array([value]), precision=precision)
      s = s.replace("[", "").replace("]", "").strip()
    else:
      if python_code:
        s = jnp.array_repr(value, precision=precision)
      else:
        s = jnp.array_str(value, precision=precision)

    if not u.is_dimensionless:
      if isinstance(u, Unit):
        if python_code:
          s += f" * {repr(u)}"
        else:
          s += f" {str(u)}"
      else:
        if python_code:
          s += f" * {repr(u.unit)}"
        else:
          s += f" {str(u.unit)}"
    elif python_code:  # Make a quantity without unit recognisable
      return f"{self.__class__.__name__}({s.strip()})"
    return s.strip()

  def get_best_unit(self, *regs):
    """
    Return the best unit for this `Array`.

    Parameters
    ----------
    regs : any number of `UnitRegistry objects
        The registries that are searched for units. If none are provided, it
        will check the standard, user and additional unit registers in turn.

    Returns
    -------
    u : `Array` or `Unit`
        The best unit for this `Array`.
    """
    if self.is_dimensionless:
      return Unit(1)
    if len(regs):
      for r in regs:
        try:
          return r[self]
        except KeyError:
          pass
      return Array(1, self.dim)
    else:
      return self.get_best_unit(
        standard_unit_register, user_unit_register, additional_unit_register
      )

  def in_best_unit(self, precision=None, python_code=False, *regs):
    """
    Represent the quantity in the "best" unit.

    Parameters
    ----------
    precision : `int`, optional
        The number of digits of precision (in the best unit, see
        Examples). If no value is given, numpy's
        `get_printoptions` value is used.
    python_code : `bool`, optional
        Whether to return a string that can be used as python code.
        If True, the string will be formatted as a python expression.
        If False, the string will be formatted as a human-readable string.
    regs : `UnitRegistry` objects
        The registries where to search for units. If none are given, the
        standard, user-defined and additional registries are searched in
        that order.

    Returns
    -------
    representation : `str`
        A string representation of this `Array`.

    Examples
    --------
    >>> from brainpy.math.units import *
    >>> x = 0.00123456 * volt
    >>> x.in_best_unit()
    '1.23456 mV'
    >>> x.in_best_unit(3)
    '1.23 mV'
    """
    u = self.get_best_unit(*regs)
    return self.in_unit(u, precision, python_code)

  def _check_tracer(self):
    self_value = self.value
    if hasattr(self_value, '_trace') and hasattr(self_value._trace.main, 'jaxpr_stack'):
      if len(self_value._trace.main.jaxpr_stack) == 0:
        raise RuntimeError('This Array is modified during the transformation. '
                           'BrainPy only supports transformations for Variable. '
                           'Please declare it as a Variable.') from jax.core.escaped_tracer_error(self_value, None)
    return self_value

  @property
  def sharding(self):
    return self._value.sharding

  @property
  def addressable_shards(self):
    return self._value.addressable_shards

  def update(self, value):
    """Update the value of this Array.
    """
    self.value = value

  @property
  def dtype(self):
    """Variable dtype."""
    return _get_dtype(self._value)

  @property
  def shape(self):
    """Variable shape."""
    return self.value.shape

  @property
  def ndim(self):
    return self.value.ndim

  @property
  def imag(self):
    return _return(self.value.image)

  @property
  def real(self):
    return _return(self.value.real)

  @property
  def size(self):
    return self.value.size

  @property
  def T(self):
    return _return(self.value.T)

  # ----------------------- #
  # Python inherent methods #
  # ----------------------- #

  def __repr__(self) -> str:
    print_code = repr(self.value)
    if ', dtype' in print_code:
      print_code = print_code.split(', dtype')[0] + ')'
    prefix = f'{self.__class__.__name__}'
    prefix2 = f'{self.__class__.__name__}(value='
    if '\n' in print_code:
      lines = print_code.split("\n")
      blank1 = " " * len(prefix2)
      lines[0] = prefix2 + lines[0]
      for i in range(1, len(lines)):
        lines[i] = blank1 + lines[i]
      lines[-1] += ","
      blank2 = " " * (len(prefix) + 1)
      lines.append(f'{blank2}dtype={self.dtype})')
      print_code = "\n".join(lines)
    else:
      print_code = prefix2 + print_code + f', dtype={self.dtype})'
    return print_code

  def __format__(self, format_spec: str) -> str:
    return format(self.value)

  def __iter__(self):
    """Solve the issue of DeviceArray.__iter__.

    Details please see JAX issues:

    - https://github.com/google/jax/issues/7713
    - https://github.com/google/jax/pull/3821
    """
    for i in range(self.value.shape[0]):
      yield self.value[i]

  def __getitem__(self, index):
    if isinstance(index, slice) and (index == _all_slice):
      return Array(self.value, self.unit)
    elif isinstance(index, tuple):
      index = tuple((x.value if isinstance(x, Array) else x) for x in index)
    elif isinstance(index, Array):
      index = index.value
    return Array(self.value[index], self.unit)

  def __setitem__(self, index, value):
    fail_for_dimension_mismatch(self, value, "Inconsistent units in assignment")
    # value is Array
    if isinstance(value, Array):
      value = value.value
    # value is numpy.ndarray
    elif isinstance(value, np.ndarray):
      value = jnp.asarray(value)

    # index is a tuple
    if isinstance(index, tuple):
      index = tuple(_check_input_array(x) for x in index)
    # index is Array
    elif isinstance(index, Array):
      index = index.value
    # index is numpy.ndarray
    elif isinstance(index, np.ndarray):
      index = jnp.asarray(index)

    # update
    self_value = self._check_tracer()
    self.value = self_value.at[index].set(value)

  # ---------- #
  # operations #
  # ---------- #

  def _binary_operation(
      self,
      other,
      operation: Callable,
      unit_operation: Callable = lambda a, b: a,
      fail_for_mismatch: bool = False,
      operator_str: str = None,
      inplace: bool = False,
  ):
    """
    General implementation for binary operations.

    Parameters
    ----------
    other : {`Array`, `ndarray`, scalar}
        The object with which the operation should be performed.
    operation : function of two variables
        The function with which the two objects are combined. For example,
        `operator.mul` for a multiplication.
    unit_operation : function of two variables, optional
        The function with which the dimension of the resulting object is
        calculated (as a function of the dimensions of the two involved
        objects). For example, `operator.mul` for a multiplication. If not
        specified, the dimensions of `self` are used for the resulting
        object.
    fail_for_mismatch : bool, optional
        Whether to fail for a dimension mismatch between `self` and `other`
        (defaults to ``False``)
    operator_str : str, optional
        The string to use for the operator in an error message.
    inplace: bool, optional
        Whether to do the operation in-place (defaults to ``False``).
    """
    _check_input_array(other)
    other_unit = None

    if fail_for_mismatch:
      if inplace:
        message = (
            "Cannot calculate ... %s {value}, units do not match" % operator_str
        )
        _, other_unit = fail_for_dimension_mismatch(
          self, other, message, value=other
        )
      else:
        message = (
            "Cannot calculate {value1} %s {value2}, units do not match"
            % operator_str
        )
        _, other_unit = fail_for_dimension_mismatch(
          self, other, message, value1=self, value2=other
        )

    if other_unit is None:
      other_unit = get_dimensions(other)

    if inplace:
      if self.shape == ():
        self_value = Array(value=self.value, unit=self.unit, copy=True)
      else:
        self_value = self
      operation(self_value, other)
      self_value.unit = unit_operation(self.unit, other_unit)
      return self_value
    else:
      newdims = unit_operation(self.dim, other_unit)
      self_arr = np.array(self, copy=False)
      other_arr = np.array(other, copy=False)
      result = operation(self_arr, other_arr)
      return Array(result, newdims)

  def __len__(self) -> int:
    return len(self.value)

  def __neg__(self):
    return _return(self.value.__neg__())

  def __pos__(self):
    return _return(self.value.__pos__())

  def __abs__(self):
    return _return(self.value.__abs__())

  def __invert__(self):
    return _return(self.value.__invert__())

  def __eq__(self, oc):
    return _return(self.value == _check_input_array(oc))

  def __ne__(self, oc):
    return _return(self.value != _check_input_array(oc))

  def __lt__(self, oc):
    return _return(self.value < _check_input_array(oc))

  def __le__(self, oc):
    return _return(self.value <= _check_input_array(oc))

  def __gt__(self, oc):
    return _return(self.value > _check_input_array(oc))

  def __ge__(self, oc):
    return _return(self.value >= _check_input_array(oc))

  def __add__(self, oc):
    return _return(self.value + _check_input_array(oc))

  def __radd__(self, oc):
    return _return(self.value + _check_input_array(oc))

  def __iadd__(self, oc):
    # a += b
    self.value = self.value + _check_input_array(oc)
    return self

  def __sub__(self, oc):
    return _return(self.value - _check_input_array(oc))

  def __rsub__(self, oc):
    return _return(_check_input_array(oc) - self.value)

  def __isub__(self, oc):
    # a -= b
    self.value = self.value - _check_input_array(oc)
    return self

  def __mul__(self, oc):
    return _return(self._binary_operation(oc, operator.mul, operator.mul))

  def __rmul__(self, oc):
    return self.__mul__(oc)

  def __imul__(self, oc):
    # a *= b
    self.value = self.value * _check_input_array(oc)
    return self

  def __rdiv__(self, oc):
    return _return(_check_input_array(oc) / self.value)

  def __truediv__(self, oc):
    return _return(self.value / _check_input_array(oc))

  def __rtruediv__(self, oc):
    return _return(_check_input_array(oc) / self.value)

  def __itruediv__(self, oc):
    # a /= b
    self.value = self.value / _check_input_array(oc)
    return self

  def __floordiv__(self, oc):
    return _return(self.value // _check_input_array(oc))

  def __rfloordiv__(self, oc):
    return _return(_check_input_array(oc) // self.value)

  def __ifloordiv__(self, oc):
    # a //= b
    self.value = self.value // _check_input_array(oc)
    return self

  def __divmod__(self, oc):
    return _return(self.value.__divmod__(_check_input_array(oc)))

  def __rdivmod__(self, oc):
    return _return(self.value.__rdivmod__(_check_input_array(oc)))

  def __mod__(self, oc):
    return _return(self.value % _check_input_array(oc))

  def __rmod__(self, oc):
    return _return(_check_input_array(oc) % self.value)

  def __imod__(self, oc):
    # a %= b
    self.value = self.value % _check_input_array(oc)
    return self

  def __pow__(self, oc):
    return _return(self.value ** _check_input_array(oc))

  def __rpow__(self, oc):
    return _return(_check_input_array(oc) ** self.value)

  def __ipow__(self, oc):
    # a **= b
    self.value = self.value ** _check_input_array(oc)
    return self

  def __matmul__(self, oc):
    return _return(self.value @ _check_input_array(oc))

  def __rmatmul__(self, oc):
    return _return(_check_input_array(oc) @ self.value)

  def __imatmul__(self, oc):
    # a @= b
    self.value = self.value @ _check_input_array(oc)
    return self

  def __and__(self, oc):
    return _return(self.value & _check_input_array(oc))

  def __rand__(self, oc):
    return _return(_check_input_array(oc) & self.value)

  def __iand__(self, oc):
    # a &= b
    self.value = self.value & _check_input_array(oc)
    return self

  def __or__(self, oc):
    return _return(self.value | _check_input_array(oc))

  def __ror__(self, oc):
    return _return(_check_input_array(oc) | self.value)

  def __ior__(self, oc):
    # a |= b
    self.value = self.value | _check_input_array(oc)
    return self

  def __xor__(self, oc):
    return _return(self.value ^ _check_input_array(oc))

  def __rxor__(self, oc):
    return _return(_check_input_array(oc) ^ self.value)

  def __ixor__(self, oc):
    # a ^= b
    self.value = self.value ^ _check_input_array(oc)
    return self

  def __lshift__(self, oc):
    return _return(self.value << _check_input_array(oc))

  def __rlshift__(self, oc):
    return _return(_check_input_array(oc) << self.value)

  def __ilshift__(self, oc):
    # a <<= b
    self.value = self.value << _check_input_array(oc)
    return self

  def __rshift__(self, oc):
    return _return(self.value >> _check_input_array(oc))

  def __rrshift__(self, oc):
    return _return(_check_input_array(oc) >> self.value)

  def __irshift__(self, oc):
    # a >>= b
    self.value = self.value >> _check_input_array(oc)
    return self

  def __round__(self, ndigits=None):
    return _return(self.value.__round__(ndigits))

  # ----------------------- #
  #       JAX methods       #
  # ----------------------- #

  @property
  def at(self):
    return self.value.at

  def block_host_until_ready(self, *args):
    return self.value.block_host_until_ready(*args)

  def block_until_ready(self, *args):
    return self.value.block_until_ready(*args)

  def device(self):
    return self.value.device()

  @property
  def device_buffer(self):
    return self.value.device_buffer

  # ----------------------- #
  #      NumPy methods      #
  # ----------------------- #

  def all(self, axis=None, keepdims=False):
    """Returns True if all elements evaluate to True."""
    r = self.value.all(axis=axis, keepdims=keepdims)
    return _return(r)

  def any(self, axis=None, keepdims=False):
    """Returns True if any of the elements of a evaluate to True."""
    r = self.value.any(axis=axis, keepdims=keepdims)
    return _return(r)

  def argmax(self, axis=None):
    """Return indices of the maximum values along the given axis."""
    return _return(self.value.argmax(axis=axis))

  def argmin(self, axis=None):
    """Return indices of the minimum values along the given axis."""
    return _return(self.value.argmin(axis=axis))

  def argpartition(self, kth, axis=-1, kind='introselect', order=None):
    """Returns the indices that would partition this array."""
    return _return(self.value.argpartition(kth=kth, axis=axis, kind=kind, order=order))

  def argsort(self, axis=-1, kind=None, order=None):
    """Returns the indices that would sort this array."""
    return _return(self.value.argsort(axis=axis, kind=kind, order=order))

  def astype(self, dtype):
    """Copy of the array, cast to a specified type.

    Parameters
    ----------
    dtype: str, dtype
      Typecode or data-type to which the array is cast.
    """
    if dtype is None:
      return _return(self.value)
    else:
      return _return(self.value.astype(dtype))

  def byteswap(self, inplace=False):
    """Swap the bytes of the array elements

    Toggle between low-endian and big-endian data representation by
    returning a byteswapped array, optionally swapped in-place.
    Arrays of byte-strings are not swapped. The real and imaginary
    parts of a complex number are swapped individually."""
    return _return(self.value.byteswap(inplace=inplace))

  def choose(self, choices, mode='raise'):
    """Use an index array to construct a new array from a set of choices."""
    return _return(self.value.choose(choices=_as_jax_array_(choices), mode=mode))

  def clip(self, min=None, max=None, out=None, ):
    """Return an array whose values are limited to [min, max]. One of max or min must be given."""
    min = _as_jax_array_(min)
    max = _as_jax_array_(max)
    r = self.value.clip(min=min, max=max)
    if out is None:
      return _return(r)
    else:
      _check_out(out)
      out.value = r

  def compress(self, condition, axis=None):
    """Return selected slices of this array along given axis."""
    return _return(self.value.compress(condition=_as_jax_array_(condition), axis=axis))

  def conj(self):
    """Complex-conjugate all elements."""
    return _return(self.value.conj())

  def conjugate(self):
    """Return the complex conjugate, element-wise."""
    return _return(self.value.conjugate())

  def copy(self):
    """Return a copy of the array."""
    return _return(self.value.copy())

  def cumprod(self, axis=None, dtype=None):
    """Return the cumulative product of the elements along the given axis."""
    return _return(self.value.cumprod(axis=axis, dtype=dtype))

  def cumsum(self, axis=None, dtype=None):
    """Return the cumulative sum of the elements along the given axis."""
    return _return(self.value.cumsum(axis=axis, dtype=dtype))

  def diagonal(self, offset=0, axis1=0, axis2=1):
    """Return specified diagonals."""
    return _return(self.value.diagonal(offset=offset, axis1=axis1, axis2=axis2))

  def dot(self, b):
    """Dot product of two arrays."""
    return _return(self.value.dot(_as_jax_array_(b)))

  def fill(self, value):
    """Fill the array with a scalar value."""
    self.value = jnp.ones_like(self.value) * value

  def flatten(self):
    return _return(self.value.flatten())

  def item(self, *args):
    """Copy an element of an array to a standard Python scalar and return it."""
    return Array(self.value.item(*args), self.unit)

  def max(self, axis=None, keepdims=False, *args, **kwargs):
    """Return the maximum along a given axis."""
    res = self.value.max(axis=axis, keepdims=keepdims, *args, **kwargs)
    return _return(res)

  def mean(self, axis=None, dtype=None, keepdims=False, *args, **kwargs):
    """Returns the average of the array elements along given axis."""
    res = self.value.mean(axis=axis, dtype=dtype, keepdims=keepdims, *args, **kwargs)
    return _return(res)

  def min(self, axis=None, keepdims=False, *args, **kwargs):
    """Return the minimum along a given axis."""
    res = self.value.min(axis=axis, keepdims=keepdims, *args, **kwargs)
    return _return(res)

  def nonzero(self):
    """Return the indices of the elements that are non-zero."""
    return tuple(_return(a) for a in self.value.nonzero())

  def prod(self, axis=None, dtype=None, keepdims=False, initial=1, where=True):
    """Return the product of the array elements over the given axis."""
    res = self.value.prod(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
    return _return(res)

  def ptp(self, axis=None, keepdims=False):
    """Peak to peak (maximum - minimum) value along a given axis."""
    r = self.value.ptp(axis=axis, keepdims=keepdims)
    return _return(r)

  def put(self, indices, values):
    """Replaces specified elements of an array with given values.

    Parameters
    ----------
    indices: array_like
      Target indices, interpreted as integers.
    values: array_like
      Values to place in the array at target indices.
    """
    self.__setitem__(indices, values)

  def ravel(self, order=None):
    """Return a flattened array."""
    return _return(self.value.ravel(order=order))

  def repeat(self, repeats, axis=None):
    """Repeat elements of an array."""
    return _return(self.value.repeat(repeats=repeats, axis=axis))

  def reshape(self, *shape, order='C'):
    """Returns an array containing the same data with a new shape."""
    return _return(self.value.reshape(*shape, order=order))

  def resize(self, new_shape):
    """Change shape and size of array in-place."""
    self.value = self.value.reshape(new_shape)

  def round(self, decimals=0):
    """Return ``a`` with each element rounded to the given number of decimals."""
    return _return(self.value.round(decimals=decimals))

  def searchsorted(self, v, side='left', sorter=None):
    """Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array `a` such that, if the
    corresponding elements in `v` were inserted before the indices, the
    order of `a` would be preserved.

    Assuming that `a` is sorted:

    ======  ============================
    `side`  returned index `i` satisfies
    ======  ============================
    left    ``a[i-1] < v <= a[i]``
    right   ``a[i-1] <= v < a[i]``
    ======  ============================

    Parameters
    ----------
    v : array_like
        Values to insert into `a`.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.  If there is no suitable
        index, return either 0 or N (where N is the length of `a`).
    sorter : 1-D array_like, optional
        Optional array of integer indices that sort array a into ascending
        order. They are typically the result of argsort.

    Returns
    -------
    indices : array of ints
        Array of insertion points with the same shape as `v`.
    """
    return _return(self.value.searchsorted(v=_as_jax_array_(v), side=side, sorter=sorter))

  def sort(self, axis=-1, kind='quicksort', order=None):
    """Sort an array in-place.

    Parameters
    ----------
    axis : int, optional
        Axis along which to sort. Default is -1, which means sort along the
        last axis.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}
        Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
        and 'mergesort' use timsort under the covers and, in general, the
        actual implementation will vary with datatype. The 'mergesort' option
        is retained for backwards compatibility.
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  A single field can
        be specified as a string, and not all fields need be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.
    """
    self.value = self.value.sort(axis=axis, kind=kind, order=order)

  def squeeze(self, axis=None):
    """Remove axes of length one from ``a``."""
    return _return(self.value.squeeze(axis=axis))

  def std(self, axis=None, dtype=None, ddof=0, keepdims=False):
    """Compute the standard deviation along the specified axis.

    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The
        default is to compute the standard deviation of the flattened array.
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it is
        the same as the array type.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `std` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard deviation,
        otherwise return a reference to the output array.
    """
    r = self.value.std(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    return _return(r)

  def sum(self, axis=None, dtype=None, keepdims=False, initial=0, where=True):
    """Return the sum of the array elements over the given axis."""
    res = self.value.sum(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
    return _return(res)

  def swapaxes(self, axis1, axis2):
    """Return a view of the array with `axis1` and `axis2` interchanged."""
    return _return(self.value.swapaxes(axis1, axis2))

  def split(self, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays as views into ``ary``.

    Parameters
    ----------
    indices_or_sections : int, 1-D array
      If `indices_or_sections` is an integer, N, the array will be divided
      into N equal arrays along `axis`.  If such a split is not possible,
      an error is raised.

      If `indices_or_sections` is a 1-D array of sorted integers, the entries
      indicate where along `axis` the array is split.  For example,
      ``[2, 3]`` would, for ``axis=0``, result in

        - ary[:2]
        - ary[2:3]
        - ary[3:]

      If an index exceeds the dimension of the array along `axis`,
      an empty sub-array is returned correspondingly.
    axis : int, optional
      The axis along which to split, default is 0.

    Returns
    -------
    sub-arrays : list of ndarrays
      A list of sub-arrays as views into `ary`.
    """
    return [_return(a) for a in jnp.split(self.value, indices_or_sections, axis=axis)]

  def take(self, indices, axis=None, mode=None):
    """Return an array formed from the elements of a at the given indices."""
    return _return(self.value.take(indices=_as_jax_array_(indices), axis=axis, mode=mode))

  def tobytes(self):
    """Construct Python bytes containing the raw data bytes in the array.

    Constructs Python bytes showing a copy of the raw contents of data memory.
    The bytes object is produced in C-order by default. This behavior is
    controlled by the ``order`` parameter."""
    return self.value.tobytes()

  def tolist(self):
    """Return the array as an ``a.ndim``-levels deep nested list of Python scalars.

    Return a copy of the array data as a (nested) Python list.
    Data items are converted to the nearest compatible builtin Python type, via
    the `~numpy.ndarray.item` function.

    If ``a.ndim`` is 0, then since the depth of the nested list is 0, it will
    not be a list at all, but a simple Python scalar.
    """
    return self.value.tolist()

  def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
    """Return the sum along diagonals of the array."""
    return _return(self.value.trace(offset=offset, axis1=axis1, axis2=axis2, dtype=dtype))

  def transpose(self, *axes):
    """Returns a view of the array with axes transposed.

    For a 1-D array this has no effect, as a transposed vector is simply the
    same vector. To convert a 1-D array into a 2D column vector, an additional
    dimension must be added. `np.atleast2d(a).T` achieves this, as does
    `a[:, np.newaxis]`.
    For a 2-D array, this is a standard matrix transpose.
    For an n-D array, if axes are given, their order indicates how the
    axes are permuted (see Examples). If axes are not provided and
    ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
    ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

    Parameters
    ----------
    axes : None, tuple of ints, or `n` ints

     * None or no argument: reverses the order of the axes.

     * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
       `i`-th axis becomes `a.transpose()`'s `j`-th axis.

     * `n` ints: same as an n-tuple of the same ints (this form is
       intended simply as a "convenience" alternative to the tuple form)

    Returns
    -------
    out : ndarray
        View of `a`, with axes suitably permuted.
    """
    return _return(self.value.transpose(*axes))

  def tile(self, reps):
    """Construct an array by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Note : Although tile may be used for broadcasting, it is strongly
    recommended to use numpy's broadcasting operations and functions.

    Parameters
    ----------
    reps : array_like
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.
    """
    return _return(self.value.tile(_as_jax_array_(reps)))

  def var(self, axis=None, dtype=None, ddof=0, keepdims=False):
    """Returns the variance of the array elements, along given axis."""
    r = self.value.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    return _return(r)

  def view(self, *args, dtype=None):
    r"""New view of array with the same data.

    This function is compatible with pytorch syntax.

    Returns a new tensor with the same data as the :attr:`self` tensor but of a
    different :attr:`shape`.

    The returned tensor shares the same data and must have the same number
    of elements, but may have a different size. For a tensor to be viewed, the new
    view size must be compatible with its original size and stride, i.e., each new
    view dimension must either be a subspace of an original dimension, or only span
    across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
    contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

    .. math::

      \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

    Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
    without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
    :meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
    returns a view if the shapes are compatible, and copies (equivalent to calling
    :meth:`contiguous`) otherwise.

    Args:
        shape (int...): the desired size

    Example::

        >>> x = brainpy.math.random.randn(4, 4)
        >>> x.size
       [4, 4]
        >>> y = x.view(16)
        >>> y.size
        [16]
        >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
        >>> z.size
        [2, 8]

        >>> a = brainpy.math.random.randn(1, 2, 3, 4)
        >>> a.size
        [1, 2, 3, 4]
        >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
        >>> b.size
        [1, 3, 2, 4]
        >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
        >>> c.size
        [1, 3, 2, 4]
        >>> brainpy.math.equal(b, c)
        False


    .. method:: view(dtype) -> Tensor
       :noindex:

    Returns a new tensor with the same data as the :attr:`self` tensor but of a
    different :attr:`dtype`.

    If the element size of :attr:`dtype` is different than that of ``self.dtype``,
    then the size of the last dimension of the output will be scaled
    proportionally.  For instance, if :attr:`dtype` element size is twice that of
    ``self.dtype``, then each pair of elements in the last dimension of
    :attr:`self` will be combined, and the size of the last dimension of the output
    will be half that of :attr:`self`. If :attr:`dtype` element size is half that
    of ``self.dtype``, then each element in the last dimension of :attr:`self` will
    be split in two, and the size of the last dimension of the output will be
    double that of :attr:`self`. For this to be possible, the following conditions
    must be true:

        * ``self.dim()`` must be greater than 0.
        * ``self.stride(-1)`` must be 1.

    Additionally, if the element size of :attr:`dtype` is greater than that of
    ``self.dtype``, the following conditions must be true as well:

        * ``self.size(-1)`` must be divisible by the ratio between the element
          sizes of the dtypes.
        * ``self.storage_offset()`` must be divisible by the ratio between the
          element sizes of the dtypes.
        * The strides of all dimensions, except the last dimension, must be
          divisible by the ratio between the element sizes of the dtypes.

    If any of the above conditions are not met, an error is thrown.


    Args:
        dtype (:class:`dtype`): the desired dtype

    Example::

        >>> x = brainpy.math.random.randn(4, 4)
        >>> x
        Array([[ 0.9482, -0.0310,  1.4999, -0.5316],
                [-0.1520,  0.7472,  0.5617, -0.8649],
                [-2.4724, -0.0334, -0.2976, -0.8499],
                [-0.2109,  1.9913, -0.9607, -0.6123]])
        >>> x.dtype
        brainpy.math.float32

        >>> y = x.view(brainpy.math.int32)
        >>> y
        tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
                [-1105482831,  1061112040,  1057999968, -1084397505],
                [-1071760287, -1123489973, -1097310419, -1084649136],
                [-1101533110,  1073668768, -1082790149, -1088634448]],
            dtype=brainpy.math.int32)
        >>> y[0, 0] = 1000000000
        >>> x
        tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
                [-0.1520,  0.7472,  0.5617, -0.8649],
                [-2.4724, -0.0334, -0.2976, -0.8499],
                [-0.2109,  1.9913, -0.9607, -0.6123]])

        >>> x.view(brainpy.math.cfloat)
        tensor([[ 0.0047-0.0310j,  1.4999-0.5316j],
                [-0.1520+0.7472j,  0.5617-0.8649j],
                [-2.4724-0.0334j, -0.2976-0.8499j],
                [-0.2109+1.9913j, -0.9607-0.6123j]])
        >>> x.view(brainpy.math.cfloat).size
        [4, 2]

        >>> x.view(brainpy.math.uint8)
        tensor([[  0, 202, 154,  59, 182, 243, 253, 188, 185, 252, 191,  63, 240,  22,
                   8, 191],
                [227, 165,  27, 190, 128,  72,  63,  63, 146, 203,  15,  63,  22, 106,
                  93, 191],
                [205,  59,  30, 192, 112, 206,   8, 189,   7,  95, 152, 190,  12, 147,
                  89, 191],
                [ 43, 246,  87, 190, 235, 226, 254,  63, 111, 240, 117, 191, 177, 191,
                  28, 191]], dtype=brainpy.math.uint8)
        >>> x.view(brainpy.math.uint8).size
        [4, 16]

    """
    if len(args) == 0:
      if dtype is None:
        raise ValueError('Provide dtype or shape.')
      else:
        return _return(self.value.view(dtype))
    else:
      if isinstance(args[0], int):  # shape
        if dtype is not None:
          raise ValueError('Provide one of dtype or shape. Not both.')
        return _return(self.value.reshape(*args))
      else:  # dtype
        assert not isinstance(args[0], int)
        assert dtype is None
        return _return(self.value.view(args[0]))

  # ------------------
  # NumPy support
  # ------------------

  def numpy(self, dtype=None):
    """Convert to numpy.ndarray."""
    # warnings.warn('Deprecated since 2.1.12. Please use ".to_numpy()" instead.', DeprecationWarning)
    return np.asarray(self.value, dtype=dtype)

  def to_numpy(self, dtype=None):
    """Convert to numpy.ndarray."""
    return np.asarray(self.value, dtype=dtype)

  def to_jax(self, dtype=None):
    """Convert to jax.numpy.ndarray."""
    if dtype is None:
      return self.value
    else:
      return jnp.asarray(self.value, dtype=dtype)

  def __array__(self, dtype=None):
    """Support ``numpy.array()`` and ``numpy.asarray()`` functions."""
    return np.asarray(self.value, dtype=dtype)

  def __jax_array__(self):
    return self.value

  def as_variable(self):
    """As an instance of Variable."""
    global bm
    if bm is None: from brainpy import math as bm
    return bm.Variable(self)

  def __format__(self, specification):
    return self.value.__format__(specification)

  def __bool__(self) -> bool:
    return self.value.__bool__()

  def __float__(self):
    return self.value.__float__()

  def __int__(self):
    return self.value.__int__()

  def __complex__(self):
    return self.value.__complex__()

  def __hex__(self):
    assert self.ndim == 0, 'hex only works on scalar values'
    return hex(self.value)  # type: ignore

  def __oct__(self):
    assert self.ndim == 0, 'oct only works on scalar values'
    return oct(self.value)  # type: ignore

  def __index__(self):
    return operator.index(self.value)

  def __dlpack__(self):
    from jax.dlpack import to_dlpack  # pylint: disable=g-import-not-at-top
    return to_dlpack(self.value)

  # ----------------------
  # PyTorch compatibility
  # ----------------------

  def unsqueeze(self, dim: int) -> 'Array':
    """
    Array.unsqueeze(dim) -> Array, or so called Tensor
    equals
    Array.expand_dims(dim)

    See :func:`brainpy.math.unsqueeze`
    """
    return _return(jnp.expand_dims(self.value, dim))

  def expand_dims(self, axis: Union[int, Sequence[int]]) -> 'Array':
    """
    self.expand_dims(axis: int|Sequence[int])

    1. 如果axis类型为int：
    返回一个在self基础上的第axis维度前插入一个维度Array，
    axis<0表示倒数第|axis|维度，
    令n=len(self._value.shape)，则axis的范围为[-(n+1),n]

    2. 如果axis类型为Sequence[int]：
    则返回依次扩展axis[i]的结果，
    即self.expand_dims(axis)==self.expand_dims(axis[0]).expand_dims(axis[1])...expand_dims(axis[len(axis)-1])


    1. If the type of axis is int:

    Returns an Array of dimensions inserted before the axis dimension based on self,

    The first | axis < 0 indicates the bottom axis | dimensions,

    Set n=len(self._value.shape), then axis has the range [-(n+1),n]


    2. If the type of axis is Sequence[int] :

    Returns the result of extending axis[i] in sequence,

    self.expand_dims(axis)==self.expand_dims(axis[0]).expand_dims(axis[1])... expand_dims(axis[len(axis)-1])

    """
    return _return(jnp.expand_dims(self.value, axis))

  def expand_as(self, array: Union['Array', jax.Array, np.ndarray]) -> 'Array':
    """
    Expand an array to a shape of another array.

    Parameters
    ----------
    array : Array

    Returns
    -------
    expanded : Array
        A readonly view on the original array with the given shape of array. It is
        typically not contiguous. Furthermore, more than one element of a
        expanded array may refer to a single memory location.
    """
    return _return(jnp.broadcast_to(self.value, array))

  def pow(self, index: int):
    return _return(self.value ** index)

  def addr(
      self,
      vec1: Union['Array', jax.Array, np.ndarray],
      vec2: Union['Array', jax.Array, np.ndarray],
      *,
      beta: float = 1.0,
      alpha: float = 1.0,
      out: Optional[Union['Array', jax.Array, np.ndarray]] = None
  ) -> Optional['Array']:
    r"""Performs the outer-product of vectors ``vec1`` and ``vec2`` and adds it to the matrix ``input``.

    Optional values beta and alpha are scaling factors on the outer product
    between vec1 and vec2 and the added matrix input respectively.

    .. math::

       out = \beta \mathrm{input} + \alpha (\text{vec1} \bigtimes \text{vec2})

    Args:
      vec1: the first vector of the outer product
      vec2: the second vector of the outer product
      beta: multiplier for input
      alpha: multiplier
      out: the output tensor.

    """
    vec1 = _as_jax_array_(vec1)
    vec2 = _as_jax_array_(vec2)
    r = alpha * jnp.outer(vec1, vec2) + beta * self.value
    if out is None:
      return _return(r)
    else:
      _check_out(out)
      out.value = r

  def addr_(
      self,
      vec1: Union['Array', jax.Array, np.ndarray],
      vec2: Union['Array', jax.Array, np.ndarray],
      *,
      beta: float = 1.0,
      alpha: float = 1.0
  ):
    vec1 = _as_jax_array_(vec1)
    vec2 = _as_jax_array_(vec2)
    r = alpha * jnp.outer(vec1, vec2) + beta * self.value
    self.value = r
    return self

  def outer(self, other: Union['Array', jax.Array, np.ndarray]) -> 'Array':
    other = _as_jax_array_(other)
    return _return(jnp.outer(self.value, other.value))

  def abs(self, *, out: Optional[Union['Array', jax.Array, np.ndarray]] = None) -> Optional['Array']:
    r = jnp.abs(self.value)
    if out is None:
      return _return(r)
    else:
      _check_out(out)
      out.value = r

  def abs_(self):
    """
    in-place version of Array.abs()
    """
    self.value = jnp.abs(self.value)
    return self

  def add_(self, value):
    self.value += value
    return self

  def absolute(self, *, out: Optional[Union['Array', jax.Array, np.ndarray]] = None) -> Optional['Array']:
    """
    alias of Array.abs
    """
    return self.abs(out=out)

  def absolute_(self):
    """
    alias of Array.abs_()
    """
    return self.abs_()

  def mul(self, value):
    return _return(self.value * value)

  def mul_(self, value):
    """
    In-place version of :meth:`~Array.mul`.
    """
    self.value *= value
    return self

  def multiply(self, value):  # real signature unknown; restored from __doc__
    """
    multiply(value) -> Tensor

    See :func:`torch.multiply`.
    """
    return self.value * value

  def multiply_(self, value):  # real signature unknown; restored from __doc__
    """
    multiply_(value) -> Tensor

    In-place version of :meth:`~Tensor.multiply`.
    """
    self.value *= value
    return self

  def sin(self, *, out: Optional[Union['Array', jax.Array, np.ndarray]] = None) -> Optional['Array']:
    r = jnp.sin(self.value)
    if out is None:
      return _return(r)
    else:
      _check_out(out)
      out.value = r

  def sin_(self):
    self.value = jnp.sin(self.value)
    return self

  def cos_(self):
    self.value = jnp.cos(self.value)
    return self

  def cos(self, *, out: Optional[Union['Array', jax.Array, np.ndarray]] = None) -> Optional['Array']:
    r = jnp.cos(self.value)
    if out is None:
      return _return(r)
    else:
      _check_out(out)
      out.value = r

  def tan_(self):
    self.value = jnp.tan(self.value)
    return self

  def tan(self, *, out: Optional[Union['Array', jax.Array, np.ndarray]] = None) -> Optional['Array']:
    r = jnp.tan(self.value)
    if out is None:
      return _return(r)
    else:
      _check_out(out)
      out.value = r

  def sinh_(self):
    self.value = jnp.tanh(self.value)
    return self

  def sinh(self, *, out: Optional[Union['Array', jax.Array, np.ndarray]] = None) -> Optional['Array']:
    r = jnp.tanh(self.value)
    if out is None:
      return _return(r)
    else:
      _check_out(out)
      out.value = r

  def cosh_(self):
    self.value = jnp.cosh(self.value)
    return self

  def cosh(self, *, out: Optional[Union['Array', jax.Array, np.ndarray]] = None) -> Optional['Array']:
    r = jnp.cosh(self.value)
    if out is None:
      return _return(r)
    else:
      _check_out(out)
      out.value = r

  def tanh_(self):
    self.value = jnp.tanh(self.value)
    return self

  def tanh(self, *, out: Optional[Union['Array', jax.Array, np.ndarray]] = None) -> Optional['Array']:
    r = jnp.tanh(self.value)
    if out is None:
      return _return(r)
    else:
      _check_out(out)
      out.value = r

  def arcsin_(self):
    self.value = jnp.arcsin(self.value)
    return self

  def arcsin(self, *, out: Optional[Union['Array', jax.Array, np.ndarray]] = None) -> Optional['Array']:
    r = jnp.arcsin(self.value)
    if out is None:
      return _return(r)
    else:
      _check_out(out)
      out.value = r

  def arccos_(self):
    self.value = jnp.arccos(self.value)
    return self

  def arccos(self, *, out: Optional[Union['Array', jax.Array, np.ndarray]] = None) -> Optional['Array']:
    r = jnp.arccos(self.value)
    if out is None:
      return _return(r)
    else:
      _check_out(out)
      out.value = r

  def arctan_(self):
    self.value = jnp.arctan(self.value)
    return self

  def arctan(self, *, out: Optional[Union['Array', jax.Array, np.ndarray]] = None) -> Optional['Array']:
    r = jnp.arctan(self.value)
    if out is None:
      return _return(r)
    else:
      _check_out(out)
      out.value = r

  def clamp(
      self,
      min_value: Optional[Union['Array', jax.Array, np.ndarray]] = None,
      max_value: Optional[Union['Array', jax.Array, np.ndarray]] = None,
      *,
      out: Optional[Union['Array', jax.Array, np.ndarray]] = None
  ) -> Optional['Array']:
    """
    return the value between min_value and max_value,
    if min_value is None, then no lower bound,
    if max_value is None, then no upper bound.
    """
    min_value = _as_jax_array_(min_value)
    max_value = _as_jax_array_(max_value)
    r = jnp.clip(self.value, max_value, max_value)
    if out is None:
      return _return(r)
    else:
      _check_out(out)
      out.value = r

  def clamp_(self,
             min_value: Optional[Union['Array', jax.Array, np.ndarray]] = None,
             max_value: Optional[Union['Array', jax.Array, np.ndarray]] = None):
    """
    return the value between min_value and max_value,
    if min_value is None, then no lower bound,
    if max_value is None, then no upper bound.
    """
    self.clamp(min_value, max_value, out=self)
    return self

  def clip_(self,
            min_value: Optional[Union['Array', jax.Array, np.ndarray]] = None,
            max_value: Optional[Union['Array', jax.Array, np.ndarray]] = None):
    """
    alias for clamp_
    """
    self.value = self.clip(min_value, max_value, out=self)
    return self

  def clone(self) -> 'Array':
    return _return(self.value.copy())

  def copy_(self, src: Union['Array', jax.Array, np.ndarray]) -> 'Array':
    self.value = jnp.copy(_as_jax_array_(src))
    return self

  def cov_with(
      self,
      y: Optional[Union['Array', jax.Array, np.ndarray]] = None,
      rowvar: bool = True,
      bias: bool = False,
      ddof: Optional[int] = None,
      fweights: Union['Array', jax.Array, np.ndarray] = None,
      aweights: Union['Array', jax.Array, np.ndarray] = None
  ) -> 'Array':
    y = _as_jax_array_(y)
    fweights = _as_jax_array_(fweights)
    aweights = _as_jax_array_(aweights)
    r = jnp.cov(self.value, y, rowvar, bias, fweights, aweights)
    return _return(r)

  def expand(self, *sizes) -> 'Array':
    """
    Expand an array to a new shape.

    Parameters
    ----------
    sizes : tuple or int
        The shape of the desired array. A single integer ``i`` is interpreted
        as ``(i,)``.

    Returns
    -------
    expanded : Array
        A readonly view on the original array with the given shape. It is
        typically not contiguous. Furthermore, more than one element of a
        expanded array may refer to a single memory location.
    """
    l_ori = len(self.shape)
    l_tar = len(sizes)
    base = l_tar - l_ori
    sizes_list = list(sizes)
    if base < 0:
      raise ValueError(f'the number of sizes provided ({len(sizes)}) must be greater or equal to the number of '
                       f'dimensions in the tensor ({len(self.shape)})')
    for i, v in enumerate(sizes[:base]):
      if v < 0:
        raise ValueError(
          f'The expanded size of the tensor ({v}) isn\'t allowed in a leading, non-existing dimension {i + 1}')
    for i, v in enumerate(self.shape):
      sizes_list[base + i] = v if sizes_list[base + i] == -1 else sizes_list[base + i]
      if v != 1 and sizes_list[base + i] != v:
        raise ValueError(
          f'The expanded size of the tensor ({sizes_list[base + i]}) must match the existing size ({v}) at non-singleton '
          f'dimension {i}.  Target sizes: {sizes}.  Tensor sizes: {self.shape}')
    return _return(jnp.broadcast_to(self.value, sizes_list))

  def tree_flatten(self):
    return (self.value,), None

  @classmethod
  def tree_unflatten(cls, aux_data, flat_contents):
    return cls(*flat_contents)

  def zero_(self):
    self.value = jnp.zeros_like(self.value)
    return self

  def fill_(self, value):
    self.fill(value)
    return self

  def uniform_(self, low=0., high=1.):
    global bm
    if bm is None: from brainpy import math as bm
    self.value = bm.random.uniform(low, high, self.shape)
    return self

  def log_normal_(self, mean=1, std=2):
    r"""Fills self tensor with numbers samples from the log-normal distribution parameterized by the given mean
    :math:`\mu` and standard deviation :math:`\sigma`. Note that mean and std are the mean and standard
    deviation of the underlying normal distribution, and not of the returned distribution:

    .. math::

       f(x)=\frac{1}{x \sigma \sqrt{2 \pi}} e^{-\frac{(\ln x-\mu)^2}{2 \sigma^2}}

    Args:
      mean: the mean value.
      std: the standard deviation.
    """
    global bm
    if bm is None: from brainpy import math as bm
    self.value = bm.random.lognormal(mean, std, self.shape)
    return self

  def normal_(self, ):
    """
    Fills self tensor with elements samples from the normal distribution parameterized by mean and std.
    """
    global bm
    if bm is None: from brainpy import math as bm
    self.value = bm.random.randn(*self.shape)
    return self

  def cuda(self):
    self.value = jax.device_put(self.value, jax.devices('cuda')[0])
    return self

  def cpu(self):
    self.value = jax.device_put(self.value, jax.devices('cpu')[0])
    return self

  # dtype exchanging #
  # ---------------- #

  def bool(self):
    return jnp.asarray(self.value, dtype=jnp.bool_)

  def int(self):
    return jnp.asarray(self.value, dtype=jnp.int32)

  def long(self):
    return jnp.asarray(self.value, dtype=jnp.int64)

  def half(self):
    return jnp.asarray(self.value, dtype=jnp.float16)

  def float(self):
    return jnp.asarray(self.value, dtype=jnp.float32)

  def double(self):
    return jnp.asarray(self.value, dtype=jnp.float64)


class Unit(Array):
  ...


class UnitRegistry:
  """
  Stores known units for printing in best units.

  All a user needs to do is to use the `register_new_unit`
  function.

  Default registries:

  The units module defines three registries, the standard units,
  user units, and additional units. Finding best units is done
  by first checking standard, then user, then additional. New
  user units are added by using the `register_new_unit` function.

  Standard units includes all the basic non-compound unit names
  built in to the module, including volt, amp, etc. Additional
  units defines some compound units like newton metre (Nm) etc.

  Methods
  -------
  add
  __getitem__
  """

  def __init__(self):
    self.units = collections.OrderedDict()
    self.units_for_dimensions = collections.defaultdict(dict)

  def add(self, u):
    """Add a unit to the registry"""
    self.units[repr(u)] = u
    self.units_for_dimensions[u.dim][float(u)] = u

  def __getitem__(self, x):
    """Returns the best unit for quantity x

    The algorithm is to consider the value:

    m=abs(x/u)

    for all matching units u. We select the unit where this ratio is the
    closest to 10 (if it is an array with several values, we select the
    unit where the deviations from that are the smallest. More precisely,
    the unit that minimizes the sum of (log10(m)-1)**2 over all entries).
    """
    matching = self.units_for_dimensions.get(x.dim, {})
    if len(matching) == 0:
      raise KeyError("Unit not found in registry.")

    matching_values = np.array(list(matching.keys()), copy=False)
    print_opts = np.get_printoptions()
    edgeitems, threshold = print_opts["edgeitems"], print_opts["threshold"]
    if x.size > threshold:
      # Only care about optimizing the units for the values that will
      # actually be shown later
      # The code looks a bit complex, but should return the same numbers
      # that are shown by numpy's string conversion
      slices = []
      for shape in x.shape:
        if shape > 2 * edgeitems:
          slices.append((slice(0, edgeitems), slice(-edgeitems, None)))
        else:
          slices.append((slice(None),))
      x_flat = np.hstack(
        [x[use_slices].flatten() for use_slices in itertools.product(*slices)]
      )
    else:
      x_flat = np.array(x, copy=False).flatten()
    floatreps = np.tile(np.abs(x_flat), (len(matching), 1)).T / matching_values
    # ignore zeros, they are well represented in any unit
    floatreps[floatreps == 0] = np.nan
    if np.all(np.isnan(floatreps)):
      return matching[1.0]  # all zeros, use the base unit

    deviations = np.nansum((np.log10(floatreps) - 1) ** 2, axis=0)
    return list(matching.values())[deviations.argmin()]


#: `UnitRegistry` containing all the standard units (metre, kilogram, um2...)
standard_unit_register = UnitRegistry()
#: `UnitRegistry` containing additional units (newton*metre, farad / metre, ...)
additional_unit_register = UnitRegistry()
#: `UnitRegistry` containing all units defined by the user
user_unit_register = UnitRegistry()
