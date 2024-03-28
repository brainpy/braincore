import collections
import itertools
import numbers
import operator
from typing import Union, Optional, Sequence, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.dtypes import canonicalize_dtype
from jax.tree_util import register_pytree_node_class

from braincore.environ import dftype

__all__ = [
  'Quantity',
  'Unit',
  'UnitRegistry',
  'DimensionMismatchError',
  'get_or_create_dimension',
  'get_dimensions',
  'is_dimensionless',
  'have_same_dimensions',
  'in_unit',
  'in_best_unit',
  'register_new_unit',
  'check_units',
  'is_scalar_type',
  'get_unit',
]

_all_slice = slice(None, None, None)
unit_checking = True


def _as_jax_array_(obj):
  return obj.value if isinstance(obj, Quantity) else obj


def _check_input_array(array):
  if isinstance(array, Quantity):
    return array
  elif isinstance(array, (numbers.Number, jax.Array, np.number, np.ndarray, list, tuple)):
    return Quantity(value=jnp.asarray(array))
  else:
    raise TypeError('Input array should be an instance of Array.')


def _check_input_not_array(array):
  if isinstance(array, Quantity):
    raise ValueError('Input array should not be an instance of Array.')
  return array


def _check_out(out):
  if not isinstance(out, Quantity):
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
  for a dimensionless array.

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

# SI unit _prefixes as integer exponents of 10, see table at end of file.
_siprefixes = {
  "y": -24,
  "z": -21,
  "a": -18,
  "f": -15,
  "p": -12,
  "n": -9,
  "u": -6,
  "m": -3,
  "c": -2,
  "d": -1,
  "": 0,
  "da": 1,
  "h": 2,
  "k": 3,
  "M": 6,
  "G": 9,
  "T": 12,
  "P": 15,
  "E": 18,
  "Z": 21,
  "Y": 24,
}


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
  Users shouldn't use this class directly, it is used internally in Array
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
    attribute -- this will return a `Dimension` object for `Array`,
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
    value = np.array(value)
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

  >>> from braincore.units import *
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

  Slightly more general than `Array.dimensions` because it will
  return `DIMENSIONLESS` if the object is of number type but not a `Array`
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
    if isinstance(obj, (numbers.Number, jax.Array, np.number, np.ndarray)):
      return DIMENSIONLESS
    try:
      return Quantity(obj).unit
    except TypeError:
      raise TypeError(f"Object of type {type(obj)} does not have dimensions")


def have_same_dimensions(obj1, obj2):
  """Test if two values have the same dimensions.

  Parameters
  ----------
  obj1, obj2 : {`Array`, array-like, number}
      The values of which to compare the dimensions.

  Returns
  -------
  same : `bool`
      ``True`` if `obj1` and `obj2` have the same dimensions.
  """

  if not unit_checking:
    return True  # ignore units when unit checking is disabled

  # If dimensions are consistently created using get_or_create_dimensions,
  # the fast "is" comparison should always return the correct result.
  # To be safe, we also do an equals comparison in case it fails. This
  # should only add a small amount of unnecessary computation for cases in
  # which this function returns False which very likely leads to a
  # DimensionMismatchError anyway.
  dim1 = get_dimensions(obj1)
  dim2 = get_dimensions(obj2)
  return (dim1 is dim2) or (dim1 == dim2) or dim1 is None or dim2 is None


def fail_for_dimension_mismatch(
    obj1, obj2=None, error_message=None, **error_arrays
):
  """
  Compare the dimensions of two objects.

  Parameters
  ----------
  obj1, obj2 : {array-like, `Array`}
      The object to compare. If `obj2` is ``None``, assume it to be
      dimensionless
  error_message : str, optional
      An error message that is used in the DimensionMismatchError
  error_arrays : dict mapping str to `Array`, optional
      Arrays in this dictionary will be converted using the `_short_str`
      helper method and inserted into the ``error_message`` (which should
      have placeholders with the corresponding names). The reason for doing
      this in a somewhat complicated way instead of directly including all the
      details in ``error_messsage`` is that converting large arrays
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
    # if it is not a Array, it has "any dimension".
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
      error_arrays = {
        name: _short_str(q) for name, q in error_arrays.items()
      }
      error_message = error_message.format(**error_arrays)
    # If we are comparing an object to a specific unit, we don't want to
    # restate this unit (it is probably mentioned in the text already)
    if obj2 is None or isinstance(obj2, (Dimension, Unit)):
      raise DimensionMismatchError(error_message, dim1)
    else:
      raise DimensionMismatchError(error_message, dim1, dim2)
  else:
    return dim1, dim2


def in_unit(x, u, precision=None):
  """
  Display a value in a certain unit with a given precision.

  Parameters
  ----------
  x : {`Array`, array-like, number}
      The value to display
  u : {`Array`, `Unit`}
      The unit to display the value `x` in.
  precision : `int`, optional
      The number of digits of precision (in the given unit, see Examples).
      If no value is given, numpy's `get_printoptions` value is used.

  Returns
  -------
  s : `str`
      A string representation of `x` in units of `u`.

  Examples
  --------
  >>> from braincore.units import *
  >>> in_unit(3 * volt, mvolt)
  '3000. mV'
  >>> in_unit(123123 * msecond, second, 2)
  '123.12 s'
  >>> in_unit(10 * uA/cm**2, nA/um**2)
  '1.00000000e-04 nA/(um^2)'
  >>> in_unit(10 * mV, ohm * amp)
  '0.01 ohm A'
  >>> in_unit(10 * nS, ohm) # doctest: +NORMALIZE_WHITESPACE
  ...                       # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
      ...
  DimensionMismatchError: Non-matching unit for method "in_unit",
  dimensions were (m^-2 kg^-1 s^3 A^2) (m^2 kg s^-3 A^-2)

  See Also
  --------
  Array.in_unit
  """
  if is_dimensionless(x):
    fail_for_dimension_mismatch(x, u, 'Non-matching unit for function "in_unit"')
    return str(np.array(x / u))
  else:
    return x.in_unit(u, precision=precision)


def in_best_unit(x, precision=None):
  """
  Represent the value in the "best" unit.

  Parameters
  ----------
  x : {`Array`, array-like, number}
      The value to display
  precision : `int`, optional
      The number of digits of precision (in the best unit, see Examples).
      If no value is given, numpy's `get_printoptions` value is used.

  Returns
  -------
  representation : `str`
      A string representation of this `Array`.

  Examples
  --------
  >>> from braincore.units import *
  >>> in_best_unit(0.00123456 * volt)
  '1.23456 mV'
  >>> in_best_unit(0.00123456 * volt, 2)
  '1.23 mV'
  >>> in_best_unit(0.123456)
  '0.123456'
  >>> in_best_unit(0.123456, 2)
  '0.12'

  See Also
  --------
  Array.in_best_unit
  """
  if is_dimensionless(x):
    if precision is None:
      precision = np.get_printoptions()["precision"]
    return str(np.round(x, precision))

  u = x.get_best_unit()
  return x.in_unit(u, precision=precision)


def array_with_units(floatval, units: Dimension, dtype=None):
  """
  Create a new `Array` with the given dimensions. Calls
  `get_or_create_dimension` with the dimension tuple of the `dims`
  argument to make sure that unpickling (which calls this function) does not
  accidentally create new Dimension objects which should instead refer to
  existing ones.

  Parameters
  ----------
  floatval : `float`
      The floating point value of the array.
  units: `Dimension`
      The unit dimensions of the array.
  dtype: `dtype`, optional
      The data type of the array.

  Returns
  -------
  array : `Quantity`
      The new `Array` object.

  Examples
  --------
  >>> from braincore.units import *
  >>> array_with_units(0.001, volt.unit)
  1. * mvolt
  """
  return Quantity(floatval, unit=get_or_create_dimension(units._dims), dtype=dtype)


def is_dimensionless(obj):
  """
  Test if a value is dimensionless or not.

  Parameters
  ----------
  obj : `object`
      The object to check.

  Returns
  -------
  dimensionless : `bool`
      ``True`` if `obj` is dimensionless.
  """
  return get_dimensions(obj) is DIMENSIONLESS


def is_scalar_type(obj):
  """
  Tells you if the object is a 1d number type.

  Parameters
  ----------
  obj : `object`
      The object to check.

  Returns
  -------
  scalar : `bool`
      ``True`` if `obj` is a scalar that can be interpreted as a
      dimensionless `Array`.
  """
  try:
    return obj.ndim == 0 and is_dimensionless(obj)
  except AttributeError:
    return jnp.isscalar(obj) and not isinstance(obj, str)


#: ufuncs that work on all dimensions and preserve the dimensions, e.g. abs
UFUNCS_PRESERVE_DIMENSIONS = [
  "absolute",
  "rint",
  "negative",
  "positive",
  "conj",
  "conjugate",
  "floor",
  "ceil",
  "trunc",
]

#: ufuncs that work on all dimensions but change the dimensions, e.g. square
UFUNCS_CHANGE_DIMENSIONS = [
  "multiply",
  "divide",
  "true_divide",
  "floor_divide",
  "sqrt",
  "square",
  "reciprocal",
  "dot",
  "matmul",
]

#: ufuncs that work with matching dimensions, e.g. add
UFUNCS_MATCHING_DIMENSIONS = [
  "add",
  "subtract",
  "maximum",
  "minimum",
  "remainder",
  "mod",
  "fmod",
]

#: ufuncs that compare values, i.e. work only with matching dimensions but do
#: not result in a value with dimensions, e.g. equals
UFUNCS_COMPARISONS = [
  "less",
  "less_equal",
  "greater",
  "greater_equal",
  "equal",
  "not_equal",
]

#: Logical operations that work on all quantities and return boolean arrays
UFUNCS_LOGICAL = [
  "logical_and",
  "logical_or",
  "logical_xor",
  "logical_not",
  "isreal",
  "iscomplex",
  "isfinite",
  "isinf",
  "isnan",
]

#: ufuncs that only work on dimensionless quantities
UFUNCS_DIMENSIONLESS = [
  "sin",
  "sinh",
  "arcsin",
  "arcsinh",
  "cos",
  "cosh",
  "arccos",
  "arccosh",
  "tan",
  "tanh",
  "arctan",
  "arctanh",
  "log",
  "log2",
  "log10",
  "log1p",
  "exp",
  "exp2",
  "expm1",
]

#: ufuncs that only work on two dimensionless quantities
UFUNCS_DIMENSIONLESS_TWOARGS = ["logaddexp", "logaddexp2", "arctan2", "hypot"]

#: ufuncs that only work on integers and therefore never on quantities
UFUNCS_INTEGERS = [
  "bitwise_and",
  "bitwise_or",
  "bitwise_xor",
  "invert",
  "left_shift",
  "right_shift",
]


def wrap_function_dimensionless(func):
  """
  Returns a new function that wraps the given function `func` so that it
  raises a DimensionMismatchError if the function is called on a array
  with dimensions (excluding dimensionless arrays). Arrays are
  transformed to unitless numpy arrays before calling `func`.

  These checks/transformations apply only to the very first argument, all
  other arguments are ignored/untouched.
  """

  def f(x, *args, **kwds):  # pylint: disable=C0111
    fail_for_dimension_mismatch(
      x,
      error_message=(
          "%s expects a dimensionless argument but got {value}" % func.__name__
      ),
      value=x,
    )
    return func(jnp.array(x), *args, **kwds)

  f._arg_units = [1]
  f._return_unit = 1
  f.__name__ = func.__name__
  f.__doc__ = func.__doc__
  f._do_not_run_doctests = True
  return f


def wrap_function_keep_dimensions(func):
  """
  Returns a new function that wraps the given function `func` so that it
  keeps the dimensions of its input. Arrays are transformed to
  unitless jax numpy arrays before calling `func`, the output is a array
  with the original dimensions re-attached.

  These transformations apply only to the very first argument, all
  other arguments are ignored/untouched, allowing to work functions like
  ``sum`` to work as expected with additional ``axis`` etc. arguments.
  """

  def f(x, *args, **kwds):  # pylint: disable=C0111
    return Quantity(func(x.value, *args, **kwds), unit=x.unit)

  f._arg_units = [None]
  f._return_unit = lambda u: u
  f.__name__ = func.__name__
  f.__doc__ = func.__doc__
  f._do_not_run_doctests = True
  return f


def wrap_function_change_dimensions(func, change_dim_func):
  """
  Returns a new function that wraps the given function `func` so that it
  changes the dimensions of its input. Arrays are transformed to
  unitless jax numpy arrays before calling `func`, the output is a array
  with the original dimensions passed through the function
  `change_dim_func`. A typical use would be a ``sqrt`` function that uses
  ``lambda d: d ** 0.5`` as ``change_dim_func``.

  These transformations apply only to the very first argument, all
  other arguments are ignored/untouched.
  """

  def f(x, *args, **kwds):  # pylint: disable=C0111
    return Quantity(func(x.value, *args, **kwds), unit=change_dim_func(x.value, x.unit))

  f._arg_units = [None]
  f._return_unit = change_dim_func
  f.__name__ = func.__name__
  f.__doc__ = func.__doc__
  f._do_not_run_doctests = True
  return f


def wrap_function_remove_dimensions(func):
  """
  Returns a new function that wraps the given function `func` so that it
  removes any dimensions from its input. Useful for functions that are
  returning integers (indices) or booleans, irrespective of the datatype
  contained in the array.

  These transformations apply only to the very first argument, all
  other arguments are ignored/untouched.
  """

  def f(x, *args, **kwds):  # pylint: disable=C0111
    return func(x.value, *args, **kwds)

  f._arg_units = [None]
  f._return_unit = 1
  f.__name__ = func.__name__
  f.__doc__ = func.__doc__
  f._do_not_run_doctests = True
  return f


def can_convert_to_dtype(elements, dtype):
  try:
    [dtype(element) for element in elements]
    return True
  except ValueError:
    return False


@register_pytree_node_class
class Quantity(object):
  """

  """
  __slots__ = ('_value', '_unit')
  _value: jax.Array
  _unit: Dimension

  __array_priority__ = 1000

  def __init__(self, value, dtype=None, unit=DIMENSIONLESS):
    dtype = dtype or dftype()

    # array value
    if isinstance(value, Quantity):
      self._unit = value.unit
      self._value = jnp.array(value.value, dtype=dtype)
      return

    elif isinstance(value, (tuple, list)):
      if len(value) == 0:
        value = jnp.asarray(1., dtype=dtype)
      else:
        # Existing logic to check for mixed types or process units
        has_units = [hasattr(v, 'unit') for v in value]
        if any(has_units) and not all(has_units):
          raise TypeError("All elements must have the same unit or no unit at all")
        if all(has_units):
          units = [v.unit for v in value]
          if not all(u == units[0] for u in units[1:]):
            raise TypeError("All elements must have the same unit")
          unit = units[0]
          value = [v.value for v in value]
          del units
        del has_units
        # Transform to jnp array
        try:
          value = jnp.array(value, dtype=dtype)
        except ValueError:
          raise TypeError("All elements must be convertible to a jax array")

    elif isinstance(value, (np.ndarray, jax.Array)):
      value = jnp.asarray(value, dtype=dtype)

    elif isinstance(value, (np.number, numbers.Number)):
      value = jnp.array(value, dtype=dtype)

    else:
      raise TypeError(f"Invalid type for value: {type(value)}")

    # value
    self._value = value
    # unit
    self._unit = unit

  @property
  def value(self):
    # return the value
    return self._value

  @value.setter
  def value(self, value):
    """
    
    Examples::
    
    >>> a = jax.numpy.array([1, 2, 3]) * mV
    >>> a[:] = jax.numpy.array([4, 5, 6]) * mV
    >>> a.value = jax.numpy.array([7, 8, 9])    
    
    Args:
      value: 

    Returns:

    """
    self_value = self._check_tracer()
    if isinstance(value, Quantity):
      raise ValueError("Cannot set the value of an Array object to another Array object.")
      # if self.unit != value.unit:
      #   raise ValueError(f"The unit of the original data is {self.unit}, "
      #                    f"while we got {value.unit}.")
      # value = value.value
    if isinstance(value, np.ndarray):
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
    raise AttributeError("Cannot set the unit of an Array object")

  ### UNITS ###

  def __array_ufunc__(self, uf, method, *inputs, **kwargs):
    if method not in ("__call__", "reduce"):
      return NotImplemented
    uf_method = getattr(uf, method)
    if "out" in kwargs:
      # In contrast to numpy, we will not change a scalar value in-place,
      # i.e. a scalar Quantity will act like a Python float and not like
      # a numpy scalar in that regard.
      if self.ndim == 0:
        del kwargs["out"]
      else:
        # The output needs to be an array to avoid infinite recursion
        # Note that it is also part of the input arguments, so we don't
        # need to check its dimensions
        assert len(kwargs["out"]) == 1
        kwargs["out"] = (jnp.asarray(kwargs["out"][0]),)
    if uf.__name__ in (UFUNCS_LOGICAL + ["sign", "ones_like"]):
      # do not touch return value
      return uf_method(*[jnp.asarray(a) for a in inputs], **kwargs)
    elif uf.__name__ in UFUNCS_PRESERVE_DIMENSIONS:
      return Quantity(
        uf_method(*[np.asarray(a) for a in inputs], **kwargs),
        unit=self.unit,
      )
    elif uf.__name__ in UFUNCS_CHANGE_DIMENSIONS + ["power"]:
      if uf.__name__ == "sqrt":
        unit = self.unit ** 0.5
      elif uf.__name__ == "power":
        fail_for_dimension_mismatch(
          inputs[1],
          error_message=(
            "The exponent for a "
            "power operation has to "
            "be dimensionless but "
            "was {value}"
          ),
          value=inputs[1],
        )
        if np.asarray(inputs[1]).size != 1:
          raise TypeError(
            "Only length-1 arrays can be used as an exponent for"
            " quantities."
          )
        unit = get_dimensions(inputs[0]) ** np.asarray(inputs[1])
      elif uf.__name__ == "square":
        unit = self.unit ** 2
      elif uf.__name__ in ("divide", "true_divide", "floor_divide"):
        unit = get_dimensions(inputs[0]) / get_dimensions(inputs[1])
      elif uf.__name__ == "reciprocal":
        unit = get_dimensions(inputs[0]) ** -1
      elif uf.__name__ in ("multiply", "dot", "matmul"):
        if method == "__call__":
          unit = get_dimensions(inputs[0]) * get_dimensions(inputs[1])
        else:
          unit = get_dimensions(inputs[0])
      else:
        return NotImplemented
      return Quantity(
        uf_method(*[np.asarray(a) for a in inputs], **kwargs), unit=unit
      )
    elif uf.__name__ in UFUNCS_INTEGERS:
      # Numpy should already raise a TypeError by itself
      raise TypeError(f"{uf.__name__} cannot be used on quantities.")
    elif uf.__name__ in UFUNCS_MATCHING_DIMENSIONS + UFUNCS_COMPARISONS:
      # Ok if dimension of arguments match (for reductions, they always do)
      if method == "__call__":
        fail_for_dimension_mismatch(
          inputs[0],
          inputs[1],
          error_message=(
                          "Cannot calculate {val1} %s {val2}, the units do not match"
                        )
                        % uf.__name__,
          val1=inputs[0],
          val2=inputs[1],
        )
      if uf.__name__ in UFUNCS_COMPARISONS:
        return uf_method(*[np.asarray(i) for i in inputs], **kwargs)
      else:
        return Quantity(
          uf_method(*[np.asarray(i) for i in inputs], **kwargs),
          unit=self.unit,
        )
    elif uf.__name__ in UFUNCS_DIMENSIONLESS:
      # Ok if argument is dimensionless
      fail_for_dimension_mismatch(
        inputs[0],
        error_message="%s expects a dimensionless argument but got {value}"
                      % uf.__name__,
        value=inputs[0],
      )
      return uf_method(np.asarray(inputs[0]), *inputs[1:], **kwargs)
    elif uf.__name__ in UFUNCS_DIMENSIONLESS_TWOARGS:
      # Ok if both arguments are dimensionless
      fail_for_dimension_mismatch(
        inputs[0],
        error_message=(
                        "Both arguments for "
                        '"%s" should be '
                        "dimensionless but "
                        "first argument was "
                        "{value}"
                      )
                      % uf.__name__,
        value=inputs[0],
      )
      fail_for_dimension_mismatch(
        inputs[1],
        error_message=(
                        "Both arguments for "
                        '"%s" should be '
                        "dimensionless but "
                        "second argument was "
                        "{value}"
                      )
                      % uf.__name__,
        value=inputs[1],
      )
      return uf_method(
        np.asarray(inputs[0]),
        np.asarray(inputs[1]),
        *inputs[2:],
        **kwargs,
      )
    else:
      return NotImplemented

  @staticmethod
  def with_units(value, *args, **keywords):
    """
    Create a `Array` object with the given units.

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
        A `Array` object with the given dim

    Examples
    --------
    All of these define an equivalent `Array` object:

    >>> from braincore.units import *
    >>> Quantity.with_units(2, get_or_create_dimension(length=1))
    2. * metre
    >>> Quantity.with_units(2, length=1)
    2. * metre
    >>> 2 * metre
    2. * metre
    """
    if len(args) and isinstance(args[0], Dimension):
      dimensions = args[0]
    else:
      dimensions = get_or_create_dimension(*args, **keywords)
    return Quantity(value, unit=dimensions)

  is_dimensionless = property(lambda self: self.unit.is_dimensionless,
                              doc="Whether the array is dimensionless")

  def has_same_unit(self, other):
    """
    Whether this Array has the same unit dimensions as another Array

    Parameters
    ----------
    other : Quantity
        The other Array to compare with

    Returns
    -------
    bool
        Whether the two Arrays have the same unit dimensions
    """
    other_unit = get_unit(other.unit)
    return (get_unit(self.unit) is other_unit) or (get_unit(self.unit) == other_unit)

  def in_unit(self, u, precision=None, python_code=False) -> str:
    """
    Represent the Array in a given unit.

    Parameters
    ----------
    u : `Quantity`
        The unit in which to show the ar.
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
    >>> from braincore.units import *
    >>> x = 25.123456 * mV
    >>> x.in_unit(volt)
    '0.02512346 V'
    >>> x.in_unit(volt, 3)
    '0.025 V'
    >>> x.in_unit(mV, 3)
    '25.123 mV'
    """
    fail_for_dimension_mismatch(self, u, 'Non-matching unit for method "in_unit"')
    value = np.asarray(self.value / u.value)
    if value.shape == ():
      s = np.array_str(jnp.array([value]), precision=precision)
      s = s.replace("[", "").replace("]", "").strip()
    else:
      if value.size > 100:
        if python_code:
          s = np.array_repr(value, precision=precision)[:100]
          s += "..."
        else:
          s = np.array_str(value, precision=precision)[:100]
          s += "..."
      else:
        if python_code:
          s = np.array_repr(value, precision=precision)
        else:
          s = np.array_str(value, precision=precision)

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
    elif python_code:  # Make a array without unit recognisable
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
    u : `Quantity` or `Unit`
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
      return Quantity(1, unit=self.unit)
    else:
      return self.get_best_unit(standard_unit_register, user_unit_register, additional_unit_register)

  def in_best_unit(self, precision=None, python_code=False, *regs):
    """
    Represent the array in the "best" unit.

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
    >>> from braincore.units import *
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
    # if hasattr(self_value, '_trace') and hasattr(self_value._trace.main, 'jaxpr_stack'):
    #   if len(self_value._trace.main.jaxpr_stack) == 0:
    #     raise RuntimeError('This Array is modified during the transformation. '
    #                        'BrainPy only supports transformations for Variable. '
    #                        'Please declare it as a Variable.') from jax.core.escaped_tracer_error(self_value, None)
    return self_value

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
    return Quantity(self.value.image, unit=self.unit)

  @property
  def real(self):
    return Quantity(self.value.real, unit=self.unit)

  @property
  def size(self):
    return self.value.size

  @property
  def T(self):
    return Quantity(self.value.T, unit=self.unit)

  # ----------------------- #
  # Python inherent methods #
  # ----------------------- #

  def __repr__(self) -> str:
    return self.in_best_unit(python_code=True)

  def __str__(self) -> str:
    return self.in_best_unit()

  def __format__(self, format_spec: str) -> str:
    # Avoid that formatted strings like f"{q}" use floating point formatting for the
    # array, i.e. discard the unit
    if format_spec == "":
      return str(self)
    else:
      return self.value.__format__(format_spec)

  def __iter__(self):
    """Solve the issue of DeviceArray.__iter__.

    Details please see JAX issues:

    - https://github.com/google/jax/issues/7713
    - https://github.com/google/jax/pull/3821
    """
    for i in range(self.value.shape[0]):
      yield Quantity(self.value[i], unit=self.unit)

  def __getitem__(self, index):
    if isinstance(index, slice) and (index == _all_slice):
      return Quantity(self.value, unit=self.unit)
    elif isinstance(index, tuple):
      for x in index:
        assert not isinstance(x, Quantity), "Array indices must be integers or slices, not Array"
    elif isinstance(index, Quantity):
      raise TypeError("Array indices must be integers or slices, not Array")
    return Quantity(self.value[index], unit=self.unit)

  def __setitem__(self, index, value: 'Quantity'):
    if not isinstance(value, Quantity):
      raise DimensionMismatchError("Only Array can be assigned to Array.")
    fail_for_dimension_mismatch(self, value, "Inconsistent units in assignment")
    value = value.value

    # index is a tuple
    assert not isinstance(index, Quantity), "Array indices must be integers or slices, not Array"
    if isinstance(index, tuple):
      index = tuple(_check_input_not_array(x) for x in index)
    # index is numpy.ndarray
    elif isinstance(index, np.ndarray):
      index = jnp.asarray(index)

    # update
    self_value = self._check_tracer()
    self.value = self_value.at[index].set(value)

  # ---------- #
  # operations #
  # ---------- #

  def __len__(self) -> int:
    return len(self.value)

  def __neg__(self):
    return Quantity(self.value.__neg__(), unit=self.unit)

  def __pos__(self):
    return Quantity(self.value.__pos__(), unit=self.unit)

  def __abs__(self):
    return Quantity(self.value.__abs__(), unit=self.unit)

  def __invert__(self):
    return Quantity(self.value.__invert__(), unit=self.unit)

  def _comparison(self, other, operator_str, operation):
    is_scalar = is_scalar_type(other)
    if not is_scalar and not isinstance(other, (jax.Array, Quantity, np.ndarray)):
      return NotImplemented
    if not is_scalar or not jnp.isinf(other):
      other = _check_input_array(other)
      message = "Cannot perform comparison {value1} %s {value2}, units do not match" % operator_str
      fail_for_dimension_mismatch(self, other, message, value1=self, value2=other)
    other = _check_input_array(other)
    return operation(self.value, other.value)

  def __eq__(self, oc):
    return self._comparison(oc, "==", operator.eq)

  def __ne__(self, oc):
    return self._comparison(oc, "!=", operator.ne)

  def __lt__(self, oc):
    return self._comparison(oc, "<", operator.lt)

  def __le__(self, oc):
    return self._comparison(oc, "<=", operator.le)

  def __gt__(self, oc):
    return self._comparison(oc, ">", operator.gt)

  def __ge__(self, oc):
    return self._comparison(oc, ">=", operator.ge)

  def _binary_operation(
      self,
      other,
      value_operation: Callable,
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
    value_operation : function of two variables
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
    other = _check_input_array(other)
    other_unit = None

    if fail_for_mismatch:
      if inplace:
        message = "Cannot calculate ... %s {value}, units do not match" % operator_str
        _, other_unit = fail_for_dimension_mismatch(self, other, message, value=other)
      else:
        message = "Cannot calculate {value1} %s {value2}, units do not match" % operator_str
        _, other_unit = fail_for_dimension_mismatch(self, other, message, value1=self, value2=other)

    if other_unit is None:
      other_unit = get_dimensions(other)

    newdims = unit_operation(self.unit, other_unit)
    result = value_operation(self.value, other.value)
    r = Quantity(result, unit=newdims)
    if inplace:
      self.value = r.value
      return self
    return r

  def __add__(self, oc):
    return self._binary_operation(oc, operator.add, fail_for_mismatch=True, operator_str="+")

  def __radd__(self, oc):
    return self.__add__(oc)

  def __iadd__(self, oc):
    # a += b
    return self._binary_operation(oc, operator.add, fail_for_mismatch=True, operator_str="+=", inplace=True)

  def __sub__(self, oc):
    return self._binary_operation(oc, operator.sub, fail_for_mismatch=True, operator_str="-")

  def __rsub__(self, oc):
    return Quantity(oc).__sub__(self)

  def __isub__(self, oc):
    # a -= b
    return self._binary_operation(oc, operator.sub, fail_for_mismatch=True, operator_str="-=", inplace=True)

  def __mul__(self, oc):
    return self._binary_operation(oc, operator.mul, operator.mul)

  def __rmul__(self, oc):
    return self.__mul__(oc)

  def __imul__(self, oc):
    # a *= b
    return self._binary_operation(oc, operator.mul, operator.mul, inplace=True)

  def __div__(self, oc):
    return self._binary_operation(oc, operator.truediv, operator.truediv)

  def __idiv__(self, oc):
    r = self.__div__(oc)
    self.value = r.value
    self.unit = r.unit
    return self

  def __truediv__(self, oc):
    return self.__div__(oc)

  def __rdiv__(self, oc):
    # division with swapped arguments
    rdiv = lambda a, b: operator.truediv(b, a)
    return self._binary_operation(oc, rdiv, rdiv)

  def __rtruediv__(self, oc):
    return self.__rdiv__(oc)

  def __itruediv__(self, oc):
    # a /= b
    return self._binary_operation(oc, operator.truediv, operator.truediv, inplace=True)

  def __floordiv__(self, oc):
    return self._binary_operation(oc, operator.floordiv, operator.truediv)

  def __rfloordiv__(self, oc):
    rdiv = lambda a, b: operator.truediv(b, a)
    rfloordiv = lambda a, b: operator.truediv(b, a)
    return self._binary_operation(oc, rfloordiv, rdiv)

  def __ifloordiv__(self, oc):
    # a //= b
    return self._binary_operation(oc, operator.floordiv, operator.truediv, inplace=True)

  def __mod__(self, oc):
    return self._binary_operation(oc, operator.mod, fail_for_mismatch=True, operator_str=r"%")

  def __rmod__(self, oc):
    oc = _check_input_array(oc)
    return oc._binary_operation(self, operator.mod, fail_for_mismatch=True, operator_str=r"%")

  def __imod__(self, oc):
    return self._binary_operation(oc, operator.mod, fail_for_mismatch=True, operator_str=r"%", inplace=True)

  def __divmod__(self, oc):
    return self.__floordiv__(oc), self.__mod__(oc)

  def __rdivmod__(self, oc):
    return self.__rfloordiv__(oc), self.__rmod__(oc)

  def __matmul__(self, oc):
    return self._binary_operation(oc, operator.matmul, operator.mul)

  def __rmatmul__(self, oc):
    oc = _check_input_array(oc)
    return oc._binary_operation(self, operator.matmul, operator.mul)

  def __imatmul__(self, oc):
    # a @= b
    raise NotImplementedError("In-place matrix multiplication is not supported")

  # -------------------- #

  def __pow__(self, oc):
    if isinstance(oc, (jax.Array, np.ndarray, numbers.Number, Quantity)) or is_scalar_type(oc):
      fail_for_dimension_mismatch(
        oc,
        error_message=(
          "Cannot calculate "
          "{base} ** {exponent}, "
          "the exponent has to be "
          "dimensionless"
        ),
        base=self,
        exponent=oc,
      )
      if isinstance(oc, Quantity):
        oc = oc.value
      return Quantity(jnp.array(self.value) ** oc, unit=self.unit ** oc)
    else:
      return TypeError('Cannot calculate {base} ** {exponent}, the '
                       'exponent has to be dimensionless'.format(base=self, exponent=oc))

  def __rpow__(self, oc):
    if self.is_dimensionless:
      if isinstance(oc, (jax.Array, np.ndarray, numbers.Number)):
        return Quantity(oc ** self.value, unit=DIMENSIONLESS)
      else:
        return oc.__pow__(self.value)
    else:
      raise DimensionMismatchError(f"Cannot calculate {_short_str(oc)} ** {_short_str(self)}, "
                                   f"the base has to be dimensionless",
                                   self.unit)

  def __ipow__(self, oc):
    # a **= b
    raise NotImplementedError("In-place power is not supported")

  def __and__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __rand__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __iand__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __or__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __ror__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __ior__(self, oc):
    # Remove the unit from the result
    # a |= b
    raise NotImplementedError("Bitwise operations are not supported")

  def __xor__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __rxor__(self, oc):
    # Remove the unit from the result
    raise NotImplementedError("Bitwise operations are not supported")

  def __ixor__(self, oc):
    # Remove the unit from the result
    # a ^= b
    raise NotImplementedError("Bitwise operations are not supported")

  def __lshift__(self, oc):
    if isinstance(oc, (jax.Array, np.ndarray, numbers.Number)) or is_scalar_type(oc):
      if isinstance(oc, Quantity):
        oc = oc.value
      return Quantity(self.value << oc, unit=self.unit)
    else:
      raise TypeError("Cannot calculate {value} << {shift}, the shift has "
                      "to be dimensionless".format(value=self, shift=oc))

  def __rlshift__(self, oc):
    if self.is_dimensionless:
      if isinstance(oc, (jax.Array, np.ndarray, numbers.Number)):
        return Quantity(oc << self.value, unit=DIMENSIONLESS)
      elif isinstance(oc, Quantity):
        return oc.__lshift__(self.value)
      else:
        raise TypeError("Cannot calculate {shift} << {value}, the value has "
                        "to be dimensionless".format(value=self, shift=oc))
    else:
      raise DimensionMismatchError(f"Cannot calculate {_short_str(oc)} << {_short_str(self)}, "
                                   f"the shift has to be dimensionless",
                                   self.unit)

  def __ilshift__(self, oc):
    # a <<= b
    r = self.__lshift__(oc)
    self.value = r.value
    return self

  def __rshift__(self, oc):
    if isinstance(oc, (jax.Array, np.ndarray, numbers.Number)) or is_scalar_type(oc):
      if isinstance(oc, Quantity):
        oc = oc.value
      return Quantity(self.value >> oc, unit=self.unit)
    else:
      raise TypeError("Cannot calculate {value} >> {shift}, the shift has "
                      "to be dimensionless".format(value=self, shift=oc))

  def __rrshift__(self, oc):
    if self.is_dimensionless:
      if isinstance(oc, (jax.Array, np.ndarray, numbers.Number)):
        return Quantity(oc >> self.value, unit=DIMENSIONLESS)
      elif isinstance(oc, Quantity):
        return oc.__rshift__(self.value)
      else:
        raise TypeError("Cannot calculate {shift} >> {value}, the value has "
                        "to be dimensionless".format(value=self, shift=oc))
    else:
      raise DimensionMismatchError(f"Cannot calculate {_short_str(oc)} >> {_short_str(self)}, "
                                   f"the shift has to be dimensionless",
                                   self.unit)

  def __irshift__(self, oc):
    # a >>= b
    r = self.__rshift__(oc)
    self.value = r.value
    return self

  def __round__(self, ndigits=None):
    return Quantity(self.value.__round__(ndigits), unit=self.unit)

  def __reduce__(self):
    return array_with_units, (self.value, self.unit, self.value.dtype)

  # ----------------------- #
  #      NumPy methods      #
  # ----------------------- #

  all = wrap_function_remove_dimensions(jnp.all)
  any = wrap_function_remove_dimensions(jnp.any)
  nonzero = wrap_function_remove_dimensions(jnp.nonzero)
  argmax = wrap_function_remove_dimensions(jnp.argmax)
  argmin = wrap_function_remove_dimensions(jnp.argmin)
  argsort = wrap_function_remove_dimensions(jnp.argsort)

  var = wrap_function_change_dimensions(jnp.var, lambda v, d: d ** 2)

  round = wrap_function_keep_dimensions(jnp.round)
  std = wrap_function_keep_dimensions(jnp.std)
  sum = wrap_function_keep_dimensions(jnp.sum)
  trace = wrap_function_keep_dimensions(jnp.trace)
  cumsum = wrap_function_keep_dimensions(jnp.cumsum)
  diagonal = wrap_function_keep_dimensions(jnp.diagonal)
  max = wrap_function_keep_dimensions(jnp.max)
  mean = wrap_function_keep_dimensions(jnp.mean)
  min = wrap_function_keep_dimensions(jnp.min)
  ptp = wrap_function_keep_dimensions(jnp.ptp)
  ravel = wrap_function_keep_dimensions(jnp.ravel)

  def astype(self, dtype):
    """Copy of the array, cast to a specified type.

    Parameters
    ----------
    dtype: str, dtype
      Typecode or data-type to which the array is cast.
    """
    if dtype is None:
      return Quantity(self.value, unit=self.unit)
    else:
      return Quantity(self.value.astype(dtype), unit=self.unit)

  def byteswap(self, inplace=False):
    """Swap the bytes of the array elements

    Toggle between low-endian and big-endian data representation by
    returning a byteswapped array, optionally swapped in-place.
    Arrays of byte-strings are not swapped. The real and imaginary
    parts of a complex number are swapped individually."""
    return Quantity(self.value.byteswap(inplace=inplace), unit=self.unit)

  def choose(self, choices, mode='raise'):
    """Use an index array to construct a new array from a set of choices."""
    return Quantity(self.value.choose(choices=_as_jax_array_(choices), mode=mode), unit=self.unit)

  def clip(self, min=None, max=None, *args, **kwds):
    """Return an array whose values are limited to [min, max]. One of max or min must be given."""
    fail_for_dimension_mismatch(self, min, "clip")
    fail_for_dimension_mismatch(self, max, "clip")
    return Quantity(
      jnp.clip(
        jnp.array(self.value),
        jnp.array(min),
        jnp.array(max),
        *args,
        **kwds,
      ),
      self.unit,
    )
    # min = _as_jax_array_(min)
    # max = _as_jax_array_(max)
    # r = self.value.clip(min=min, max=max)
    # if out is None:
    #   return _return(r)
    # else:
    #   _check_out(out)
    #   out.value = r

  def compress(self, condition, axis=None):
    """Return selected slices of this array along given axis."""
    return Quantity(self.value.compress(condition=_as_jax_array_(condition), axis=axis), unit=self.unit)

  def conj(self):
    """Complex-conjugate all elements."""
    return Quantity(self.value.conj(), unit=self.unit)

  def conjugate(self):
    """Return the complex conjugate, element-wise."""
    return Quantity(self.value.conjugate(), unit=self.unit)

  def copy(self):
    """Return a copy of the array."""
    return Quantity(self.value.copy(), unit=self.unit)

  def cumprod(self, axis=None, dtype=None):
    """Return the cumulative product of the elements along the given axis."""
    if not self.is_dimensionless:
      raise TypeError(
        "cumprod over array elements on arrays "
        "with units is not possible"
      )
    return Quantity(self.value.cumprod(axis=axis, dtype=dtype), unit=self.unit)

  def dot(self, b):
    """Dot product of two arrays."""
    return self._binary_operation(b, jnp.dot, operator.mul)

  def fill(self, value):
    """Fill the array with a scalar value."""
    fail_for_dimension_mismatch(self, value, "fill")
    self.value = jnp.ones_like(self.value) * value

  def flatten(self):
    return Quantity(self.value.flatten(), unit=self.unit)

  def item(self, *args):
    """Copy an element of an array to a standard Python scalar and return it."""
    return Quantity(self.value.item(*args), unit=self.unit)

  def prod(self, *args, **kwds):
    """Return the product of the array elements over the given axis."""
    prod_res = self.value.prod(*args, **kwds)
    # Calculating the correct dimensions is not completly trivial (e.g.
    # like doing self.dim**self.size) because prod can be called on
    # multidimensional arrays along a certain axis.
    # Our solution: Use a "dummy matrix" containing a 1 (without units) at
    # each entry and sum it, using the same keyword arguments as provided.
    # The result gives the exponent for the dimensions.
    # This relies on sum and prod having the same arguments, which is true
    # now and probably remains like this in the future
    dim_exponent = jnp.ones_like(self.value).sum(*args, **kwds)
    # The result is possibly multidimensional but all entries should be
    # identical
    if dim_exponent.size > 1:
      dim_exponent = dim_exponent[0]
    return Quantity(jnp.array(prod_res), unit=self.unit ** dim_exponent)

  def put(self, indices, values):
    """Replaces specified elements of an array with given values.

    Parameters
    ----------
    indices: array_like
      Target indices, interpreted as integers.
    values: array_like
      Values to place in the array at target indices.
    """
    fail_for_dimension_mismatch(self, values, "put")
    self.__setitem__(indices, values)

  def repeat(self, repeats, axis=None):
    """Repeat elements of an array."""
    return Quantity(self.value.repeat(repeats=repeats, axis=axis), unit=self.unit)

  def reshape(self, *shape, order='C'):
    """Returns an array containing the same data with a new shape."""
    return Quantity(self.value.reshape(*shape, order=order), unit=self.unit)

  def resize(self, new_shape):
    """Change shape and size of array in-place."""
    self.value = self.value.reshape(new_shape)

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
    fail_for_dimension_mismatch(self, v, "searchsorted")
    return self.value.searchsorted(v=_as_jax_array_(v), side=side, sorter=sorter)

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
    return Quantity(self.value.squeeze(axis=axis), unit=self.unit)

  def swapaxes(self, axis1, axis2):
    """Return a view of the array with `axis1` and `axis2` interchanged."""
    return Quantity(self.value.swapaxes(axis1, axis2), unit=self.unit)

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
    return [Quantity(a, unit=self.unit) for a in jnp.split(self.value, indices_or_sections, axis=axis)]

  def take(self, indices, axis=None, mode=None):
    """Return an array formed from the elements of a at the given indices."""
    return Quantity(self.value.take(indices=indices, axis=axis, mode=mode), unit=self.unit)

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

    def replace_with_array(seq, unit):
      """
      Replace all the elements in the list with an equivalent `Array`
      with the given `unit`.
      """
      # No recursion needed for single values
      if not isinstance(seq, list):
        return Quantity(seq, unit=unit)

      def top_replace(s):
        """
        Recursively descend into the list.
        """
        for i in s:
          if not isinstance(i, list):
            yield Quantity(i, unit=unit)
          else:
            yield type(i)(top_replace(i))

      return type(seq)(top_replace(seq))

    return replace_with_array(self.value.tolist(), self.unit)

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
    return Quantity(self.value.transpose(*axes), unit=self.unit)

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
    return Quantity(self.value.tile(_as_jax_array_(reps)), unit=self.unit)

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

        >>> x = braincore.math.random.randn(4, 4)
        >>> x.size
       [4, 4]
        >>> y = x.view(16)
        >>> y.size
        [16]
        >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
        >>> z.size
        [2, 8]

        >>> a = braincore.math.random.randn(1, 2, 3, 4)
        >>> a.size
        [1, 2, 3, 4]
        >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
        >>> b.size
        [1, 3, 2, 4]
        >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
        >>> c.size
        [1, 3, 2, 4]
        >>> braincore.math.equal(b, c)
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

        >>> x = braincore.math.random.randn(4, 4)
        >>> x
        Array([[ 0.9482, -0.0310,  1.4999, -0.5316],
                [-0.1520,  0.7472,  0.5617, -0.8649],
                [-2.4724, -0.0334, -0.2976, -0.8499],
                [-0.2109,  1.9913, -0.9607, -0.6123]])
        >>> x.dtype
        braincore.math.float32

        >>> y = x.view(braincore.math.int32)
        >>> y
        tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
                [-1105482831,  1061112040,  1057999968, -1084397505],
                [-1071760287, -1123489973, -1097310419, -1084649136],
                [-1101533110,  1073668768, -1082790149, -1088634448]],
            dtype=braincore.math.int32)
        >>> y[0, 0] = 1000000000
        >>> x
        tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
                [-0.1520,  0.7472,  0.5617, -0.8649],
                [-2.4724, -0.0334, -0.2976, -0.8499],
                [-0.2109,  1.9913, -0.9607, -0.6123]])

        >>> x.view(braincore.math.cfloat)
        tensor([[ 0.0047-0.0310j,  1.4999-0.5316j],
                [-0.1520+0.7472j,  0.5617-0.8649j],
                [-2.4724-0.0334j, -0.2976-0.8499j],
                [-0.2109+1.9913j, -0.9607-0.6123j]])
        >>> x.view(braincore.math.cfloat).size
        [4, 2]

        >>> x.view(braincore.math.uint8)
        tensor([[  0, 202, 154,  59, 182, 243, 253, 188, 185, 252, 191,  63, 240,  22,
                   8, 191],
                [227, 165,  27, 190, 128,  72,  63,  63, 146, 203,  15,  63,  22, 106,
                  93, 191],
                [205,  59,  30, 192, 112, 206,   8, 189,   7,  95, 152, 190,  12, 147,
                  89, 191],
                [ 43, 246,  87, 190, 235, 226, 254,  63, 111, 240, 117, 191, 177, 191,
                  28, 191]], dtype=braincore.math.uint8)
        >>> x.view(braincore.math.uint8).size
        [4, 16]

    """
    if len(args) == 0:
      if dtype is None:
        raise ValueError('Provide dtype or shape.')
      else:
        return Quantity(self.value.view(dtype), unit=self.unit)
    else:
      if isinstance(args[0], int):  # shape
        if dtype is not None:
          raise ValueError('Provide one of dtype or shape. Not both.')
        return Quantity(self.value.reshape(*args), unit=self.unit)
      else:  # dtype
        assert not isinstance(args[0], int)
        assert dtype is None
        return Quantity(self.value.view(args[0]), unit=self.unit)

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

  # def __bool__(self) -> bool:
  #   return self.value.__bool__()

  def __float__(self):
    return self.value.__float__()

  #
  # def __int__(self):
  #   return self.value.__int__()
  #
  # def __complex__(self):
  #   return self.value.__complex__()
  #
  # def __hex__(self):
  #   assert self.ndim == 0, 'hex only works on scalar values'
  #   return hex(self.value)  # type: ignore
  #
  # def __oct__(self):
  #   assert self.ndim == 0, 'oct only works on scalar values'
  #   return oct(self.value)  # type: ignore

  def __index__(self):
    return operator.index(self.value)

  # ----------------------
  # PyTorch compatibility
  # ----------------------

  def unsqueeze(self, dim: int) -> 'Quantity':
    """
    Array.unsqueeze(dim) -> Array, or so called Tensor
    equals
    Array.expand_dims(dim)

    See :func:`braincore.math.unsqueeze`
    """
    return Quantity(jnp.expand_dims(self.value, dim), unit=self.unit)

  def expand_dims(self, axis: Union[int, Sequence[int]]) -> 'Quantity':
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
    return Quantity(jnp.expand_dims(self.value, axis), unit=self.unit)

  def expand_as(self, array: Union['Quantity', jax.Array, np.ndarray]) -> 'Quantity':
    """
    Expand an array to a shape of another array.

    Parameters
    ----------
    array : Quantity

    Returns
    -------
    expanded : Quantity
        A readonly view on the original array with the given shape of array. It is
        typically not contiguous. Furthermore, more than one element of a
        expanded array may refer to a single memory location.
    """
    return Quantity(jnp.broadcast_to(self.value, array), unit=self.unit)

  def pow(self, oc):
    return self.__pow__(oc)

  def addr(
      self,
      vec1: Union['Quantity', jax.Array, np.ndarray],
      vec2: Union['Quantity', jax.Array, np.ndarray],
      *,
      beta: float = 1.0,
      alpha: float = 1.0,
      out: Optional[Union['Quantity', jax.Array, np.ndarray]] = None
  ) -> Optional['Quantity']:
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
    try:
      out_unit = vec1.unit * vec2.unit
    except:
      out_unit = DIMENSIONLESS
    vec1 = _as_jax_array_(vec1)
    vec2 = _as_jax_array_(vec2)
    r = alpha * jnp.outer(vec1, vec2) + beta * self.value
    if out is None:
      return Quantity(r, unit=out_unit)
    else:
      _check_out(out)
      out.value = r
      out.unit = out_unit

  def addr_(
      self,
      vec1: Union['Quantity', jax.Array, np.ndarray],
      vec2: Union['Quantity', jax.Array, np.ndarray],
      *,
      beta: float = 1.0,
      alpha: float = 1.0
  ):
    vec1 = _as_jax_array_(vec1)
    vec2 = _as_jax_array_(vec2)
    r = alpha * jnp.outer(vec1, vec2) + beta * self.value
    self.value = r
    try:
      self.unit = vec1.unit * vec2.unit
    except:
      self.unit = DIMENSIONLESS
    return self

  def outer(self, other: Union['Quantity', jax.Array, np.ndarray]) -> 'Quantity':
    other = _as_jax_array_(other)
    return Quantity(jnp.outer(self.value, other.value), unit=self.unit * other.unit)

  def abs(self, *, out: Optional[Union['Quantity', jax.Array, np.ndarray]] = None) -> Optional['Quantity']:
    r = jnp.abs(self.value)
    if out is None:
      return Quantity(r, unit=self.unit)
    else:
      _check_out(out)
      out.value = r
      out.unit = self.unit

  def abs_(self):
    """
    in-place version of Array.abs()
    """
    self.value = jnp.abs(self.value)
    return self

  def add_(self, value):
    self.value += value
    return self

  def absolute(self, *, out: Optional[Union['Quantity', jax.Array, np.ndarray]] = None) -> Optional['Quantity']:
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
    if isinstance(value, Quantity):
      return Quantity(self.value * value.value, unit=self.unit * value.unit)
    else:
      return Quantity(self.value * value, unit=self.unit)

  def mul_(self, value):
    """
    In-place version of :meth:`~Array.mul`.
    """
    if isinstance(value, Quantity):
      self.value *= value.value
      self.unit *= value.unit
    else:
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

  def sin(self, *, out: Optional[Union['Quantity', jax.Array, np.ndarray]] = None) -> Optional['Quantity']:
    if self.is_dimensionless:
      r = jnp.sin(self.value)
      if out is None:
        return Quantity(r)
      else:
        _check_out(out)
        out.value = r
    else:
      raise ValueError("Cannot take sin of a Array with units.")

  def sin_(self):
    if self.is_dimensionless:
      self.value = jnp.sin(self.value)
      return self
    else:
      raise ValueError("Cannot take sin of a Array with units.")

  def cos_(self):
    if self.is_dimensionless:
      self.value = jnp.cos(self.value)
      return self
    else:
      raise ValueError("Cannot take cos of a Array with units.")

  def cos(self, *, out: Optional[Union['Quantity', jax.Array, np.ndarray]] = None) -> Optional['Quantity']:
    if self.is_dimensionless:
      r = jnp.cos(self.value)
      if out is None:
        return Quantity(r)
      else:
        _check_out(out)
        out.value = r
    else:
      raise ValueError("Cannot take cos of a Array with units.")

  def tan_(self):
    if self.is_dimensionless:
      self.value = jnp.tan(self.value)
      return self
    else:
      raise ValueError("Cannot take tan of a Array with units.")

  def tan(self, *, out: Optional[Union['Quantity', jax.Array, np.ndarray]] = None) -> Optional['Quantity']:
    if self.is_dimensionless:
      r = jnp.tan(self.value)
      if out is None:
        return Quantity(r)
      else:
        _check_out(out)
        out.value = r
    else:
      raise ValueError("Cannot take tan of a Array with units.")

  def sinh_(self):
    if self.is_dimensionless:
      self.value = jnp.sinh(self.value)
      return self
    else:
      raise ValueError("Cannot take sinh of a Array with units.")

  def sinh(self, *, out: Optional[Union['Quantity', jax.Array, np.ndarray]] = None) -> Optional['Quantity']:
    if self.is_dimensionless:
      r = jnp.tanh(self.value)
      if out is None:
        return Quantity(r)
      else:
        _check_out(out)
        out.value = r
    else:
      raise ValueError("Cannot take sinh of a Array with units.")

  def cosh_(self):
    if self.is_dimensionless:
      self.value = jnp.cosh(self.value)
      return self
    else:
      raise ValueError("Cannot take cosh of a Array with units.")

  def cosh(self, *, out: Optional[Union['Quantity', jax.Array, np.ndarray]] = None) -> Optional['Quantity']:
    if self.is_dimensionless:
      r = jnp.cosh(self.value)
      if out is None:
        return Quantity(r)
      else:
        _check_out(out)
        out.value = r
    else:
      raise ValueError("Cannot take cosh of a Array with units.")

  def tanh_(self):
    if self.is_dimensionless:
      self.value = jnp.tanh(self.value)
      return self
    else:
      raise ValueError("Cannot take tanh of a Array with units.")

  def tanh(self, *, out: Optional[Union['Quantity', jax.Array, np.ndarray]] = None) -> Optional['Quantity']:
    if self.is_dimensionless:
      r = jnp.tanh(self.value)
      if out is None:
        return Quantity(r)
      else:
        _check_out(out)
        out.value = r
    else:
      raise ValueError("Cannot take tanh of a Array with units.")

  def arcsin_(self):
    if self.is_dimensionless:
      self.value = jnp.arcsin(self.value)
      return self
    else:
      raise ValueError("Cannot take arcsin of a Array with units.")

  def arcsin(self, *, out: Optional[Union['Quantity', jax.Array, np.ndarray]] = None) -> Optional['Quantity']:
    if self.is_dimensionless:
      r = jnp.arcsin(self.value)
      if out is None:
        return Quantity(r)
      else:
        _check_out(out)
        out.value = r
    else:
      raise ValueError("Cannot take arcsin of a Array with units.")

  def arccos_(self):
    if self.is_dimensionless:
      self.value = jnp.arccos(self.value)
      return self
    else:
      raise ValueError("Cannot take arccos of a Array with units.")

  def arccos(self, *, out: Optional[Union['Quantity', jax.Array, np.ndarray]] = None) -> Optional['Quantity']:
    if self.is_dimensionless:
      r = jnp.arccos(self.value)
      if out is None:
        return Quantity(r)
      else:
        _check_out(out)
        out.value = r
    else:
      raise ValueError("Cannot take arccos of a Array with units.")

  def arctan_(self):
    if self.is_dimensionless:
      self.value = jnp.arctan(self.value)
      return self
    else:
      raise ValueError("Cannot take arctan of a Array with units.")

  def arctan(self, *, out: Optional[Union['Quantity', jax.Array, np.ndarray]] = None) -> Optional['Quantity']:
    if self.is_dimensionless:
      r = jnp.arctan(self.value)
      if out is None:
        return Quantity(r)
      else:
        _check_out(out)
        out.value = r
    else:
      raise ValueError("Cannot take arctan of a Array with units.")

  def clamp(
      self,
      min_value: Optional[Union['Quantity', jax.Array, np.ndarray]] = None,
      max_value: Optional[Union['Quantity', jax.Array, np.ndarray]] = None,
      *,
      out: Optional[Union['Quantity', jax.Array, np.ndarray]] = None
  ) -> Optional['Quantity']:
    """
    return the value between min_value and max_value,
    if min_value is None, then no lower bound,
    if max_value is None, then no upper bound.
    """
    min_value = _as_jax_array_(min_value)
    max_value = _as_jax_array_(max_value)
    r = jnp.clip(self.value, min_value, max_value)
    if out is None:
      return Quantity(r, unit=self.units)
    else:
      _check_out(out)
      out.value = r
      out.units = self.units

  def clamp_(self,
             min_value: Optional[Union['Quantity', jax.Array, np.ndarray]] = None,
             max_value: Optional[Union['Quantity', jax.Array, np.ndarray]] = None):
    """
    return the value between min_value and max_value,
    if min_value is None, then no lower bound,
    if max_value is None, then no upper bound.
    """
    self.clamp(min_value, max_value, out=self)
    return self

  def clip_(self,
            min_value: Optional[Union['Quantity', jax.Array, np.ndarray]] = None,
            max_value: Optional[Union['Quantity', jax.Array, np.ndarray]] = None):
    """
    alias for clamp_
    """
    self.value = self.clip(min_value, max_value, out=self)
    return self

  def clone(self) -> 'Quantity':
    return Quantity(self.value.copy(), unit=self.units)

  def copy_(self, src: Union['Quantity', jax.Array, np.ndarray]) -> 'Quantity':
    self.value = jnp.copy(_as_jax_array_(src))
    return self

  def cov_with(
      self,
      y: Optional[Union['Quantity', jax.Array, np.ndarray]] = None,
      rowvar: bool = True,
      bias: bool = False,
      ddof: Optional[int] = None,
      fweights: Union['Quantity', jax.Array, np.ndarray] = None,
      aweights: Union['Quantity', jax.Array, np.ndarray] = None
  ) -> 'Quantity':
    y = _as_jax_array_(y)
    fweights = _as_jax_array_(fweights)
    aweights = _as_jax_array_(aweights)
    r = jnp.cov(self.value, y, rowvar, bias, fweights, aweights)
    return Quantity(r)

  def expand(self, *sizes) -> 'Quantity':
    """
    Expand an array to a new shape.

    Parameters
    ----------
    sizes : tuple or int
        The shape of the desired array. A single integer ``i`` is interpreted
        as ``(i,)``.

    Returns
    -------
    expanded : Quantity
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
    return Quantity(jnp.broadcast_to(self.value, sizes_list), unit=self.unit)

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
    # TODO: Not finish braincore.math yet
    global random
    if random is None: from braincore import random
    self.value = random.uniform(low, high, self.shape)
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
    # TODO: Not finish braincore.math yet
    global random
    if random is None: from braincore import random
    self.value = random.lognormal(mean, std, self.shape)
    return self

  def normal_(self, ):
    """
    Fills self tensor with elements samples from the normal distribution parameterized by mean and std.
    """
    global random
    if random is None: from braincore import random
    self.value = random.randn(*self.shape)
    return self

  def cuda(self, deice=None):
    deice = jax.devices('cuda')[0] if deice is None else deice
    self.value = jax.device_put(self.value, deice)
    return self

  def cpu(self, device=None):
    device = jax.devices('cpu')[0] if device is None else device
    self.value = jax.device_put(self.value, device)
    return self

  # dtype exchanging #
  # ---------------- #

  def bool(self):
    return Quantity(jnp.asarray(self.value, dtype=jnp.bool_), unit=self.unit)

  def int(self):
    return Quantity(jnp.asarray(self.value, dtype=jnp.int32), unit=self.unit)

  def long(self):
    return Quantity(jnp.asarray(self.value, dtype=jnp.int64), unit=self.unit)

  def half(self):
    return Quantity(jnp.asarray(self.value, dtype=jnp.float16), unit=self.unit)

  def float(self):
    return Quantity(jnp.asarray(self.value, dtype=jnp.float32), unit=self.unit)

  def double(self):
    return Quantity(jnp.asarray(self.value, dtype=jnp.float64), unit=self.unit)


class Unit(Quantity):
  r"""
   A physical unit.

   Normally, you do not need to worry about the implementation of
   units. They are derived from the `Array` object with
   some additional information (name and string representation).

   Basically, a unit is just a number with given dimensions, e.g.
   mvolt = 0.001 with the dimensions of voltage. The units module
   defines a large number of standard units, and you can also define
   your own (see below).

   The unit class also keeps track of various things that were used
   to define it so as to generate a nice string representation of it.
   See below.

   When creating scaled units, you can use the following prefixes:

    ======     ======  ==============
    Factor     Name    Prefix
    ======     ======  ==============
    10^24      yotta   Y
    10^21      zetta   Z
    10^18      exa     E
    10^15      peta    P
    10^12      tera    T
    10^9       giga    G
    10^6       mega    M
    10^3       kilo    k
    10^2       hecto   h
    10^1       deka    da
    1
    10^-1      deci    d
    10^-2      centi   c
    10^-3      milli   m
    10^-6      micro   u (\mu in SI)
    10^-9      nano    n
    10^-12     pico    p
    10^-15     femto   f
    10^-18     atto    a
    10^-21     zepto   z
    10^-24     yocto   y
    ======     ======  ==============

  **Defining your own**

   It can be useful to define your own units for printing
   purposes. So for example, to define the newton metre, you
   write

   >>> from braincore import units as U
   >>> Nm = U.newton * U.metre

   You can then do

   >>> (1*Nm).in_unit(Nm)
   '1. N m'

   New "compound units", i.e. units that are composed of other units will be
   automatically registered and from then on used for display. For example,
   imagine you define total conductance for a membrane, and the total area of
   that membrane:

   >>> conductance = 10.*U.nS
   >>> area = 20000*U.um**2

   If you now ask for the conductance density, you will get an "ugly" display
   in basic SI dimensions, as Brian does not know of a corresponding unit:

   >>> conductance/area
   0.5 * metre ** -4 * kilogram ** -1 * second ** 3 * amp ** 2

   By using an appropriate unit once, it will be registered and from then on
   used for display when appropriate:

   >>> U.usiemens/U.cm**2
   usiemens / (cmetre ** 2)
   >>> conductance/area  # same as before, but now Brian knows about uS/cm^2
   50. * usiemens / (cmetre ** 2)

   Note that user-defined units cannot override the standard units (`volt`,
   `second`, etc.) that are predefined by Brian. For example, the unit
   ``Nm`` has the dimensions "length²·mass/time²", and therefore the same
   dimensions as the standard unit `joule`. The latter will be used for display
   purposes:

   >>> 3*U.joule
   3. * joule
   >>> 3*Nm
   3. * joule

  """
  __slots__ = ["_value", "_unit", "scale", "_dispname", "_name", "iscompound"]

  __array_priority__ = 1000

  automatically_register_units = True

  def __init__(
      self,
      value,
      unit=None,
      scale=0,
      name=None,
      dispname=None,
      iscompound=None,
      dtype=None,
  ):
    if unit is None:
      unit = DIMENSIONLESS
    if value != 10.0 ** scale:
      raise AssertionError(f"Unit value has to be 10**scale (scale={scale}, value={value})")
    # self._unit = unit
    # self._value = value

    # The scale for this unit (as the integer exponent of 10), i.e.
    # a scale of 3 means 10^3, for a "k" prefix.
    self.scale = scale
    if name is None:
      if unit is DIMENSIONLESS:
        name = "Unit(1)"
      else:
        name = repr(unit)
    # The full name of this unit
    self._name = name
    # The display name of this unit
    if dispname is None:
      dispname = name
    self._dispname = dispname
    # Whether this unit is a combination of other units
    self.iscompound = iscompound

    super().__init__(value, dtype=dtype, unit=unit)

    if Unit.automatically_register_units:
      register_new_unit(self)

  @staticmethod
  def create(unit, name, dispname, scale=0):
    """
    Create a new named unit.

    Parameters
    ----------
    dim : `Dimension`
        The dimensions of the unit.
    name : `str`
        The full name of the unit, e.g. ``'volt'``
    dispname : `str`
        The display name, e.g. ``'V'``
    scale : int, optional
        The scale of this unit as an exponent of 10, e.g. -3 for a unit that
        is 1/1000 of the base scale. Defaults to 0 (i.e. a base unit).

    Returns
    -------
    u : `Unit`
        The new unit.
    """
    name = str(name)
    dispname = str(dispname)

    u = Unit(
      10.0 ** scale,
      unit=unit,
      scale=scale,
      name=name,
      dispname=dispname,
    )

    return u

  @staticmethod
  def create_scaled_unit(baseunit, scalefactor):
    """
    Create a scaled unit from a base unit.

    Parameters
    ----------
    baseunit : `Unit`
        The unit of which to create a scaled version, e.g. ``volt``,
        ``amp``.
    scalefactor : `str`
        The scaling factor, e.g. ``"m"`` for mvolt, mamp

    Returns
    -------
    u : `Unit`
        The new unit.
    """
    name = scalefactor + baseunit.name
    dispname = scalefactor + baseunit.dispname
    scale = _siprefixes[scalefactor] + baseunit.scale

    u = Unit(
      10.0 ** scale,
      unit=baseunit.unit,
      name=name,
      dispname=dispname,
      scale=scale,
    )

    return u

  def set_name(self, name):
    """Sets the name for the unit.

    .. deprecated:: 2.1
        Create a new unit with `Unit.create` instead.
    """
    raise NotImplementedError(
      "Setting the name for a unit after"
      "its creation is no longer supported, use"
      "'Unit.create' to create a new unit."
    )

  def set_display_name(self, name):
    """Sets the display name for the unit.

    .. deprecated:: 2.1
        Create a new unit with `Unit.create` instead.
    """
    raise NotImplementedError(
      "Setting the display name for a unit after"
      "its creation is no longer supported, use"
      "'Unit.create' to create a new unit."
    )

  name = property(
    fget=lambda self: self._name, fset=set_name, doc="The name of the unit"
  )

  dispname = property(
    fget=lambda self: self._dispname,
    fset=set_display_name,
    doc="The display name of the unit",
  )

  def __repr__(self):
    return self.name

  def __str__(self):
    return self.dispname

  def __mul__(self, other):
    if isinstance(other, Unit):
      name = f"{self.name} * {other.name}"
      dispname = f"{self.dispname} * {other.dispname}"
      scale = self.scale + other.scale
      u = Unit(
        10.0 ** scale,
        unit=self.unit * other.unit,
        name=name,
        dispname=dispname,
        iscompound=True,
        scale=scale,
      )
      return u
    else:
      return super().__mul__(other)

  def __div__(self, other):
    if isinstance(other, Unit):
      if self.iscompound:
        dispname = f"({self.dispname})"
        name = f"({self.name})"
      else:
        dispname = self.dispname
        name = self.name
      dispname += "/"
      name += " / "
      if other.iscompound:
        dispname += f"({other.dispname})"
        name += f"({other.name})"
      else:
        dispname += other.dispname
        name += other.name

      scale = self.scale - other.scale
      u = Unit(
        10.0 ** scale,
        unit=self.unit / other.unit,
        name=name,
        dispname=dispname,
        scale=scale,
        iscompound=True,
      )
      return u
    else:
      return super().__div__(other)

  def __pow__(self, other):
    if is_scalar_type(other):
      if self.iscompound:
        dispname = f"({self.dispname})"
        name = f"({self.name})"
      else:
        dispname = self.dispname
        name = self.name
      dispname += f"^{str(other)}"
      name += f" ** {repr(other)}"
      scale = self.scale * other
      u = Unit(
        10.0 ** scale,
        unit=self.unit ** other,
        name=name,
        dispname=dispname,
        scale=scale,
        iscompound=True,
      )  # To avoid issues with units like (second ** -1) ** -1
      return u
    else:
      return super().__pow__(other)

  def __iadd__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __isub__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __imul__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __idiv__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __itruediv__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __ifloordiv__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __imod__(self, other):
    raise TypeError("Units cannot be modified in-place")

  def __ipow__(self, other, modulo=None):
    raise TypeError("Units cannot be modified in-place")

  def __eq__(self, other):
    if isinstance(other, Unit):
      return other.unit is self.unit and other.scale == self.scale
    else:
      return Quantity.__eq__(self, other)

  def __neq__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash((self.unit, self.scale))


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
    self.units_for_dimensions[u.unit][float(u)] = u

  def __getitem__(self, x):
    """Returns the best unit for array x

    The algorithm is to consider the value:

    m=abs(x/u)

    for all matching units u. We select the unit where this ratio is the
    closest to 10 (if it is an array with several values, we select the
    unit where the deviations from that are the smallest. More precisely,
    the unit that minimizes the sum of (log10(m)-1)**2 over all entries).
    """
    matching = self.units_for_dimensions.get(x.unit, {})
    if len(matching) == 0:
      raise KeyError("Unit not found in registry.")

    matching_values = np.array(list(matching.keys()))
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
      x_flat = np.array(x).flatten()
    floatreps = np.tile(np.abs(x_flat), (len(matching), 1)).T / matching_values
    # ignore zeros, they are well represented in any unit
    floatreps[floatreps == 0] = np.nan
    if np.all(np.isnan(floatreps)):
      return matching[1.0]  # all zeros, use the base unit

    deviations = np.nansum((np.log10(floatreps) - 1) ** 2, axis=0)
    return list(matching.values())[deviations.argmin()]


def register_new_unit(u):
  """Register a new unit for automatic displaying of arrays

  Parameters
  ----------
  u : `Unit`
      The unit that should be registered.

  Examples
  --------
  >>> from braincore.units import *
  >>> 2.0*farad/metre**2
  2. * metre ** -4 * kilogram ** -1 * second ** 4 * amp ** 2
  >>> register_new_unit(pfarad / mmetre**2)
  >>> 2.0*farad/metre**2
  2000000. * pfarad / (mmetre ** 2)
  """
  user_unit_register.add(u)


#: `UnitRegistry` containing all the standard units (metre, kilogram, um2...)
standard_unit_register = UnitRegistry()
#: `UnitRegistry` containing additional units (newton*metre, farad / metre, ...)
additional_unit_register = UnitRegistry()
#: `UnitRegistry` containing all units defined by the user
user_unit_register = UnitRegistry()


def get_unit(d):
  """
  Find an unscaled unit (e.g. `volt` but not `mvolt`) for a `Dimension`.

  Parameters
  ----------
  d : `Dimension`
      The dimension to find a unit for.

  Returns
  -------
  u : `Unit`
      A registered unscaled `Unit` for the dimensions ``d``, or a new `Unit`
      if no unit was found.
  """
  for unit_register in [
    standard_unit_register,
    user_unit_register,
    additional_unit_register,
  ]:
    if 1.0 in unit_register.units_for_dimensions[d]:
      return unit_register.units_for_dimensions[d][1.0]
  return Unit(1.0, unit=d)


def get_unit_for_display(d):
  """
  Return a string representation of an appropriate unscaled unit or ``'1'``
  for a dimensionless array.

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


def check_units(**au):
  """Decorator to check units of arguments passed to a function

  Examples
  --------
  >>> from braincore.units import *
  >>> @check_units(I=amp, R=ohm, wibble=metre, result=volt)
  ... def getvoltage(I, R, **k):
  ...     return I*R

  You don't have to check the units of every variable in the function, and
  you can define what the units should be for variables that aren't
  explicitly named in the definition of the function. For example, the code
  above checks that the variable wibble should be a length, so writing

  >>> getvoltage(1*amp, 1*ohm, wibble=1)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  ...
  DimensionMismatchError: Function "getvoltage" variable "wibble" has wrong dimensions, dimensions were (1) (m)

  fails, but

  >>> getvoltage(1*amp, 1*ohm, wibble=1*metre)
  1. * volt

  passes. String arguments or ``None`` are not checked

  >>> getvoltage(1*amp, 1*ohm, wibble='hello')
  1. * volt

  By using the special name ``result``, you can check the return value of the
  function.

  You can also use ``1`` or ``bool`` as a special value to check for a
  unitless number or a boolean value, respectively:

  >>> @check_units(value=1, absolute=bool, result=bool)
  ... def is_high(value, absolute=False):
  ...     if absolute:
  ...         return abs(value) >= 5
  ...     else:
  ...         return value >= 5

  This will then again raise an error if the argument if not of the expected
  type:

  >>> is_high(7)
  True
  >>> is_high(-7, True)
  True
  >>> is_high(3, 4)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  ...
  TypeError: Function "is_high" expected a boolean value for argument "absolute" but got 4.

  If the return unit depends on the unit of an argument, you can also pass
  a function that takes the units of all the arguments as its inputs (in the
  order specified in the function header):

  >>> @check_units(result=lambda d: d**2)
  ... def square(value):
  ...     return value**2

  If several arguments take arbitrary units but they have to be
  consistent among each other, you can state the name of another argument as
  a string to state that it uses the same unit as that argument.

  >>> @check_units(summand_1=None, summand_2='summand_1')
  ... def multiply_sum(multiplicand, summand_1, summand_2):
  ...     "Calculates multiplicand*(summand_1 + summand_2)"
  ...     return multiplicand*(summand_1 + summand_2)
  >>> multiply_sum(3, 4*mV, 5*mV)
  27. * mvolt
  >>> multiply_sum(3*nA, 4*mV, 5*mV)
  27. * pwatt
  >>> multiply_sum(3*nA, 4*mV, 5*nA)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  ...
  DimensionMismatchError: Function 'multiply_sum' expected the same arguments for arguments 'summand_1', 'summand_2', but argument 'summand_1' has unit V, while argument 'summand_2' has unit A.

  Raises
  ------

  DimensionMismatchError
      In case the input arguments or the return value do not have the
      expected dimensions.
  TypeError
      If an input argument or return value was expected to be a boolean but
      is not.

  Notes
  -----
  This decorator will destroy the signature of the original function, and
  replace it with the signature ``(*args, **kwds)``. Other decorators will
  do the same thing, and this decorator critically needs to know the signature
  of the function it is acting on, so it is important that it is the first
  decorator to act on a function. It cannot be used in combination with
  another decorator that also needs to know the signature of the function.

  Note that the ``bool`` type is "strict", i.e. it expects a proper
  boolean value and does not accept 0 or 1. This is not the case the other
  way round, declaring an argument or return value as "1" *does* allow for a
  ``True`` or ``False`` value.
  """

  def do_check_units(f):
    def new_f(*args, **kwds):
      newkeyset = kwds.copy()
      arg_names = f.__code__.co_varnames[0: f.__code__.co_argcount]
      for n, v in zip(arg_names, args[0: f.__code__.co_argcount]):
        if (
            not isinstance(v, (Quantity, str, bool))
            and v is not None
            and n in au
        ):
          try:
            # allow e.g. to pass a Python list of values
            v = Quantity(v)
          except TypeError:
            if have_same_dimensions(au[n], 1):
              raise TypeError(
                f"Argument {n} is not a unitless value/array."
              )
            else:
              raise TypeError(
                f"Argument '{n}' is not a array, "
                "expected a array with dimensions "
                f"{au[n]}"
              )
        newkeyset[n] = v

      for k in newkeyset:
        # string variables are allowed to pass, the presumption is they
        # name another variable. None is also allowed, useful for
        # default parameters
        if (
            k in au
            and not isinstance(newkeyset[k], str)
            and not newkeyset[k] is None
            and not au[k] is None
        ):
          if au[k] == bool:
            if not isinstance(newkeyset[k], bool):
              value = newkeyset[k]
              error_message = (
                f"Function '{f.__name__}' "
                "expected a boolean value "
                f"for argument '{k}' but got "
                f"'{value}'"
              )
              raise TypeError(error_message)
          elif isinstance(au[k], str):
            if not au[k] in newkeyset:
              error_message = (
                f"Function '{f.__name__}' "
                "expected its argument to have the "
                f"same units as argument '{k}', but "
                "there is no argument of that name"
              )
              raise TypeError(error_message)
            if not have_same_dimensions(newkeyset[k], newkeyset[au[k]]):
              d1 = get_dimensions(newkeyset[k])
              d2 = get_dimensions(newkeyset[au[k]])
              error_message = (
                f"Function '{f.__name__}' expected "
                f"the argument '{k}' to have the same "
                f"units as argument '{au[k]}', but "
                f"argument '{k}' has "
                f"unit {get_unit_for_display(d1)}, "
                f"while argument '{au[k]}' "
                f"has unit {get_unit_for_display(d2)}."
              )
              raise DimensionMismatchError(error_message)
          elif not have_same_dimensions(newkeyset[k], au[k]):
            unit = repr(au[k])
            value = newkeyset[k]
            error_message = (
              f"Function '{f.__name__}' "
              "expected a array with unit "
              f"{unit} for argument '{k}' but got "
              f"'{value}'"
            )
            raise DimensionMismatchError(
              error_message, get_dimensions(newkeyset[k])
            )

      result = f(*args, **kwds)
      if "result" in au:
        if isinstance(au["result"], Callable) and au["result"] != bool:
          expected_result = au["result"](*[get_dimensions(a) for a in args])
        else:
          expected_result = au["result"]
        if au["result"] == bool:
          if not isinstance(result, bool):
            error_message = (
              "The return value of function "
              f"'{f.__name__}' was expected to be "
              "a boolean value, but was of type "
              f"{type(result)}"
            )
            raise TypeError(error_message)
        elif not have_same_dimensions(result, expected_result):
          unit = get_unit_for_display(expected_result)
          error_message = (
            "The return value of function "
            f"'{f.__name__}' was expected to have "
            f"unit {unit} but was "
            f"'{result}'"
          )
          raise DimensionMismatchError(error_message, get_dimensions(result))
      return result

    new_f._orig_func = f
    new_f.__doc__ = f.__doc__
    new_f.__name__ = f.__name__
    # store the information in the function, necessary when using the
    # function in expressions or equations
    if hasattr(f, "_orig_arg_names"):
      arg_names = f._orig_arg_names
    else:
      arg_names = f.__code__.co_varnames[: f.__code__.co_argcount]
    new_f._arg_names = arg_names
    new_f._arg_units = [au.get(name, None) for name in arg_names]
    return_unit = au.get("result", None)
    if return_unit is None:
      new_f._return_unit = None
    else:
      new_f._return_unit = return_unit
    if return_unit == bool:
      new_f._returns_bool = True
    else:
      new_f._returns_bool = False
    new_f._orig_arg_names = arg_names

    # copy any annotation attributes
    if hasattr(f, "_annotation_attributes"):
      for attrname in f._annotation_attributes:
        setattr(new_f, attrname, getattr(f, attrname))
    new_f._annotation_attributes = getattr(f, "_annotation_attributes", []) + [
      "_arg_units",
      "_arg_names",
      "_return_unit",
      "_orig_func",
      "_returns_bool",
    ]
    return new_f

  return do_check_units