# -*- coding: utf-8 -*-


"""
All the basic classes for the ``brainpy.core``.

The basic classes include:

- ``Module``: The base class for all the objects in the ecosystem.
- ``module_list``: A sequence of :py:class:`~.Module`.
- ``module_dict``: A dictionary of :py:class:`~.Module`.
- ``Sequential``: The class for a sequential of modules, which update the modules sequentially.
- ``ModuleGroup``: The class for a group of modules, which update ``Projection`` first,
                   then ``Dynamics``, finally others.

For handling dynamical systems:

- ``Projection``: The class for the synaptic projection.
- ``Dynamics``: The class for the dynamical system.

For handling the delays:

- ``StateDelay``: The class for the state delay.
- ``DataDelay``: The class for the data delay.
- ``DelayAccess``: The class for the delay access.

"""

import inspect
import math
import numbers
from collections import namedtuple
from functools import partial
from typing import Sequence, Any, Tuple, Union, Dict, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from brainpy.core import environ, share
from .math import get_dtype
from .mixin import Mixin, Mode, ParamDesc, AllOfTypes, Batching, UpdateReturn
from ._state import State, StateStack
from ._utils import unique_name, Stack, jit_error, get_unique_name


Shape = Union[int, Sequence[int]]
PyTree = Any
ArrayLike = jax.typing.ArrayLike

delay_identifier = '_*_delay_of_'
ROTATE_UPDATE = 'rotation'
CONCAT_UPDATE = 'concat'

StateLoadResult = namedtuple('StateLoadResult', ['missing_keys', 'unexpected_keys'])

# the maximum order
_max_order = 10

__all__ = [
  # basic classes
  'Module', 'module_list', 'module_dict', 'ModuleGroup', 'Sequential',

  # dynamical systems
  'Projection', 'Dynamics',

  # delay handling
  'Delay', 'DelayAccess',

  # helper functions
  'call_order',

  # state processing
  'init_states', 'load_states', 'save_states',
]


class Module(object):
  """
  The Module class for the whole ecosystem.

  The ``Module`` is the base class for all the objects in the ecosystem. It
  provides the basic functionalities for the objects, including:

  - ``states()``: Collect all states in this node and the children nodes.
  - ``nodes()``: Collect all children nodes.
  - ``update()``: The function to specify the updating rule.
  - ``init_state()``: State initialization function.
  - ``save_state()``: Save states as a dictionary.
  - ``load_state()``: Load states from the external objects.

  """

  __module__ = 'brainpy.core'

  # the excluded states
  _invisible_states = ()

  # the excluded nodes
  _invisible_nodes = ()

  # # the supported computing modes
  # supported_modes: Optional[Sequence[Mode]] = None

  def __init__(self, name: str = None, mode: Mode = None):
    super().__init__()

    # check whether the object has a unique name.
    self._name = unique_name(self=self, name=name)

    # mode setting
    self._mode = None
    self.mode = mode if mode is not None else environ.get('mode')

  def __repr__(self):
    return f'{self.__class__.__name__}({self.name}, mode={self.mode})'

  @property
  def name(self):
    """Name of the model."""
    return self._name

  @name.setter
  def name(self, name: str = None):
    raise AttributeError('The name of the model is read-only.')

  @property
  def mode(self):
    """Mode of the model, which is useful to control the multiple behaviors of the model."""
    return self._mode

  @mode.setter
  def mode(self, value):
    if not isinstance(value, Mode):
      raise ValueError(f'Must be instance of {Mode.__name__}, '
                       f'but we got {type(value)}: {value}')
    self._mode = value

  def states(self, method: str = 'absolute', level: int = -1, include_self: bool = True) -> StateStack:
    """
    Collect all states in this node and the children nodes.

    Parameters
    ----------
    method : str
      The method to access the variables.
    level: int
      The hierarchy level to find variables.
    include_self: bool
      Whether include the variables in the self.

    Returns
    -------
    states : StateStack
      The collection contained (the path, the variable).
    """
    nodes = self.nodes(method=method, level=level, include_self=include_self)

    # state stacks
    states = StateStack()
    for node_path, node in nodes.items():
      for k in node.__dict__.keys():
        if k in node._invisible_states:
          continue
        v = getattr(node, k)
        if isinstance(v, State):
          states[f'{node_path}.{k}' if node_path else k] = v
    return states

  def nodes(self, method='absolute', level=-1, include_self=True) -> Stack:
    """
    Collect all children nodes.

    Parameters
    ----------
    method : str
      The method to access the nodes.
    level: int
      The hierarchy level to find nodes.
    include_self: bool
      Whether include the self.

    Returns
    -------
    gather : Stack
      The collection contained (the path, the node).
    """
    return _find_nodes(self, method=method, level=level, include_self=include_self)

  def update(self, *args, **kwargs):
    """
    The function to specify the updating rule.
    """
    raise NotImplementedError(f'Subclass of {self.__class__.__name__} must '
                              f'implement "update" function.')

  def __call__(self, *args, **kwargs):
    return self.update(*args, **kwargs)

  def __rrshift__(self, other):
    """
    Support using right shift operator to call modules.

    Examples
    --------

    >>> import brainpy.core as bc
    >>> x = bc.random.rand((10, 10))
    >>> l = bp.dnn.Activation(jax.numpy.tanh)
    >>> y = x >> l
    """
    return self.__call__(other)

  def init_state(self, *args, **kwargs):
    """
    State initialization function.
    """
    pass

  def save_state(self, **kwargs) -> Dict:
    """Save states as a dictionary. """
    return self.states(include_self=True, level=0, method='absolute')

  def load_state(self, state_dict: Dict, **kwargs) -> Optional[Tuple[Sequence[str], Sequence[str]]]:
    """Load states from the external objects."""
    variables = self.states(include_self=True, level=0, method='absolute')
    keys1 = set(state_dict.keys())
    keys2 = set(variables.keys())
    for key in keys2.intersection(keys1):
      variables[key].value = jax.numpy.asarray(state_dict[key])
    unexpected_keys = list(keys1 - keys2)
    missing_keys = list(keys2 - keys1)
    return unexpected_keys, missing_keys


def _find_nodes(self, method='absolute', level=-1, include_self=True, _lid=0, _edges=None) -> Stack:
  if _edges is None:
    _edges = set()
  gather = Stack()
  if include_self:
    if method == 'absolute':
      gather[self.name] = self
    elif method == 'relative':
      gather[''] = self
    else:
      raise ValueError(f'No support for the method of "{method}".')
  if (level > -1) and (_lid >= level):
    return gather
  if method == 'absolute':
    nodes = []
    for k, v in self.__dict__.items():
      if k not in self._invisible_nodes:
        continue
      if isinstance(v, Module):
        _add_node_absolute(self, v, _edges, gather, nodes)
      elif isinstance(v, module_list):
        for v2 in v:
          _add_node_absolute(self, v2, _edges, gather, nodes)
      elif isinstance(v, module_dict):
        for v2 in v.values():
          if isinstance(v2, Module):
            _add_node_absolute(self, v2, _edges, gather, nodes)

    # finding nodes recursively
    for v in nodes:
      gather.update(_find_nodes(v,
                                method=method,
                                level=level,
                                _lid=_lid + 1,
                                _edges=_edges,
                                include_self=include_self))

  elif method == 'relative':
    nodes = []
    for k, v in self.__dict__.items():
      if isinstance(v, Module):
        _add_node_relative(self, k, v, _edges, gather, nodes)
      elif isinstance(v, module_list):
        for i, v2 in enumerate(v):
          _add_node_relative(self, f'{k}-list:{i}', v2, _edges, gather, nodes)
      elif isinstance(v, module_dict):
        for k2, v2 in v.items():
          if isinstance(v2, Module):
            _add_node_relative(self, f'{k}-dict:{k2}', v2, _edges, gather, nodes)

    # finding nodes recursively
    for k1, v1 in nodes:
      for k2, v2 in _find_nodes(v1,
                                method=method,
                                _edges=_edges,
                                _lid=_lid + 1,
                                level=level,
                                include_self=include_self).items():
        if k2:
          gather[f'{k1}.{k2}'] = v2

  else:
    raise ValueError(f'No support for the method of "{method}".')
  return gather


def _add_node_absolute(self, v, _paths, gather, nodes):
  path = (id(self), id(v))
  if path not in _paths:
    _paths.add(path)
    gather[v.name] = v
    nodes.append(v)


def _add_node_relative(self, k, v, _paths, gather, nodes):
  path = (id(self), id(v))
  if path not in _paths:
    _paths.add(path)
    gather[k] = v
    nodes.append((k, v))


class Projection(Module):
  """
  Base class to model synaptic projections.
  """

  __module__ = 'brainpy.core'

  def update(self, *args, **kwargs):
    nodes = tuple(self.nodes(level=1, include_self=False).values())
    if len(nodes):
      for node in nodes:
        node(*args, **kwargs)
    else:
      raise ValueError('Do not implement the update() function.')


class module_list(list):
  """
  A sequence of :py:class:`~.Module`, which is compatible with
  :py:func:`~.vars()` and :py:func:`~.nodes()` operations in a :py:class:`~.Module`.

  That is to say, any nodes that are wrapped into :py:class:`~.NodeList` will be automatically
  retieved when using :py:func:`~.nodes()` function.

  >>> import brainpy.core as bc
  >>> l = bc.module_list([bp.dnn.Dense(1, 2),
  >>>                   bp.dnn.LSTMCell(2, 3)])
  """

  __module__ = 'brainpy.core'

  def __init__(self, seq=()):
    super().__init__()
    self.extend(seq)

  def append(self, element) -> 'module_list':
    if isinstance(element, State):
      raise TypeError(f'Cannot append a state into a node list. ')
    super().append(element)
    return self

  def extend(self, iterable) -> 'module_list':
    for element in iterable:
      self.append(element)
    return self


class module_dict(dict):
  """
  A dictionary of :py:class:`~.Module`, which is compatible with
  :py:func:`.vars()` operation in a :py:class:`~.Module`.

  """

  __module__ = 'brainpy.core'

  def __init__(self, *args, check_unique: bool = False, **kwargs):
    super().__init__()
    self.check_unique = check_unique
    self.update(*args, **kwargs)

  def update(self, *args, **kwargs) -> 'module_dict':
    for arg in args:
      if isinstance(arg, dict):
        for k, v in arg.items():
          self[k] = v
      elif isinstance(arg, tuple):
        assert len(arg) == 2
        self[arg[0]] = args[1]
    for k, v in kwargs.items():
      self[k] = v
    return self

  def __setitem__(self, key, value) -> 'module_dict':
    if self.check_unique:
      exist = self.get(key, None)
      if id(exist) != id(value):
        raise KeyError(f'Duplicate usage of key "{key}". "{key}" has been used for {value}.')
    super().__setitem__(key, value)
    return self


class ReceiveInputProj(Mixin):
  """
  The :py:class:`~.Mixin` that receives the input projections.

  Note that the subclass should define a ``cur_inputs`` attribute. Otherwise,
  the input function utilities cannot be used.

  """
  _current_inputs: Optional[module_dict]
  _delta_inputs: Optional[module_dict]

  @property
  def current_inputs(self):
    """
    The current inputs of the model. It should be a dictionary of the input data.
    """
    return self._current_inputs

  @property
  def delta_inputs(self):
    """
    The delta inputs of the model. It should be a dictionary of the input data.
    """

    return self._delta_inputs

  def add_input_fun(self, key: str, fun: Callable, label: Optional[str] = None, category: str = 'current'):
    """Add an input function.

    Args:
      key: str. The dict key.
      fun: Callable. The function to generate inputs.
      label: str. The input label.
      category: str. The input category, should be ``current`` (the current) or
         ``delta`` (the delta synapse, indicating the delta function).
    """
    if not callable(fun):
      raise TypeError('Must be a function.')

    key = _input_label_repr(key, label)
    if category == 'current':
      if self._current_inputs is None:
        self._current_inputs = module_dict()
      if key in self._current_inputs:
        raise ValueError(f'Key "{key}" has been defined and used.')
      self._current_inputs[key] = fun

    elif category == 'delta':
      if self._delta_inputs is None:
        self._delta_inputs = module_dict()
      if key in self._delta_inputs:
        raise ValueError(f'Key "{key}" has been defined and used.')
      self._delta_inputs[key] = fun

    else:
      raise NotImplementedError(f'Unknown category: {category}. Only support "current" and "delta".')

  def get_input_fun(self, key: str):
    """Get the input function.

    Args:
      key: str. The key.

    Returns:
      The input function which generates currents.
    """
    if self._current_inputs is not None and key in self._current_inputs:
      return self._current_inputs[key]

    elif self._delta_inputs is not None and key in self._delta_inputs:
      return self._delta_inputs[key]

    else:
      raise ValueError(f'Unknown key: {key}')

  def sum_current_inputs(self, *args, init: Any = 0., label: Optional[str] = None, **kwargs):
    """
    Summarize all current inputs by the defined input functions ``.current_inputs``.

    Args:
      *args: The arguments for input functions.
      init: The initial input data.
      label: str. The input label.
      **kwargs: The arguments for input functions.

    Returns:
      The total currents.
    """
    if self._current_inputs is None:
      return init
    if label is None:
      for key, out in self._current_inputs.items():
        init = init + out(*args, **kwargs)
    else:
      label_repr = _input_label_start(label)
      for key, out in self._current_inputs.items():
        if key.startswith(label_repr):
          init = init + out(*args, **kwargs)
    return init

  def sum_delta_inputs(self, *args, init: Any = 0., label: Optional[str] = None, **kwargs):
    """
    Summarize all delta inputs by the defined input functions ``.delta_inputs``.

    Args:
      *args: The arguments for input functions.
      init: The initial input data.
      label: str. The input label.
      **kwargs: The arguments for input functions.

    Returns:
      The total currents.
    """
    if self._delta_inputs is None:
      return init
    if label is None:
      for key, out in self._delta_inputs.items():
        init = init + out(*args, **kwargs)
    else:
      label_repr = _input_label_start(label)
      for key, out in self._delta_inputs.items():
        if key.startswith(label_repr):
          init = init + out(*args, **kwargs)
    return init


class Container(Mixin):
  """Container :py:class:`~.MixIn` which wrap a group of objects.
  """
  children: module_dict

  def __getitem__(self, item):
    """Overwrite the slice access (`self['']`). """
    if item in self.children:
      return self.children[item]
    else:
      raise ValueError(f'Unknown item {item}, we only found {list(self.children.keys())}')

  def __getattr__(self, item):
    """Overwrite the dot access (`self.`). """
    children = super().__getattribute__('children')
    if item == 'children':
      return children
    else:
      if item in children:
        return children[item]
      else:
        return super().__getattribute__(item)

  def __repr__(self):
    cls_name = self.__class__.__name__
    indent = ' ' * len(cls_name)
    child_str = [_repr_context(repr(val), indent) for val in self.children.values()]
    string = ", \n".join(child_str)
    return f'{cls_name}({string})'

  def __get_elem_name(self, elem):
    if isinstance(elem, Module):
      return elem.name
    else:
      return get_unique_name('ContainerElem')

  def format_elements(self, child_type: type, *children_as_tuple, **children_as_dict):
    res = dict()

    # add tuple-typed components
    for module in children_as_tuple:
      if isinstance(module, child_type):
        res[self.__get_elem_name(module)] = module
      elif isinstance(module, (list, tuple)):
        for m in module:
          if not isinstance(m, child_type):
            raise TypeError(f'Should be instance of {child_type.__name__}. '
                            f'But we got {type(m)}')
          res[self.__get_elem_name(m)] = m
      elif isinstance(module, dict):
        for k, v in module.items():
          if not isinstance(v, child_type):
            raise TypeError(f'Should be instance of {child_type.__name__}. '
                            f'But we got {type(v)}')
          res[k] = v
      else:
        raise TypeError(f'Cannot parse sub-systems. They should be {child_type.__name__} '
                        f'or a list/tuple/dict of {child_type.__name__}.')
    # add dict-typed components
    for k, v in children_as_dict.items():
      if not isinstance(v, child_type):
        raise TypeError(f'Should be instance of {child_type.__name__}. '
                        f'But we got {type(v)}')
      res[k] = v
    return res

  def add_elem(self, *elems, **elements):
    """
    Add new elements.

    >>> obj = Container()
    >>> obj.add_elem(a=1.)

    Args:
      elements: children objects.
    """
    self.children.update(self.format_elements(object, *elems, **elements))


class ExtendedUpdateWithBA(Module):
  """
  The extended update with before and after updates.
  """

  _before_updates: Optional[module_dict]
  _after_updates: Optional[module_dict]

  def __init__(self, *args, **kwargs):

    # -- Attribute for "BeforeAfterMixIn" -- #
    # the before- / after-updates used for computing
    self._before_updates: Optional[Dict[str, Callable]] = None
    self._after_updates: Optional[Dict[str, Callable]] = None

    super().__init__(*args, **kwargs)

  @property
  def before_updates(self):
    """
    The before updates of the model. It should be a dictionary of the updating functions.
    """
    return self._before_updates

  @property
  def after_updates(self):
    """
    The after updates of the model. It should be a dictionary of the updating functions.
    """
    return self._after_updates

  def add_before_update(self, key: Any, fun: Callable):
    """
    Add the before update into this node.
    """
    if self._before_updates is None:
      self._before_updates = module_dict()
    if key in self.before_updates:
      raise KeyError(f'{key} has been registered in before_updates of {self}')
    self.before_updates[key] = fun

  def add_after_update(self, key: Any, fun: Callable):
    """Add the after update into this node"""
    if self._after_updates is None:
      self._after_updates = module_dict()
    if key in self.after_updates:
      raise KeyError(f'{key} has been registered in after_updates of {self}')
    self.after_updates[key] = fun

  def get_before_update(self, key: Any):
    """Get the before update of this node by the given ``key``."""
    if self._before_updates is None:
      raise KeyError(f'{key} is not registered in before_updates of {self}')
    if key not in self.before_updates:
      raise KeyError(f'{key} is not registered in before_updates of {self}')
    return self.before_updates.get(key)

  def get_after_update(self, key: Any):
    """Get the after update of this node by the given ``key``."""
    if self._after_updates is None:
      raise KeyError(f'{key} is not registered in after_updates of {self}')
    if key not in self.after_updates:
      raise KeyError(f'{key} is not registered in after_updates of {self}')
    return self.after_updates.get(key)

  def has_before_update(self, key: Any):
    """Whether this node has the before update of the given ``key``."""
    if self._before_updates is None:
      return False
    return key in self.before_updates

  def has_after_update(self, key: Any):
    """Whether this node has the after update of the given ``key``."""
    if self._after_updates is None:
      return False
    return key in self.after_updates

  def __call__(self, *args, **kwargs):
    """The shortcut to call ``update`` methods."""

    # ``before_updates``
    if self.before_updates is not None:
      for model in self.before_updates.values():
        if hasattr(model, '_receive_update_input'):
          model(*args, **kwargs)
        else:
          model()

    # update the model self
    ret = self.update(*args, **kwargs)

    # ``after_updates``
    if self.after_updates is not None:
      for model in self.after_updates.values():
        if hasattr(model, '_not_receive_update_output'):
          model()
        else:
          model(ret)
    return ret


class Dynamics(ExtendedUpdateWithBA, ReceiveInputProj, UpdateReturn):
  """
  Dynamical System class.

  .. note::
     In general, every instance of :py:class:`~.Module` implemented in
     BrainPy only defines the evolving function at each time step :math:`t`.

     If users want to define the logic of running models across multiple steps,
     we recommend users to use :py:func:`~.for_loop`, :py:class:`~.LoopOverTime`,
     :py:class:`~.DSRunner`, or :py:class:`~.DSTrainer`.

     To be compatible with previous APIs, :py:class:`~.Module` inherits
     from the :py:class:`~.DelayRegister`. It's worthy to note that the methods of
     :py:class:`~.DelayRegister` will be removed in the future, including:

     - ``.register_delay()``
     - ``.get_delay_data()``
     - ``.update_local_delays()``
     - ``.reset_local_delays()``


  There are several essential attributes:

  - ``size``: the geometry of the neuron group. For example, `(10, )` denotes a line of
    neurons, `(10, 10)` denotes a neuron group aligned in a 2D space, `(10, 15, 4)` denotes
    a 3-dimensional neuron group.
  - ``num``: the flattened number of neurons in the group. For example, `size=(10, )` => \
    `num=10`, `size=(10, 10)` => `num=100`, `size=(10, 15, 4)` => `num=600`.

  Args:
    size: The neuron group geometry.
    name: The name of the dynamic system.
    keep_size: Whether keep the geometry information.
    mode: The computing mode.
  """

  __module__ = 'brainpy.core'

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
      method: str = 'exp_auto'
  ):
    # size
    if isinstance(size, (list, tuple)):
      if len(size) <= 0:
        raise ValueError(f'size must be int, or a tuple/list of int. '
                         f'But we got {type(size)}')
      if not isinstance(size[0], (int, np.integer)):
        raise ValueError('size must be int, or a tuple/list of int.'
                         f'But we got {type(size)}')
      size = tuple(size)
    elif isinstance(size, (int, np.integer)):
      size = (size,)
    else:
      raise ValueError('size must be int, or a tuple/list of int.'
                       f'But we got {type(size)}')
    self.size = size
    self.keep_size = keep_size

    # number of neurons
    self.num = np.prod(size)

    # integration method
    self.method = method

    # -- Attribute for "InputProjMixIn" -- #
    # each instance of "SupportInputProj" should have
    # "_current_inputs" and "_delta_inputs" attributes
    self._current_inputs: Optional[Dict[str, Callable]] = None
    self._delta_inputs: Optional[Dict[str, Callable]] = None

    # initialize
    super().__init__(name=name, mode=mode)

  @property
  def varshape(self):
    """The shape of variables in the neuron group."""
    return self.size if self.keep_size else (self.num,)

  def __repr__(self):
    return f'{self.name}(mode={self.mode}, size={self.size})'

  def update_return_info(self) -> PyTree:
    raise NotImplementedError(f'Subclass of {self.__class__.__name__}'
                              'must implement "update_return_info" function.')

  def update_return(self) -> PyTree:
    raise NotImplementedError(f'Subclass of {self.__class__.__name__}'
                              'must implement "update_return" function.')

  def register_return_delay(
      self,
      delay_name: str,
      delay_time: ArrayLike = None,
      delay_step: ArrayLike = None,
  ):
    """Register local relay at the given delay time.

    Args:
      delay_name: str. The name of the current delay data.
      delay_time: The delay time. Float.
      delay_step: The delay step. Int. ``delay_step`` and ``delay_time`` are exclusive. ``delay_step = delay_time / dt``.
    """
    if not self.has_after_update(delay_identifier):
      # add a model to receive the return of the target model
      model = Delay(self.update_return_info())
      # register the model
      self.add_after_update(delay_identifier, model)
    delay_cls: Delay = self.get_after_update(delay_identifier)
    delay_cls.register_entry(delay_name, delay_time=delay_time, delay_step=delay_step)
    return delay_cls

  def get_return_delay_at(self, delay_name):
    """Get the state delay at the given identifier (`name`).

    See also :py:meth:`~.Module.register_state_delay`.

    Args:
      delay_name: The identifier of the delay.

    Returns:
      The delayed data at the given delay position.
    """
    return self.get_after_update(delay_identifier).at(delay_name)


class ModuleGroup(Module, Container):
  """A group of :py:class:`~.Module`s in which the updating order does not matter.

  Args:
    children_as_tuple: The children objects.
    children_as_dict: The children objects.
    name: The object name.
    mode: The mode which controls the model computation.
    child_type: The type of the children object. Default is :py:class:`Module`.
  """

  __module__ = 'brainpy.core'

  def __init__(
      self,
      *children_as_tuple,
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
      child_type: type = Module,
      **children_as_dict
  ):
    super().__init__(name=name, mode=mode)

    # Attribute of "Container"
    self.children = module_dict(self.format_elements(child_type, *children_as_tuple, **children_as_dict))

  def update(self, *args, **kwargs):
    """
    Step function of a network.

    In this update function, the update functions in children systems are
    iteratively called.
    """
    projs, dyns, others = self.nodes(level=1, include_self=False).split(Projection, Dynamics)

    # update nodes of projections
    for node in projs.values():
      node()

    # update nodes of dynamics
    for node in dyns.values():
      node()

    # update nodes with other types, including delays, ...
    for node in others.values():
      node()


class Sequential(Module, UpdateReturn, Container):
  """A sequential `input-output` module.

  Modules will be added to it in the order they are passed in the
  constructor. Alternatively, an ``dict`` of modules can be
  passed in. The ``update()`` method of ``Sequential`` accepts any
  input and forwards it to the first module it contains. It then
  "chains" outputs to inputs sequentially for each subsequent module,
  finally returning the output of the last module.

  The value a ``Sequential`` provides over manually calling a sequence
  of modules is that it allows treating the whole container as a
  single module, such that performing a transformation on the
  ``Sequential`` applies to each of the modules it stores (which are
  each a registered submodule of the ``Sequential``).

  What's the difference between a ``Sequential`` and a
  :py:class:`Container`? A ``Container`` is exactly what it
  sounds like--a container to store :py:class:`Module` s!
  On the other hand, the layers in a ``Sequential`` are connected
  in a cascading way.

  Examples
  --------

  >>> import brainpy.core as bc
  >>>
  >>> # composing ANN models
  >>> l = bc.Sequential(bp.layers.Dense(100, 10),
  >>>                   bm.relu,
  >>>                   bp.layers.Dense(10, 2))
  >>> l(bc.random.random((256, 100)))
  >>>
  >>> # Using Sequential with Dict. This is functionally the
  >>> # same as the above code
  >>> l = bc.Sequential(l1=bp.layers.Dense(100, 10),
  >>>                   l2=bm.relu,
  >>>                   l3=bp.layers.Dense(10, 2))
  >>> l(bc.random.random((256, 100)))


  Args:
    modules_as_tuple: The children modules.
    modules_as_dict: The children modules.
    name: The object name.
    mode: The object computing context/mode. Default is ``None``.
  """

  __module__ = 'brainpy.core'

  def __init__(
      self,
      *modules_as_tuple,
      name: str = None,
      mode: Mode = None,
      **modules_as_dict
  ):
    # Attribute of "Container"
    self.children = module_dict(self.format_elements(object, *modules_as_tuple, **modules_as_dict))

    # super init
    super().__init__(name=name, mode=mode)

  def update(self, x):
    """Update function of a sequential model.
    """
    for m in self.children.values():
      x = m(x)
    return x

  def update_return(self):
    """
    The return information of the sequence according to the final model.
    """
    last = self[-1]
    if not isinstance(last, UpdateReturn):
      raise NotImplementedError(f'The last element in the sequence is not an instance of {UpdateReturn.__name__}')
    return last.update_return()

  def update_return_info(self):
    """
    The return information of the sequence according to the final model.
    """
    last = self[-1]
    if not isinstance(last, UpdateReturn):
      raise NotImplementedError(f'The last element in the sequence is not an instance of {UpdateReturn.__name__}')
    return last.update_return_info()

  def __getitem__(self, key: Union[int, slice, str]):
    if isinstance(key, str):
      if key in self.children:
        return self.children[key]
      else:
        raise KeyError(f'Does not find a component named {key} in\n {str(self)}')
    elif isinstance(key, slice):
      return Sequential(**dict(tuple(self.children.items())[key]))
    elif isinstance(key, int):
      return tuple(self.children.values())[key]
    elif isinstance(key, (tuple, list)):
      _all_nodes = tuple(self.children.items())
      return Sequential(**dict(_all_nodes[k] for k in key))
    else:
      raise KeyError(f'Unknown type of key: {type(key)}')

  def __repr__(self):
    nodes = self.children.values()
    entries = '\n'.join(f'  [{i}] {_repr_object(x)}' for i, x in enumerate(nodes))
    return f'{self.__class__.__name__}(\n{entries}\n)'


def receive_update_output(cls: object):
  """
  The decorator to mark the object (as the after updates) to receive the output of the update function.

  That is, the `aft_update` will receive the return of the update function::

    ret = model.update(*args, **kwargs)
    for fun in model.aft_updates:
      fun(ret)

  """
  # assert isinstance(cls, Module), 'The input class should be instance of Module.'
  if hasattr(cls, '_not_receive_update_output'):
    delattr(cls, '_not_receive_update_output')
  return cls


def not_receive_update_output(cls: object):
  """
  The decorator to mark the object (as the after updates) to not receive the output of the update function.

  That is, the `aft_update` will not receive the return of the update function::

    ret = model.update(*args, **kwargs)
    for fun in model.aft_updates:
      fun()

  """
  # assert isinstance(cls, Module), 'The input class should be instance of Module.'
  cls._not_receive_update_output = True
  return cls


def receive_update_input(cls: object):
  """
  The decorator to mark the object (as the before updates) to receive the input of the update function.

  That is, the `bef_update` will receive the input of the update function::


    for fun in model.bef_updates:
      fun(*args, **kwargs)
    model.update(*args, **kwargs)

  """
  # assert isinstance(cls, Module), 'The input class should be instance of Module.'
  cls._receive_update_input = True
  return cls


def not_receive_update_input(cls: object):
  """
  The decorator to mark the object (as the before updates) to not receive the input of the update function.

  That is, the `bef_update` will not receive the input of the update function::

      for fun in model.bef_updates:
        fun()
      model.update()

  """
  # assert isinstance(cls, Module), 'The input class should be instance of Module.'
  if hasattr(cls, '_receive_update_input'):
    delattr(cls, '_receive_update_input')
  return cls


class Delay(ExtendedUpdateWithBA, ParamDesc):
  """
  Generate Delays for the given :py:class:`~.State` instance.

  The data in this delay variable is arranged as::

       delay = 0             [ data
       delay = 1               data
       delay = 2               data
       ...                     ....
       ...                     ....
       delay = length-1        data
       delay = length          data ]

  Args:
    time: int, float. The delay time.
    init: Any. The delay data. It can be a Python number, like float, int, boolean values.
      It can also be arrays. Or a callable function or instance of ``Connector``.
      Note that ``initial_delay_data`` should be arranged as the following way::

         delay = 1             [ data
         delay = 2               data
         ...                     ....
         ...                     ....
         delay = length-1        data
         delay = length          data ]
    entries: optional, dict. The delay access entries.
    name: str. The delay name.
    method: str. The method used for updating delay. Default None.
    mode: Mode. The computing mode. Default None.
  """

  __module__ = 'brainpy.core'

  not_desc_params = ('time', 'entries', 'name')
  max_time: float
  max_length: int
  history: Optional[State]

  def __init__(
      self,
      target_info: PyTree,
      time: Optional[Union[int, float]] = None,  # delay time
      init: Optional[Union[ArrayLike, Callable]] = None,  # delay data init
      entries: Optional[Dict] = None,  # delay access entry
      method: Optional[str] = ROTATE_UPDATE,  # delay method
      # others
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
  ):

    # target information
    self.target_info = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, get_dtype(a)), target_info)

    # delay method
    assert method in [ROTATE_UPDATE, CONCAT_UPDATE]
    self.method = method

    # delay length and time
    self.max_time, delay_length = _get_delay(time, None)
    self.max_length = delay_length + 1

    super().__init__(name=name, mode=mode)

    # delay data
    if init is not None:
      assert isinstance(init, (numbers.Number, jax.Array, Callable))
    self._init = init
    self._history = None

    # other info
    self._registered_entries = dict()

    # delay data
    if delay_length > 0:
      self._init_history(self.max_length)

    # other info
    if entries is not None:
      for entry, delay_time in entries.items():
        self.register_entry(entry, delay_time)

  def __repr__(self):
    name = self.__class__.__name__
    return f'{name}(delay_length={self.max_length}, target_info={self.target_info}, method="{self.method}")'

  @property
  def history(self):
    return self._history

  @history.setter
  def history(self, value):
    self._history = value

  def _f_to_init(self, a, batch_size, length):
    shape = list(a.shape)
    if batch_size is not None:
      shape.insert(self.mode.batch_axis, batch_size)
    shape.insert(0, length)
    if isinstance(self._init, (jax.Array, numbers.Number)):
      data = jnp.broadcast_to(jnp.asarray(self._init, a.dtype), shape)
    elif callable(self._init):
      data = self._init(shape, dtype=a.dtype)
    else:
      assert self._init is None, f'init should be Array, Callable, or None. but got {self._init}'
      data = jnp.zeros(shape, dtype=a.dtype)
    return data

  def _init_history(self, length: int, batch_size: int = None):
    if batch_size is not None:
      assert self.mode.has(Batching), 'The mode should have Batching behavior when batch_size is not None.'
    self.history = State(jax.tree.map(partial(self._f_to_init, length=length, batch_size=batch_size),
                                      self.target_info))

  def register_entry(
      self,
      entry: str,
      delay_time: Optional[Union[int, float]] = None,
      delay_step: Optional[int] = None,
  ) -> 'Delay':
    """Register an entry to access the data.

    Args:
      entry: str. The entry to access the delay data.
      delay_time: The delay time of the entry (can be a float).
      delay_step: The delay step of the entry (must be an int). ``delat_step = delay_time / dt``.

    Returns:
      Return the self.
    """
    if entry in self._registered_entries:
      raise KeyError(f'Entry {entry} has been registered. '
                     f'The existing delay for the key {entry} is {self._registered_entries[entry]}. '
                     f'The new delay for the key {entry} is {delay_time}. '
                     f'You can use another key. ')

    if isinstance(delay_time, (np.ndarray, jax.Array)):
      assert delay_time.size == 1 and delay_time.ndim == 0
      delay_time = delay_time.item()

    _, delay_step = _get_delay(delay_time, delay_step)

    # delay variable
    if self.max_length <= delay_step + 1:
      self._init_history(delay_step + 1)
      self.max_length = delay_step + 1
      self.max_time = delay_time
    self._registered_entries[entry] = delay_step
    return self

  def at(self, entry: str, *indices) -> ArrayLike:
    """Get the data at the given entry.

    Args:
      entry: str. The entry to access the data.
      *indices: The slicing indices. Not include the slice at the batch dimension.

    Returns:
      The data.
    """
    assert isinstance(entry, str), (f'entry should be a string for describing the '
                                    f'entry of the delay data. But we got {entry}.')
    if entry not in self._registered_entries:
      raise KeyError(f'Does not find delay entry "{entry}".')
    delay_step = self._registered_entries[entry]
    if delay_step is None:
      delay_step = 0
    return self.retrieve(delay_step, *indices)

  def retrieve(self, delay_step, *indices):
    """Retrieve the delay data according to the delay length.

    Parameters
    ----------
    delay_step: int
      The delay length used to retrieve the data.
    """
    assert self.history is not None, 'The delay history is not initialized.'
    assert delay_step is not None, 'The delay step should be given.'

    if share.get(share.JIT_ERROR_CHECK, False):
      def _check_delay(delay_len):
        raise ValueError(f'The request delay length should be less than the '
                         f'maximum delay {self.max_length}. But we got {delay_len}')

      jit_error(delay_step >= self.max_length, _check_delay, delay_step)

    # rotation method
    if self.method == ROTATE_UPDATE:
      i = share.get(share.I)
      di = i - delay_step
      delay_idx = jnp.asarray(di % self.max_length, dtype=jnp.int32)
      delay_idx = jax.lax.stop_gradient(delay_idx)

    elif self.method == CONCAT_UPDATE:
      delay_idx = delay_step

    else:
      raise ValueError(f'Unknown updating method "{self.method}"')

    # the delay index
    if hasattr(delay_idx, 'dtype') and not jnp.issubdtype(delay_idx.dtype, jnp.integer):
      raise ValueError(f'"delay_len" must be integer, but we got {delay_idx}')
    indices = (delay_idx,) + indices

    # the delay data
    return jax.tree.map(lambda a: a[indices], self.history.value)

  def update(self, current: PyTree) -> None:
    """
    Update delay variable with the new data.
    """
    assert self.history is not None, 'The delay history is not initialized.'

    # update the delay data at the rotation index
    if self.method == ROTATE_UPDATE:
      i = share.get(share.I)
      idx = jnp.asarray(i % self.max_length, dtype=environ.dutype())
      idx = jax.lax.stop_gradient(idx)
      self.history.value = jax.tree.map(lambda hist, cur: hist.at[idx].set(cur),
                                        self.history.value,
                                        current)
    # update the delay data at the first position
    elif self.method == CONCAT_UPDATE:
      current = jax.tree_map(lambda a: jnp.expand_dims(a, 0), current)
      if self.max_length > 1:
        self.history.value = jax.tree_map(lambda hist, cur: jnp.concatenate([cur, hist[:-1]], axis=0),
                                          self.history.value,
                                          current)
      else:
        self.history.value = current

    else:
      raise ValueError(f'Unknown updating method "{self.method}"')


class _StateDelay(Delay):
  """
  The state delay class.

  Args:
    target: The target state instance.
    init: The initial delay data.
  """

  __module__ = 'brainpy.core'
  _invisible_states = ('target',)

  def __init__(
      self,
      target: State,
      time: Optional[Union[int, float]] = None,  # delay time
      init: Optional[Union[ArrayLike, Callable]] = None,  # delay data init
      entries: Optional[Dict] = None,  # delay access entry
      method: Optional[str] = ROTATE_UPDATE,  # delay method
      # others
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
  ):
    super().__init__(target_info=target.value,
                     time=time, init=init, entries=entries,
                     method=method, name=name, mode=mode)
    self.target = target

  def update(self, *args, **kwargs):
    super().update(self.target.value)


class DelayAccess(Module):
  """
  The delay access class.

  Args:
    delay: The delay instance.
    time: The delay time.
    indices: The indices of the delay data.
    delay_entry: The delay entry.
  """

  __module__ = 'brainpy.core'

  def __init__(
      self,
      delay: Delay,
      time: Union[None, int, float],
      *indices,
      delay_entry: str = None
  ):
    super().__init__(mode=delay.mode)
    self.refs = {'delay': delay}
    assert isinstance(delay, Delay)
    self._delay_entry = delay_entry or self.name
    delay.register_entry(self._delay_entry, time)
    self.indices = indices

  def update(self):
    return self.refs['delay'].at(self._delay_entry, *self.indices)


def register_delay_of_target(target: AllOfTypes[ExtendedUpdateWithBA, UpdateReturn]):
  """Register delay class for the given target.

  Args:
    target: The target class to register delay.

  Returns:
    The delay registered for the given target.
  """
  if not target.has_aft_update(delay_identifier):
    assert isinstance(target, AllOfTypes[ExtendedUpdateWithBA, UpdateReturn])
    target.add_after_update(delay_identifier, Delay(target.update_return_info()))
  delay_cls = target.get_after_update(delay_identifier)
  return delay_cls


def call_order(level: int = 0):
  """The decorator for indicating the resetting level.

  The function takes an optional integer argument level with a default value of 0.

  The lower the level, the earlier the function is called.

  >>> import brainpy.core as bc
  >>> bc.call_order(0)
  >>> bc.call_order(-1)
  >>> bc.call_order(-2)

  """
  if level < 0:
    level = _max_order + level
  if level < 0 or level >= _max_order:
    raise ValueError(f'"reset_level" must be an integer in [0, 10). but we got {level}')

  def wrap(fun: Callable):
    fun.call_order = level
    return fun

  return wrap


def init_states(target: Module, *args, **kwargs):
  """
  Reset states of all children nodes in the given target.

  Args:
    target: The target Module.
  """
  nodes_with_order = []

  # reset node whose `reset_state` has no `reset_level`
  for node in list(target.nodes().values()):
    if not hasattr(node.reset_state, 'call_order'):
      node.reset_state(*args, **kwargs)
    else:
      nodes_with_order.append(node)

  # reset the node's states
  for l in range(_max_order):
    for node in nodes_with_order:
      if node.reset_state.call_order == l:
        node.reset_state(*args, **kwargs)


def load_states(target: Module, state_dict: Dict, **kwargs):
  """Copy parameters and buffers from :attr:`state_dict` into
  this module and its descendants.

  Args:
    target: Module. The dynamical system to load its states.
    state_dict: dict. A dict containing parameters and persistent buffers.

  Returns:
  -------
    ``NamedTuple``  with ``missing_keys`` and ``unexpected_keys`` fields:

    * **missing_keys** is a list of str containing the missing keys
    * **unexpected_keys** is a list of str containing the unexpected keys
  """
  missing_keys = []
  unexpected_keys = []
  for name, node in target.nodes().items():
    r = node.load_state(state_dict[name], **kwargs)
    if r is not None:
      missing, unexpected = r
      missing_keys.extend([f'{name}.{key}' for key in missing])
      unexpected_keys.extend([f'{name}.{key}' for key in unexpected])
  return StateLoadResult(missing_keys, unexpected_keys)


def save_states(target: Module, **kwargs) -> Dict:
  """Save all states in the ``target`` as a dictionary for later disk serialization.

  Args:
    target: Module. The node to save its states.

  Returns:
    Dict. The state dict for serialization.
  """
  return {key: node.save_state(**kwargs) for key, node in target.nodes().items()}


def _input_label_start(label: str):
  # unify the input label repr.
  return f'{label} // '


def _input_label_repr(name: str, label: Optional[str] = None):
  # unify the input label repr.
  return name if label is None else (_input_label_start(label) + str(name))


def _repr_context(repr_str, indent):
  splits = repr_str.split('\n')
  splits = [(s if i == 0 else (indent + s)) for i, s in enumerate(splits)]
  return '\n'.join(splits)


def _repr_object(x):
  if isinstance(x, Module):
    return repr(x)
  elif callable(x):
    signature = inspect.signature(x)
    args = [f'{k}={v.default}' for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty]
    args = ', '.join(args)
    while not hasattr(x, '__name__'):
      if not hasattr(x, 'func'):
        break
      x = x.func  # Handle functools.partial
    if not hasattr(x, '__name__') and hasattr(x, '__class__'):
      return x.__class__.__name__
    if args:
      return f'{x.__name__}(*, {args})'
    return x.__name__
  else:
    x = repr(x).split('\n')
    x = [x[0]] + ['  ' + y for y in x[1:]]
    return '\n'.join(x)


def _get_delay(delay_time, delay_step):
  if delay_time is None:
    if delay_step is None:
      return 0., 0
    else:
      assert isinstance(delay_step, int), '"delay_step" should be an integer.'
      if delay_step == 0:
        return 0., 0
      delay_time = delay_step * environ.get_dt()
  else:
    assert delay_step is None, '"delay_step" should be None if "delay_time" is given.'
    assert isinstance(delay_time, (int, float))
    delay_step = math.ceil(delay_time / environ.get_dt())
  return delay_time, delay_step
