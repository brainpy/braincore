from typing import Any, Tuple, Dict, List, Callable

import jax
from jax.api_util import shaped_abstractify

from ._utils import DictManager

__all__ = [
  'State', 'ShortTermState', 'LongTermState', 'ParamState',
  'StateManager', 'visible_state_dict',
]

PyTree = Any
_pytree_registered_objects = set()


def _register_pytree_cls(cls):
  if cls not in _pytree_registered_objects:
    jax.tree_util.register_pytree_node_class(cls)
    _pytree_registered_objects.add(cls)


state_stack_list: List['StateTrace'] = []


class State(object):
  """
  The pointer to specify the dynamical state.

  To implement a new subclass of :py:class:`~.State`, you only need to inherent this class:

  Example::

    class MyState(State):
      pass

  The typical examples of states are:

  - :py:class:`~.ShortTermState`: The short-term state, which is used to store the short-term data in the program.
  - :py:class:`~.LongTermState`: The long-term state, which is used to store the long-term data in the program.
  - :py:class:`~.ParamState`: The parameter state, which is used to store the parameters in the program.
  - :py:class:`~.RandomState`: The random generator state, which is used to store the random key in the program.

  Args:
    value: PyTree. It can be anything as a pyTree.
  """
  __module__ = 'braincore'
  __slots__ = ('_value', '_tree')

  def __init__(self, value: PyTree):
    if isinstance(value, State):
      value = value.value
    self._value = value
    self._tree = jax.tree.structure(value)
    _register_pytree_cls(self.__class__)

  @property
  def value(self):
    """
    The data and its value.
    """
    trace: StateTrace
    for trace in state_stack_list:
      trace.read_its_value(self)
    return self._value

  @value.setter
  def value(self, v):
    v = v.value if isinstance(v, State) else v
    self.__check_value(v)

    trace: StateTrace
    for trace in state_stack_list:
      trace.write_its_value(self)
    self._value = v

  def __check_value(self, v):
    in_tree = jax.tree_util.tree_structure(v)
    if in_tree != self._tree:
      raise ValueError(f'The given value {in_tree} does not match with the origin tree structure {self._tree}.')

  def tree_flatten(self):
    """Flattens this variable.

    Returns:
      A pair where the first element is a list of leaf values
      and the second element is a treedef representing the
      structure of the flattened tree.
    """
    return (self._value,), None

  @classmethod
  def tree_unflatten(cls, aux_data, flat_contents):
    """Reconstructs a variable from the aux_data and the leaves.

    Args:
      aux_data:
      flat_contents:

    Returns:
      The variable.
    """
    return cls(flat_contents[0], )

  def __repr__(self):
    return f'{self.__class__.__name__}({self._value})'


class ShortTermState(State):
  """
  The short-term state, which is used to store the short-term data in the program.

  For example, in a training process, the gradients of the model are short-term states.
  """

  __module__ = 'braincore'


class LongTermState(State):
  """
  The long-term state, which is used to store the long-term data in the program.

  For example, in a training process, the weights of the model are long-term states.

  """

  __module__ = 'braincore'


class ParamState(LongTermState):
  __module__ = 'braincore'


@jax.tree_util.register_pytree_node_class
class StateManager(DictManager):
  """
  State stack, for collecting all :py:class:`~.State` used in the program.

  :py:class:`~.StateManager` supports all features of python dict.
  """

  __module__ = 'braincore'

  def assign_values(self, *args: Dict) -> None:
    """
    Assign the value for each element according to the given ``data``.
    """
    for arg in args:
      assert isinstance(arg, dict), 'Must be an instance of dict.'
      for k, v in arg.items():
        self._set_elem(k, v)

  def split_values(self, *filters: type) -> Tuple[Dict, ...]:
    """
    Split the values into several subsets of stack by the given types.
    """
    results = tuple(DictManager() for _ in range(len(filters) + 1))
    for k, v in self.items():
      for i, filt in enumerate(filters):
        if isinstance(v, filt):
          results[i][k] = v.value
          break
      else:
        results[-1][k] = v.value
    return results

  def collect_values(self) -> Dict:
    """
    Collect the values by the given types.
    """
    results = DictManager()
    for k, v in self.items():
      results[k] = v.value
    return results

  def split(self, first: type, *others: type) -> Tuple['StateManager', ...]:
    return super().split(first, *others)

  def _check_elem(self, elem):
    assert isinstance(elem, State), f'must be instance of {State}'

  def _set_elem(self, key: Any, value: Any) -> None:
    self[key].value = value


class visible_state_dict(StateManager):
  """
  The state dictionary whose elements are visible to ``.states()`` collection functions.
  """
  pass


class StateTrace(object):
  """
  The auto stack, which is used to store the states automatically.
  """

  def __init__(self, new_arg: Callable = None):
    self.reads = []
    self.writes = []
    self._jax_trace_new_arg = new_arg
    self._org_values_of_reads = []
    self._org_values_of_writes = []

  def new_arg(self, state: State):
    if self._jax_trace_new_arg is not None:
      # internal use
      state._value = jax.tree.map(lambda x: self._jax_trace_new_arg(shaped_abstractify(x)), state._value)

  def __enter__(self) -> 'StateTrace':
    state_stack_list.append(self)
    return self

  def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
    state_stack_list.pop()

  def read_its_value(self, state: State) -> None:
    if state not in self.reads:
      self.reads.append(state)
      self._org_values_of_reads.append(state._value)  # internal use
      self.new_arg(state)

  def write_its_value(self, state: State) -> None:
    if state not in self.writes:
      self.writes.append(state)
      self._org_values_of_writes.append(state._value)  # internal use
      if state not in self.reads:
        self.new_arg(state)

  def collect_values(self, *categories: str) -> Tuple:
    results = []
    ids = set()
    if 'read' in categories:
      for st in self.reads:
        results.append(st.value)
        ids.add(id(st))
    if 'write' in categories:
      for st in self.writes:
        if id(st) not in ids:
          results.append(st.value)
          ids.add(id(st))
    return tuple(results)

  def recovery_original_values(self):
    ids = set()
    for st, v in zip(self.reads, self._org_values_of_reads):
      if id(st) not in ids:
        st._value = v
        ids.add(id(st))
    for st, v in zip(self.writes, self._org_values_of_writes):
      if id(st) not in ids:
        st._value = v
        ids.add(id(st))


def sate_compose(first: dict, *others: dict):
  """
  Compose multiple dictionaries as a ``DictManager``.

  Args:
    first: The dict.
    others: Dicts.

  Returns:
    stack: The composed stack.
  """
  stack = StateManager(first)
  for oth in others:
    stack.update(oth)
  return stack
