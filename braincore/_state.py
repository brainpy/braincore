from typing import Any, Tuple, Dict

import jax

from ._utils import Stack

__all__ = [
  'State', 'ShortTermState', 'LongTermState', 'ParamState',
  'StateStack', 'visible_state_dict',
]

PyTree = Any
_pytree_registered_objects = set()


def _register_pytree_cls(cls):
  if cls not in _pytree_registered_objects:
    jax.tree_util.register_pytree_node_class(cls)
    _pytree_registered_objects.add(cls)



state_stack_list = []



class State(object):
  """
  The pointer to specify the dynamical state.

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
    for stack in state_stack_list:
      stack.add_read(self)
    return self._value

  @value.setter
  def value(self, v):
    v = v.value if isinstance(v, State) else v
    self.__check_value(v)
    for stack in state_stack_list:
      stack.add_write(self)
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
class StateStack(Stack):
  """
  State stack, for collecting all :py:class:`~.State` used in the program.

  :py:class:`~.StateStack` supports all features of python dict.
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
    results = tuple(Stack() for _ in range(len(filters) + 1))
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
    results = Stack()
    for k, v in self.items():
      results[k] = v.value
    return results

  def split(self, first: type, *others: type) -> Tuple['StateStack', ...]:
    return super().split(first, *others)

  def _check_elem(self, elem):
    assert isinstance(elem, State), f'must be instance of {State}'

  def _set_elem(self, key: Any, value: Any) -> None:
    self[key].value = value


class visible_state_dict(StateStack):
  """
  The state dictionary whose elements are visible to ``.states()`` collection functions.
  """
  pass


class state_auto_stack(object):
  """
  The auto stack, which is used to store the states automatically.
  """

  def __init__(self):
    self.reads = StateStack()
    self.writes = StateStack()

  def __enter__(self) -> 'state_auto_stack':
    state_stack_list.append(self)
    return self

  def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
    state_stack_list.pop()

  def add_read(self, state: State) -> None:
    self.reads.add(state)

  def add_write(self, state: State) -> None:
    self.writes.add(state)


def sate_compose(first: dict, *others: dict):
  """
  Compose multiple dictionaries as a ``Stack``.

  Args:
    first: The dict.
    others: Dicts.

  Returns:
    stack: The composed stack.
  """
  stack = StateStack(first)
  for oth in others:
    stack.update(oth)
  return stack
