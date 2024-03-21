"""
The core system for the next-generation BrainPy framework.
"""

__version__ = "0.1.1"

from . import environ
from . import math
from . import mixin
from . import random
from . import share
from . import surrogate
from ._module import *
from ._module import __all__ as _module_all
from ._projection import *
from ._projection import __all__ as _projection_all
from ._state import *
from ._state import __all__ as _state_all
from ._transform import *
from ._transform import __all__ as _transform_all
from ._utils import *
from ._utils import __all__ as _utils_all

__all__ = (
    ['environ', 'share', 'surrogate', 'random', 'mixin', 'math'] +
    _projection_all + _module_all +
    _state_all + _transform_all + _utils_all
)
del _projection_all, _module_all, _state_all, _transform_all, _utils_all
