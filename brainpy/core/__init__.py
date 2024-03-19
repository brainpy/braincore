"""
The core system for the next-generation BrainPy framework.
"""

__version__ = "0.1.0"

from .projection import *
from .module import *
from .state import *
from .transform import *
from .utils import *

from .projection import __all__ as _projection_all
from .module import __all__ as _module_all
from .state import __all__ as _state_all
from .transform import __all__ as _transform_all
from .utils import __all__ as _utils_all

from . import environ
from . import share
from . import surrogate
from . import random
from . import mixin
from . import math

__all__ = (
    ['environ', 'share', 'surrogate', 'random', 'mixin', 'math'] +
    _projection_all + _module_all +
    _state_all + _transform_all + _utils_all
)
