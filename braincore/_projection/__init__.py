"""

This module defines the basic classes for synaptic projections.

"""

from .align_post import *
from .align_post import __all__ as align_post_all
from .align_pre import *
from .align_pre import __all__ as align_pre_all
from .delta import *
from .delta import __all__ as delta_all
from .vanilla import *
from .vanilla import __all__ as vanilla_all

__all__ = align_post_all + align_pre_all + delta_all + vanilla_all
del align_post_all, align_pre_all, delta_all, vanilla_all
