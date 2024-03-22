"""

This module defines the basic classes for synaptic projections.

"""

from ._align_post import *
from ._align_post import __all__ as align_post_all
from ._align_pre import *
from ._align_pre import __all__ as align_pre_all
from ._delta import *
from ._delta import __all__ as delta_all
from ._vanilla import *
from ._vanilla import __all__ as vanilla_all

__all__ = align_post_all + align_pre_all + delta_all + vanilla_all
del align_post_all, align_pre_all, delta_all, vanilla_all
