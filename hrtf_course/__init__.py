"""hrtf_course — teaching layer over `hrtf_relearning`.

Public API
----------
Manipulations (return a new ``slab.HRTF``)::

    from hrtf_course import shift_band, widen_band, octave_band

Conditions + study::

    from hrtf_course import Condition, run_condition

Analysis + preview::

    from hrtf_course import collect_results, plot_compare, preview

The teaching layer wraps existing primitives in ``hrtf_relearning`` —
nothing in that package is modified.
"""

from hrtf_course.manipulations import (
    shift_band,
    widen_band,
    octave_band,
    erb_bandwidth,
)
from hrtf_course.conditions import Condition
from hrtf_course.study import run_condition, default_hrir_settings, default_loc_settings
from hrtf_course.analysis import collect_results, plot_compare
from hrtf_course import preview

__all__ = [
    "shift_band",
    "widen_band",
    "octave_band",
    "erb_bandwidth",
    "Condition",
    "run_condition",
    "default_hrir_settings",
    "default_loc_settings",
    "collect_results",
    "plot_compare",
    "preview",
]

