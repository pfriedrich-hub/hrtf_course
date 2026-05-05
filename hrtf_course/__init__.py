"""hrtf_course — teaching layer for the practical psychoacoustics course.

Public API
----------
Manipulations (return a new ``slab.HRTF``)::

    from hrtf_course import shift_band, widen_band, octave_band

Conditions + study::

    from hrtf_course import Condition, run_condition

Analysis + preview::

    from hrtf_course import collect_results, plot_compare, preview

Data layout
-----------
``hrtf_course.PATH`` points at the repo root (one level above this
package).  By default, SOFA files and Subject pickles live under
``hrtf_course.PATH / "data" / {sofa,results}``.  To analyse rig data
on a laptop, copy ``{subject}.sofa`` and ``{subject}.pkl`` into those
folders.

Override locations with the environment variables
``HRTF_COURSE_SOFA_DIR`` and ``HRTF_COURSE_RESULTS_DIR`` if needed.
"""

from pathlib import Path

# Absolute path to the repo root — one level above this package.
# Used as the anchor for the default data layout (data/sofa, data/results).
PATH: Path = Path(__file__).resolve().parent.parent

from hrtf_course.manipulations import (
    shift_band,
    octave_band,
    erb_bandwidth,
)
from hrtf_course.conditions import Condition
from hrtf_course.analysis import (
    list_runs,
    collect_results,
    plot_compare,
    plot_tolerance_curve,
)
from hrtf_course import preview

__all__ = [
    "PATH",
    "shift_band",
    "octave_band",
    "erb_bandwidth",
    "Condition",
    "list_runs",
    "collect_results",
    "plot_compare",
    "plot_tolerance_curve",
    "preview",
]

