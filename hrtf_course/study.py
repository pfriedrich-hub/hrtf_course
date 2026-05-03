"""Run one localization block per :class:`Condition`.

This is a thin wrapper around
``hrtf_relearning.experiment.Localization.Localization_AR.Localization``.
It builds the manipulated SOFA, sets ``hrir_settings["name"]`` so the
sequence is tagged with the condition, and runs the standard 5-min
localization block.

Course-day usage
----------------

>>> from hrtf_course import Condition, run_condition, shift_band, default_loc_settings
>>> cond = Condition(
...     name="shift_8kHz_+10pct",
...     base_subject="AGV",
...     fn=lambda h: shift_band(h, 5500, 11500, factor=1.10),
... )
>>> filename = run_condition("AGV", cond)        # blocks ~5 min
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from hrtf_course.conditions import Condition

logger = logging.getLogger(__name__)


# Sensible course defaults — students can override per-call.
default_hrir_settings: dict[str, Any] = {
    "ear": None,            # binaural
    "mirror": False,
    "reverb": True,
    "drr": 20,
    "hp_filter": True,
    "hp": "DT990",          # change to MYSPHERE if needed
    "convolution": "cpu",
    "storage": "cpu",
}


# 5-min sector test — covers central ±35° az/el grid with 3 trials/sector.
default_loc_settings: dict[str, Any] = {
    "kind": "sectors",
    "azimuth_range": (-35, 35),
    "elevation_range": (-35, 35),
    "sector_size": (14, 14),
    "targets_per_sector": 3,
    "replace": False,
    "min_distance": 20,
    "gain": 0.2,
    "stim": "noise",
}


def run_condition(
    subject_id: str,
    condition: Condition,
    *,
    hrir_kwargs: Optional[dict[str, Any]] = None,
    loc_kwargs: Optional[dict[str, Any]] = None,
    rebuild_sofa: bool = True,
) -> str:
    """Build the manipulated SOFA and run a localization block.

    Parameters
    ----------
    subject_id : str
        The *human* subject sitting in the chair — typically equal to
        ``condition.base_subject`` (testing students on their own HRTF) but
        can differ if you want to test e.g. subject A on subject B's HRTF.
    condition : Condition
        Manipulation to apply to ``condition.base_subject``'s recorded HRTF.
    hrir_kwargs : dict, optional
        Extra keys merged into :data:`default_hrir_settings`.
    loc_kwargs : dict, optional
        Extra keys merged into :data:`default_loc_settings`.  Pass e.g.
        ``targets_per_sector=2`` to shorten the run.
    rebuild_sofa : bool
        If False, reuses the existing SOFA when present.  Default True so
        that re-runs after parameter changes always reflect the current
        ``Condition.fn``.

    Returns
    -------
    str
        The sequence filename that was written to the Subject pickle.  Use
        this to look the run up later in
        ``Subject(subject_id).localization``.
    """
    # Lazy imports — these pull in slab/freefield/pybinsim/etc. and are not
    # needed at import time of the teaching package.
    from hrtf_relearning.experiment.Subject import Subject
    from hrtf_relearning.experiment.Localization.Localization_AR import (
        Localization,
    )

    sofa_name = condition.build_sofa(overwrite=rebuild_sofa)

    hrir_settings = {
        **default_hrir_settings,
        **(hrir_kwargs or {}),
        "name": sofa_name,
        "subject_id": condition.base_subject,
    }
    loc_settings = {**default_loc_settings, **(loc_kwargs or {})}

    logger.info(
        "Running condition %s | tester=%s | HRTF=%s",
        condition.name, subject_id, sofa_name,
    )

    subject = Subject(subject_id)
    loc = Localization(subject, hrir_settings, loc_settings=loc_settings)
    loc.run()

    return loc.filename
