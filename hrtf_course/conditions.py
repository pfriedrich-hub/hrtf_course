"""Condition objects bind a manipulation to a base HRTF.

A :class:`Condition` knows how to:

* build the manipulated ``slab.HRTF`` (in memory),
* write it as a SOFA file with a deterministic name, and
* describe itself for plots and reports.

The SOFA file name is ``{base_subject}_{condition.name}.sofa``.  This name
flows through ``hrir_settings["name"]`` into ``hrtf_relearning``'s
``Localization`` machinery, which means the resulting localization run is
tagged with the condition automatically — no changes are needed to the
research package's data model.

Typical usage::

    from hrtf_course import Condition, shift_band, smooth_band

    base = "AGV"

    cond = Condition(
        name="shift_8kHz_+10pct",
        base_subject=base,
        fn=lambda h: shift_band(h, 5500, 11500, factor=1.10),
    )
    sofa_name = cond.build_sofa()   # writes data/hrtf/sofa/AGV_shift_8kHz_+10pct.sofa
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import slab

logger = logging.getLogger(__name__)


def _default_sofa_dir() -> Path:
    """Where SOFA files are read from / written to.

    Resolution order:

    1. ``HRTF_COURSE_SOFA_DIR`` environment variable (explicit override).
    2. ``<hrtf_course repo>/data/sofa`` — the project's own data tree.
       To analyse rig data, copy ``{subject}.sofa`` files here from
       ``hrtf_relearning/data/hrtf/sofa/``.
    """
    env = os.environ.get("HRTF_COURSE_SOFA_DIR")
    if env:
        return Path(env).expanduser()
    # Lazy import to avoid a circular reference at package init time.
    import hrtf_course
    return hrtf_course.PATH / "data" / "sofa"


SOFA_DIR: Path = _default_sofa_dir()


# Names end up as both filenames *and* tokens used to parse condition out of
# saved sequences.  Keep them filesystem-friendly.
_NAME_RE = re.compile(r"^[A-Za-z0-9_+\-.]+$")


def _validate_name(name: str) -> None:
    if not name:
        raise ValueError("Condition name must be non-empty.")
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Condition name {name!r} contains characters that are unsafe in "
            "SOFA filenames.  Allowed: letters, digits, _ + - ."
        )


def _load_base_hrtf(base_subject: str) -> slab.HRTF:
    sofa_path = SOFA_DIR / f"{base_subject}.sofa"
    if not sofa_path.exists():
        raise FileNotFoundError(
            f"Base SOFA not found: {sofa_path}.  Did you record the HRTF on "
            "Day 1, or set base_subject to a SOFA that exists in "
            f"{SOFA_DIR}?"
        )
    return slab.HRTF(sofa_path)


@dataclass
class Condition:
    """An HRTF manipulation tied to a specific base subject.

    Parameters
    ----------
    name : str
        Short, filename-safe label, e.g. ``"shift_8kHz_+10pct"``.  This name
        appears in SOFA filenames, ``Localization`` sequence names, and as
        the grouping variable in the analysis DataFrame.
    base_subject : str
        Subject id whose recorded SOFA will be loaded as input.
    fn : callable
        ``fn(slab.HRTF) -> slab.HRTF``.  Typically a one-line lambda over a
        ``hrtf_course`` manipulation, e.g.
        ``lambda h: shift_band(h, 5500, 11500, factor=1.10)``.
    description : str, optional
        Free-form one-liner displayed in plots and reports.
    """

    name: str
    base_subject: str
    fn: Callable[[slab.HRTF], slab.HRTF]
    description: str = ""
    _identity: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        _validate_name(self.name)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls, base_subject: str) -> "Condition":
        """A do-nothing baseline condition.

        Builds a SOFA whose contents are identical to the recorded one,
        renamed so it appears as its own condition in the analysis.  Useful
        as the within-subject reference in tolerance plots.
        """
        cond = cls(
            name="baseline",
            base_subject=base_subject,
            fn=lambda h: h,
            description="Subject's own (unmanipulated) HRTF.",
        )
        cond._identity = True
        return cond

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @property
    def sofa_name(self) -> str:
        """The filename stem written by :meth:`build_sofa` (no extension)."""
        if self._identity:
            # The baseline is just a renamed copy so it shows up in analysis.
            return f"{self.base_subject}_{self.name}"
        return f"{self.base_subject}_{self.name}"

    @property
    def sofa_path(self) -> Path:
        return SOFA_DIR / f"{self.sofa_name}.sofa"

    def build_hrtf(self) -> slab.HRTF:
        """Apply ``fn`` to the base HRTF and return the result (no I/O)."""
        base = _load_base_hrtf(self.base_subject)
        out = self.fn(base)
        if not isinstance(out, slab.HRTF):
            raise TypeError(
                f"Condition.fn must return slab.HRTF, got {type(out).__name__}"
            )
        return out

    def build_sofa(self, *, overwrite: bool = True) -> str:
        """Build the manipulated HRTF and write it to ``data/hrtf/sofa/``.

        Returns the SOFA *name* (without extension) so it can be passed
        straight to ``hrir_settings["name"]`` for the localization run.
        """
        if self.sofa_path.exists() and not overwrite:
            logger.info("SOFA exists, not rebuilt: %s", self.sofa_path.name)
            return self.sofa_name

        hrtf = self.build_hrtf()
        SOFA_DIR.mkdir(parents=True, exist_ok=True)
        hrtf.write_sofa(self.sofa_path)
        logger.info("Wrote %s", self.sofa_path.name)
        return self.sofa_name

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        suffix = f" — {self.description}" if self.description else ""
        return f"Condition({self.base_subject}, {self.name!r}){suffix}"
