"""Vendored ``Subject`` class for storing per-subject localization runs.

Originally lived in ``hrtf_relearning.experiment.Subject``.  Inlined so
the laptop profile can load and analyse rig-side pickles without needing
the full research package installed.

The on-disk format is unchanged — pickles written by either copy of
``Subject`` are mutually readable.

Results directory resolution (in order):
  1. ``HRTF_COURSE_RESULTS_DIR`` env var.
  2. ``hrtf_relearning.PATH / "data" / "results"`` if hrtf_relearning is
     installed (rig setup — keeps existing pickles discoverable).
  3. ``~/.hrtf_course/results``.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

__all__ = ["Subject", "results_dir"]


def _default_results_dir() -> Path:
    env = os.environ.get("HRTF_COURSE_RESULTS_DIR")
    if env:
        return Path(env).expanduser()
    try:
        import hrtf_relearning  # type: ignore[import-not-found]
        return Path(hrtf_relearning.__file__).resolve().parent / "data" / "results"
    except ImportError:
        return Path.home() / ".hrtf_course" / "results"


results_dir: Path = _default_results_dir()
results_dir.mkdir(parents=True, exist_ok=True)


class Subject:
    """Per-subject store: localization sequences + trial logs.

    Attributes
    ----------
    id : str
    file_path : Path
        ``results_dir / f"{id}.pkl"``.
    backup_path : Path
        Sister JSON backup, written best-effort.
    localization : dict[str, slab.Trialsequence]
        Saved localization runs keyed by sequence filename.
    trials : list
    last_sequence : object | None
    highscore : int
    """

    def __init__(self, id: str):
        self.file_path = results_dir / f"{id}.pkl"
        self.backup_path = self.file_path.with_suffix(".json")
        self.id = id
        if self.file_path.exists():
            self._load()
        else:
            logging.info("Creating new subject.")
            self.localization = {}
            self.trials = []
            self.last_sequence = None
            self.highscore = 0

    def _load(self):
        logging.info("Loading subject data.")
        with open(self.file_path, "rb") as f:
            data = pickle.load(f)
        self.id = data.get("id", self.id)
        self.localization = data.get("localization", {})
        self.trials = data.get("trials", [])
        self.last_sequence = data.get("last_sequence", None)
        self.highscore = data.get("highscore", 0)

    def write(self):
        logging.debug("Writing subject data.")
        data = {
            "id": self.id,
            "localization": self.localization,
            "trials": self.trials,
            "last_sequence": self.last_sequence,
            "highscore": self.highscore,
        }
        with open(self.file_path, "wb") as f:
            pickle.dump(data, f)
        self._write_backup()

    def _write_backup(self):
        """Write a plain-text JSON backup of the localization data.

        Read-only mirror — nothing in this module reads it back.  Use a
        separate restore script if the pickle ever gets corrupted.
        Failures here are logged but never raised so a backup problem
        can't prevent the authoritative pickle write from succeeding.
        """
        try:
            payload = {
                "id": self.id,
                "localization": _to_jsonable(self.localization),
            }
            tmp_path = self.backup_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            tmp_path.replace(self.backup_path)
        except Exception:
            logging.exception(
                "Failed to write JSON backup for subject %s", self.id
            )


def _to_jsonable(obj):
    """Recursively convert obj into JSON-serialisable primitives.

    Handles numpy scalars/arrays and custom objects (like
    ``slab.Trialsequence``) by dumping their ``__dict__`` under a
    ``__class__`` tag so the backup is self-describing enough for a
    restore script to reconstruct it.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)

    if hasattr(obj, "__dict__"):
        return {
            "__class__": f"{type(obj).__module__}.{type(obj).__name__}",
            **{k: _to_jsonable(v) for k, v in vars(obj).items()
               if not k.startswith("_")},
        }

    return repr(obj)
