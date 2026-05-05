"""Pool localization results across subjects and conditions.

Workflow assumed by this module:

1.  Localization tests are run **at the rig** with ``hrtf_relearning``.
    Each test stores its sequence on the ``Subject`` pickle, tagged with
    the SOFA name of the HRTF used (``sequence.hrir``).  By convention
    the SOFA name is ``{base_subject}_{condition}.sofa``, so the
    condition is recoverable from the run.

2.  The Subject pickles (``{id}.pkl``) and the SOFA files
    (``{id}.sofa`` and the manipulated variants) are **copied over to
    this repo's** ``data/results/`` and ``data/sofa/`` folders for
    analysis.

3.  This module loads those pickles, lists the runs, lets you select
    which runs to keep, and computes per-run summary statistics
    (elevation/azimuth gain, RMSE, SD) plus VSI dissimilarity vs. the
    subject's own baseline HRTF.

Multiple runs per condition are normal — each run becomes its own row
in the returned DataFrame.  Use :func:`list_runs` to inspect what's
available, then filter ``collect_results`` by ``condition_names`` or
``runs`` to select the ones you want to plot.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import slab

from hrtf_course.vsi import vsi_dissimilarity
from hrtf_course.conditions import SOFA_DIR
from hrtf_course._subject import Subject
from hrtf_course._localization_metrics import localization_accuracy

logger = logging.getLogger(__name__)


__all__ = [
    "list_runs",
    "collect_results",
    "plot_compare",
    "plot_tolerance_curve",
]


# ---------------------------------------------------------------------------
# Inspecting what's inside a Subject pickle
# ---------------------------------------------------------------------------


def _condition_from_sofa_name(sofa_name: str, base_subject: str) -> str:
    """Strip the ``{base_subject}_`` prefix off a SOFA name.

    Sequences whose SOFA is exactly ``{base_subject}.sofa`` are returned
    as ``"recorded"`` so the unmanipulated baseline shows up as its own
    condition in the DataFrame.
    """
    prefix = f"{base_subject}_"
    if sofa_name == base_subject:
        return "recorded"
    if sofa_name.startswith(prefix):
        return sofa_name[len(prefix):]
    return sofa_name  # foreign HRTF (e.g. testing A on B) — keep as-is


def list_runs(subject_id: str) -> pd.DataFrame:
    """List every localization run on a Subject pickle.

    Use this to see what's in a ``data/results/{id}.pkl`` before
    deciding which runs to feed into :func:`collect_results`.

    Parameters
    ----------
    subject_id : str
        Subject id whose pickle to load (must be in ``data/results/``).

    Returns
    -------
    pandas.DataFrame
        One row per saved localization sequence:

        * ``run``        — sequence filename (timestamped, unique)
        * ``base_hrtf``  — SOFA name embedded in the sequence
        * ``condition``  — parsed from the SOFA name
        * ``n_trials``   — number of completed trials
    """
    subj = Subject(subject_id)
    rows = []
    for run_name, sequence in (subj.localization or {}).items():
        sofa_name = getattr(sequence, "hrir", None) or ""
        base_subject = sofa_name.split("_")[0] if sofa_name else ""
        cond = _condition_from_sofa_name(sofa_name, base_subject) if sofa_name else "?"
        n_trials = len(getattr(sequence, "data", []) or [])
        rows.append(dict(
            run=run_name,
            base_hrtf=sofa_name,
            condition=cond,
            n_trials=n_trials,
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["condition", "run"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _vsi_diss_safe(target_sofa: Path, baseline_sofa: Path,
                   bandwidth: tuple[float, float] = (5700, 11300)) -> float:
    """VSI dissimilarity, NaN if either SOFA is missing or unreadable."""
    if not target_sofa.exists() or not baseline_sofa.exists():
        return float("nan")
    try:
        h_target = slab.HRTF(target_sofa)
        h_base = slab.HRTF(baseline_sofa)
        return float(vsi_dissimilarity(h_base, h_target, bandwidth=bandwidth))
    except Exception:
        logger.exception("VSI dissimilarity failed for %s vs %s",
                         target_sofa.name, baseline_sofa.name)
        return float("nan")


def collect_results(
    subject_ids: Iterable[str],
    *,
    condition_names: Optional[Sequence[str]] = None,
    runs: Optional[Sequence[str]] = None,
    vsi_bandwidth: tuple[float, float] = (5700, 11300),
) -> pd.DataFrame:
    """Pool localization runs across subjects into one DataFrame.

    Parameters
    ----------
    subject_ids : iterable of str
        Subjects whose pickles to load from ``data/results/``.
    condition_names : sequence of str, optional
        Restrict to these conditions (after the ``{base_subject}_``
        prefix is stripped).  ``None`` returns every condition found.
    runs : sequence of str, optional
        Restrict to these run filenames.  Use :func:`list_runs` to see
        what's available; pass ``None`` to keep all.
    vsi_bandwidth : (low_hz, high_hz)
        Frequency band for VSI dissimilarity.  Default 5700–11300 Hz,
        the peak VSI band from Trapeau & Schönwiesner (2016).

    Returns
    -------
    pandas.DataFrame
        Columns:

        * ``subject``     — tester id
        * ``base_hrtf``   — SOFA name written by the rig
        * ``base_subject``— subject id whose SOFA was manipulated
        * ``condition``   — condition name (e.g. ``"shift_+10pct"``)
        * ``run``         — sequence filename (unique per run)
        * ``n_trials``    — number of recorded trials
        * ``ele_gain``    — slope of response_el vs. target_el
        * ``ele_rmse``    — elevation RMSE in degrees
        * ``ele_sd``      — within-sector elevation SD
        * ``az_gain``     — slope of response_az vs. target_az (NaN if midline)
        * ``az_rmse``     — azimuth RMSE in degrees
        * ``az_sd``       — within-sector azimuth SD
        * ``vsi_diss``    — VSI dissimilarity vs. ``{base_subject}.sofa``
                            (NaN if either SOFA isn't in ``data/sofa/``)
    """
    rows: list[dict] = []
    keep_conds = set(condition_names) if condition_names is not None else None
    keep_runs = set(runs) if runs is not None else None
    # Cache VSI computations per (target, base) SOFA pair — they're expensive.
    vsi_cache: dict[tuple[str, str], float] = {}

    for sid in subject_ids:
        try:
            subj = Subject(sid)
        except Exception:
            logger.exception("Could not load subject %s", sid)
            continue

        for run_name, sequence in (subj.localization or {}).items():
            if keep_runs is not None and run_name not in keep_runs:
                continue

            sofa_name = getattr(sequence, "hrir", None)
            if sofa_name is None:
                logger.debug("Skipping %s — no .hrir on sequence", run_name)
                continue

            base_subject = sofa_name.split("_")[0]
            cond = _condition_from_sofa_name(sofa_name, base_subject)
            if keep_conds is not None and cond not in keep_conds:
                continue

            try:
                eg, e_rmse, e_sd, ag, a_rmse, a_sd = localization_accuracy(sequence)
            except Exception:
                logger.exception("localization_accuracy failed for %s", run_name)
                eg = e_rmse = e_sd = ag = a_rmse = a_sd = float("nan")

            n_trials = len(getattr(sequence, "data", []) or [])

            target_sofa = SOFA_DIR / f"{sofa_name}.sofa"
            baseline_sofa = SOFA_DIR / f"{base_subject}.sofa"
            cache_key = (str(target_sofa), str(baseline_sofa))
            if cache_key not in vsi_cache:
                vsi_cache[cache_key] = _vsi_diss_safe(
                    target_sofa, baseline_sofa, bandwidth=vsi_bandwidth,
                )

            rows.append(dict(
                subject=sid,
                base_hrtf=sofa_name,
                base_subject=base_subject,
                condition=cond,
                run=run_name,
                n_trials=n_trials,
                ele_gain=eg,
                ele_rmse=e_rmse,
                ele_sd=e_sd,
                az_gain=ag,
                az_rmse=a_rmse,
                az_sd=a_sd,
                vsi_diss=vsi_cache[cache_key],
            ))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["subject", "condition", "run"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


_METRIC_LABELS = {
    "ele_gain": "Elevation gain",
    "ele_rmse": "Elevation RMSE (°)",
    "ele_sd":   "Elevation SD (°)",
    "az_gain":  "Azimuth gain",
    "az_rmse":  "Azimuth RMSE (°)",
    "az_sd":    "Azimuth SD (°)",
    "vsi_diss": "VSI dissimilarity",
}


def _check_metric(metric: str) -> None:
    if metric not in _METRIC_LABELS:
        raise ValueError(
            f"unknown metric {metric!r}.  Try one of: "
            f"{sorted(_METRIC_LABELS)}"
        )


def plot_compare(
    df: pd.DataFrame,
    *,
    metric: str = "ele_rmse",
    group: str = "condition",
    axis: Optional[plt.Axes] = None,
    filepath: Optional[Path] = None,
):
    """Per-group point cloud + group mean overlay for ``metric``.

    One dot per run; horizontal bar = mean across runs (and subjects)
    in that group.  Use ``group="condition"`` for the standard
    condition-vs-condition view, ``group="subject"`` for a per-subject
    view, etc.
    """
    _check_metric(metric)
    if df.empty:
        raise ValueError("df is empty — nothing to plot.")
    if group not in df.columns:
        raise ValueError(f"group {group!r} not in df columns ({list(df.columns)}).")

    if axis is None:
        fig, axis = plt.subplots(figsize=(7, 4))
    else:
        fig = axis.figure

    groups = list(df[group].unique())
    rng = numpy.random.default_rng(0)
    for x_idx, gname in enumerate(groups):
        sub = df[df[group] == gname][metric].astype(float)
        x_jit = rng.uniform(-0.12, 0.12, size=len(sub)) + x_idx
        axis.scatter(x_jit, sub, alpha=0.7)
        if len(sub):
            axis.hlines(
                float(numpy.nanmean(sub)),
                x_idx - 0.25, x_idx + 0.25,
                colors="black", linewidth=2,
            )

    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels(groups, rotation=20, ha="right")
    axis.set_ylabel(_METRIC_LABELS[metric])
    axis.set_xlabel(group)
    axis.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    if filepath is not None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=150)
    return fig


def plot_tolerance_curve(
    df: pd.DataFrame,
    *,
    x: str = "vsi_diss",
    y: str = "ele_rmse",
    axis: Optional[plt.Axes] = None,
    filepath: Optional[Path] = None,
    color_by_subject: bool = True,
):
    """Behavioural metric vs. VSI dissimilarity — the tolerance plot.

    Each point is one localization run.  By default, runs from the
    same subject share a colour, so you can see whether a condition is
    consistently harder or just noisy.
    """
    _check_metric(x)
    _check_metric(y)
    if df.empty:
        raise ValueError("df is empty — nothing to plot.")

    if axis is None:
        fig, axis = plt.subplots(figsize=(6, 5))
    else:
        fig = axis.figure

    if color_by_subject and "subject" in df.columns:
        for sid, sub in df.groupby("subject"):
            axis.scatter(sub[x], sub[y], label=sid, alpha=0.8)
        axis.legend(title="subject", fontsize=8)
    else:
        axis.scatter(df[x], df[y], alpha=0.8)

    if "condition" in df.columns:
        for _, row in df.iterrows():
            try:
                axis.annotate(
                    str(row["condition"]),
                    (row[x], row[y]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=6, alpha=0.6,
                )
            except Exception:
                pass

    axis.set_xlabel(_METRIC_LABELS[x])
    axis.set_ylabel(_METRIC_LABELS[y])
    axis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if filepath is not None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=150)
    return fig
