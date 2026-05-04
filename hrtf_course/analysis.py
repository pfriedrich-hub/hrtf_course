"""Pool localization results across subjects and conditions.

The flow is:

1.  :func:`collect_results` walks every ``Subject`` pickle for the requested
    subject ids, finds every saved localization sequence, parses the
    condition name out of the SOFA name embedded in the sequence, and
    computes per-run summary stats (re-using
    ``hrtf_relearning.localization_accuracy``) plus the Trapeau & Schönwiesner
    (2016) VSI dissimilarity vs. the subject's own baseline HRTF.

2.  :func:`plot_compare` plots a chosen metric (``ele_rmse``,
    ``ele_gain``, ``ele_sd``, …) per condition with one point per run and
    a mean overlay.

3.  :func:`plot_tolerance_curve` plots a behavioural metric *against*
    VSI dissimilarity — the headline "tolerance" plot for the seminar.

4.  :func:`add_baumgartner_predictions` (opt-in) wraps the existing
    Baumgartner 2014 model so a comparison column can be added to ``df``.
    Currently a thin stub — see its docstring for the next step.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import slab

# Vendored helpers — work on the laptop profile without hrtf_relearning.
from hrtf_course.vsi import vsi_dissimilarity
from hrtf_course.conditions import SOFA_DIR
from hrtf_course._subject import Subject
from hrtf_course._localization_metrics import localization_accuracy

# The Baumgartner 2014 model lives in hrtf_relearning and is the only
# thing in this module that still requires the research package — it's
# lazy-imported inside add_baumgartner_predictions().

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _condition_from_sofa_name(sofa_name: str, base_subject: str) -> str:
    """Strip the ``{base_subject}_`` prefix off a SOFA name.

    Sequences whose SOFA is exactly ``{base_subject}.sofa`` (i.e. the
    untouched recorded HRTF) are returned as ``"recorded"`` so they appear
    as their own condition in the DataFrame.
    """
    prefix = f"{base_subject}_"
    if sofa_name == base_subject:
        return "recorded"
    if sofa_name.startswith(prefix):
        return sofa_name[len(prefix):]
    return sofa_name  # foreign HRTF (e.g. testing A on B) — keep as-is


def _vsi_diss_safe(target_sofa: Path, baseline_sofa: Path,
                   bandwidth: tuple[float, float] = (5700, 11300)) -> float:
    """Compute VSI dissimilarity, returning NaN if the input SOFA is missing."""
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
    vsi_bandwidth: tuple[float, float] = (5700, 11300),
) -> pd.DataFrame:
    """Pool localization runs across subjects into one DataFrame.

    Parameters
    ----------
    subject_ids : iterable of str
        Subjects whose pickles to load.  Each must have a recorded
        ``{id}.sofa`` for VSI dissimilarity to be computed (skipped otherwise).
    condition_names : sequence of str, optional
        If given, restrict to these conditions (after the
        ``{base_subject}_`` prefix is stripped).  ``None`` returns every
        condition found.
    vsi_bandwidth : (low_hz, high_hz)
        Frequency band for VSI computation.  Default 5700–11300 Hz, the peak
        VSI band from Trapeau & Schönwiesner (2016).

    Returns
    -------
    pandas.DataFrame
        Columns:

        * ``subject``     — tester id (the human in the chair)
        * ``base_hrtf``   — SOFA name written to disk (e.g. ``AGV_shift...``)
        * ``base_subject``— subject id whose SOFA was manipulated
        * ``condition``   — condition name (``"baseline"``, ``"shift_..."``)
        * ``run``         — sequence filename (timestamped, unique)
        * ``n_trials``    — number of completed trials in the sequence
        * ``ele_gain``    — slope of response_el vs. target_el
        * ``ele_rmse``    — elevation RMSE in degrees
        * ``ele_sd``      — within-sector elevation SD
        * ``az_gain``     — slope of response_az vs. target_az (NaN if midline)
        * ``az_rmse``     — azimuth RMSE in degrees
        * ``az_sd``       — within-sector azimuth SD
        * ``vsi_diss``    — VSI dissimilarity vs. ``{base_subject}.sofa``
    """
    rows: list[dict] = []
    requested = set(condition_names) if condition_names is not None else None
    # Cache VSI computations per (target, base) SOFA pair — they're expensive.
    vsi_cache: dict[tuple[str, str], float] = {}

    for sid in subject_ids:
        try:
            subj = Subject(sid)
        except Exception:
            logger.exception("Could not load subject %s", sid)
            continue

        for run_name, sequence in (subj.localization or {}).items():
            sofa_name = getattr(sequence, "hrir", None)
            if sofa_name is None:
                logger.debug("Skipping %s — no .hrir on sequence", run_name)
                continue

            base_subject = sofa_name.split("_")[0]
            cond = _condition_from_sofa_name(sofa_name, base_subject)
            if requested is not None and cond not in requested:
                continue

            try:
                eg, e_rmse, e_sd, ag, a_rmse, a_sd = localization_accuracy(sequence)
            except Exception:
                logger.exception("localization_accuracy failed for %s", run_name)
                eg = e_rmse = e_sd = ag = a_rmse = a_sd = float("nan")

            # n_trials: count of recorded trials, regardless of whether the
            # sequence was completed (lets you compare partial runs too).
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
        # Stable sort so plots come out in a sensible order
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

    Each subject contributes one dot per run; the mean across runs (and
    subjects) is shown as a horizontal bar.  Useful for the seminar slide
    that compares conditions at a glance.
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
        # Light horizontal jitter to separate overlapping runs
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
    """Headline plot — behavioural metric vs. VSI dissimilarity.

    Each point is one localization run.  By default, runs from the same
    subject share a colour, so you can read off whether a condition is
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

    # Optional condition labels next to each dot (small, ~6pt) so the plot
    # is self-documenting in seminar slides.
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


# ---------------------------------------------------------------------------
# Optional model comparison
# ---------------------------------------------------------------------------


def add_baumgartner_predictions(
    df: pd.DataFrame,
    *,
    sig_path: Optional[Path] = None,
    do_dtf: bool = False,
    column: str = "model_pe",
) -> pd.DataFrame:
    """Add a Baumgartner-2014 polar-error column to ``df`` (opt-in).

    For each row, run
    ``hrtf_relearning.hrtf.models.baumgartner2014.baumgartner2014`` with
    the manipulated SOFA as the *target* and the subject's baseline SOFA
    as the *template*.  The returned polar error is added as ``column``.

    .. note::

       This is intentionally a thin wrapper.  The Baumgartner module's
       current entry point in this repo expects positional path arguments
       and saves figures to disk; it returns a dictionary of results that
       includes a polar-error scalar.  The exact extraction may need
       adjusting as the model module evolves — the implementation below
       guards against that and returns NaN if it cannot find a scalar.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`collect_results`.
    sig_path : Path, optional
        Stimulus WAV used by the model.  If None, a built-in pink-noise
        stimulus from ``hrtf_relearning/data/sounds`` is used.
    do_dtf : bool
        Whether the model should diffuse-field-equalize.  See the model
        docstring.
    column : str
        Name of the new column.

    Returns
    -------
    DataFrame
        Same as input, with one extra column.
    """
    try:
        import hrtf_relearning  # type: ignore[import-not-found]
        from hrtf_relearning.hrtf.models.baumgartner2014 import baumgartner2014
    except Exception:
        logger.exception("Could not import baumgartner2014 model.")
        df = df.copy()
        df[column] = float("nan")
        return df

    if sig_path is None:
        candidate = (
            hrtf_relearning.PATH / "data" / "sounds" / "1s_pinknoise_44100.wav"
        )
        sig_path = candidate if candidate.exists() else None

    df = df.copy()
    preds: list[float] = []
    for _, row in df.iterrows():
        target = SOFA_DIR / f"{row['base_hrtf']}.sofa"
        template = SOFA_DIR / f"{row['base_subject']}.sofa"
        if not target.exists() or not template.exists():
            preds.append(float("nan"))
            continue
        try:
            result = baumgartner2014(
                target, template, sig_path,
                shutup=True, do_dtf=do_dtf,
                fig_savepath=None, is_save=False,
            )
        except TypeError:
            # Older signatures expect different positional args — give up
            # gracefully and let the user wire it in by hand.
            logger.warning(
                "baumgartner2014 signature mismatch.  Fill in this wrapper "
                "for your installed version of hrtf_relearning."
            )
            preds.append(float("nan"))
            continue
        except Exception:
            logger.exception("baumgartner2014 failed for %s", row["base_hrtf"])
            preds.append(float("nan"))
            continue

        # Try to pull a scalar polar-error metric out of whatever the model
        # returned.  Adjust as needed for your concrete model output shape.
        scalar = _extract_polar_error(result)
        preds.append(scalar)

    df[column] = preds
    return df


def _extract_polar_error(result):
    """Best-effort scalar extraction from the Baumgartner model output."""
    if result is None:
        return float("nan")
    if isinstance(result, (int, float, numpy.floating)):
        return float(result)
    if isinstance(result, dict):
        for key in ("polar_error", "pe", "PE", "rmse_polar"):
            if key in result:
                try:
                    return float(numpy.asarray(result[key]).squeeze())
                except Exception:
                    pass
    return float("nan")
