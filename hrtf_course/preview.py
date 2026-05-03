"""Visual sanity checks before committing to a 5-minute localization run.

Two functions:

* :func:`show` — original vs. manipulated spectrogram side by side, with
  the chosen band marked, and the VSI dissimilarity in the figure title.
  Use this in the exploration script (Day 2 / Day 3).

* :func:`plot_spectrogram` — quick mid-line spectrogram of a single HRTF,
  with optional shading of one or more candidate bands.  Use this to
  *pick* a band before designing manipulations.

Both use the vendored image/contour helpers in :mod:`hrtf_course._plot`
so they work without ``hrtf_relearning`` installed (laptop profile).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy
import slab

from hrtf_course._plot import plot as _modify_plot, _build_tf_image
from hrtf_course.vsi import vsi as _vsi, vsi_dissimilarity

from hrtf_course.conditions import Condition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-HRTF spectrogram (for picking a band)
# ---------------------------------------------------------------------------


def plot_spectrogram(
    hrtf: slab.HRTF,
    *,
    ear: str = "left",
    n_bins: Optional[int] = None,
    xlim: tuple[float, float] = (1000, 18000),
    bands: Optional[Sequence[tuple[float, float]]] = None,
    axis: Optional[plt.Axes] = None,
    title: Optional[str] = None,
):
    """Vertical-midline spectrogram of one HRTF (frequency × elevation).

    Parameters
    ----------
    hrtf : slab.HRTF
    ear : ``"left"`` | ``"right"``
    n_bins : int, optional
        FFT size for the spectrogram.  None uses the IR's natural length.
    xlim : (float, float)
        Frequency axis limits in Hz.
    bands : sequence of (low, high), optional
        Bands to shade with a vertical span — useful when picking the
        manipulation band.  Pass e.g. ``[octave_band(8000)]``.
    axis : matplotlib.axes.Axes, optional
        Existing axis to draw into.
    title : str, optional
    """
    sources = hrtf.cone_sources(0)
    freqs, elevations, img = _build_tf_image(hrtf, sources, ear, n_bins, xlim)

    if axis is None:
        fig, axis = plt.subplots(figsize=(7, 4.5))
    else:
        fig = axis.figure

    levels = numpy.linspace(float(img.min()), float(img.max()), 21)
    ct = axis.contourf(freqs, elevations, img.T, levels=levels, cmap="hot")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Elevation (°)")
    axis.set_xlim(xlim)
    if title is None:
        title = f"{getattr(hrtf, 'name', 'HRTF')} — {ear} ear"
    axis.set_title(title)
    fig.colorbar(ct, ax=axis, shrink=0.85, label="dB")

    if bands:
        for low, high in bands:
            axis.axvspan(low, high, color="cyan", alpha=0.18, lw=0)
            for f in (low, high):
                axis.axvline(f, color="cyan", linestyle="--", linewidth=1)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pre/post comparison (the real preview)
# ---------------------------------------------------------------------------


def show(
    condition: Condition,
    *,
    ear: str = "right",
    xlim: tuple[float, float] = (1000, 18000),
    vsi_bandwidth: tuple[float, float] = (5700, 11300),
    save: bool = False,
    save_dir: Optional[Path] = None,
):
    """Side-by-side spectrograms of the recorded vs. manipulated HRTF.

    Computes Trapeau & Schönwiesner (2016) VSI metrics in ``vsi_bandwidth``
    and prints them in the figure footer.  Does *not* write a SOFA — only
    the in-memory manipulation.  Use this to iterate on parameters before
    committing to a study run.

    Parameters
    ----------
    condition : Condition
    ear, xlim : passed to the underlying plotting helper.
    vsi_bandwidth : (low_hz, high_hz)
        Frequency band for VSI.  Default is the Trapeau peak band.
    save : bool
        If True, save under ``hrtf_relearning/data/results/plot/{base}/``.
    save_dir : Path, optional
        Override the save directory.

    Returns
    -------
    matplotlib.figure.Figure
    """
    base = condition.fn.__self__ if hasattr(condition.fn, "__self__") else None
    # Always reload from disk for the baseline so the comparison reflects
    # whatever's currently on the SOFA store.
    base_hrtf = slab.HRTF(condition.sofa_path.parent / f"{condition.base_subject}.sofa")
    mod_hrtf = condition.build_hrtf()

    v_orig = _vsi(base_hrtf, bandwidth=vsi_bandwidth)
    v_mod = _vsi(mod_hrtf, bandwidth=vsi_bandwidth)
    v_diss = vsi_dissimilarity(base_hrtf, mod_hrtf, bandwidth=vsi_bandwidth)

    fig = _modify_plot(
        base_hrtf, mod_hrtf, kind="image", ear=ear, xlim=xlim,
        vsi_orig=v_orig, vsi_mod=v_mod, vsi_dis=v_diss, vsi_bw=vsi_bandwidth,
    )
    fig.suptitle(
        f"{condition.base_subject}  |  {condition.name}",
        y=1.02, fontsize=11,
    )

    if save:
        if save_dir is None:
            # Default to the rig-side results tree if hrtf_relearning is
            # available; otherwise drop alongside the SOFA file (laptop).
            try:
                import hrtf_relearning
                save_dir = (
                    hrtf_relearning.PATH / "data" / "results" / "plot"
                    / condition.base_subject
                )
            except ImportError:
                save_dir = condition.sofa_path.parent / "previews"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{condition.sofa_name}_preview.png",
                    dpi=150, bbox_inches="tight")

    return fig
