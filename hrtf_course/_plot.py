"""Vendored plotting helpers — frequency × elevation contour plots.

Originally lived in ``hrtf_relearning.hrtf.processing.modify``.  Inlined
here so previews work on student laptops without the research package.
"""
from __future__ import annotations

import numpy
import matplotlib
import matplotlib.ticker
from matplotlib import pyplot as plt

__all__ = ["_build_tf_image", "plot"]


def _build_tf_image(hrtf, sourceidx, ear, n_bins, xlim, floor_db=-25):
    """Build a (freqs, elevations, dB image) triple for a contour plot.

    Returns
    -------
    freqs      : 1-D array, frequency axis trimmed to xlim[1]
    elevations : 1-D array, elevation for each source
    img        : 2-D array (n_freq_bins, n_sources), clipped at floor_db
    """
    chan = {"left": 0, "right": 1}[ear]
    n_b = n_bins if n_bins is not None else hrtf[sourceidx[0]].n_taps
    tfs = hrtf.tfs_from_sources(sourceidx, n_bins=n_b, ear=ear)
    img = numpy.clip(tfs.squeeze(-1).T, floor_db, None)
    freqs, _ = hrtf[sourceidx[0]].tf(channels=chan, n_bins=n_bins, show=False)
    elevations = hrtf.sources.vertical_polar[sourceidx, 1]
    mask = freqs <= xlim[1]
    return freqs[mask], elevations, img[mask, :]


def plot(hrtf, hrtf_modified, kind="image", ear="left", n_bins=None,
         xlim=(1000, 18000),
         vsi_orig=None, vsi_mod=None, vsi_dis=None, vsi_bw=None):
    """Side-by-side spectrogram comparison (original vs. modified).

    Parameters mirror the original ``modify.plot`` helper.
    """
    sources = hrtf.cone_sources(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if kind == "image":
        freqs, elevations, img_orig = _build_tf_image(hrtf, sources, ear, n_bins, xlim)
        _, _, img_mod = _build_tf_image(hrtf_modified, sources, ear, n_bins, xlim)

        vmin = float(min(img_orig.min(), img_mod.min()))
        vmax = float(max(img_orig.max(), img_mod.max()))
        levels = numpy.linspace(vmin, vmax, 21)

        ct = None
        for ax, img, title, vsi_val in zip(
                axes,
                [img_orig, img_mod],
                ["original", "modified"],
                [vsi_orig, vsi_mod]):
            ct = ax.contourf(freqs, elevations, img.T, cmap="hot", levels=levels)
            ax.set_title(title)
            xlabel = "Frequency [kHz]"
            if vsi_val is not None:
                xlabel += f"\nVSI = {vsi_val:.3f}"
            ax.set(xlabel=xlabel, ylabel="Elevation [°]", xlim=xlim)
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(x / 1000))))
            ax.autoscale(tight=True)
            ax.tick_params("both", length=2, pad=2)

        cbar_ticks = numpy.arange(vmin, vmax, 6)
        cax_pos = list(axes[-1].get_position().bounds)
        cax_pos[2] = cax_pos[2] * 0.06
        cax_pos[0] = 0.92
        cbar_axis = fig.add_axes(cax_pos)
        fig.colorbar(ct, cbar_axis, orientation="vertical", ticks=cbar_ticks)

        if vsi_dis is not None:
            bw_str = (f"{vsi_bw[0]/1000:.1f}–{vsi_bw[1]/1000:.1f} kHz"
                      if vsi_bw is not None else "")
            fig.text(0.5, 0.0,
                     f"VSI dissimilarity = {vsi_dis:.3f}   ({bw_str}, Trapeau et al. 2016)",
                     ha="center", va="bottom", fontsize=9)
        else:
            plt.tight_layout()
    else:
        hrtf.plot_tf(sources, kind=kind, axis=axes[0], ear=ear, xlim=xlim, show=False)
        hrtf_modified.plot_tf(sources, kind=kind, axis=axes[1], ear=ear, xlim=xlim, show=False)
        axes[0].set_title("original")
        axes[1].set_title("modified")
        plt.tight_layout()

    plt.show(block=False)
    plt.pause(0.1)
    return fig
