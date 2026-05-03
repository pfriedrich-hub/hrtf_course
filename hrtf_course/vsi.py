"""Vendored Vertical Spectral Information (VSI) metrics.

Originally from ``hrtf_relearning.hrtf.analysis.vsi``.  Inlined so
``hrtf_course`` works without the research package.

Reference
---------
Trapeau & Schönwiesner (2016). Fast and persistent adaptation to new
spectral cues for sound localization suggests a many-to-one mapping
mechanism.  J. Acoust. Soc. Am. 140(2), 879–890.

The default bandwidth (5700–11300 Hz) corresponds to the octave band with
the highest VSI and the strongest correlation with vertical localization
performance (Trapeau & Schönwiesner 2016, Fig. 2 & 5).
"""
from __future__ import annotations

import numpy

__all__ = ["vsi", "vsi_dissimilarity"]


def vsi(hrtf, bandwidth=(5700, 11300)):
    """VSI for one HRTF.

    VSI = 1 − mean of all off-diagonal entries of the autocorrelation matrix
    (Pearson r between every pair of median-plane DTFs in the band),
    averaged over left and right ear.  0 = no spectral discriminability;
    higher = better.
    """
    freqs, _ = hrtf[0].tf(show=False)
    freq_idx = numpy.logical_and(freqs >= bandwidth[0], freqs <= bandwidth[1])
    sources = hrtf.cone_sources(0)
    n = len(sources)
    n_bins = len(freqs)

    ear_vsi = []
    for ear in ("left", "right"):
        dtfs = hrtf.tfs_from_sources(sources, n_bins=n_bins, ear=ear).squeeze()[:, freq_idx]
        off_diag = [
            float(numpy.corrcoef(dtfs[i], dtfs[j])[0, 1])
            for i in range(n) for j in range(n) if i != j
        ]
        ear_vsi.append(1.0 - float(numpy.mean(off_diag)))

    return float(numpy.mean(ear_vsi))


def vsi_dissimilarity(hrtf_1, hrtf_2, bandwidth=(5700, 11300)):
    """VSI dissimilarity between two HRTFs.

    RMS distance between cross-correlation matrix (hrtf_1 vs hrtf_2) and
    autocorrelation matrix (hrtf_1 vs hrtf_1), averaged over ears.  0 =
    identical DTF correlation structure.
    """
    freqs, _ = hrtf_1[0].tf(show=False)
    freq_idx = numpy.logical_and(freqs >= bandwidth[0], freqs <= bandwidth[1])
    sources = hrtf_1.cone_sources(0)
    n = len(sources)
    n_bins = len(freqs)

    ear_dissim = []
    for ear in ("left", "right"):
        d1 = hrtf_1.tfs_from_sources(sources, n_bins=n_bins, ear=ear).squeeze()[:, freq_idx]
        d2 = hrtf_2.tfs_from_sources(sources, n_bins=n_bins, ear=ear).squeeze()[:, freq_idx]

        cross = numpy.array(
            [[numpy.corrcoef(d1[i], d2[j])[0, 1] for j in range(n)] for i in range(n)]
        )
        auto = numpy.array(
            [[numpy.corrcoef(d1[i], d1[j])[0, 1] for j in range(n)] for i in range(n)]
        )

        ear_dissim.append(float(numpy.sqrt(numpy.mean((cross - auto) ** 2))))

    return float(numpy.mean(ear_dissim))
