"""Vendored localization-accuracy metrics.

Originally lived in
``hrtf_relearning.experiment.analysis.localization.localization_analysis``.
Inlined so the laptop profile can compute per-run summary stats from
``Subject`` pickles without needing the full research package.

The matplotlib backend setup from the original module is dropped — this
module is pure NumPy / SciPy so it works headless on any install.
"""
from __future__ import annotations

import numpy
import scipy.stats

__all__ = ["localization_accuracy", "compute_sector_precision"]


def localization_accuracy(sequence):
    """Per-run summary statistics for one localization sequence.

    Returns
    -------
    (elevation_gain, ele_rmse, ele_sd, azimuth_gain, az_rmse, az_sd)
        All NaN if the sequence is empty / not yet started.
        ``azimuth_gain`` is ``None`` when all targets share one azimuth
        (median plane only).
    """
    if (sequence.this_n == -1
            or sequence.n_remaining == len(sequence.data)
            or not sequence.data):
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan

    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets = loc_data[:, 1]   # [az, ele]
    responses = loc_data[:, 0]

    # elevation gain — slope of response_el vs. target_el
    try:
        elevation_gain, _ = scipy.stats.linregress(
            targets[:, 1], responses[:, 1])[:2]
    except ValueError:
        elevation_gain = 0

    # azimuth gain — None on the median plane
    if not len(numpy.unique(targets[:, 0])) == 1:
        azimuth_gain, _ = scipy.stats.linregress(
            targets[:, 0], responses[:, 0])[:2]
    else:
        azimuth_gain = None

    rmse = numpy.sqrt(numpy.mean(numpy.square(targets - responses), axis=0))
    az_rmse, ele_rmse = rmse[0], rmse[1]

    az_sd, ele_sd = compute_sector_precision(
        targets, responses,
        sequence.settings["sector_centers"],
        sequence.settings["sector_size"],
    )
    return elevation_gain, ele_rmse, ele_sd, azimuth_gain, az_rmse, az_sd


def compute_sector_precision(targets, responses, sector_centers, sector_size):
    """Mean within-sector response SD (azimuth, elevation), in degrees.

    For each sector with at least 2 targets, shift responses to a common
    origin and take the SD of the shifted responses; then average over
    sectors.
    """
    az_size, el_size = sector_size
    per_sector_std = []
    for center in sector_centers:
        az_min = center[0] - az_size / 2
        az_max = center[0] + az_size / 2
        el_min = center[1] - el_size / 2
        el_max = center[1] + el_size / 2
        in_sector = numpy.where(
            (targets[:, 0] >= az_min) & (targets[:, 0] < az_max) &
            (targets[:, 1] >= el_min) & (targets[:, 1] < el_max)
        )[0]
        if len(in_sector) >= 2:
            response_shift = responses[in_sector] - targets[in_sector]
            az_std = numpy.std(response_shift[:, 0])
            el_std = numpy.std(response_shift[:, 1])
            per_sector_std.append((az_std, el_std))

    if per_sector_std:
        per_sector_std = numpy.array(per_sector_std)
        return tuple(numpy.mean(per_sector_std, axis=0))
    return (numpy.nan, numpy.nan)
