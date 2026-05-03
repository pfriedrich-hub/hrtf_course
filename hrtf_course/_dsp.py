"""Internal DSP helpers for band-limited HRTF manipulations.

These are *not* part of the public student API.  Public functions live in
``hrtf_course.manipulations``.

The strategy is identical for both ``shift_band`` and ``smooth_band``:

1.  Take the rFFT magnitude of each HRIR (left/right, per source).
2.  Build a smooth band window ``w(f)`` that is 1 inside ``[low, high]`` and
    tapers to 0 in a small skirt outside.
3.  Compute a *modified* magnitude ``mag_mod`` (warped or smoothed).
4.  Blend in the magnitude domain::

        mag_out = (1 - w) * mag_in  +  w * mag_mod

5.  Reconstruct a minimum-phase HRIR and restore the original ITD via the
    onset-shift helper from ``hrtf_relearning.hrtf.processing.modify``.

We deliberately re-use the modify.py primitives rather than reimplementing
them, so the teaching layer benefits from any future improvements to the
underlying DSP.
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy

# Vendored DSP primitives — see _dsp_primitives.py for provenance.
from hrtf_course._dsp_primitives import (
    _smooth as _cepstral_smooth,
    minimum_phase_from_magnitude,
    restore_itd_from_onsets,
)


__all__ = [
    "band_window",
    "rebuild_hrir",
    "apply_per_source",
]


def band_window(
    freqs: numpy.ndarray,
    low_hz: float,
    high_hz: float,
    skirt_octaves: float = 0.25,
) -> numpy.ndarray:
    """Smooth in-band window on a log-frequency axis.

    The window is 1 inside ``[low_hz, high_hz]`` and tapers to 0 over a skirt
    of ``skirt_octaves`` on each side using a raised-cosine in log frequency.
    DC and Nyquist are forced to 0 so the manipulation is phase-coherent and
    cannot blow up the bandlimits during minimum-phase reconstruction.

    Parameters
    ----------
    freqs : array
        Frequency axis (Hz), monotonically increasing.  Typically the output
        of ``numpy.fft.rfftfreq``.
    low_hz, high_hz : float
        Pass-band edges (Hz).  ``high_hz > low_hz > 0`` required.
    skirt_octaves : float
        Width of the cosine taper on each side, in octaves.  Default 0.25
        (≈ a 1/4-octave skirt).

    Returns
    -------
    array of shape ``freqs.shape``, values in [0, 1].
    """
    if not (0 < low_hz < high_hz):
        raise ValueError(f"need 0 < low_hz < high_hz, got {low_hz}, {high_hz}")
    if skirt_octaves < 0:
        raise ValueError(f"skirt_octaves must be >= 0, got {skirt_octaves}")

    f = numpy.asarray(freqs, dtype=float)
    w = numpy.zeros_like(f)

    # Avoid log(0) by treating DC separately
    pos = f > 0
    if not numpy.any(pos):
        return w

    f_pos = f[pos]
    log_f = numpy.log2(f_pos)
    log_lo = numpy.log2(low_hz)
    log_hi = numpy.log2(high_hz)

    skirt = float(skirt_octaves)

    if skirt == 0:
        # hard band — still leave Nyquist untouched
        w_pos = ((log_f >= log_lo) & (log_f <= log_hi)).astype(float)
    else:
        w_pos = numpy.zeros_like(f_pos)

        # raised-cosine ramp up on the low side
        ramp_up = (log_f >= log_lo - skirt) & (log_f < log_lo)
        x = (log_f[ramp_up] - (log_lo - skirt)) / skirt
        w_pos[ramp_up] = 0.5 * (1 - numpy.cos(numpy.pi * x))

        # flat top
        flat = (log_f >= log_lo) & (log_f <= log_hi)
        w_pos[flat] = 1.0

        # raised-cosine ramp down on the high side
        ramp_dn = (log_f > log_hi) & (log_f <= log_hi + skirt)
        x = (log_f[ramp_dn] - log_hi) / skirt
        w_pos[ramp_dn] = 0.5 * (1 + numpy.cos(numpy.pi * x))

    w[pos] = w_pos

    # Force DC and Nyquist to 0 — never touch them
    w[0] = 0.0
    if f.size and f[-1] > 0:
        w[-1] = 0.0

    return w


def rebuild_hrir(
    ir_original: numpy.ndarray,
    mag_processed: numpy.ndarray,
    onset_threshold_db: float = 15.0,
) -> numpy.ndarray:
    """Reconstruct a (n_samples, 2) HRIR from a modified magnitude spectrum.

    Wraps ``minimum_phase_from_magnitude`` and ``restore_itd_from_onsets`` from
    ``hrtf_relearning.hrtf.processing.modify``.

    Parameters
    ----------
    ir_original : array, shape (n_samples, 2)
        The original HRIR — used only to recover the inter-aural onset offset.
    mag_processed : array, shape (n_bins, 2)
        Modified one-sided magnitude spectrum.  ``n_bins == n_samples//2 + 1``.
    onset_threshold_db : float
        Threshold for onset detection used by the ITD restoration step.

    Returns
    -------
    array, shape (n_samples, 2)
    """
    n_samples = ir_original.shape[0]
    spec = minimum_phase_from_magnitude(mag_processed)
    ir = numpy.fft.irfft(spec, n=n_samples, axis=0)
    return restore_itd_from_onsets(
        ir_original, ir, threshold_db=onset_threshold_db
    )


def apply_per_source(
    hrtf,
    process_mag: Callable[[numpy.ndarray, numpy.ndarray, float, float], numpy.ndarray],
    *,
    onset_threshold_db: float = 15.0,
):
    """Apply ``process_mag`` to every source of an ``slab.HRTF``.

    ``process_mag(mag_in, freqs, azimuth, elevation) -> mag_out`` is called
    for each source's HRIR.  ``mag_in`` is the one-sided magnitude (``n_bins,
    2``); ``freqs`` is the corresponding frequency axis.  The function
    returns a modified magnitude of the same shape.  The HRIR is then
    rebuilt with minimum-phase + ITD restoration.

    A deep copy of the HRTF is returned — the input is never mutated.
    """
    out = copy.deepcopy(hrtf)
    for filt, source in zip(out, out.sources.vertical_polar):
        az, el = float(source[0]), float(source[1])
        ir = numpy.asarray(filt.data, dtype=float)
        if ir.ndim != 2 or ir.shape[1] != 2:
            raise ValueError(
                f"each HRIR must have shape (n_samples, 2), got {ir.shape}"
            )

        n_samples = ir.shape[0]
        fs = filt.samplerate
        freqs = numpy.fft.rfftfreq(n_samples, d=1.0 / fs)

        spec = numpy.fft.rfft(ir, axis=0)
        mag_in = numpy.abs(spec)

        mag_out = process_mag(mag_in, freqs, az, el)

        if mag_out.shape != mag_in.shape:
            raise ValueError(
                f"process_mag must preserve shape, got {mag_out.shape} from "
                f"{mag_in.shape}"
            )

        filt.data = rebuild_hrir(
            ir, mag_out, onset_threshold_db=onset_threshold_db
        )
    return out


# Re-export for sub-modules that need the cepstral smoother.
__all__ += ["_cepstral_smooth"]
