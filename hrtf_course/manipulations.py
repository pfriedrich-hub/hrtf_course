"""Student-facing HRTF manipulations.

All functions take a ``slab.HRTF`` and return a *new* ``slab.HRTF`` — the
input is never modified.  They are pure (no I/O), so students can chain
them, plot intermediate results, and explore freely without polluting the
SOFA directory.

Currently implemented:

* :func:`shift_band` — extract the high-quefrency detail (sharp peaks
  and notches) from the cepstrum, warp **only that detail** in
  log-frequency inside a chosen band by a multiplicative ``factor``,
  and recombine with the unshifted envelope.  Tests tolerance for
  *cue frequency* (elevation cues move; broad envelope /
  externalisation cues stay put).

Helper utilities :func:`octave_band` and :func:`erb_bandwidth` make it
easy to choose sensible band edges from a single centre frequency.
"""
from __future__ import annotations

from typing import Tuple

import numpy

from hrtf_course._dsp import (
    band_window,
    apply_per_source,
    _cepstral_smooth,
)


__all__ = [
    "shift_band",
    "octave_band",
    "erb_bandwidth",
]


# ---------------------------------------------------------------------------
# Band helpers
# ---------------------------------------------------------------------------


def octave_band(center_hz: float, fraction: float = 1.0) -> Tuple[float, float]:
    """Return ``(low, high)`` Hz for an octave (or fractional octave) band.

    >>> octave_band(8000)
    (5656.854249..., 11313.708498...)
    >>> octave_band(8000, fraction=1/3)   # third-octave
    (7127.18..., 8979.69...)
    """
    if center_hz <= 0:
        raise ValueError(f"center_hz must be positive, got {center_hz}")
    if fraction <= 0:
        raise ValueError(f"fraction must be positive, got {fraction}")
    factor = 2.0 ** (fraction / 2.0)
    return float(center_hz / factor), float(center_hz * factor)


def erb_bandwidth(center_hz: float) -> float:
    """Glasberg & Moore (1990) equivalent rectangular bandwidth, in Hz.

    Useful for choosing band widths comparable to a single auditory filter.
    """
    f_kHz = center_hz / 1000.0
    return 24.7 * (4.37 * f_kHz + 1.0)


# ---------------------------------------------------------------------------
# Day 2 — frequency-shift inside a chosen band
# ---------------------------------------------------------------------------


def shift_band(
    hrtf,
    low_hz: float,
    high_hz: float,
    factor: float,
    *,
    envelope_n_keep: int = 3,
    skirt_octaves: float = 0.25,
    onset_threshold_db: float = 15.0,
):
    """Shift only the *fine spectral structure* inside ``[low_hz, high_hz]``
    in frequency, while leaving the broad spectral envelope in place.

    Cepstral split (matched to :func:`widen_band` so the two manipulations
    operate on the same decomposition):

    1.  ``envelope = cepstral_lifter(log|H|, n_keep=envelope_n_keep)`` —
        the broad spectral slope, low-quefrency components.  Carries
        coarse / general-externalisation information.
    2.  ``detail = log|H| - envelope`` — the high-quefrency residual:
        sharp peaks and notches that carry vertical-localisation cues.
    3.  Warp only ``detail`` in log-frequency by ``factor`` (each output
        frequency ``f`` takes ``detail`` at ``f / factor``).
    4.  ``new_log_mag = envelope + window * detail_warped`` — recombine
        with the in-band window so the warped detail vanishes outside the
        band; the envelope itself is left untouched.
    5.  Reconstruct minimum-phase HRIR; restore original ITD.

    Compared to a plain magnitude warp inside the band, this redesign
    moves *only* the elevation cues and leaves the broad envelope
    (externalisation, spectral colouration) where it was — closer to the
    perceptual question Day 2 is asking.

    Parameters
    ----------
    hrtf : slab.HRTF
        Input HRTF.  Not modified.
    low_hz, high_hz : float
        Band edges in Hz.
    factor : float
        Multiplicative shift applied to the high-quefrency detail.

        * ``factor > 1`` shifts the cues **up** in frequency
          (e.g. ``1.10`` ≈ +10 %, roughly +1.4 semitones).
        * ``factor < 1`` shifts the cues **down**.
        * ``factor == 1`` is a no-op (still re-builds the HRIR, so it can
          be used as a sanity check).
    envelope_n_keep : int
        Cepstral coefficients retained for the envelope (default 3 —
        matches :func:`widen_band`).  Lower → more "detail" gets shifted;
        higher → only the very sharpest peaks/notches move.
    skirt_octaves : float
        Width of the cosine taper outside the band (default 0.25 octave).
    onset_threshold_db : float
        Threshold for onset-based ITD restoration.

    Returns
    -------
    slab.HRTF
        New HRTF with the band-shifted detail.
    """
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    if envelope_n_keep < 1:
        raise ValueError(f"envelope_n_keep must be >= 1, got {envelope_n_keep}")

    def _process(mag_in: numpy.ndarray, freqs: numpy.ndarray, az: float, el: float):
        eps = numpy.finfo(float).tiny
        log_mag_db = 20.0 * numpy.log10(numpy.maximum(mag_in, eps))

        # 1. envelope (low-quefrency)
        envelope_mag = _cepstral_smooth(mag_in, n_keep=int(envelope_n_keep))
        envelope_db = 20.0 * numpy.log10(numpy.maximum(envelope_mag, eps))

        # 2. detail (high-quefrency)
        detail_db = log_mag_db - envelope_db

        # 3. warp detail in log-frequency by `factor`
        src_freqs = freqs / factor
        detail_warped = numpy.empty_like(detail_db)
        for ch in range(detail_db.shape[1]):
            detail_warped[:, ch] = numpy.interp(
                src_freqs, freqs, detail_db[:, ch],
                left=detail_db[0, ch], right=detail_db[-1, ch],
            )

        # 4. taper warped detail with the band window so it vanishes
        #    outside; the envelope is preserved everywhere.
        w = band_window(freqs, low_hz, high_hz, skirt_octaves=skirt_octaves)
        new_log_mag = envelope_db + w[:, None] * detail_warped
        new_mag = 10.0 ** (new_log_mag / 20.0)

        # blend in-band only; outside the skirt w=0 ⇒ mag_out = mag_in
        mag_out = (1.0 - w[:, None]) * mag_in + w[:, None] * new_mag
        return mag_out

    return apply_per_source(
        hrtf, _process, onset_threshold_db=onset_threshold_db
    )


