"""Vendored DSP primitives.

Originally lived in ``hrtf_relearning.hrtf.processing.modify``.  Inlined
here so ``hrtf_course`` is self-contained and student laptops don't need
the heavy research package installed.

Anything that's *only* used at the rig (Localization runner, Subject
loader, recording pipeline) is still pulled from ``hrtf_relearning`` via
lazy imports — see ``study.py`` and ``analysis.py``.

If a future version of ``hrtf_relearning`` improves these primitives,
re-sync them by re-copying.
"""
from __future__ import annotations

import numpy

__all__ = [
    "_smooth",
    "minimum_phase_from_magnitude",
    "find_ir_onsets",
    "restore_itd_from_onsets",
]


def _smooth(mag, n_keep):
    """Smooth a one-sided magnitude spectrum via truncated cosine-series
    reconstruction of log-magnitude (Kulkarni & Colburn 1998).

    Parameters
    ----------
    mag : array (n_bins, n_channels)
    n_keep : int
        Cosine coefficients retained.  Lower → smoother spectrum.
    """
    mag = numpy.asarray(mag, dtype=float)
    if mag.ndim != 2:
        raise ValueError("mag must have shape (n_bins, n_channels)")
    n_bins, n_channels = mag.shape
    n_samples = 2 * (n_bins - 1)
    n_keep = int(n_keep)
    if n_keep < 1 or n_keep > n_bins:
        raise ValueError(f"n_keep must be between 1 and {n_bins}, got {n_keep}")

    log_mag = numpy.log(numpy.maximum(mag, numpy.finfo(float).tiny))
    k = numpy.arange(n_bins, dtype=float)[:, None]
    n = numpy.arange(n_bins, dtype=float)[None, :]
    basis = numpy.cos(2.0 * numpy.pi * k * n / float(n_samples))

    log_mag_smooth = numpy.empty_like(log_mag)
    for ch in range(n_channels):
        coeffs, _, _, _ = numpy.linalg.lstsq(basis, log_mag[:, ch], rcond=None)
        coeffs[n_keep:] = 0.0
        log_mag_smooth[:, ch] = basis @ coeffs

    return numpy.exp(log_mag_smooth)


def minimum_phase_from_magnitude(mag):
    """Real-cepstrum minimum-phase reconstruction from a one-sided magnitude
    spectrum.

    Parameters
    ----------
    mag : array (n_bins, n_channels)

    Returns
    -------
    spec_min : array (n_bins, n_channels), complex
    """
    mag = numpy.asarray(mag, dtype=float)
    n_bins, n_channels = mag.shape
    n_samples = 2 * (n_bins - 1)
    tiny = numpy.finfo(float).tiny
    spec_min = numpy.empty((n_bins, n_channels), dtype=complex)

    for ch in range(n_channels):
        mag_ch = numpy.maximum(mag[:, ch], tiny)
        log_mag_half = numpy.log(mag_ch)
        log_mag_full = numpy.concatenate((log_mag_half, log_mag_half[-2:0:-1]))
        cep = numpy.fft.ifft(log_mag_full).real
        cep_min = numpy.zeros_like(cep)
        cep_min[0] = cep[0]
        cep_min[1:n_samples // 2] = 2.0 * cep[1:n_samples // 2]
        cep_min[n_samples // 2] = cep[n_samples // 2]
        spec_min[:, ch] = numpy.exp(numpy.fft.fft(cep_min))[:n_bins]

    return spec_min


def find_ir_onsets(ir, threshold_db=15.0):
    """Onset sample for each channel — first sample within ``threshold_db``
    of the peak."""
    ir = numpy.asarray(ir, dtype=float)
    n_samples, n_channels = ir.shape
    onsets = numpy.zeros(n_channels, dtype=int)
    for ch in range(n_channels):
        x = numpy.abs(ir[:, ch])
        peak_idx = int(numpy.argmax(x))
        peak_val = float(x[peak_idx])
        if peak_val <= 0:
            continue
        threshold = peak_val / (10.0 ** (float(threshold_db) / 20.0))
        above = numpy.where(x[:peak_idx + 1] >= threshold)[0]
        onsets[ch] = int(above[0]) if len(above) else 0
    return onsets


def restore_itd_from_onsets(ir_original, ir_processed, threshold_db=15.0):
    """Time-shift each channel of ``ir_processed`` so the per-ear onset
    sample matches ``ir_original``.  Preserves ITD across the
    minimum-phase reconstruction.
    """
    ir_original = numpy.asarray(ir_original, dtype=float)
    ir_processed = numpy.asarray(ir_processed, dtype=float)
    n_samples, n_channels = ir_original.shape
    out = numpy.zeros_like(ir_processed)
    on_orig = find_ir_onsets(ir_original, threshold_db=threshold_db)
    on_proc = find_ir_onsets(ir_processed, threshold_db=threshold_db)
    for ch in range(n_channels):
        delta = int(on_orig[ch] - on_proc[ch])
        if delta > 0:
            out[:, ch] = numpy.concatenate(
                (numpy.zeros(delta), ir_processed[:-delta, ch]))
        elif delta < 0:
            d = -delta
            out[:, ch] = numpy.concatenate(
                (ir_processed[d:, ch], numpy.zeros(d)))
        else:
            out[:, ch] = ir_processed[:, ch]
    return out
