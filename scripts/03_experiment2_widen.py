# %% [markdown]
# # Day 3 — Experiment 2: tolerance for **cue shape**
#
# Question: how much can we *widen* the spectral cues in a chosen band
# (i.e. blur peaks/notches in frequency, keeping their centre frequency
# and dB depth fixed) before vertical localization breaks down?  This is
# the shape-only counterpart to Day 2's frequency-shift experiment:
#
# | Day | manipulation         | what changes                    | what's preserved                  |
# |----:|----------------------|----------------------------------|-----------------------------------|
# | 2   | ``shift_band``       | centre frequency                 | shape (sharpness) + depth         |
# | 3   | ``widen_band``       | sharpness (Q) of each feature    | centre frequency + dB depth       |
#
# Plan (≈ 2 h, 2–3 students):
#
# 1. Use the *same* band as Day 2 so the comparison is clean.
# 2. Define 4 conditions: baseline + three widening levels.
# 3. Preview, then run the localization block per condition.
# 4. Pool with Day 2 in ``04_analyze_pooled.py``.

# %% --- 1. Settings ----------------------------------------------------------
SUBJECT_ID = "STUDENT_01"

from hrtf_course import octave_band
BAND_LOW, BAND_HIGH = octave_band(8000.0)   # same band you used on Day 2

# Gaussian feature widths in **octaves** (sigma).  Each detected
# peak/notch is reinjected as a Gaussian in log-frequency at its original
# centre and signed dB amplitude — sharpness changes, centre + depth are
# preserved as long as features are isolated; once sigma exceeds the
# inter-feature spacing, neighbouring peaks merge into broader shapes.
#
# These three sigmas were calibrated on KEMAR so the VSI dissimilarity
# range roughly matches the Day-2 shift conditions:
#
#   sigma=0.10 oct  ->  VSI diss ~0.23   (≈ shift ±5%)
#   sigma=0.20 oct  ->  VSI diss ~0.42   (≈ shift +10%)
#   sigma=0.40 oct  ->  VSI diss ~0.70   (≈ shift +18%)
WIDEN_LEVELS = {
    "widen_w10": 0.10,
    "widen_w20": 0.20,
    "widen_w40": 0.40,
}

# %% --- 2. Build the conditions ---------------------------------------------
from hrtf_course import Condition, widen_band

baseline = Condition.identity(SUBJECT_ID)

widen_conds = [
    Condition(
        name=cname,
        base_subject=SUBJECT_ID,
        fn=lambda h, w=width: widen_band(h, BAND_LOW, BAND_HIGH, width_octaves=w),
        description=(
            f"peak/notch widening inside ({BAND_LOW:.0f}–{BAND_HIGH:.0f}) Hz, "
            f"sigma={width:.2f} octaves; centre freq + dB depth preserved"
        ),
    )
    for cname, width in WIDEN_LEVELS.items()
]

conditions = [baseline] + widen_conds
for c in conditions:
    print(c)

# %% --- 3. Preview each condition -------------------------------------------
from hrtf_course import preview
import matplotlib.pyplot as plt

baseline.build_sofa(overwrite=True)

for c in widen_conds:
    print(f"\n--- {c.name} ---")
    preview.show(c, save=True)
plt.show()

# %% [markdown]
# Look at the previews:
#
# * Each peak/notch should sit at the *same frequency* as in baseline,
#   with the *same dB depth* — only the slope around it should be
#   shallower as ``width_octaves`` grows.
# * VSI dissimilarity should grow monotonically with ``width_octaves``
#   (less elevation discriminability).
# * The spectrum *outside* the band should be untouched.

# %% --- 4. Run the localization study ---------------------------------------
from hrtf_course import run_condition

for c in conditions:
    input(f"\nReady for condition '{c.name}'?  Press Enter to start.")
    run_condition(SUBJECT_ID, c)

# %% --- 5. Quick within-subject summary -------------------------------------
from hrtf_course import collect_results, plot_compare, plot_tolerance_curve

df = collect_results([SUBJECT_ID],
                     condition_names=[c.name for c in conditions])
print(df[["condition", "n_trials", "ele_gain", "ele_rmse", "vsi_diss"]])

plot_compare(df, metric="ele_rmse", group="condition")
plot_tolerance_curve(df, x="vsi_diss", y="ele_rmse")
plt.show()
