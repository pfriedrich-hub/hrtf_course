# %% [markdown]
# # Day 2 — Experiment 1: tolerance for **cue frequency**
#
# Question: how much can we shift the spectral cues in a chosen band
# before vertical localization breaks down?
#
# Plan (≈ 2 h, 2–3 students):
#
# 1. Inspect your recorded HRTF, pick a band that contains prominent cues
#    (typically a 1-octave band centred somewhere between 6 and 10 kHz).
# 2. Define 4 conditions: a baseline + three frequency shifts.
# 3. Preview each condition (spectrogram + VSI dissimilarity) — make sure
#    the manipulation does what you expect *before* running the test.
# 4. Run the localization block (≈ 5 min) for each condition.
# 5. Record what each student did in the lab notebook.

# %% --- 1. Settings ----------------------------------------------------------
SUBJECT_ID = "STUDENT_01"            # student doing the localization

# Band you picked from the spectrogram — adjust to your subject!
# octave_band(8000) gives (5657, 11314) Hz — a 1-octave band around 8 kHz.
from hrtf_course import octave_band

BAND_LOW, BAND_HIGH = octave_band(8000.0)

# Shift factors to test — keep the list short, ~5 min per condition.
#
# shift_band uses a cepstral split: the broad spectral envelope (low
# quefrency, carrying externalisation) is left untouched, and only the
# fine peaks/notches (high quefrency, the actual elevation cues) are
# warped in frequency by `factor`.  Reference VSI-dissimilarity values
# on KEMAR (1-octave band around 8 kHz):
#
#   factor=0.90  -> VSI diss ~0.49
#   factor=1.10  -> VSI diss ~0.41
#   factor=1.20  -> VSI diss ~0.78
SHIFT_FACTORS = {
    "shift_-10pct": 0.90,   # elevation cues 10 % lower
    "shift_+10pct": 1.10,   # elevation cues 10 % higher
    "shift_+20pct": 1.20,
}

# %% --- 2. Build the conditions ---------------------------------------------
from hrtf_course import Condition, shift_band

# Baseline = subject's own recorded HRTF, renamed so it lines up as a
# condition in the analysis DataFrame.
baseline = Condition.identity(SUBJECT_ID)

shift_conds = [
    Condition(
        name=cname,
        base_subject=SUBJECT_ID,
        fn=lambda h, f=factor: shift_band(h, BAND_LOW, BAND_HIGH, factor=f),
        description=f"shift inside ({BAND_LOW:.0f}–{BAND_HIGH:.0f}) Hz by ×{factor}",
    )
    for cname, factor in SHIFT_FACTORS.items()
]

conditions = [baseline] + shift_conds
for c in conditions:
    print(c)

# %% --- 3. Preview each condition (no test yet) -----------------------------
from hrtf_course import preview
import matplotlib.pyplot as plt

# Build the baseline SOFA once so 'baseline' is a real file the analysis
# step can re-load for VSI dissimilarity.
baseline.build_sofa(overwrite=True)

for c in shift_conds:
    print(f"\n--- {c.name} ---")
    preview.show(c, save=True)
plt.show()

# %% [markdown]
# Stop here and check the previews:
#
# * Did the chosen band actually shift the way you intended?
# * Is the VSI dissimilarity comparable across conditions?
# * Are the spectra outside the band untouched?
#
# If anything looks wrong, tweak ``BAND_LOW``, ``BAND_HIGH`` or
# ``SHIFT_FACTORS`` and re-run cells 2–3.

# %% --- 4. Run the localization study ---------------------------------------
from hrtf_course import run_condition

# Warning: run_condition() blocks ~5 min per call and requires the
# headphones + head tracker.  Make sure the subject is comfortable.
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

# %% [markdown]
# ## Done
#
# Save the plots into the lab notebook.  After the second student has run
# their session, run ``04_analyze_pooled.py`` for the group-level view.
