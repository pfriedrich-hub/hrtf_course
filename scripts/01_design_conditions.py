# %% [markdown]
# # Design cue-frequency-shift conditions
#
# Pick a band on the recorded HRTF, build a few shift conditions, and
# preview them.  The manipulated SOFAs are written into
# ``data/sofa/`` — copy them over to the rig machine so the
# localization tests can be run from there.
#
# This script runs entirely on a laptop (no rig deps needed).  It
# expects a recorded ``data/sofa/{SUBJECT_ID}.sofa`` to exist; copy
# the recording from the rig before starting.

# %% --- 1. Settings ---------------------------------------------------------
SUBJECT_ID = "AGV"   # base SOFA file: data/sofa/AGV.sofa

from hrtf_course import octave_band

# 1-octave band centred on 8 kHz — adjust to where your subject's
# prominent peaks/notches actually sit (use plot_spectrogram below).
BAND_LOW, BAND_HIGH = octave_band(8000.0)

# Shift factors to test — keep the list short, ~5 min per condition at
# the rig.
SHIFT_FACTORS = {
    "shift_-10pct": 0.90,   # cues 10 % lower
    "shift_+10pct": 1.10,   # cues 10 % higher
    "shift_+20pct": 1.20,
}

# %% --- 2. Look at the recorded HRTF ---------------------------------------
import slab
import matplotlib.pyplot as plt
from hrtf_course import preview, PATH

sofa_path = PATH / "data" / "sofa" / f"{SUBJECT_ID}.sofa"
hrtf = slab.HRTF(sofa_path)

# Spectrogram with the chosen band shaded — sanity-check the band edges
preview.plot_spectrogram(hrtf, bands=[(BAND_LOW, BAND_HIGH)])
plt.show()

# %% --- 3. Build the conditions --------------------------------------------
from hrtf_course import Condition, shift_band

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

# %% --- 4. Preview each condition (no SOFA written yet) --------------------
for c in shift_conds:
    print(f"\n--- {c.name} ---")
    preview.show(c, save=True)
plt.show()

# %% [markdown]
# Things to check in the previews:
#
# * The chosen band actually shifted the way you intended.
# * VSI dissimilarity grows with the shift magnitude.
# * The spectrum *outside* the band is untouched.
#
# If something looks wrong, adjust ``BAND_LOW`` / ``BAND_HIGH`` /
# ``SHIFT_FACTORS`` and re-run cells 2–4.

# %% --- 5. Write the SOFAs --------------------------------------------------
# Once you're happy with the previews, write the SOFAs to disk.  Copy
# them onto the rig machine for localization testing.
for c in conditions:
    sofa_name = c.build_sofa(overwrite=True)
    print(f"wrote {sofa_name}.sofa")
