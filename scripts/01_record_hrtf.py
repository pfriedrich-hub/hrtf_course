# %% [markdown]
# # Day 1 — Record your partner's HRTF
#
# Pair up.  One person sits in the dome with the in-ear mics, one drives
# the rig.  Then swap.  At the end of the session every student has their
# own SOFA file at::
#
#     hrtf_relearning/data/hrtf/sofa/{SUBJECT_ID}.sofa
#
# That file is the input to Days 2 and 3.
#
# This script just orchestrates the existing ``hrtf_relearning`` recording
# pipeline — no new code.  Edit the constants in the first cell, then run
# the cells in order.

# %% --- 1. Settings ----------------------------------------------------------
SUBJECT_ID    = "STUDENT_01"           # change to your initials
REFERENCE_ID  = "ref_03.04"            # latest reference recording
HEAD_RADIUS   = 0.078                  # metres
N_DIRECTIONS  = 3                      # repetitions per direction (rig setting)
N_RECORDINGS  = 10                     # sweeps per direction
FS            = 48828
HP_FREQ       = 120                    # spherical-head extension cutoff
SHOW_PLOTS    = True

# Headphone models you want to calibrate for the next two days.
HEADPHONES    = ["DT990"]              # add "MYSPHERE" if you'll use it

# %% --- 2. Record HRIRs ------------------------------------------------------
import logging
logging.getLogger().setLevel("INFO")

from hrtf_relearning.hrtf.record.record_hrir import record_hrir

hrir = record_hrir(
    subject_id     = SUBJECT_ID,
    reference_id   = REFERENCE_ID,
    n_directions   = N_DIRECTIONS,
    n_recordings   = N_RECORDINGS,
    fs             = FS,
    hp_freq        = HP_FREQ,
    head_radius    = HEAD_RADIUS,
    show           = SHOW_PLOTS,
    overwrite_rec  = False,
    overwrite_hrir = True,
)
print("HRIR written:", hrir.name)

# %% --- 3. Calibrate headphones ----------------------------------------------
from hrtf_relearning.hrtf.record.calibration.calibrate_headphones import (
    calibrate_headphones,
)

for hp in HEADPHONES:
    print(f"--- calibrating {hp} ---")
    calibrate_headphones(SUBJECT_ID, hp, n_recordings=3, show=SHOW_PLOTS, save=True)

# %% --- 4. Quick spectrogram check -------------------------------------------
# Sanity-check what was recorded — same view you'll use to *pick* a
# manipulation band on Days 2 and 3.
import slab
from hrtf_course import preview

hrtf = slab.HRTF.kemar() if SUBJECT_ID.upper() == "KEMAR" else hrir
preview.plot_spectrogram(hrtf, ear="left", title=f"{SUBJECT_ID} — recorded HRTF")

import matplotlib.pyplot as plt
plt.show()

# %% [markdown]
# ## Done
#
# Take a screenshot of the spectrogram for your lab notebook.  Look for
# clear notches/peaks between roughly 5 and 15 kHz — those are the cues
# you'll be perturbing on Days 2 and 3.
