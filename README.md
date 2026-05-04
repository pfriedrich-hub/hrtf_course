# hrtf_course

Teaching layer over [`hrtf_relearning`](https://github.com/pfriedrich-hub/hrtf_relearning) for the practical psychoacoustics course.

The course runs over three days:

| Day | What students do                                                              | Script                          |
|----:|-------------------------------------------------------------------------------|---------------------------------|
| 1   | Record each other's HRTFs on the dome                                         | `scripts/01_record_hrtf.py`     |
| 2   | Test tolerance for **cue frequency** (band-shift)                             | `scripts/02_experiment1_freq_shift.py` |
| 3   | Test tolerance for **cue shape** (peak/notch widening, fixed centre+depth)    | `scripts/03_experiment2_widen.py`  |
| —   | Pool results, prepare seminar plots                                           | `scripts/04_analyze_pooled.py`  |

The two experimental days are designed as a clean **position vs. shape** contrast:

| Day | manipulation     | what changes                  | what is preserved               |
|----:|------------------|-------------------------------|---------------------------------|
| 2   | `shift_band`     | centre frequency of cues      | shape (sharpness) + dB depth    |
| 3   | `widen_band`     | sharpness (Q) of each feature | centre frequency + dB depth     |

The `hrtf_course` package wraps existing functionality in `hrtf_relearning` — no research code is modified.

## Install

The project ships with a populated `.venv/` containing all
dependencies.  The student workflow is just:

1. Open the `hrtf_course/` folder in PyCharm.
2. PyCharm auto-detects `.venv/` as the project interpreter (if it
   doesn't, point it at `hrtf_course/.venv/bin/python` via
   Settings → Project → Python Interpreter → Add → Existing).
3. Open a script in `scripts/` and run the `# %%` cells.

That's it.

### Re-creating `.venv/` (one-time provisioning)

If the venv is missing or broken — e.g. on a fresh clone — recreate it
with [`uv`](https://docs.astral.sh/uv/):

```bash
cd ~/projects/hrtf_course
uv sync           # default: laptop profile (cue editing + preview only)
```

That's enough for everything in `scripts/02_*.py` and `scripts/03_*.py`
(cue editing, preview, VSI).  No `hrtf_relearning` clone needed, no
rig-only deps (Qt, PortAudio, MetaWear, pybinsim).

### Rig install (Day 1 recording + Day 2/3 localization)

The rig machine additionally needs `hrtf_relearning` for the recording
pipeline and the localization runner.  Clone it as a sibling and sync
the `rig` extra:

```
~/projects/
├── hrtf_relearning/    # research code (rig only)
└── hrtf_course/        # this repo
```

```bash
cd ~/projects/hrtf_course
uv sync --extra rig
```

You may need system packages first:

* macOS: `brew install qt portaudio`
* Linux: `apt install qt5-qmake portaudio19-dev`

### Bringing rig data over

The course expects SOFA recordings and Subject pickles to live inside
the course repo so the analysis path is self-contained:

```
hrtf_course/
└── data/
    ├── sofa/           # {subject}.sofa  + manipulated SOFAs
    └── results/        # {subject}.pkl   (Subject pickle from rig runs)
```

After Day 1 (HRTF recording at the rig) and Days 2–3 (localization
tests at the rig), copy the relevant files from the rig machine over:

```bash
# from hrtf_relearning's data tree → hrtf_course's data tree
cp ~/projects/hrtf_relearning/data/hrtf/sofa/AGV.sofa  ~/projects/hrtf_course/data/sofa/
cp ~/projects/hrtf_relearning/data/results/AGV.pkl    ~/projects/hrtf_course/data/results/
```

You can also override the locations entirely with the env vars
`HRTF_COURSE_SOFA_DIR` and `HRTF_COURSE_RESULTS_DIR`.

### Trying it out without a recording

Before you have your own HRTF, you can explore the manipulations on
the KEMAR reference HRTF — works on the default laptop install:

```python
import slab
from hrtf_course import shift_band, widen_band, octave_band, preview

hrtf = slab.HRTF.kemar()
low, high = octave_band(8000)

shifted = shift_band(hrtf, low, high, factor=1.10)
widened = widen_band(hrtf, low, high, width_octaves=0.20)

# Quick visual sanity check
preview.plot_spectrogram(hrtf, bands=[(low, high)])
```

## What the package gives you

```python
from hrtf_course import (
    # Manipulations (return new slab.HRTF — input not modified)
    shift_band,        # cepstral split: warp ONLY the high-quefrency
                       #   detail (sharp peaks/notches) inside [low, high];
                       #   broad envelope (externalisation) untouched
    widen_band,        # widen peaks/notches inside [low, high]
                       #   (centre freq + dB depth preserved)
    octave_band,       # convenience: octave_band(8000) -> (5657, 11314) Hz

    # Conditions + study runner
    Condition,         # name + base subject + manipulation function
    run_condition,     # build SOFA + run 5-min localization block (rig only)

    # Analysis  (works on rig-data pickles copied into data/results/)
    collect_results,   # walk Subject pickles -> tidy DataFrame
    plot_compare,      # per-condition point clouds (one row per run)

    # Preview
    preview,           # spectrogram + VSI dissimilarity, no SOFA written
)
```

### Designing a condition (laptop)

Iterate on a manipulation against a recorded subject's SOFA — picking
a band, previewing, and writing the manipulated SOFA out so you can
take it to the rig:

```python
from hrtf_course import Condition, shift_band, octave_band, preview

low, high = octave_band(8000)

# Assumes data/sofa/AGV.sofa exists (copy it in from the rig).
cond = Condition(
    name="shift_8kHz_+10pct",
    base_subject="AGV",
    fn=lambda h: shift_band(h, low, high, factor=1.10),
)

# Visual sanity check — original vs. manipulated, with VSI dissimilarity
preview.show(cond)

# Write the manipulated SOFA to data/sofa/AGV_shift_8kHz_+10pct.sofa
sofa_name = cond.build_sofa()
print(sofa_name)
```

Carry the resulting SOFA over to the rig machine and run the
localization block there — `run_condition` is wired up for rig use
only, so it's invoked from the rig install, not the laptop.

### Pooling results across the class

After running localization blocks at the rig, copy each subject's
``{id}.pkl`` pickle into ``data/results/`` here.  Then:

```python
from hrtf_course import collect_results
from hrtf_course.analysis import plot_tolerance_curve

df = collect_results(["AGV", "NKa", "VD"])
plot_tolerance_curve(df, x="vsi_diss", y="ele_rmse")
```

## How condition tagging works

When `Condition.build_sofa()` is called, the manipulated HRTF is written to:

```
hrtf_relearning/data/hrtf/sofa/{base_subject}_{condition.name}.sofa
```

`run_condition` passes that name into `hrtf_relearning`'s `Localization` machinery as `hrir_settings["name"]`, which means the saved sequence ends up tagged with the condition automatically. `collect_results` parses the condition back out by stripping the `{base_subject}_` prefix.

This is why **condition names must be filename-safe**: only `A–Z`, `a–z`, `0–9`, `_`, `+`, `-`, and `.` are allowed.

## VSI and the Baumgartner 2014 model

`collect_results` automatically computes [Trapeau & Schönwiesner (2016)](https://doi.org/10.1121/1.4960400) VSI dissimilarity vs. each subject's baseline HRTF, in the 5.7–11.3 kHz peak band. This gives a single scalar per condition that quantifies *how much the spectral cues changed*. `plot_tolerance_curve` plots a behavioural metric against it — the headline figure for the seminar.

`add_baumgartner_predictions(df)` is an opt-in extension that wraps the existing Baumgartner 2014 model in `hrtf_relearning/hrtf/models/` and adds a `model_pe` column. It's a thin stub today — see the docstring for what to fill in.

## Time budget for the course

About 2 h per session, 5 min per localization block, 2–3 students. The example scripts are sized to **4 conditions per student per day** (1 baseline + 3 manipulation levels). At 5 min each plus repositioning, that's ~25 min per student, leaving time for setup, instructions, and one or two repeat blocks if something goes wrong.
