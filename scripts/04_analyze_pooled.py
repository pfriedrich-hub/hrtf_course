# %% [markdown]
# # Group analysis — pooled across students and both experiments
#
# Run this after every student has done Day 2 + Day 3.  Generates the
# headline plots for the seminar.

# %% --- 1. Settings ----------------------------------------------------------
SUBJECT_IDS = [
    "STUDENT_01",
    "STUDENT_02",
    # add the rest of your group...
]

# Conditions to keep — match the names from scripts 02 + 03.
DAY2_CONDITIONS = ["baseline", "shift_-10pct", "shift_+10pct", "shift_+20pct"]
DAY3_CONDITIONS = ["baseline", "widen_w10", "widen_w20", "widen_w40"]

# %% --- 2. Pool the data ----------------------------------------------------
from hrtf_course import collect_results

df = collect_results(
    SUBJECT_IDS,
    condition_names=DAY2_CONDITIONS + DAY3_CONDITIONS,
)
print(df.shape, "rows pooled.")
print(df[["subject", "condition", "ele_rmse", "vsi_diss"]].head(20))

# Tag each row with its experiment so the plots can split by manipulation.
import pandas as pd
df["experiment"] = "other"
df.loc[df["condition"].str.startswith("shift"), "experiment"] = "shift"
df.loc[df["condition"].str.startswith("widen"), "experiment"] = "widen"
df.loc[df["condition"] == "baseline",           "experiment"] = "baseline"

# %% --- 3. Per-condition comparison plots -----------------------------------
from hrtf_course import plot_compare
import matplotlib.pyplot as plt

day2 = df[df["condition"].isin(DAY2_CONDITIONS)]
day3 = df[df["condition"].isin(DAY3_CONDITIONS)]

fig1 = plot_compare(day2, metric="ele_rmse", group="condition")
fig1.suptitle("Day 2 — frequency-shift tolerance")

fig2 = plot_compare(day3, metric="ele_rmse", group="condition")
fig2.suptitle("Day 3 — peak/notch widening tolerance")

plt.show()

# %% --- 4. Tolerance curves (the seminar headline) --------------------------
from hrtf_course import plot_tolerance_curve

fig3 = plot_tolerance_curve(
    df[df["experiment"].isin(("shift", "baseline"))],
    x="vsi_diss", y="ele_rmse",
)
fig3.suptitle("Cue-frequency tolerance (Day 2)")

fig4 = plot_tolerance_curve(
    df[df["experiment"].isin(("widen", "baseline"))],
    x="vsi_diss", y="ele_rmse",
)
fig4.suptitle("Cue-shape tolerance (Day 3)")

plt.show()

# %% --- 5. Optional — Baumgartner 2014 model comparison ---------------------
# This is the more advanced extension — comparing behaviour to a
# physiological model.  Read the function's docstring first; you may need
# to wire up the exact return shape of the model in your install.
#
# from hrtf_course import analysis
# df_with_pe = analysis.add_baumgartner_predictions(df)
# fig5 = plot_tolerance_curve(df_with_pe, x="vsi_diss", y="model_pe")
# fig5.suptitle("Model-predicted error vs. VSI dissimilarity")
# plt.show()

# %% --- 6. Save the table for the lab notebook -----------------------------
from pathlib import Path
import hrtf_course
out = hrtf_course.PATH / "data" / "results" / "course_summary.csv"
df.to_csv(out, index=False)
print("Wrote", out)
