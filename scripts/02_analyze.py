# %% [markdown]
# # Analyse localization results
#
# After the localization tests have been run **at the rig**, copy each
# subject's pickle (``{id}.pkl``) into ``data/results/`` and the
# corresponding SOFAs (``{id}.sofa`` and the manipulated ones) into
# ``data/sofa/``.  This script then loads them, lets you pick which
# runs to include, and produces the seminar plots.

# %% --- 1. Inspect what's in a Subject pickle -------------------------------
SUBJECTS = ["AGV"]    # add the rest of your group

from hrtf_course import list_runs

for sid in SUBJECTS:
    df = list_runs(sid)
    print(f"\n=== {sid} ===")
    print(df)

# %% [markdown]
# ``list_runs`` shows every saved sequence with its condition and trial
# count.  If you ran a condition multiple times, multiple rows appear —
# you can keep all, or pick a subset by run filename in
# ``collect_results(..., runs=[...])``.

# %% --- 2. Pool the runs you want into one DataFrame -----------------------
from hrtf_course import collect_results

# Filter to the conditions you ran in this experiment.  Adjust the
# names to match what you used at the rig (must match the SOFA name
# suffix exactly — e.g. shift_-10pct).
CONDITIONS = ["recorded", "shift_-10pct", "shift_+10pct", "shift_+20pct"]

df = collect_results(SUBJECTS, condition_names=CONDITIONS)
print(df.shape, "rows pooled.")
print(df[["subject", "condition", "run", "n_trials", "ele_rmse", "vsi_diss"]])

# %% [markdown]
# To restrict to specific runs (e.g. drop a partial / failed one),
# re-run ``collect_results`` with the ``runs=[...]`` argument:
#
# ```python
# df = collect_results(
#     SUBJECTS,
#     condition_names=CONDITIONS,
#     runs=["AGV_2026-04-29_14-05.pkl", "AGV_2026-04-29_14-22.pkl"],
# )
# ```

# %% --- 3. Per-condition comparison ----------------------------------------
import matplotlib.pyplot as plt
from hrtf_course import plot_compare

fig1 = plot_compare(df, metric="ele_rmse", group="condition")
fig1.suptitle("Elevation RMSE by condition")

fig2 = plot_compare(df, metric="ele_gain", group="condition")
fig2.suptitle("Elevation gain by condition")

plt.show()

# %% --- 4. Tolerance curve (the seminar headline) --------------------------
from hrtf_course import plot_tolerance_curve

fig3 = plot_tolerance_curve(df, x="vsi_diss", y="ele_rmse")
fig3.suptitle("Cue-frequency tolerance — elevation RMSE vs. VSI dissimilarity")

plt.show()

# %% --- 5. Save the summary -------------------------------------------------
from hrtf_course import PATH
out = PATH / "data" / "results" / "summary.csv"
df.to_csv(out, index=False)
print("Wrote", out)
