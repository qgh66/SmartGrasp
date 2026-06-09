"""Full analysis report for runs/."""
import pandas as pd
import glob
from pathlib import Path

OUT_DIR = Path("analysis_report")
OUT_DIR.mkdir(exist_ok=True)

# ---- Load ----
all_df = pd.concat([
    pd.read_csv(p).assign(model=Path(p).parent.name)
    for p in sorted(glob.glob("runs/*/results.csv"))
], ignore_index=True)

details_df = pd.concat([
    pd.read_csv(p).assign(model=Path(p).parents[1].name)
    for p in sorted(glob.glob("runs/*/scene_details/*.csv"))
], ignore_index=True)

print(f"Loaded {len(all_df)} main rows, {len(details_df)} detail rows")
print(f"Models: {sorted(all_df['model'].unique())}")

# ---- 1. Per-model summary ----
summary = all_df.groupby("model").agg(
    rows=("target_id", "count"),
    success_rate=("cl_success", "mean"),
    avg_steps=("cl_num_steps", "mean"),
    max_steps=("cl_num_steps", "max"),
).round(3)
summary.to_csv(OUT_DIR / "summary_per_model.csv")
print("\n=== Per-model summary ===")
print(summary)

# ---- 2. Decision pivot ----
decision_pivot = all_df.pivot_table(
    index=["scene_id", "target_id", "target_label"],
    columns="model",
    values="cl_action_seq",
    aggfunc="first",
)
decision_pivot.to_csv(OUT_DIR / "decisions_pivot.csv")

steps_pivot = all_df.pivot_table(
    index=["scene_id", "target_id"],
    columns="model",
    values="cl_num_steps",
    aggfunc="first",
)
steps_pivot["disagreement"] = steps_pivot.max(axis=1) - steps_pivot.min(axis=1)
steps_pivot.to_csv(OUT_DIR / "steps_pivot.csv")

# ---- 3. Disagreement cases ----
disagree = decision_pivot[decision_pivot.nunique(axis=1) > 1]
disagree.to_csv(OUT_DIR / "disagreement_cases.csv")
print(f"\nDisagreement cases: {len(disagree)}")

# ---- 4. P_s distribution ----
print("\n=== P_s extreme value frequency ===")
ps_dist = []
for model in sorted(details_df["model"].unique()):
    sub = details_df[details_df["model"] == model]["P_s"]
    ps_dist.append({
        "model": model,
        "p_s_near_0_pct": (sub <= 0.05).mean(),
        "p_s_mid_pct":    ((sub > 0.05) & (sub < 0.95)).mean(),
        "p_s_near_1_pct": (sub >= 0.95).mean(),
        "p_s_mean":       sub.mean(),
        "p_s_std":        sub.std(),
    })
ps_dist_df = pd.DataFrame(ps_dist).round(3)
ps_dist_df.to_csv(OUT_DIR / "ps_distribution.csv", index=False)
print(ps_dist_df)

# ---- 5. Pairwise agreement ----
from itertools import combinations
models = sorted(all_df["model"].unique())
agree = pd.DataFrame(index=models, columns=models, dtype=float)
for m1, m2 in combinations(models, 2):
    d1 = all_df[all_df["model"] == m1].set_index(["scene_id", "target_id"])
    d2 = all_df[all_df["model"] == m2].set_index(["scene_id", "target_id"])
    common = d1.index.intersection(d2.index)
    same = (d1.loc[common, "cl_action_seq"] == 
            d2.loc[common, "cl_action_seq"]).mean()
    agree.loc[m1, m2] = same
    agree.loc[m2, m1] = same
for m in models:
    agree.loc[m, m] = 1.0
agree.to_csv(OUT_DIR / "pairwise_agreement.csv")
print("\n=== Pairwise agreement ===")
print(agree.round(3))

# ---- 6. Final report ----
print(f"\nAll analysis written to {OUT_DIR}/")
print("Files:")
for f in sorted(OUT_DIR.glob("*.csv")):
    print(f"  {f}")