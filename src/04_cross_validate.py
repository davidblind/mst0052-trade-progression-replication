"""STEP 04: model selection on expanding-window folds.

  1. For each validation year t in VALID_YEARS, train on base years <= t - HORIZON
     (every TRAIN_EVERY-th), evaluate on base year t (common.cv_folds).
  2. Every model x every grid point x every fold, with the same folds for every
     model: one GridSearchCV per model, exactly as Assignment 3's evaluate()
     ("use the same resampling folds for every candidate").
  3. Summarise mean +- sd per (model, params), apply the written selection rule,
     save the choice. The test year is NOT touched here.

Outputs: 
results/tables/cv_folds.csv
results/tables/cv_summary.csv 
results/tables/selected_model.json 
results/tables/cv_keep_summary.csv 
results/tables/rung_comparison.csv
results/tables/ablation_folds.csv
results/tables/ablation.csv

Run:  cd src && python 04_cross_validate.py [--only logit_ridge random_forest]

"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.base import clone

from common import cv_folds, cv_panel, load_data
from config import TABLES_DIR, VALID_YEARS, NUMERIC, MACRO, NEIGHBOUR
from models import MODELS, preprocess, preprocess_poly

# PR-AUC selects (refit is not needed: 05 refits the chosen setting); ROC-AUC is recorded
# alongside, as in L7's scoring={...} example.
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}

# D10 ablation: nested feature sets, each adds one group to the one before.
FEATURE_SETS = {"trade":       NUMERIC,
                "+macro":      NUMERIC + MACRO,
                "+neighbours": NUMERIC + MACRO + NEIGHBOUR}


# Ablated on top of the rung winners: boosting ties with the forest in D8, so the
# rung-3 conclusion should not depend on which of the two ensembles was picked.
EXTRA_ABLATION = ["gradient_boosting"]


def run_cv(data, names):
    """One GridSearchCV per model on the same three expanding-window folds (L7, Assignment 3).

    data:  the CV panel from common.cv_panel(), with a 0..n-1 index (the folds are positions).
    names: keys of models.MODELS.
    """
    tables = []
    for name in names:
        spec = MODELS[name]
        search = GridSearchCV(spec["pipe"], spec["grid"], cv = cv_folds(data, spec["slow"]),
                              scoring = SCORING, refit = False, n_jobs = -1)
        search.fit(data, data["y"])
        res = pd.DataFrame(search.cv_results_)
        res["model"] = name
        res["rung"] = spec["rung"]
        res["params"] = res["params"].map(json.dumps)      # dict -> text, so it survives a CSV
        tables.append(res)
        for _, row in res.iterrows():
            print(f"  {name:18s} {row['params']:55s} PR-AUC "
                  f"{row['mean_test_pr_auc']:.4f} +- {row['std_test_pr_auc']:.4f}")
    return pd.concat(tables, ignore_index = True)

def fold_table(cv):
    """cv_results_ rows -> one row per (model, grid point, validation year): the old cv_folds.csv.
    """
    return pd.concat([pd.DataFrame({"model": cv["model"], "rung": cv["rung"], "params": cv["params"],
                                    "fold": t,
                                    "pr_auc": cv[f"split{k}_test_pr_auc"],
                                    "roc_auc": cv[f"split{k}_test_roc_auc"],
                                    "fit_seconds": cv["mean_fit_time"]})
                      for k, t in enumerate(VALID_YEARS)], ignore_index = True)

def summarise_and_select(cv):
    """cv_results_ table -> summary, the best setting per model, and ONE selected model (rule from D2).

    1. summary: one row per (model, params) with mean and sd of PR-AUC over the three folds.
    2. keep: the best grid point per model (sort by mean, keep the first row of each model).
    3. Tie flag over eligible models (rung != "comparison"): tied with the leader if the gap
       is smaller than the larger of the two fold sds (a tolerance fixed in advance, not a test).
    4. Tie-break: among tied models the lowest complexity, then the highest mean.
    5. Save cv_summary.csv, cv_keep_summary.csv, selected_model.json.
    """
    # 1. GridSearchCV's own mean and sd (ddof = 0) per grid point
    summary = pd.DataFrame({"model": cv["model"],
                            "rung": cv["rung"],
                            "params": cv["params"],
                            "pr_auc_mean": cv["mean_test_pr_auc"],
                            "pr_auc_sd": cv["std_test_pr_auc"],
                            "roc_auc_mean": cv["mean_test_roc_auc"],
                            "fit_seconds": cv["mean_fit_time"],
                            "complexity": [MODELS[m]["complexity"] for m in cv["model"]]})

    # 2. best grid point per model
    keep = (summary.sort_values("pr_auc_mean", ascending = False)
                   .drop_duplicates("model")
                   .reset_index(drop = True))

    # 3. tie flag among eligible models
    keep["eligible"] = keep["rung"] != "comparison"
    leader = keep[keep["eligible"]].iloc[0]                 # sorted, so the first eligible row leads
    keep["tied"] = keep["eligible"] & (
        leader["pr_auc_mean"] - keep["pr_auc_mean"] < np.maximum(leader["pr_auc_sd"], keep["pr_auc_sd"]))

    # 4. tie-break: lowest complexity, then highest mean
    best = (keep[keep["tied"]]
            .sort_values(["complexity", "pr_auc_mean"], ascending = [True, False])
            .iloc[0])
    print(f"Leader: {leader['model']} ({leader['pr_auc_mean']:.4f}); "
          f"selected: {best['model']} ({best['pr_auc_mean']:.4f})")

    # 5. save
    summary.to_csv(TABLES_DIR / "cv_summary.csv", index = False)
    keep.to_csv(TABLES_DIR / "cv_keep_summary.csv", index = False)
    (TABLES_DIR / "selected_model.json").write_text(
        json.dumps({"model": best["model"], "params": json.loads(best["params"])}, indent = 2))
    return keep


def rung_table(keep):
    """D9, the core table: the best model per rung and whether each step up the ladder is a real gain.

      gain        mean PR-AUC minus that of the rung below (.diff())
      tie_window  max(this rung's sd, the rung below's sd) (.shift() = the row above)
      real_gain   gain > tie_window; NaN for the baseline (nothing below it)
      x_random    mean PR-AUC / majority's PR-AUC (= the positive rate)
    """

    best = keep.drop_duplicates("rung").set_index("rung")
    rungs = best.loc[["baseline", "1_linear", "2_poly", "3_trees"], ["model", "pr_auc_mean", "pr_auc_sd"]]

    rungs["gain"] = rungs["pr_auc_mean"].diff()
    rungs["tie_window"] = np.maximum(rungs["pr_auc_sd"], rungs["pr_auc_sd"].shift())
    rungs["real_gain"] = (rungs["gain"] > rungs["tie_window"]).where(rungs["gain"].notna())

    base_rate = keep.loc[keep["model"] == "majority", "pr_auc_mean"].item()
    rungs["x_random"] = rungs["pr_auc_mean"] / base_rate

    rungs.to_csv(TABLES_DIR / "rung_comparison.csv")
    return rungs.round(4)


def prep_for(rung, columns):
    """The preprocessor a rung uses in models.py, on other numeric columns.
    """
    if rung == "2_poly":
        return preprocess_poly(numeric = columns)
    return preprocess(scale = (rung == "1_linear"), numeric = columns)      # trees: no scaling


def run_ablation(data, keep, rungs):
    """D10: does adding macro and neighbour variables help the best model of each rung?

    Models: the rung winners from rung_table() (rungs 1-3; the baseline uses fixed
    columns) + EXTRA_ABLATION. Each keeps its D8 best hyperparameters, so only the
    columns change (fixed, not retuned: cheaper and isolates the effect of the columns;
    it slightly favours the trade set, on which the parameters were tuned).

    For each model and each nested feature set: the model's pipeline with the "prep"
    step swapped for its rung's preprocessor on those columns (prep_for()), scored with
    cross_validate (one fixed setting, as in Assignment 1 Part 4) on the same folds as D8.
    Mean and sd come from the three fold scores (numpy .std() is ddof=0, as in GridSearchCV).

    Per model, each set is compared with the set before it, by the same rule as D9:
      gain        mean PR-AUC minus that of the previous set
      tie_window  max(this set's sd, the previous set's sd)
      real_gain   gain > tie_window (NaN for the trade set: nothing to compare with)

    Check: each model's trade row must equal its D8 score (same pipeline, same folds).

    Saves ablation_folds.csv (one row per model x set x fold) and ablation.csv;
    returns the table rounded to 4 decimals.
    """
    models = list(rungs.loc[["1_linear", "2_poly", "3_trees"], "model"]) + EXTRA_ABLATION
    params = keep.set_index("model")["params"].map(json.loads)      # D8 best, text -> dict

    fold_rows, rows = [], []
    for name in models:
        spec = MODELS[name]
        folds = cv_folds(data, spec["slow"])
        for set_name, columns in FEATURE_SETS.items():
            pipe = clone(spec["pipe"]).set_params(prep = prep_for(spec["rung"], columns), **params[name])
            res = cross_validate(pipe, data, data["y"], cv = folds, scoring = SCORING, n_jobs = -1)
            pr, roc = res["test_pr_auc"], res["test_roc_auc"]          # one score per validation year
            for k, t in enumerate(VALID_YEARS):
                fold_rows.append({"model": name, "features": set_name, "fold": t,
                                  "pr_auc": pr[k], "roc_auc": roc[k]})
            rows.append({"model": name, "features": set_name,
                         "pr_auc_mean": pr.mean(), "pr_auc_sd": pr.std(), "roc_auc_mean": roc.mean()})
            print(f"  {name:18s} {set_name:12s} PR-AUC {pr.mean():.4f} +- {pr.std():.4f}")
    table = pd.DataFrame(rows)

    # gain over the previous set within each model, and the noise it must beat
    by_model = table.groupby("model", sort = False)
    table["gain"] = by_model["pr_auc_mean"].diff()
    table["tie_window"] = np.maximum(table["pr_auc_sd"], by_model["pr_auc_sd"].shift())
    table["real_gain"] = (table["gain"] > table["tie_window"]).where(table["gain"].notna())

    # check: the trade set is exactly the D8 model
    d8 = keep.set_index("model")["pr_auc_mean"]
    trade = table[table["features"] == "trade"].set_index("model")["pr_auc_mean"]
    assert np.allclose(trade, d8[trade.index]), "trade rows do not reproduce the D8 scores"

    pd.DataFrame(fold_rows).to_csv(TABLES_DIR / "ablation_folds.csv", index = False)
    table.to_csv(TABLES_DIR / "ablation.csv", index = False)
    return table.round(4)


def main():
    """CV for every model (or a few with --only), select one, the rung table, the ablation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs = "*", help = "run only these models; saves nothing")
    args = parser.parse_args()
    names = args.only or list(MODELS)

    panel, baskets = load_data()
    print(f"Panel: {len(panel):,} rows, base years {panel['base_year'].min()}-{panel['base_year'].max()}, "
          f"{panel['y'].mean():.2%} positives")
    data, fit_years = cv_panel(panel, baskets)
    print(f"Country features (PCA, k-means) fitted on base years {fit_years}")
    for t, (train, val) in zip(VALID_YEARS, cv_folds(data, slow = False)):
        print(f"Fold {t}: {len(train):,} train rows, {len(val):,} validation rows, "
              f"{data['y'].iloc[val].mean():.2%} positives")

    cv = run_cv(data, names)

    # --only is a smoke test: print, save nothing (a partial run must not overwrite the full results)
    if args.only:
        print(f"\n--only run: {len(cv)} grid points, nothing saved.")
        print(cv[["model", "params", "mean_test_pr_auc", "std_test_pr_auc"]].round(4).to_string(index = False))
        return

    fold_table(cv).to_csv(TABLES_DIR / "cv_folds.csv", index = False)

    # D8: one model by the rule from D2
    keep = summarise_and_select(cv)
    print(keep.to_string(index = False, float_format = "%.4f"))

    # D9: best model per rung and the gain at each step
    rungs = rung_table(keep)
    print("\nRung table:")
    print(rungs.to_string())

    # D10: feature-group ablation on the rung winners
    print("\nAblation (fixed D8 hyperparameters):")
    print(run_ablation(data, keep, rungs).to_string(index = False))



if __name__ == "__main__":
    main()
