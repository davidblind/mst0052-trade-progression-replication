"""STEP 06: interpret, calibrate and stress-test the selected model.

Outputs:
results/tables/importance_<name>.csv              permutation importance (forest, ridge)
results/tables/poly_terms_<name>.csv              rung 2's coefficients (poly ridge, poly lasso)
results/tables/error_slices_<name>.csv            slices (forest, ridge, poly ridge)
results/tables/calibration_brier.csv, calibration_reliability.csv
results/tables/reseed_folds.csv, reseed_summary.csv, reseed_gaps.csv
results/figures/pdp_ridge_vs_forest.png, reliability.png


"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # no window: save to file only (as in 03)
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from common import build_split, cv_folds, cv_panel, load_data, metrics, refit, score, subsample
from config import (CATEGORICAL, FIGURES_DIR, NUMERIC, PCS, RESEEDS, SEED, TABLES_DIR,
                    TEST_YEAR, VALID_YEARS)
from models import MODELS

FEATURES = NUMERIC + PCS + CATEGORICAL        # the 27 input columns the pipeline reads
PDP_FEATURES = ["rca", "rca_volatility", "density", "years_competitive_last5"]
RESEED_MODELS = ["random_forest", "gradient_boosting", "logit_ridge", "logit_poly_ridge"]

def permutation_importance_table(model, test, name):
    """Mean drop in test PR-AUC when one feature is shuffled (L12). Saves importance_<name>.csv.

    Uses the full test set (2019: 150,332 rows, 1,438 positives), 5 shuffles per feature.
    A 50k sample was tried first, but PR-AUC rests on few positives and varied from
    0.055 to 0.071 between samples, so the full set is used. drop_sd therefore measures
    only the shuffle noise.

    Correlated features (rca and its lags) share importance, so a low value for one of them
    does not mean the information is unused (L10 pitfall 3).
    """
    result = permutation_importance(model,
                                    test[FEATURES],       # all test rows, no sample
                                    test["y"],
                                    scoring = "average_precision",
                                    n_repeats = 5, 
                                    random_state = SEED, 
                                    n_jobs = -1)

    table = pd.DataFrame({"feature": FEATURES, 
                          "drop_mean": result.importances_mean,
                          "drop_sd": result.importances_std}).sort_values("drop_mean", ascending = False)

    table.to_csv(TABLES_DIR / f"importance_{name}.csv", index = False)

    print(f"\nPermutation importance, {name} (drop in PR-AUC, all {len(test):,} test rows):")
    print(table.round(4).to_string(index = False))
    return table


def partial_dependence_plots(models, test):
    """Average predicted probability as one feature varies, for each model on the same axes (L10).
    Ridge can only draw S-curves; where the forest's curve departs from it, the forest uses
    a shape the linear model cannot fit. Saves figures/pdp_ridge_vs_forest.png.
    """
    X = test[FEATURES].sample(20_000, random_state = SEED)
    X = X.astype({c: "float64" for c in NUMERIC + PCS})    # pandas 3: PDP needs float columns

    fig, axes = plt.subplots(1, len(PDP_FEATURES), figsize = (14, 3.6))
    for (label, model), color in zip(models.items(), ["tab:blue", "tab:orange"]):
        PartialDependenceDisplay.from_estimator(model, X, PDP_FEATURES, ax = axes, grid_resolution = 50,
                                                line_kw = {"label": label, "color": color})
    for ax in axes:                     # each panel gets the y-range of its own curves
        ax.relim()
        ax.autoscale(axis = "y")
    fig.suptitle(f"Partial dependence on the {TEST_YEAR} test year (predicted probability)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pdp_ridge_vs_forest.png", dpi = 150)
    plt.close(fig)


def surviving_polynomial_terms(pipeline, name):
    """Which squares/interactions does the L1 penalty keep? And how do the ridge and lasso coefficients compare (rung 2's own interpretation)
    """
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coef  = pipeline.named_steps["model"].coef_[0]

    terms = pd.DataFrame({"term": names, "coef": coef})
    terms = terms[terms["term"].str.startswith("poly__")]          # the 44 terms built from KEY_NONLINEAR
    terms["term"] = terms["term"].str.replace("poly__", "")        # "poly__rca^2" -> "rca^2"

    terms["kind"] = "linear"
    terms.loc[terms["term"].str.contains("^", regex = False), "kind"] = "square"
    terms.loc[terms["term"].str.contains(" "), "kind"] = "interaction"
    terms["kept"] = terms["coef"] != 0

    terms["abs_coef"] = terms["coef"].abs()
    terms = terms.sort_values("abs_coef", ascending = False).drop(columns = "abs_coef")

    terms.to_csv(TABLES_DIR / f"poly_terms_{name}.csv", index = False)

    print(f"\nPolynomial terms, {name}: {terms['kept'].sum()} of {len(terms)} non-zero")
    print(terms.groupby("kind")["kept"].agg(["size", "sum"]))
    print(terms[terms["kept"]].head(15).round(3).to_string(index = False))
    return terms
    
def reliability(y, p, bins = 10):
    """Mean prediction vs share of positives in 10 equal-sized bins of the prediction."""
    table = pd.DataFrame({"pred": p, "y": np.asarray(y)})
    table["bin"] = pd.qcut(table["pred"], bins, labels = False)
    return table.groupby("bin").agg(mean_pred = ("pred", "mean"),
                                    share_pos = ("y", "mean"),
                                    rows = ("y", "size"))

def calibration_check(model, train, test, name, benchmarks):
    """Are the predicted probabilities honest? (L10 'Calibrating forest probabilities', L5 Brier)

    Brier and a reliability table (10 deciles of the prediction, pd.qcut) for the raw forest,
    the forest + Platt scaling (CalibratedClassifierCV, sigmoid, cv=3, ensemble=False: the
    same forest plus a sigmoid fitted on out-of-fold predictions, so the ranking and PR-AUC
    are unchanged) and the log-loss benchmarks. Saves calibration_brier.csv,
    calibration_reliability.csv and figures/reliability.png.

    Result on 2019: the raw forest is close to calibrated in shape (min_samples_leaf=50 makes
    leaf fractions local frequencies) and Platt makes it worse (Brier 0.00918 -> 0.00949 =
    constant): a sigmoid in a score in [0, 0.24] has a floor near 0.007. Every model
    overpredicts the level (1.16-1.28% vs 0.96%), because the training rows have 1.35%
    successes: drift, which no calibration on the training years can see. Raw scores kept.
    """
    p_raw = model.predict_proba(test)[:, 1]
    rows = subsample(train, MODELS[name]["slow"])           # the same 150k rows refit() trained on
    p_const = np.full(len(test), rows["y"].mean())           # no skill: always the training base rate

    print(f"Brier on {TEST_YEAR}: constant {brier_score_loss(test['y'], p_const):.5f}, "
          f"{name} raw {brier_score_loss(test['y'], p_raw):.5f}")

    # Platt scaling (L10): one forest on all rows + a sigmoid fitted on out-of-fold predictions
    platt = CalibratedClassifierCV(model, method = "sigmoid", cv = 3, ensemble = False)
    platt.fit(rows, rows["y"])
    p_cal = platt.predict_proba(test)[:, 1]

    print(f"Brier after Platt scaling: {brier_score_loss(test['y'], p_cal):.5f}")
    print(f"PR-AUC raw {average_precision_score(test['y'], p_raw):.4f}, "
          f"after {average_precision_score(test['y'], p_cal):.4f}")

    # log-loss models: calibrated on their training data by construction, so a level error there is drift
    p_bench = {b: m.predict_proba(test)[:, 1] for b, m in benchmarks.items()}

    brier = pd.Series({"constant (training base rate)": brier_score_loss(test["y"], p_const),
                       f"{name} raw": brier_score_loss(test["y"], p_raw),
                       f"{name} Platt": brier_score_loss(test["y"], p_cal),
                       **{b: brier_score_loss(test["y"], p) for b, p in p_bench.items()}}, name = "brier")
    mean_pred = pd.Series({"actual (test base rate)": test["y"].mean(),
                           f"{name} raw": p_raw.mean(),
                           f"{name} Platt": p_cal.mean(),
                           **{b: p.mean() for b, p in p_bench.items()}}, name = "mean_pred")

    table = pd.concat({"raw": reliability(test["y"], p_raw),
                       "platt": reliability(test["y"], p_cal),
                       "ridge": reliability(test["y"], p_bench["logit_ridge"])}, axis = 1)
    table.columns = [f"{m}_{c}" for m, c in table.columns]     # ("raw", "mean_pred") -> "raw_mean_pred"

    brier.to_csv(TABLES_DIR / "calibration_brier.csv")
    table.to_csv(TABLES_DIR / "calibration_reliability.csv")
    print(brier.round(5).to_string(), mean_pred.round(4).to_string(), sep = "\n\n")

    # reliability diagram: deciles of each model's predictions against the share of successes
    fig, ax = plt.subplots(figsize = (5.5, 5))
    for key, label in [("raw", f"{name} (raw)"), ("platt", f"{name} + Platt"), ("ridge", "logit_ridge")]:
        ax.plot(table[f"{key}_mean_pred"], table[f"{key}_share_pos"], marker = "o", label = label)
    lims = [1e-4, 0.1]
    ax.plot(lims, lims, color = "grey", linestyle = "--", label = "perfectly calibrated")
    ax.set(xscale = "log", yscale = "log", xlim = lims, ylim = lims,
           xlabel = "mean predicted probability (per decile)",
           ylabel = f"share of successes, {TEST_YEAR}",
           title = f"Reliability on the {TEST_YEAR} test year")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reliability.png", dpi = 150)
    plt.close(fig)

    return table


def reseed_sensitivity(panel, baskets, keep):
    """Does the conclusion survive another random draw? (A3 optional stress tests; L15/L16 audit)

    Repeats 04's CV for RESEED_MODELS (forest, boosting, ridge, poly ridge) at their fixed
    CV-best settings, under SEED and each of RESEEDS (fixed in config.py before the run).
    A seed changes the 150,000-row training draw (cv_folds(..., seed=seed); 67-79% new rows)
    and the model's random_state; the validation rows stay the same. One cross_validate per
    model and seed (one fixed setting), as in 04's ablation. Seed 42 must reproduce cv_folds.csv.
    Saves reseed_folds.csv; summarise_reseeds() adds reseed_summary.csv and reseed_gaps.csv.

    Result (2026-11-14): forest seed sd 0.0005 vs fold sd 0.0047; ahead of boosting in 11 of 12
    and of ridge in 12 of 12 seed x fold pairs; selected under every seed; poly ridge - ridge
    <= 0.0012. Seed 42 was the forest's best draw (CV score about 0.001 optimistic).
    """
    data, _ = cv_panel(panel, baskets)               # the same CV panel as 04
    params = keep["params"].map(json.loads)          # each model's CV-best setting, text -> dict

    records = []
    for seed in [SEED] + RESEEDS:                     # 42 first: must reproduce cv_folds.csv
        folds = cv_folds(data, slow = False, seed = seed)
        for model_name in RESEED_MODELS:
            pipe = clone(MODELS[model_name]["pipe"]).set_params(**params[model_name],
                                                                model__random_state = seed)
            res = cross_validate(pipe, data, data["y"], cv = folds,
                                 scoring = "average_precision", n_jobs = -1)
            for k, t in enumerate(VALID_YEARS):
                records.append({"seed": seed, "model": model_name, "fold": t,
                                "pr_auc": res["test_score"][k]})
            print(f"seed {seed:6d} {model_name:18s} PR-AUC {res['test_score'].mean():.4f} "
                  f"+- {res['test_score'].std():.4f}")
    folds_table = pd.DataFrame(records)
    folds_table.to_csv(TABLES_DIR / "reseed_folds.csv", index = False)
    return summarise_reseeds(folds_table) 


def summarise_reseeds(folds):
    """Mean PR-AUC per seed and model, the spread over seeds, and the paired gaps behind the conclusions.
    Separate from reseed_sensitivity() so it can be rerun from reseed_folds.csv without refitting."""
    seed_means = folds.groupby(["seed", "model"])["pr_auc"].mean().unstack()    # rows: seeds, columns: models

    summary = pd.DataFrame({"mean_over_seeds": seed_means.mean(),
                            "sd_over_seeds": seed_means.std(ddof = 0),
                            "min": seed_means.min(),
                            "max": seed_means.max(),
                            "sd_over_folds": folds[folds["seed"] == SEED].groupby("model")["pr_auc"].std(ddof = 0)})

    # the three comparisons the conclusions rest on, per seed (positive = first model ahead)
    gaps = pd.DataFrame({"forest - boosting": seed_means["random_forest"] - seed_means["gradient_boosting"],
                         "forest - ridge": seed_means["random_forest"] - seed_means["logit_ridge"],
                         "poly ridge - ridge": seed_means["logit_poly_ridge"] - seed_means["logit_ridge"]})

    # paired per fold and seed: in how many of the 12 (seed, fold) pairs is the first model ahead?
    by_fold = folds.pivot_table(index = ["seed", "fold"], columns = "model", values = "pr_auc")
    ahead = pd.Series({"forest - boosting": (by_fold["random_forest"] > by_fold["gradient_boosting"]).sum(),
                       "forest - ridge": (by_fold["random_forest"] > by_fold["logit_ridge"]).sum(),
                       "poly ridge - ridge": (by_fold["logit_poly_ridge"] > by_fold["logit_ridge"]).sum()},
                      name = f"ahead, of {len(by_fold)}")

    summary.to_csv(TABLES_DIR / "reseed_summary.csv")
    pd.concat([gaps, ahead.to_frame().T]).to_csv(TABLES_DIR / "reseed_gaps.csv")
    print(seed_means.round(4).to_string(), summary.round(4).to_string(), gaps.round(4).to_string(),
          ahead.to_string(), sep = "\n\n")
    return summary, gaps


def error_slices(test, scores, name):
    """Where does the model fail? (Required element 6; Assignment 3 Part 6)

    PR-AUC, ROC-AUC and x random within each sector, RCA band at t, income third
    of countries (+ missing) and history (comeback vs first time), with a pooled
    'all' row. top1pct_recall: share of the group's successes on the pooled top-1%
    shortlist (the list behind precision_top1pct in 05). Saves error_slices_<name>.csv.

    Failure mode: first-time specialisations (never competitive since 1995) are 43%
    of the 2019 successes but fill 25 of the forest's 1,503 top-ranked pairs, so the
    shortlist catches 4 of 617. Within their group they are ranked well (ROC 0.82),
    but their scores sit below the comebacks'. Ranking per group recovers them.
    """
    scored = test.assign(score = scores)
    # RCA at t in four bands (the at-risk set is RCA < 0.5, so 0.5 is the top edge)
    scored["rca_band"] = pd.cut(scored["rca"], bins = [0, 0.01, 0.1, 0.25, 0.5],
                            labels = ["< 0.01", "0.01-0.1", "0.1-0.25", "0.25-0.5"], right = False)

    # income: thirds of COUNTRIES (one value per country in the test year), then mapped to rows
    income = scored.drop_duplicates("country").set_index("country")["log_gdpcap_ppp"]
    groups = pd.qcut(income, 3, labels = ["low", "middle", "high"]).astype(str)
    scored["income"] = scored["country"].map(groups).fillna("missing")
    
    # history: has the pair been competitive before (since 1995)? splits comebacks from first-timers
    scored["history"] = np.where(scored["ever_competitive"] > 0, "comeback", "first time")

    # the pooled top 1% (the shortlist from common.metrics): which rows are on it?
    top = scored.nlargest(len(scored) // 100, "score")
    scored["in_top"] = scored.index.isin(top.index)

    def one_row(slice_name, group, rows):
        return {"model": name, "slice": slice_name, "group": group,
                "rows": len(rows),
                "positives": int(rows["y"].sum()),
                "base_rate": rows["y"].mean(),
                "pr_auc": average_precision_score(rows["y"], rows["score"]),
                "roc_auc": roc_auc_score(rows["y"], rows["score"]), 
                "top1pct_rows": int(rows["in_top"].sum()),
                "top1pct_hits": int(rows.loc[rows["in_top"], "y"].sum())}

    records = [one_row("all", "all", scored)]                 # the pooled score, as a reference line
    for col in ["sector", "rca_band", "income", "history"]:
        for group, rows in scored.groupby(col, observed = True):
            records.append(one_row(col, group, rows))

    table = pd.DataFrame(records)
    table["x_random"] = table["pr_auc"] / table["base_rate"]
    table["top1pct_recall"] = table["top1pct_hits"] / table["positives"]
    print(table.round(4).to_string(index = False))
    table.to_csv(TABLES_DIR / f"error_slices_{name}.csv", index = False)
    return table


def main():
    choice = json.loads((TABLES_DIR / "selected_model.json").read_text())
    name, params = choice["model"], choice["params"]
    panel, baskets = load_data()
    years, train, test = build_split(panel, baskets, TEST_YEAR)
    
    # the selected model (rung 3) and the linear benchmark (rung 1), refitted as in 05
    selected = refit(MODELS[name], params, train) #Random Forest as the selected model

    # keep summary to fetch the best specifications for the other models.
    keep = pd.read_csv(TABLES_DIR / "cv_keep_summary.csv").set_index("model") 
    #refit model based on best specification in keep:
    ridge = refit(MODELS["logit_ridge"], 
                  json.loads(keep.loc["logit_ridge", 
                                      "params"]), train)
    poly_ridge = refit(MODELS["logit_poly_ridge"], 
                       json.loads(keep.loc["logit_poly_ridge", 
                                           "params"]), train)
    poly_lasso = refit(MODELS["logit_poly_lasso"], 
                       json.loads(keep.loc["logit_poly_lasso", 
                                           "params"]), train)

    s, is_proba = score(selected, test)
    print(f"{name} on {TEST_YEAR}: PR-AUC {metrics(test['y'], s, is_proba)['pr_auc']:.4f}")  # must be 0.0652

    permutation_importance_table(selected, test, name) # passing inn forest
    permutation_importance_table(ridge, test, "logit_ridge") # passing in ridge
    partial_dependence_plots({"ridge (rung 1)": ridge, "forest (rung 3)": selected}, test)
    surviving_polynomial_terms(poly_ridge, "logit_poly_ridge")
    surviving_polynomial_terms(poly_lasso, "logit_poly_lasso")

    # Error slices:
    s_ridge, _ = score(ridge, test)
    s_poly, _ = score(poly_ridge, test)
    slices = pd.concat([error_slices(test, s, name),
                        error_slices(test, s_ridge, "logit_ridge"),
                        error_slices(test, s_poly, "logit_poly_ridge")])
    
    side = slices.pivot_table(index = ["slice", "group"], 
                              columns = "model",
                              values = ["pr_auc", "roc_auc", "top1pct_recall"], 
                              sort = False)
    print("\nError slices, forest vs ridge vs poly ridge:")
    print(side.round(3).to_string())
    calibration_check(selected, train, test, name,
                      {"logit_ridge": ridge, 
                       "logit_poly_ridge": poly_ridge})

    reseed_sensitivity(panel, baskets, keep)

if __name__ == "__main__":
    main()
