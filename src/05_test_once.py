"""STEP 05: open the test year once.

The model was chosen in 04 from CV scores alone. Here it is refitted on every
base year whose label was known in TEST_YEAR (<= TEST_YEAR - HORIZON) and
scored on TEST_YEAR. Nothing in this script may feed back into the choice.

Outputs: 
results/tables/test_selected.csv 
results/tables/test_predeclared.csv
results/tables/case_study_nordic.csv
results/tables/case_study_hits.csv

"""
import json

import pandas as pd

from common import build_split, load_data, metrics, refit, score
from config import CASE_COUNTRIES, HORIZON, PRODUCT_FILE, TABLES_DIR, HS_DIGITS, TEST_YEAR
from models import MODELS



def predeclared_table(keep, train, test):
    """Test scores of every model's best CV setting, for the report only; never used to reselect.

    Saves test_predeclared.csv.
    """
    records = []
    for _, row in keep.iterrows():
        name = row["model"]
        model = refit(MODELS[name], json.loads(row["params"]), train)   # L7: clone, set_params, fit
        s, is_proba = score(model, test)          # SVM has no predict_proba: decision_function instead
        records.append({"model": name,
                        "rung": row["rung"],
                        "cv_pr_auc": row["pr_auc_mean"],
                        **metrics(test["y"], s, is_proba)})      # ** unpacks the metrics into this dict
        print(f"  {name} done")

    table = pd.DataFrame(records)
    table["x_random"] = table["pr_auc"] / table["base_rate"]
    table.to_csv(TABLES_DIR / "test_predeclared.csv", index = False)
    print(table.round(4).to_string(index = False))
    return table


def case_study(test, scores):
    """Top-20 candidate products per case country in TEST_YEAR, by the selected model vs density.
    Saves case_study_nordic.csv (top 20 per country) and case_study_hits.csv (hits per country).
    """
    names = pd.read_csv(PRODUCT_FILE, dtype = str)
    names = names[names["product_level"] == str(HS_DIGITS)]
    name_of = dict(zip(names["product_hs92_code"], names["product_name_short"]))

    scored = test.assign(score = scores)
    tops, hits = [], []
    for country in CASE_COUNTRIES:
        case = scored[scored["country"] == country]
        by_model   = case.nlargest(20, "score")        # the model's 20 most likely products
        by_density = case.nlargest(20, "density")      # the 20 closest to current exports

        hits.append({"country": country,
                     "candidates": len(case),
                     "positives": int(case["y"].sum()),
                     "model_top20": int(by_model["y"].sum()),
                     "density_top20": int(by_density["y"].sum()),
                     "overlap": len(set(by_model["product"]) & set(by_density["product"]))})
        tops.append(by_model.assign(name = by_model["product"].map(name_of)))

    hits = pd.DataFrame(hits)
    hits.loc[len(hits)] = ["total", *hits.drop(columns = "country").sum()]   # total row at the bottom
    print(hits.to_string(index = False))

    tops = pd.concat(tops)[["country", "product", "name", "score", "density", "rca", "y"]]
    tops.to_csv(TABLES_DIR / "case_study_nordic.csv", index = False)
    hits.to_csv(TABLES_DIR / "case_study_hits.csv", index = False)
    return tops, hits


def main():
    choice = json.loads((TABLES_DIR / "selected_model.json").read_text())
    name, params = choice["model"], choice["params"]
    panel, baskets = load_data()
    years, train, test = build_split(panel, baskets, TEST_YEAR)
    print(f"Test {TEST_YEAR}: train base years {years} (labels known by {max(years) + HORIZON}), "
          f"{len(test):,} test rows")

    # the selected model, refitted on the training years and scored once
    model = refit(MODELS[name], params, train)
    s, is_proba = score(model, test)
    result = pd.DataFrame([{"model": name, "params": json.dumps(params), **metrics(test["y"], s, is_proba)}])
    result.to_csv(TABLES_DIR / "test_selected.csv", index = False)
    print("Selected model on the test year:")
    print(result.to_string(index = False, float_format = "%.4f"))

    predeclared_table(pd.read_csv(TABLES_DIR / "cv_keep_summary.csv"), train, test)
    case_study(test, s)

if __name__ == "__main__":
    main()
