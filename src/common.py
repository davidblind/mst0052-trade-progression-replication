"""Helpers shared by 04_cross_validate.py, 05_test_once.py and 06_interpret.py.

"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config import (PANEL_FILE, BASKETS_FILE, HORIZON, TRAIN_EVERY, VALID_YEARS, N_PCA, N_CLUSTERS,
                    PCS, MAX_TRAIN_ROWS, MAX_SLOW_ROWS, SEED)


def load_data():
    """The modelling panel and the country export baskets."""
    panel = pd.read_parquet(PANEL_FILE)
    baskets = pd.read_parquet(BASKETS_FILE)
    return panel, baskets


def train_years(target_year, available):
    """Base years whose labels were known at `target_year`, every TRAIN_EVERY-th counting back.

    A label for base year s is observed at s + HORIZON, so it may be used to
    predict target_year only if s + HORIZON <= target_year.
    """
    last = target_year - HORIZON
    return [int(y) for y in available if y <= last and (last - y) % TRAIN_EVERY == 0]


def add_country_features(panel, baskets, fit_years):
    """PCA of each country's export basket + a k-means cluster label (L9, L13).

    PCA and k-means are FITTED on the baskets of `fit_years` only, then applied
    to every row (L3: split first, then fit). Returns a copy of `panel` with
    basket_pc1..N_PCA and country_cluster added.
    """
    cols = baskets.columns[2:]                      # columns after base_year, country
    fit_rows = baskets[baskets["base_year"].isin(fit_years)]
    pca = PCA(n_components=N_PCA, random_state=SEED).fit(fit_rows[cols])
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit(pca.transform(fit_rows[cols]))
    country = baskets[["base_year", "country"]].copy()
    country[PCS] = pca.transform(baskets[cols])
    country["country_cluster"] = km.predict(country[PCS].to_numpy()).astype(str)
    return panel.merge(country, on=["base_year", "country"], how="left")


def cv_panel(panel, baskets):
    """The panel with country features for model selection (04).

    The features are fitted once, on the base years of the FIRST fold's training
    set (1999-2005). Every fold trains on those years, so no fold's validation
    year is ever seen by PCA or k-means.
    """
    fit_years = train_years(min(VALID_YEARS), sorted(panel["base_year"].unique()))
    return add_country_features(panel, baskets, fit_years), fit_years


def cv_folds(data, slow, seed=SEED):
    """Expanding-window folds as (train, validation) index pairs for GridSearchCV(cv=...).

    The same idea as TimeSeriesSplit (L7): train on the past, validate on a later
    year. Here the split is by base year, with a gap of HORIZON years so every
    training label was already known in the validation year. Training rows are
    a random subsample (MAX_SLOW_ROWS for slow models, else MAX_TRAIN_ROWS).
    `data` must have a 0..n-1 index (the positions GridSearchCV uses).
    """
    n_train = MAX_SLOW_ROWS if slow else MAX_TRAIN_ROWS
    rng = np.random.default_rng(seed)
    folds = []
    for t in VALID_YEARS:
        years = train_years(t, sorted(data["base_year"].unique()))
        train = np.flatnonzero(data["base_year"].isin(years).to_numpy())
        if len(train) > n_train:
            train = np.sort(rng.choice(train, size=n_train, replace=False))
        val = np.flatnonzero(data["base_year"].eq(t).to_numpy())
        folds.append((train, val))
    return folds


def build_split(panel, baskets, target_year):
    """(years, train, eval) for the test year: country features fitted on the training years."""
    years = train_years(target_year, sorted(panel["base_year"].unique()))
    rows = panel[panel["base_year"].isin(years + [target_year])]
    rows = add_country_features(rows, baskets, years)
    return years, rows[rows["base_year"].isin(years)], rows[rows["base_year"] == target_year]


def subsample(train, slow, seed = SEED):
    """Random training subsample: MAX_SLOW_ROWS for slow models, else MAX_TRAIN_ROWS."""
    n = MAX_SLOW_ROWS if slow else MAX_TRAIN_ROWS
    return train.sample(n=min(n, len(train)), random_state=seed)


def refit(spec, params, train, seed = SEED):
    """The chosen setting, refitted on the training rows (L7: clone(...).set_params(...))."""
    rows = subsample(train, spec["slow"], seed)
    return clone(spec["pipe"]).set_params(**params).fit(rows, rows["y"])


def score(model, X):
    """Ranking score for the positive class, and whether it is a probability."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1], True
    return model.decision_function(X), False


def metrics(y, s, is_proba):
    """PR-AUC (main), ROC-AUC, precision in the top 1% of scores, Brier (probabilities only)."""
    y = np.asarray(y)
    # TODO (optional): precision in the top-k per country -> needs the country column passed in
    top = np.argsort(-s)[: max(1, len(s) // 100)]
    return {"pr_auc": average_precision_score(y, s),
            "roc_auc": roc_auc_score(y, s),
            "precision_top1pct": y[top].mean(),
            "brier": brier_score_loss(y, s) if is_proba else np.nan,
            "base_rate": y.mean()}
