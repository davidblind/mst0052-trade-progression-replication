"""Every candidate model: an unfitted scikit-learn Pipeline plus the grid GridSearchCV searches.

Pipeline([("prep", <ColumnTransformer>), ("model", <estimator>)])

MODELS[name] is a dict with five keys:
  pipe        the unfitted Pipeline
  grid        what GridSearchCV searches; {} = one fixed setting.
  rung        "baseline", "1_linear", "2_poly", "3_trees" or "comparison"
  complexity  the tie-break: among tied models the lowest number wins
  slow        True -> trained on MAX_SLOW_ROWS (40,000) instead of MAX_TRAIN_ROWS (150,000)

"""
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import NUMERIC, PCS, CATEGORICAL, HISTORY, KEY_NONLINEAR, SEED


# ----------------------------------------------------------------------------
# Preprocessing (L3): fitted inside the pipeline, so on each fold's training rows only
# ("split first, then fit").
# ----------------------------------------------------------------------------

def preprocess(scale = True, numeric = NUMERIC):
    """Numeric columns: median imputation, then standardising (if scale). Categorical: one-hot.

    The same preprocessor as Assignment 1 Part 3, with one switch.
    numeric: the numeric columns besides PCS. Only 04's ablation changes it (adds MACRO, NEIGHBOUR).
    scale=False for trees: a split "x_j > threshold" does not care about units. Scaling matters
    for penalised models (one penalty for all coefficients), k-NN and SVM (distances).
    """
    numeric_steps = [("imputer", SimpleImputer(strategy = "median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric + PCS),
        ("cat", OneHotEncoder(handle_unknown = "ignore"), CATEGORICAL),
    ])


def preprocess_poly(numeric = NUMERIC):
    """Rung 2: squares and pairwise products of the 8 KEY_NONLINEAR columns; the rest as in preprocess().

    8 columns -> 8 linear + 8 squares + 28 products = 44 columns. Any extra columns (MACRO,
    NEIGHBOUR in the ablation) go into `others`: linear, standardised.
    Order matters: the terms are built from the raw (imputed) values and THEN standardised,
    so x^2 and x*z get the same penalty scale as x (L4: "always pair polynomial features
    with a penalty").
    """
    others = [c for c in numeric + PCS if c not in KEY_NONLINEAR]
    return ColumnTransformer([
        ("poly", Pipeline([("imputer", SimpleImputer(strategy = "median")),
                           ("poly", PolynomialFeatures(degree = 2, include_bias = False)),
                           ("scaler", StandardScaler())]), KEY_NONLINEAR),
        ("num", Pipeline([("imputer", SimpleImputer(strategy = "median")),
                          ("scaler", StandardScaler())]), others),
        ("cat", OneHotEncoder(handle_unknown = "ignore"), CATEGORICAL),
    ])


def only(columns):
    """Standardise a few named columns and drop the rest (the two single-idea baselines)."""
    return ColumnTransformer([("num", StandardScaler(), columns)])


# Every penalised logit searches the same C values: a fair comparison between rungs 1 and 2.
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0]       # small C = strong penalty (C ~ 1/alpha, L5)


MODELS = {
    # ------------------------------------------------------------------------
    # Baselines (spec §3): the bars every real model must clear (Assignment 3 Part 3)
    # ------------------------------------------------------------------------
    # §3.1 Always predicts "no". Its PR-AUC is the share of positives: the floor.
    "majority": {
        "pipe": DummyClassifier(strategy = "most_frequent"),
        "grid": {}, "rung": "baseline", "complexity": 0, "slow": False},

    # §3.2 The economists' baseline: logistic regression on relatedness density only (L5).
    "logit_density": {
        "pipe": Pipeline([("prep", only(["density"])),
                          ("model", LogisticRegression(max_iter = 1000))]),
        "grid": {}, "rung": "baseline", "complexity": 1, "slow": False},

    # §3.3 Persistence baseline: the pair's own RCA history only.
    "logit_history": {
        "pipe": Pipeline([("prep", only(HISTORY)),
                          ("model", LogisticRegression(max_iter = 1000))]),
        "grid": {}, "rung": "baseline", "complexity": 2, "slow": False},

    # ------------------------------------------------------------------------
    # Rung 1: linear in the raw features (spec §4): L4 penalties, L5 logistic regression
    # ------------------------------------------------------------------------
    # §4.1 L2 penalty (the default).
    "logit_ridge": {
        "pipe": Pipeline([("prep", preprocess()),
                          ("model", LogisticRegression(max_iter = 2000))]),
        "grid": {"model__C": C_GRID}, "rung": "1_linear", "complexity": 3, "slow": False},

    # §4.2 L1 penalty: can set coefficients exactly to zero (L4 lasso).
    "logit_lasso": {
        "pipe": Pipeline([("prep", preprocess()),
                          ("model", LogisticRegression(l1_ratio = 1.0, solver = "saga",
                                                       max_iter = 2000, random_state = SEED))]),
        "grid": {"model__C": C_GRID}, "rung": "1_linear", "complexity": 3, "slow": False},

    # §4.3 Elastic net: half L1, half L2.
    "logit_elastic_net": {
        "pipe": Pipeline([("prep", preprocess()),
                          ("model", LogisticRegression(l1_ratio = 0.5, solver = "saga",
                                                       max_iter = 2000, random_state = SEED))]),
        "grid": {"model__C": C_GRID}, "rung": "1_linear", "complexity": 3, "slow": False},

    # ------------------------------------------------------------------------
    # Rung 2: linear in hand-made nonlinear features (spec §5): L4, L6
    # ------------------------------------------------------------------------
    # §5 Ridge on raw features + squares + products of KEY_NONLINEAR.
    "logit_poly_ridge": {
        "pipe": Pipeline([("prep", preprocess_poly()),
                          ("model", LogisticRegression(max_iter = 3000))]),
        "grid": {"model__C": C_GRID}, "rung": "2_poly", "complexity": 4, "slow": False},

    # §5 Lasso: the surviving terms show WHICH curves and interactions matter (06).
    "logit_poly_lasso": {
        "pipe": Pipeline([("prep", preprocess_poly()),
                          ("model", LogisticRegression(l1_ratio = 1.0, solver = "saga",
                                                       max_iter = 3000, random_state = SEED))]),
        "grid": {"model__C": C_GRID}, "rung": "2_poly", "complexity": 4, "slow": False},

    # §5 Elastic net.
    "logit_poly_elastic_net": {
        "pipe": Pipeline([("prep", preprocess_poly()),
                          ("model", LogisticRegression(l1_ratio = 0.5, solver = "saga",
                                                       max_iter = 3000, random_state = SEED))]),
        "grid": {"model__C": C_GRID}, "rung": "2_poly", "complexity": 4, "slow": False},

    # ------------------------------------------------------------------------
    # Rung 3: models that find the nonlinearity themselves (spec §6): L10, L12
    # ------------------------------------------------------------------------
    # §6.1 One classification tree (L10 workflow 1).
    "decision_tree": {
        "pipe": Pipeline([("prep", preprocess(scale = False)),
                          ("model", DecisionTreeClassifier(min_samples_leaf = 50, random_state = SEED))]),
        "grid": {"model__max_depth": [3, 5, 8, 12]},
        "rung": "3_trees", "complexity": 5, "slow": False},

    # §6.2 Bootstrap samples + a random subset of features at every split (L10).
    "random_forest": {
        "pipe": Pipeline([("prep", preprocess(scale = False)),
                          ("model", RandomForestClassifier(n_estimators = 300, random_state = SEED))]),
        "grid": {"model__min_samples_leaf": [10, 50, 200, 300], "model__max_features": ["sqrt", 0.5]},
        "rung": "3_trees", "complexity": 6, "slow": False},

    # §6.3 Histogram gradient boosting, L12 recipe: early stopping picks the number of trees,
    # the grid tunes learning_rate x max_depth. Never put the number of trees in the grid.
    "gradient_boosting": {
        "pipe": Pipeline([("prep", preprocess(scale = False)),
                          ("model", HistGradientBoostingClassifier(max_iter = 2000, early_stopping = True,
                                                                   random_state = SEED))]),
        "grid": {"model__learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                 "model__max_depth": [3, 4, 6, 9, 12]},
        "rung": "3_trees", "complexity": 7, "slow": False},

    # ------------------------------------------------------------------------
    # Comparison points (spec §7): reported, never selected. L5, L11
    # complexity is listed for the cv_summary table only: comparison models are never
    # eligible, so they never enter the tie-break in 04's summarise_and_select().
    # ------------------------------------------------------------------------
    # §7.1 Score = share of positives among the k nearest training rows (L5).
    "knn": {
        "pipe": Pipeline([("prep", preprocess()), ("model", KNeighborsClassifier())]),
        "grid": {"model__n_neighbors": [25, 50, 100, 200, 400, 800, 1200, 1600]},
        "rung": "comparison", "complexity": 8, "slow": True},

    # §7.2 Gaussian naive Bayes (L5). No grid.
    "naive_bayes": {
        "pipe": Pipeline([("prep", preprocess(scale = False)), ("model", GaussianNB())]),
        "grid": {}, "rung": "comparison", "complexity": 2, "slow": False},

    # §7.3 Soft-margin SVM, RBF kernel, balanced class weights (L11; class_weight from L5).
    "svm": {
        "pipe": Pipeline([("prep", preprocess()),
                          ("model", SVC(kernel = "rbf", class_weight = "balanced"))]),
        "grid": {"model__C": [0.001, 0.01, 0.1, 1, 10],
                 "model__gamma": [0.0003, 0.001, 0.003, 0.01, 0.03]},
        "rung": "comparison", "complexity": 9, "slow": True},
}
