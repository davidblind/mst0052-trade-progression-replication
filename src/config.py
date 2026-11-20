"""Every setting of the project in one place.

Change a number here, not inside a script, so that every step agrees and the
report can quote one source of truth.

"""
from pathlib import Path

# --- Project paths --------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = ROOT / "data" / "interim"        # output of 01
PROCESSED_DIR = ROOT / "data" / "processed"    # output of 02
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
for d in (INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Raw data (downloaded by the user, see README.md) ------------------------------
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "hs92_country_product_year_4.csv"     # Atlas trade, HS4
PRODUCT_FILE = RAW_DIR / "product_hs92.csv"                 # Atlas product classification
GRAVITY_ZIP = RAW_DIR / "Gravity_csv_V202211.zip"           # CEPII Gravity, macro + geography

# Outputs of 01_ingest.py and 01b_ingest_gravity.py
TRADE_FILE = INTERIM_DIR / "trade_cpy_hs4.parquet"          # country x HS4 x year: exports, imports
MACRO_FILE = INTERIM_DIR / "country_year.parquet"           # country x year: macro variables
GEO_FILE = INTERIM_DIR / "pairs_geo.parquet"                # country x partner: contig, dist

# --- Panel design (02_build_panel.py) -------------------------------------------
HS_DIGITS = 4                 # fixed by the raw file; 6 needs the bilateral route
FIRST_YEAR, LAST_YEAR = 1995, 2024
HORIZON = 5                   # label looks HORIZON-1 and HORIZON years ahead
LAGS = 4                      # years of RCA history used as features
BASE_YEARS = list(range(FIRST_YEAR + LAGS, LAST_YEAR - HORIZON + 1))   # 1999..2019
AT_RISK_MAX_RCA = 0.5         # a row is "at risk" only if RCA < this at t
SUCCESS_MIN_RCA = 1.0         # y = 1 if RCA >= this at both t+4 and t+5
MIN_COUNTRY_EXPORTS = 1e9     # drop countries exporting < $1bn at t (noisy RCA)
MIN_PRODUCT_EXPORTS = 10e6    # drop products with world exports < $10m at t
EXAMPLE_COUNTRY, EXAMPLE_PRODUCT = "NOR", "0303"   # followed through every step

PANEL_FILE = PROCESSED_DIR / f"panel_hs{HS_DIGITS}_h{HORIZON}.parquet"
BASKETS_FILE = PROCESSED_DIR / f"baskets_hs{HS_DIGITS}.parquet"

# --- Features -------------------------------------------------------------------
HISTORY = ["rca", "rca_lag1", "rca_lag2", "rca_lag3", "rca_lag4"]

# Features that get squares and pairwise interactions in the polynomial rung.
# Kept short on purpose: 8 features -> 8 + 8 squares + 28 interactions = 44 terms.
KEY_NONLINEAR = HISTORY + ["density", "export_growth", "rca_volatility"]

NUMERIC = HISTORY + [
    "years_competitive_last5", "ever_competitive", "log_exports", "export_growth",
    "log_imports", "rca_imports", "density",
    "diversity", "log_country_exports",
    "ubiquity", "log_world_exports", "world_growth",
    "natural_resource", "green_product", "rca_volatility",
]

CATEGORICAL = ["sector", "country_cluster"]

# Feature groups for the ablation in 04: trade (above) -> + MACRO -> + NEIGHBOUR.
MACRO = ["log_gdpcap_ppp", 
         "log_gdp", 
         "log_pop",
         "gdp_growth",
         "wto",
         "eu"]

NEIGHBOUR = ["neighbour_share", 
             "neighbour_share_dist"]

# --- Unsupervised country features (03_unsupervised.py decides these) -----------
# Evidence: results/tables/unsupervised_summary.csv.
# N_PCA = 5: the scree plot bends at 3-5 components (5 PCs = 26.8% of basket variance, each further
#   PC adds ~2% or less). The k = 3 partition barely changes with 3-21 PCs (ARI >= 0.92 vs 5 PCs).
# N_CLUSTERS = 3: highest silhouette for k = 2..10 (0.348), stable across seeds (ARI 0.980, min 0.964),
#   moderate Ward agreement (0.545). k = 4+ is not reproducible (seed ARI <= 0.76); k = 2 is stable
#   but almost duplicates basket_pc1 (ARI 0.91 with a PC1 median split) and splits the apparel group.
N_PCA = 5
N_CLUSTERS = 3
PCS = [f"basket_pc{k + 1}" for k in range(N_PCA)]

# --- Validation (04, 05) ----------------------------------------------------------
# To predict base year t honestly, every training label must be known at t:
# train only on base years <= t - HORIZON.
VALID_YEARS = [2010, 2012, 2014]   # expanding-window folds for model selection
TEST_YEAR = 2019                   # labels from 2023-24; opened ONCE, at the end
TRAIN_EVERY = 2                    # every 2nd base year: consecutive years are near-duplicates
MAX_TRAIN_ROWS = 150_000           # random training subsample per fit (speed)
MAX_SLOW_ROWS = 40_000             # k-NN and kernel SVM scale badly
SEED = 42
RESEEDS = [404, 31415, 301]                # sensitivity check in 06_interpret.py
#CASE_COUNTRY = "NOR"
CASE_COUNTRIES = ["NOR", "SWE", "DNK", "FIN", "ISL"]
