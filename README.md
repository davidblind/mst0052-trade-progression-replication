# Which products will a country start exporting?

Predicts whether a country becomes competitive in an HS4 product (RCA < 0.5 at base year *t*, RCA ≥ 1 at *t*+4 and *t*+5), comparing linear, polynomial and tree models on temporal folds, with base year 2019 as the test year.

## Data

Download three files into `data/raw/` (create the folder):

| File | Source |
|---|---|
| `hs92_country_product_year_4.csv` | Atlas of Economic Complexity, HS92 country × product × year, 4-digit: <https://doi.org/10.7910/DVN/T4CHWJ> |
| `product_hs92.csv` | HS92 product classification, same source |
| `Gravity_csv_V202211.zip` | CEPII Gravity database, version V202211, CSV (keep it zipped): <https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8> |

## Run

Python 3.12.

```bash
pip install -r requirements.txt
cd src
python 01_ingest.py
python 01b_ingest_gravity.py
python 02_build_panel.py
python 02b_figure_example_row.py
python 02c_rca_stability.py
python 03_unsupervised.py
python 04_cross_validate.py   # slow
python 05_test_once.py
python 06_interpret.py
```

Outputs go to `data/interim/`, `data/processed/` and `results/`. All settings are in `src/config.py`.
