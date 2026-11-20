"""STEP 02c: is RCA less stable for small exporters and small product markets?

Reads TRADE_FILE (01). Prints one table and saves it

Stability = mean of |log(1 + RCA(t+1)) - log(1 + RCA(t))|, over all (country, product)
pairs, zeros included. The mean, not the median: 18-78% of rows per group are zero in
both years, which puts the median at or near 0. Rows are split by the two panel filters at t:

country total exports          >=   MIN_COUNTRY_EXPORTS ($1bn), 
world exports of the product   >=   MIN_PRODUCT_EXPORTS ($10m).

Outputs:
results/tables/rca_stability.csv
"""

import numpy as np
import pandas as pd

from config import (FIRST_YEAR, LAST_YEAR, MIN_COUNTRY_EXPORTS, MIN_PRODUCT_EXPORTS,
                    TABLES_DIR, TRADE_FILE)


def rca_change_table():
    trade = pd.read_parquet(TRADE_FILE, columns=["country", "product", "year", "exports"])

    # RCA, same formula as 02 add_rca(): (x_cp / X_c) / (X_p / X_w)
    X_c = trade.groupby(["year", "country"])["exports"].transform("sum")
    X_p = trade.groupby(["year", "product"])["exports"].transform("sum")
    X_w = trade.groupby("year")["exports"].transform("sum")
    trade["rca"] = (trade["exports"] / X_c) / (X_p / X_w)
    rca = trade[["country", "product", "year", "rca"]]

    # Same pair one year later: shift the year back by one and merge.
    # Outer merge + fill 0: a pair missing from the Atlas in one of the years has no
    # exports that year (as in the 02 grid), so entries and exits count as change.
    next_year = rca.assign(year=rca["year"] - 1)
    df = rca.merge(next_year, on=["country", "product", "year"], how="outer", suffixes=("", "_next"))
    df = df.fillna({"rca": 0.0, "rca_next": 0.0})
    df = df[df["year"].between(FIRST_YEAR, LAST_YEAR - 1)]   # t+1 must exist in the data

    # Panel filters at t, from country and product totals in year t
    country_total = trade.groupby(["year", "country"], as_index=False)["exports"].sum()
    country_total["country_1bn"] = country_total["exports"] >= MIN_COUNTRY_EXPORTS
    product_total = trade.groupby(["year", "product"], as_index=False)["exports"].sum()
    product_total["product_10m"] = product_total["exports"] >= MIN_PRODUCT_EXPORTS
    df = df.merge(country_total[["year", "country", "country_1bn"]], on=["year", "country"], how="left")
    df = df.merge(product_total[["year", "product", "product_10m"]], on=["year", "product"], how="left")
    df = df.fillna({"country_1bn": False, "product_10m": False})   # no rows in year t = no exports

    # log1p, as for the linear models: defined at RCA = 0, so no pair is dropped
    df["abs_change"] = (np.log1p(df["rca_next"]) - np.log1p(df["rca"])).abs()

    table = df.pivot_table(index="country_1bn", columns="product_10m",
                           values="abs_change", aggfunc=["mean", "size"])
    return table


def main():
    table = rca_change_table()
    print("Mean |change in log(1 + RCA)|, t to t+1 (size = number of rows)")
    print(table.round(3).to_string())
    table.to_csv(TABLES_DIR / "rca_stability.csv")


if __name__ == "__main__":
    main()
