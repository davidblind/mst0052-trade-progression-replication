"""STEP 02b: Figure 1, one row of the data (Norway, HS 0301 live fish, base year 2011).

Reads TRADE_FILE (01) and PANEL_FILE (02); writes nothing else.

Outputs:
results/figures/example_row_nor0301.pdf
results/figures/example_row_nor0301.png

  [x] rca_series()        RCA per year for one country x product, same formula as 02 add_rca()
  [x] check_panel_row()   the drawn series must agree with the panel row (rca at t, y)
  [x] draw()              series, two threshold lines, base-year line, two shaded windows, legend
  [x] main()

"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import (AT_RISK_MAX_RCA, FIGURES_DIR, FIRST_YEAR, HORIZON, LAGS, PANEL_FILE,
                    SUCCESS_MIN_RCA, TRADE_FILE)

COUNTRY, PRODUCT, BASE_YEAR = "NOR", "0301", 2011
END_YEAR = 2019


def rca_series(country, product):
    """Balassa RCA = (x_cp / X_c) / (X_p / X_w) for every year, one country x product.

    World and country totals are sums over ALL rows of TRADE_FILE (the zero-filled grid
    in 02 adds only zeros, so the sums are the same). Returns a Series indexed by year.
    """
    trade = pd.read_parquet(TRADE_FILE, columns = ["country", "product", "year", "exports"])
    X_c = trade.groupby(["year", "country"])["exports"].transform("sum")   # country total
    X_p = trade.groupby(["year", "product"])["exports"].transform("sum")   # world exports of the product
    X_w = trade.groupby("year")["exports"].transform("sum")                # world total
    trade["rca"] = (trade["exports"] / X_c) / (X_p / X_w)                  # x_cp = trade["exports"]

    # Then pick the one country × product and index by year:
    pick = trade["country"].eq(country) & trade["product"].eq(product)
    return trade.loc[pick].set_index("year")["rca"].sort_index()



def check_panel_row(series):
    """assert that the series agrees with the panel: rca at BASE_YEAR, and the label y."""
    panel = pd.read_parquet(PANEL_FILE, columns = ["country", "product", "base_year", "rca", "y"])
    pick = (panel["country"].eq(COUNTRY) & panel["product"].eq(PRODUCT)
            & panel["base_year"].eq(BASE_YEAR))
    row = panel.loc[pick].iloc[0]
    assert abs(row["rca"] - series[BASE_YEAR]) < 1e-4, "RCA at t differs from the panel"

    # the label rule on one row: RCA >= 1 at both t+4 and t+5
    t4, t5 = BASE_YEAR + HORIZON - 1, BASE_YEAR + HORIZON
    y = int(series[t4] >= SUCCESS_MIN_RCA and series[t5] >= SUCCESS_MIN_RCA)
    assert y == row["y"], "label recomputed from the series differs from the panel"
    print(f"Panel row {COUNTRY} {PRODUCT} {BASE_YEAR}: rca {row['rca']:.4f}, "
          f"RCA {t4} {series[t4]:.2f}, {t5} {series[t5]:.2f}, y = {y}")


def draw(series):
    """Default matplotlib colours (C0, C1, ...); every element has a legend entry."""
    t = BASE_YEAR
    s = series.loc[FIRST_YEAR:END_YEAR]
    fig, ax = plt.subplots(figsize = (8, 3.5))

    ax.plot(s.index, s.values, marker = "o", label = "RCA")
    ax.axhline(SUCCESS_MIN_RCA, ls = "--", color = "grey", label = f"competitive (RCA = {SUCCESS_MIN_RCA:g})")
    ax.axhline(AT_RISK_MAX_RCA, ls = ":", color = "grey", label = f"at risk (RCA < {AT_RISK_MAX_RCA:g})")
    ax.axvline(t, color = "black", lw = 1, label = f"base year t = {t}")
    ax.axvspan(t - LAGS, t, alpha = 0.15, color = "C1", label = f"features (t-{LAGS} to t)")
    ax.axvspan(t + HORIZON - 1, t + HORIZON, alpha = 0.15, color = "C2",
               label = f"label years (t+{HORIZON - 1}, t+{HORIZON})")

    ax.set_xticks(range(FIRST_YEAR, END_YEAR + 1, 2))
    ax.set_xlabel("Year")
    ax.set_ylabel("RCA")
    ax.set_ylim(bottom = 0)
    ax.legend(fontsize = "small", loc = "upper left")   # no title: the caption goes in LaTeX

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"example_row_{COUNTRY.lower()}{PRODUCT}.{ext}", dpi = 200)
    plt.close(fig)


def main():
    series = rca_series(COUNTRY, PRODUCT)
    print(series.round(2).to_string())
    check_panel_row(series)
    draw(series)


if __name__ == "__main__":
    main()
