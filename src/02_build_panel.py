"""STEP 02: turn trade totals into the modelling panel.

One row per (country, product, base year t) where the country is clearly not
competitive in the product at t (RCA < 0.5), with features known at t and
y = 1 if RCA >= 1 at both t+4 and t+5.

Outputs: PANEL_FILE and BASKETS_FILE (see config.py)

  [X] B1  load_trade()                  read 01's table; list countries and products
  [X] B2  complete_grid()               every country x product x year, zeros included
  [X] B3  add_rca()                     Balassa RCA, import RCA, competitive flag
  [X] B4  add_history()                 lags, years competitive, ever competitive, growth
  [X] B5  add_label()                   y from t+4 and t+5, NaN when unknown
  [X] B6  country_and_product_tables()  size and growth of country and world market, + macro (01b)
  [X] B7  density()                     relatedness density of one year's matrix
  [X] B7  network_and_baskets()         per-year filters, density, diversity, ubiquity, neighbour shares, baskets
  [X] B8  finalise()                    at-risk filter, product info, dtypes, leakage test
  [ ] (validation 2026-11-10)           the saved panel has 2,799,624 rows, 1.30%, 1.54% in 1999 -> 0.96% in 2019
"""
import numpy as np
import pandas as pd

from config import (TRADE_FILE, MACRO_FILE, GEO_FILE, PRODUCT_FILE, PANEL_FILE, BASKETS_FILE,
                    HS_DIGITS, FIRST_YEAR, LAST_YEAR, HORIZON, LAGS, BASE_YEARS, HISTORY,
                    AT_RISK_MAX_RCA, SUCCESS_MIN_RCA, MIN_COUNTRY_EXPORTS, MIN_PRODUCT_EXPORTS,
                    EXAMPLE_COUNTRY, EXAMPLE_PRODUCT, MACRO, NEIGHBOUR)


def header(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def show_example(df, columns, years=range(2012, 2025), year_col="year"):
    """Print the example (country, product) pair for some years."""
    rows = df[(df["country"] == EXAMPLE_COUNTRY) & (df["product"] == EXAMPLE_PRODUCT) & df[year_col].isin(years)]
    print(f"\nExample: {EXAMPLE_COUNTRY}, product {EXAMPLE_PRODUCT}")
    print(rows[[year_col] + columns].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


def load_trade():
    # B1. Exports and imports per (year, country, HS4 product), from 01.
    trade = pd.read_parquet(TRADE_FILE)
    print(trade.head())
    countries = trade["country"].drop_duplicates().sort_values().tolist()
    exported = trade["exports"].gt(0)   # True where the row has exports
    products = trade.loc[exported, "product"].drop_duplicates().sort_values().tolist()

    print("No of countries: ", len(countries))
    print("No of products: ", len(products))
    print("First 10 countries: ", countries[:10])
    print("First 10 products: ", products[:10])
    
    return trade, countries, products


def complete_grid(trade, countries, products):
    """B2. Every country x product x year, zeros filled in. Asserts to check balance and prints underway to show share of rown before and after fillna
    Missing Atlas rows becomes 0 in the grid
    """
    years = range(FIRST_YEAR, LAST_YEAR + 1)
    grid = pd.MultiIndex.from_product([years, countries, products], names=["year", "country", "product"]).to_frame(index=False)
    df = grid.merge(trade, on=["year", "country", "product"], how="left") #, validate="one_to_one") assert below is basically reporting on_to_one – clearer more readbable text if the merge contains duplicates
    assert len(df) == len(years) * len(countries) * len(products), "grid is not balanced: expected one row per year x country x product (duplicate keys in trade?)"
    print(f"Rows filled with 0 (no Atlas row): {df['exports'].isna().mean():.1%}")
    df[["exports", "imports"]] = df[["exports", "imports"]].fillna(0.0)
    print(f"Share of rows with zero exports {df['exports'].eq(0).mean():.1%}\nShare of rows with zero imports {df['imports'].eq(0).mean():.1%}")
    show_example(df, ["exports", "imports"])
    return df


def add_rca(df):
    """B3. Balassa RCA for exports and imports, and the competitive flag."""

    def rca(value):
        country_total = df.groupby(["year", "country"])[value].transform("sum") # sum of exports and imports per country and year
        world_product = df.groupby(["year", "product"])[value].transform("sum") # sum of exports and imports per product 
        world_total = df.groupby(["year"])[value].transform("sum") # sum of world exports and imports
        ratio = (df[value]/country_total) /(world_product / world_total)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0) # NaN and zero-handling
    
        return ratio, country_total, world_product

    # creating adding rca function output to df, with column names as follows:
    df["rca"], df["country_total"], df["world_product"] = rca("exports")
    df["rca_imports"] = rca("imports")[0] # index 0 since i only need the ratio
    # adding competitive flag
    df["competitive"] = (df["rca"] >= SUCCESS_MIN_RCA).astype(int) # where SUCCESS_MIN_RCA is defined in config as 1 (competetive in exports for a given product at time t)

    
    # Several cheks on my own rca measure compared to the altas_rca
    has_atlas = df["atlas_rca"].notna() # only rows with atlas_rca
    mine  = df.loc[has_atlas, "rca"] 
    atlas = df.loc[has_atlas, "atlas_rca"]
    pearson  = mine.corr(atlas)                       #
    spearman = mine.corr(atlas, method="spearman")    
    atlas_competitive = df.loc[has_atlas, "atlas_rca"].ge(SUCCESS_MIN_RCA).astype(int)
    agree = df.loc[has_atlas, "competitive"].eq(atlas_competitive).mean()   # If rca and atlas_rca agrees. Important distinction, this is mainly driven bt dropping product 9999 and XXXX
    print(f"Correlation with atlas_rca: {pearson:.4f} (Spearman {spearman:.4f}), "
      f"same competitive flag: {agree:.2%}") 
    print(f"Rows with RCA >= 1 {df['competitive'].mean():.2%}")
    show_example(df, ["rca", "atlas_rca", "rca_imports", "competitive", "country_total", "world_product"])
    return df


def add_history(df):
    # sorts the df to country, products, year and constructs LAGS for each products and adds it to a column.  each row gets its pair's past as features: every model sees one row at a time"
    df = df.sort_values(["country", "product", "year"], ignore_index=True)
    pair = df.groupby(["country", "product"], sort=False) 

    #Creating rca lags, k range 1-4
    for k in range(1, LAGS + 1):
        df[f"rca_lag{k}"] = pair["rca"].shift(k)

    count = df["competitive"].astype(float)          # year t itself
    for k in range(1, LAGS + 1):
        count = count + pair["competitive"].shift(k)  # add t-1, t-2, ...
    df["years_competitive_last5"] = count 
    df["ever_competitive"] = pair["competitive"].cummax() # has the country even been comp in a given product?
    df["log_exports"] = np.log1p(df["exports"]) # to account for 0 values which crashes the log calculation
    df["log_imports"] = np.log1p(df["imports"])
    df["export_growth"] = df["log_exports"] - pair["log_exports"].shift(LAGS) # growth rate in exports, log-difference as growth
    
    # add rca volatility across the rca and its LAGS. It can be argued that for countries with large volatility for a given products rca, they can by "mistake" become competitive, or jump back and forth. 
    # Especially if the rca hovers around 1
    window = ["rca"] + [f"rca_lag{k}" for k in range(1, LAGS + 1)]
    df["rca_volatility"] = df[window].std(axis=1, skipna=False) # axis does the cstandard deviatioin calculations on our lag columns

    show_example(df, ["rca", "rca_lag1", "rca_lag2", "rca_lag3", "rca_lag4", "years_competitive_last5",
                      "ever_competitive", "export_growth", "rca_volatility"], years = range(1995, 2025))
    return df


def add_label(df):
    """y = competitive at t+5 AND t+4 
    NaN where the future is unknown(read: observations in years that hasn't happened yet)."""
    pair = df.groupby(["country", "product"], sort=False)
    ahead_4 = pair["competitive"].shift(-(HORIZON - 1)) # Negative shift to look forwards
    ahead_5 = pair["competitive"].shift(-HORIZON)
    df["y"] = ((ahead_4 == 1) & (ahead_5 == 1)).astype(float) # 1 for competitive at t+4 & t+5 for country, products pair at time t
    df.loc[ahead_5.isna(), "y"] = np.nan

    show_example(df, ["rca", "competitive", "y"])

    return df


def country_and_product_tables(df):

    cy = (df.groupby(["country", "year"], as_index=False)["country_total"].first().sort_values(["country", "year"])) #country x year
    assert len(cy["country"].unique())*len(cy["year"].unique()) == len(cy), "countries x years != len(cy)"
    
    # Shifts totalt exports from t+5 to t, this will be used to drop countries that do not longer export at t+5. future value: a sample filter in B7 only, never a feature.
    cy["exports_at_horizon"] = cy.groupby("country")["country_total"].shift(-HORIZON) 
    cy["log_country_exports"] = np.log(cy["country_total"].where(cy["country_total"] > 0))

    py = (df.groupby(["product", "year"], as_index=False)["world_product"].first().sort_values(["product", "year"])) # product x year
    py["log_world_exports"] = np.log(py["world_product"].where(py["world_product"] > 0))
    lag = py.groupby("product")["world_product"].shift(LAGS) # helper to make world_growth more readable. We use the same np.log logic as in log _country_exports above
    py["world_growth"] = py["log_world_exports"] - np.log(lag.where(lag > 0)) # log difference for world product at t minus same product at time t-4

    macro = pd.read_parquet(MACRO_FILE)[["country", "year"] + MACRO + ["legal_origin"]]
    cy = cy.merge(macro, on = ["country", "year"], how = "left", validate = "one_to_one")


    print(f"cy: {cy.shape} | py: {py.shape}")
    print("example country: \n", cy[cy["country"] == EXAMPLE_COUNTRY].tail(7).to_string(index=False))
    print("example product: \n", py[py["product"] == EXAMPLE_PRODUCT].tail(7).to_string(index=False))

    in_panel = (
        (cy["year"].isin(BASE_YEARS))
        & (cy["country_total"] >= MIN_COUNTRY_EXPORTS)
        & (cy["exports_at_horizon"] > 0)
    )

    print(f"\nCountry-years that can reach the panel: {in_panel.sum():,} "
          f"({cy.loc[in_panel, 'country'].nunique()} countries)")
    print("Missing share of each macro column in those country-years:")
    print(cy.loc[in_panel, MACRO + ["legal_origin"]].isna().mean().round(4).to_string())
    print("Panel countries with no Gravity row:", set(cy.loc[in_panel, "country"]) - set(macro["country"]))
    print(f"world_growth: {np.isinf(py['world_growth']).sum()} inf, "
          f"{py.loc[py['year'].isin(BASE_YEARS), 'world_growth'].isna().sum()} NaN in base years")

    return cy, py 


def density(M):
    ubiquity = M.sum(axis = 0)
    co = M.T @ M                                       # co[p, k] = # countries competitive in both
    phi = np.nan_to_num(co / np.maximum.outer(ubiquity, ubiquity))
    np.fill_diagonal(phi, 0)                           # a product is not related to itself
    return np.nan_to_num((M @ phi) / phi.sum(axis=0))


def network_and_baskets(df, cy, py):
    network_rows, baskets = [], []
    geo = pd.read_parquet(GEO_FILE)
    contig_all = geo.pivot(index = "country", columns = "partner", values = "contig").fillna(0)
    dist_all = geo.pivot(index = "country", columns = "partner", values = "dist")


    for t in BASE_YEARS:
        cyt = cy[cy["year"] == t].set_index("country")
        pyt = py[py["year"] == t].set_index("product")
        year_t = df[df["year"] == t]
    
        keep_c = sorted(cyt.index[(cyt["country_total"] >= MIN_COUNTRY_EXPORTS) & (cyt["exports_at_horizon"] > 0)]) # Keeps countries that exports above the minimum level and is still exporting at horizon t+5
        keep_p = sorted(pyt.index[pyt["world_product"] >= MIN_PRODUCT_EXPORTS]) # keeps products that is exported above the minimum threshold
        M = (year_t.pivot(index="country", columns="product", values="competitive").loc[keep_c, keep_p].to_numpy(dtype=float))
        contig = contig_all.reindex(index=keep_c, columns=keep_c).fillna(0).to_numpy()
        dist = dist_all.reindex(index=keep_c, columns=keep_c).to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            neighbour_share = (contig @ M) / contig.sum(axis=1, keepdims=True)
            W = np.where(np.eye(len(keep_c), dtype=bool) | np.isnan(dist), 0.0, 1.0 / dist)
            neighbour_share_dist = (W @ M) / W.sum(axis=1, keepdims=True)
        
        network_rows.append(pd.DataFrame({
            "year": t,
            "country": np.repeat(keep_c, len(keep_p)),
            "product": np.tile(keep_p, len(keep_c)),
            "density": density(M).ravel(),
            "diversity": np.repeat(M.sum(axis=1), len(keep_p)),   # # products c is competitive in
            "ubiquity": np.tile(M.sum(axis=0), len(keep_c)),      # # countries competitive in p
            "neighbour_share": neighbour_share.ravel(), 
            "neighbour_share_dist": neighbour_share_dist.ravel()
    
        }))
        basket = np.log1p(year_t.pivot(index="country", columns="product", values="rca").loc[keep_c])
        basket = basket.astype(np.float32).reset_index()
        basket.insert(0, "base_year", t)
        baskets.append(basket)
        print(f"{t}: {len(keep_c)} countries x {len(keep_p)} products")
        
    network = pd.concat(network_rows, ignore_index=True)
    baskets = pd.concat(baskets, ignore_index=True)
    


    return network, baskets


def finalise(network, df, cy, py, products):
    """ At-risk rows, product information, and final columns. Returns the panel for further analysis"""
    print(network.shape[:])
    panel = network.merge(df, on = ["year", "country", "product"], how = "inner", validate = "one_to_one") 
    print(panel.shape[:])
    panel = panel.merge(cy[["country", "year", "log_country_exports"] + MACRO + ["legal_origin"]], on = ["country", "year"], how = "left", validate="many_to_one")
    print(panel.shape[:])
    panel = panel.merge(py[["product", "year", "log_world_exports", "world_growth"]], on = ["product", "year"], how = "left", validate="many_to_one")
    print(panel.shape[:])
    assert len(panel) == len(network)

    panel = panel[panel["rca"] < AT_RISK_MAX_RCA] # Saving observations where rca < 0.5. When inspecting the dta it severely limits the potential candidates. Should be experimented with. 
    panel["y"] = panel["y"].astype(int)

    #product metadata
    meta = pd.read_csv(PRODUCT_FILE, dtype=str)
    meta["root"] = meta["product_id_hierarchy"].str.split(".").str[0]
    sector_names = meta[meta["product_level"] == "1"].set_index("product_id")["product_name_short"]
    hs4 = meta[meta["product_level"] == "4"].set_index("product_hs92_code")
    info = pd.DataFrame({"product": products})
    parent = hs4.reindex(info["product"].str[:4])
    info["sector"] = parent["root"].map(sector_names).fillna("Unknown").to_numpy()
    info["natural_resource"] = (parent["natural_resource"] == "True").astype(int).to_numpy()
    info["green_product"] = (parent["green_product"] == "True").astype(int).to_numpy()
    panel = panel.merge(info, on="product", how="left").rename(columns={"year": "base_year"})

    keep = (["country", "product", "base_year"] + HISTORY 
            + ["years_competitive_last5", "ever_competitive", "log_exports", "export_growth",
               "log_imports", "rca_imports", "rca_volatility", "density", "diversity", "log_country_exports",
               "ubiquity", "log_world_exports", "world_growth", "sector", "natural_resource",
               "green_product"] + MACRO + ["legal_origin"] + NEIGHBOUR + ["y"])
    panel = panel[keep].sort_values(["base_year", "country", "product"], ignore_index=True)
    print(panel.shape[:])
    floats = panel.select_dtypes("float64").columns
    panel[floats] = panel[floats].astype(np.float32)   # lower precision to 32

    return panel


def main():
    header("B1  Load trade")
    trade, countries, products = load_trade()
    header("B2  Complete grid")
    df = complete_grid(trade, countries, products)
    header("B3  RCA")
    df = add_rca(df)
    header("B4  History features")
    df = add_history(df)
    header("B5  Label")
    df = add_label(df)
    header("B6  Country-year and product-year features")
    cy, py = country_and_product_tables(df)
    header("B7  Filters, density, baskets (one base year at a time)")
    network, baskets = network_and_baskets(df, cy, py)
    header("B8  At-risk rows, product info, save")
    panel = finalise(network, df, cy, py, products)
    panel.to_parquet(PANEL_FILE, index=False)
    baskets.to_parquet(BASKETS_FILE, index=False)
    print(f"Saved {PANEL_FILE.name}: {len(panel):,} rows, {panel['y'].mean():.2%} positives")


if __name__ == "__main__":
    main()
