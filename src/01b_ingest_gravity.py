"""STEP 01b: macro and geography variables from CEPII Gravity (V202211).

  country_year.parquet : country x year -> GDP per capita (PPP), population, GDP growth, GDP (nominal), WTO, EU, legal origin (categorical)
  pairs_geo.parquet    : country x partner -> contig (shared border), dist (km)


CEPII: Only the country and geography columns are used. The Gravity trade columns
(tradeflow_baci, ...) are note: trade comes from the Atlas only (considerations.md, 1).

  [x] A4  read_gravity()      read the CSV inside the zip, only COLUMNS
  [x] A4  clean_gravity()     keep years >= 1990 and pairs where both countries exist
  [x] A5  country_year()      origin-side variables, one row per (country, year), derived columns
  [x] A6  pairs_geo()         one row per (country, partner): contig and distance
  
"""
import zipfile

import numpy as np
import pandas as pd
from config import GRAVITY_ZIP, MACRO_FILE, GEO_FILE, LAGS

COLUMNS = ["year", "iso3_o", "iso3_d", "country_exists_o", "country_exists_d",
           "contig", "distw_harmonic", "dist",
           "pop_o", "gdp_o", "gdpcap_ppp_o", "wto_o", "eu_o", "legal_old_o"]


def read_gravity(path=GRAVITY_ZIP):
    with zipfile.ZipFile(path) as z, z.open("Gravity_V202211.csv") as f:
       df_gravity = pd.read_csv(f, usecols = COLUMNS)
    return df_gravity


def clean_gravity(df_gravity):
    """A4. Years >= 1990 and pairs where both countries exist."""
    recent_years = df_gravity["year"].ge(1990)
    origin_exists = df_gravity["country_exists_o"].eq(1)
    partner_exists = df_gravity["country_exists_d"].eq(1)
    keep = recent_years & origin_exists & partner_exists
    g = df_gravity[keep]
    print("Length gravity", len(df_gravity))
    print("Length cleaned", len(g))
    print("Year min", g["year"].min())
    print("Year max", g["year"].max())
    assert not g.duplicated(["year", "iso3_o", "iso3_d"]).any(), "duplicate pairs left after cleaning"
    return g


def country_year(g):
    """A5. One row per (country, year) with the origin-side variables and the derived features."""
    cy = (g.groupby(["iso3_o", "year"], as_index=False).first()
       .rename(columns={"iso3_o": "country", "wto_o": "wto", "eu_o": "eu"})
       .sort_values(["country", "year"], ignore_index=True))
    cy = cy.astype({"wto": int, "eu": int})
    cy["log_gdpcap_ppp"] = np.log(cy["gdpcap_ppp_o"])
    cy["log_pop"] = np.log(cy["pop_o"])
    cy["log_gdp"] = np.log(cy["gdp_o"])
    cy["gdp_growth"] = cy["log_gdp"] - cy["log_gdp"].groupby(cy["country"]).shift(LAGS)
    # Origin of the country’s legal system. Possible values are british, french, german, scandinavian, or socialist. 
    cy["legal_origin"] = cy["legal_old_o"].map({1: "fr", 2: "so", 3: "uk", 4: "ge", 5: "sc"}) 
    keep = ["country", "year", "log_gdpcap_ppp", "log_gdp", "log_pop", "gdp_growth", "wto", "eu", "legal_origin"]
    print("Share missing rows in country-year between 1990-2021 ", cy[keep].isna().mean().round(4).to_string()) # mean of binary variable gives us the missing share for each variable between 1990 – 2021

    return cy[keep]


def pairs_geo(g):
    geo = g.loc[g["year"].eq(2019), ["iso3_o", "iso3_d", "contig", "distw_harmonic", "dist"]]
    geo = geo.rename(columns={"iso3_o": "country", "iso3_d": "partner"})
    geo = geo[geo["country"].ne(geo["partner"])]
    geo["dist"] = geo["distw_harmonic"].fillna(geo["dist"])
    geo["contig"] = geo["contig"].astype(int)
    assert not geo.duplicated(["country", "partner"]).any(), "duplicate pairs left after cleaning"
    keep = ["country", "partner", "contig", "dist"]
    print("No unique countries: ", geo["country"].nunique()) # 235
    borders = geo.groupby("country")["contig"].sum()
    print("Countries without border:", borders.eq(0).sum())
    print("Country pairs: ", len(geo)) # 235*234
    return geo[keep].reset_index(drop=True)


def main():
    g = clean_gravity(read_gravity())
    country_year(g).to_parquet(MACRO_FILE, index=False)
    pairs_geo(g).to_parquet(GEO_FILE, index=False)
    print(f"Saved {MACRO_FILE.name}, {GEO_FILE.name}")


if __name__ == "__main__":
    main()
