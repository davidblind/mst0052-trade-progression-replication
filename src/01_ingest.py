"""STEP 01: read the Atlas country x HS4 x year file and save one clean table.

  trade_cpy_hs4.parquet : year x country x HS4 product -> exports, imports
                          (+ atlas_rca, atlas_distance: CHECKS ONLY, never features)

"""
import pandas as pd
from config import RAW_FILE, TRADE_FILE

COLUMNS = ["year", 
           "country_iso3_code", 
           "product_hs92_code",
           "export_value", 
           "import_value", 
           "export_rca", 
           "distance"]
DTYPES = {"year": "int16", 
          "country_iso3_code": "string", 
          "product_hs92_code": "string",
          "export_value": "float64", 
          "import_value": "float64"}


def read_raw(path=RAW_FILE):
    """A1. Read the raw file with product codes as text and only the columns in COLUMNS."""
    df = pd.read_csv(path, usecols=COLUMNS, dtype=DTYPES)
    # Assert every product code has length 4 
    assert df['product_hs92_code'].str.len().eq(4).all()
    return df


def clean(raw):
    """A2. Drop rows that are not a real (country, product).
    Returns the cleaned DataFrame and a dict {reason: share of export value dropped}."""
    # Boolean flag per reason to drop value
    unspecified_product = ~raw["product_hs92_code"].str.fullmatch(r"\d{4}")
    unclassified_trade = raw["product_hs92_code"].str.startswith("99")
    bad_country = raw["country_iso3_code"] == "ANS"
    
    # Print the dropped share of export value per reason, and in total
    total_value = raw["export_value"].sum()
    drop = unspecified_product | unclassified_trade | bad_country
    dropped = {}
    print("Share of export value dropped per reason:")
    for reason, flag in [("unspecified product", unspecified_product),
                         ("unclassified trade", unclassified_trade),
                         ("bad country", bad_country)]:
        dropped[reason] = raw.loc[flag, "export_value"].sum() / total_value
        print(f"{reason}: {dropped[reason]:.2%}")
    dropped["total"] = raw.loc[drop, "export_value"].sum() / total_value
    dropped["rows"] = int(drop.sum())
    print(f"Total export share dropped: {dropped['total']:.2%}, rows dropped: {dropped['rows']:,}")
    df = raw[~drop]
    return df, dropped



def tidy_and_save(df, out=TRADE_FILE):
    """A3. Panel names, checks, parquet. Do NOT fill in the grid here: that is B2's job."""
    # Renaming columns
    df = df.rename(columns={"country_iso3_code": "country", 
                            "product_hs92_code": "product", 
                            "export_value": "exports",
                            "import_value": "imports",
                            "export_rca": "atlas_rca",
                            "distance": "atlas_distance"
                            })
    # assert no duplicated (year, country, product); assert exports, imports >= 0
    assert not df.duplicated(subset=["year", "country", "product"]).any(), "Duplicate (year, country, product) found"
    assert (df["exports"] >= 0).all(), "Export values must be non-negative"
    assert (df["imports"] >= 0).all(), "Import values must be non-negative"
    # sort by keys; if out is not None: write parquet and print the file name
    df = df.sort_values(by=["year", "country", "product"], ignore_index=True)
    if out is not None:
        df.to_parquet(out, index=False)
        print(f"File saved: {out}")
    return df


def main():
    raw = read_raw()
    df, dropped = clean(raw)
    tidy_and_save(df)


if __name__ == "__main__":
    main()
