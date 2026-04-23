"""
data_loader.py
==============
Step 1 of the AI-Based Cost & Profit Optimization Pipeline.

Responsibilities:
    - Load the raw Big Mart dataset from CSV.
    - Report data quality (shape, dtypes, null counts).
    - Perform all data cleaning:
        * Impute Item_Weight with column mean.
        * Impute Outlet_Size with mode per Outlet_Type.
        * Harmonize Item_Fat_Content labels.
    - Save the cleaned dataset to data/processed/.

Usage:
    python src/data_loader.py
"""

import os
import pandas as pd

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
RAW_TRAIN_PATH = os.path.join("big mart dataset", "train_dataset(big mart).csv")
RAW_TEST_PATH  = os.path.join("big mart dataset", "test_dataset(big mart).csv")
PROCESSED_DIR  = os.path.join("data", "processed")
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "bigmart_cleaned.csv")


def load_raw(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    print(f"\n{'='*55}")
    print(f"  Loading data from: {path}")
    print(f"{'='*55}")
    df = pd.read_csv(path)
    print(f"  ✔ Loaded  →  {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    return df


def report_quality(df: pd.DataFrame) -> None:
    """
    Print a data-quality summary: dtypes and missing value counts.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to inspect.
    """
    print("\n── Column Info ──────────────────────────────────")
    info = pd.DataFrame({
        "dtype":   df.dtypes,
        "nulls":   df.isnull().sum(),
        "null_%":  (df.isnull().mean() * 100).round(2),
    })
    print(info.to_string())
    print("─────────────────────────────────────────────────\n")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning transformations to the Big Mart dataset.

    Transformations
    ---------------
    1. Item_Weight   → fill NaN with column mean.
    2. Outlet_Size   → fill NaN with mode per Outlet_Type group.
    3. Item_Fat_Content → harmonize label variants.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset (copy of input with modifications).
    """
    df = df.copy()

    # ── 1. Item_Weight: fill with mean ─────────────────────
    weight_nulls_before = df["Item_Weight"].isnull().sum()
    df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].mean())
    print(f"  ✔ Item_Weight : filled {weight_nulls_before} nulls with mean "
          f"({df['Item_Weight'].mean():.2f})")

    # ── 2. Outlet_Size: fill with mode per Outlet_Type ─────
    size_nulls_before = df["Outlet_Size"].isnull().sum()
    df["Outlet_Size"] = df.groupby("Outlet_Type")["Outlet_Size"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Small")
    )
    print(f"  ✔ Outlet_Size : filled {size_nulls_before} nulls using "
          f"per-group mode")

    # ── 3. Item_Fat_Content: harmonize labels ───────────────
    fat_map = {
        "LF":       "Low Fat",
        "low fat":  "Low Fat",
        "reg":      "Regular",
    }
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(fat_map)
    unique_cats = df["Item_Fat_Content"].unique().tolist()
    print(f"  ✔ Item_Fat_Content: harmonized → {unique_cats}")

    return df


def save_cleaned(df: pd.DataFrame, path: str) -> None:
    """
    Save the cleaned dataframe to a CSV file.

    Parameters
    ----------
    df   : pd.DataFrame
    path : str  — destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n  ✔ Cleaned data saved → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> pd.DataFrame:
    print("\n" + "█"*55)
    print("  STEP 1 — DATA LOADING & CLEANING")
    print("█"*55)

    # Load
    df = load_raw(RAW_TRAIN_PATH)

    # Inspect
    print("\n── Before Cleaning ──────────────────────────────")
    report_quality(df)

    # Clean
    print("── Applying Transformations ─────────────────────")
    df_clean = clean(df)

    # Verify
    print("\n── After Cleaning ───────────────────────────────")
    remaining_nulls = df_clean.isnull().sum().sum()
    print(f"  Total remaining nulls: {remaining_nulls}")

    # Save
    save_cleaned(df_clean, PROCESSED_PATH)

    print("\n  STEP 1 COMPLETE ✓\n")
    return df_clean


if __name__ == "__main__":
    main()
