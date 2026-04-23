"""
feature_engineering.py
=======================
Step 2 of the AI-Based Cost & Profit Optimization Pipeline.

NOTE: The new clean Big Mart dataset lacks cost columns.
We engineer a realistic, deterministic cost model based on industry
retail standards for each outlet and item type.

Usage:
    python src/feature_engineering.py
"""

import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CURRENT_YEAR = 2026

# Realistic COGS percentages
ITEM_TYPE_MATERIAL_RATE = {
    "Health and Hygiene":    0.50,
    "Hard Drinks":           0.52,
    "Soft Drinks":           0.53,
    "Household":             0.55,
    "Others":                0.55,
    "Snack Foods":           0.58,
    "Baking Goods":          0.58,
    "Breakfast":             0.58,
    "Starchy Foods":         0.60,
    "Breads":                0.60,
    "Canned":                0.61,
    "Frozen Foods":          0.62,
    "Dairy":                 0.63,
    "Meat":                  0.65,
    "Seafood":               0.67,
    "Fruits and Vegetables": 0.68,
}
DEFAULT_MATERIAL_RATE = 0.60

OUTLET_LABOR_RATE = {
    "Grocery Store":      0.08,
    "Supermarket Type1":  0.10,
    "Supermarket Type2":  0.11,
    "Supermarket Type3":  0.12,
}

OUTLET_OVERHEAD_RATE = {
    "Grocery Store":      0.05,
    "Supermarket Type1":  0.07,
    "Supermarket Type2":  0.08,
    "Supermarket Type3":  0.09,
}

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
CLEANED_PATH  = os.path.join("data", "processed", "bigmart_cleaned.csv")
OUTPUT_DIR    = "output"
ENRICHED_PATH = os.path.join(OUTPUT_DIR, "enriched_bigmart.csv")


def apply_cost_model(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the deterministic cost model to create financial columns."""
    df = df.copy()

    # Material Cost: based on Sales (COGS = % of revenue)
    mat_rate = df["Item_Type"].map(ITEM_TYPE_MATERIAL_RATE).fillna(DEFAULT_MATERIAL_RATE)
    df["Material_Cost"] = (df["Item_Outlet_Sales"] * mat_rate).round(4)

    # Labor & Overhead: based on Sales
    lab_rate = df["Outlet_Type"].map(OUTLET_LABOR_RATE).fillna(0.10)
    df["Labor_Cost"] = (df["Item_Outlet_Sales"] * lab_rate).round(4)

    ovh_rate = df["Outlet_Type"].map(OUTLET_OVERHEAD_RATE).fillna(0.07)
    df["Overhead_Cost"] = (df["Item_Outlet_Sales"] * ovh_rate).round(4)

    df["Total_Cost"] = df["Material_Cost"] + df["Labor_Cost"] + df["Overhead_Cost"]
    df["Profit"]     = df["Item_Outlet_Sales"] - df["Total_Cost"]

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Avoid div by zero
    df["Profit_Margin_Pct"] = np.where(
        df["Item_Outlet_Sales"] > 0, 
        df["Profit"] / df["Item_Outlet_Sales"] * 100, 
        0
    ).round(2)
    
    df["Is_Loss"] = df["Profit"] < 0
    df["Outlet_Age"] = CURRENT_YEAR - df["Outlet_Establishment_Year"]
    
    def bucket(profit):
        if profit < 0: return "Loss"
        elif profit <= 500: return "Low"
        else: return "High"
    df["Profit_Bucket"] = df["Profit"].apply(bucket)
    df["Units_Sold"] = np.where(df["Item_MRP"] > 0, df["Item_Outlet_Sales"] / df["Item_MRP"], 0).round(4)
    
    return df


def print_financial_summary(df: pd.DataFrame) -> None:
    total_sales  = df["Item_Outlet_Sales"].sum()
    total_cost   = df["Total_Cost"].sum()
    total_profit = df["Profit"].sum()
    avg_margin   = df["Profit_Margin_Pct"].mean()

    print("\n── Financial Summary (from REAL dataset values) ─────────────")
    print(f"  Total Sales          : Rs {total_sales:>14,.0f}")
    print(f"  Total Cost           : Rs {total_cost:>14,.0f}")
    print(f"  Total Profit         : Rs {total_profit:>14,.0f}")
    print(f"  Avg Profit Margin    : {avg_margin:>14.1f}%")
    print(f"  Loss Records         : {df['Is_Loss'].sum():>14,}")
    print("─────────────────────────────────────────────────────────────\n")


def main() -> pd.DataFrame:
    print("\n" + "#"*55)
    print("  STEP 2 — FEATURE ENGINEERING")
    print("#"*55)

    print(f"\n  Loading cleaned data from: {CLEANED_PATH}")
    df = pd.read_csv(CLEANED_PATH)
    
    print("\n── Applying Deterministic Cost Model ──────────────────────")
    df = apply_cost_model(df)

    print("\n── Adding Derived Columns ───────────────────────────────────")
    df = add_derived_columns(df)

    print_financial_summary(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(ENRICHED_PATH, index=False)
    print(f"  Enriched dataset saved -> {ENRICHED_PATH}")
    print("  STEP 2 COMPLETE\n")
    return df


if __name__ == "__main__":
    main()
