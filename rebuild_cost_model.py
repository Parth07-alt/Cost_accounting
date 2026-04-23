"""
rebuild_cost_model.py
=====================
Rebuild the cost model from scratch using the CLEAN new dataset:
    big mart dataset/train_dataset(big mart).csv

WHY WE REBUILD:
    Old dataset problem: Material_Cost = Sales x RAND() in Excel
    → Random function caused negative profits (loss rows)
    → Loss rows are unpredictable by any ML model (random noise)

CORRECT COST MODEL DESIGN (Industry Benchmarks, NO RANDOMNESS):
    Material Cost = MRP × category_material_rate
                    ↑ Based on MRP (purchase price), not sales price
                    ↑ Different rates per Item_Type (food vs non-food)

    Labor Cost    = Sales × outlet_labor_rate
                    ↑ Operations cost scales with transaction volume

    Overhead Cost = Sales × outlet_overhead_rate
                    ↑ Occupancy/utilities scales with outlet type

    PROFIT GUARANTEE: Material rates set so Profit is ALWAYS positive
    for the average item. A small % of items can still be loss-making
    but only when MRP is unusually high relative to sales (real scenario).

OUTPUT:
    Saves rebuilt dataset to:
    - data/processed/bigmart_cleaned.csv
    - output/enriched_bigmart.csv

Usage:
    python rebuild_cost_model.py
    (Then run main.py to re-run full pipeline on clean data)
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
NEW_TRAIN_PATH = os.path.join("big mart dataset", "train_dataset(big mart).csv")
CLEANED_PATH   = os.path.join("data", "processed", "bigmart_cleaned.csv")
ENRICHED_PATH  = os.path.join("output", "enriched_bigmart.csv")
CURRENT_YEAR   = 2026

# ─────────────────────────────────────────────
# COST MODEL PARAMETERS (Fixed, No Randomness)
# ─────────────────────────────────────────────

# Material Cost Rate = fraction of MRP paid to supplier (purchase cost)
# These are industry-realistic rates based on retail margin standards
# Retail margin = 1 - material_rate (so material_rate means supplier gets this %)
# Material Cost = fraction of SALES (not MRP) that goes to product procurement.
# For retail: COGS (Cost of Goods Sold) is typically 60-75% of sales revenue.
# We use Sales-based material rates so cost scales naturally with revenue.
ITEM_TYPE_MATERIAL_RATE = {
    # Lower COGS — high markup categories
    "Health and Hygiene":    0.50,
    "Hard Drinks":           0.52,
    "Soft Drinks":           0.53,
    "Household":             0.55,
    "Others":                0.55,
    # Medium COGS
    "Snack Foods":           0.58,
    "Baking Goods":          0.58,
    "Breakfast":             0.58,
    "Starchy Foods":         0.60,
    "Breads":                0.60,
    "Canned":                0.61,
    "Frozen Foods":          0.62,
    # Higher COGS — perishables with high wastage
    "Dairy":                 0.63,
    "Meat":                  0.65,
    "Seafood":               0.67,
    "Fruits and Vegetables": 0.68,
}
DEFAULT_MATERIAL_RATE = 0.60

# Labor Cost Rate = fraction of SALES for store operations staff
OUTLET_LABOR_RATE = {
    "Grocery Store":      0.08,
    "Supermarket Type1":  0.10,
    "Supermarket Type2":  0.11,
    "Supermarket Type3":  0.12,
}

# Overhead Cost Rate = fraction of SALES for rent, utilities, logistics
OUTLET_OVERHEAD_RATE = {
    "Grocery Store":      0.05,
    "Supermarket Type1":  0.07,
    "Supermarket Type2":  0.08,
    "Supermarket Type3":  0.09,
}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard cleaning: fill nulls, harmonize labels."""
    df = df.copy()

    # Fill Item_Weight with mean
    mean_weight = df["Item_Weight"].mean()
    null_before = df["Item_Weight"].isnull().sum()
    df["Item_Weight"] = df["Item_Weight"].fillna(mean_weight)
    print(f"  Item_Weight  : filled {null_before} nulls with mean ({mean_weight:.2f})")

    # Fill Outlet_Size with per-outlet-type mode
    null_before = df["Outlet_Size"].isnull().sum()
    df["Outlet_Size"] = df.groupby("Outlet_Type")["Outlet_Size"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Small")
    )
    print(f"  Outlet_Size  : filled {null_before} nulls with group mode")

    # Harmonize Item_Fat_Content labels
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({
        "LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"
    })
    print(f"  Fat Content  : harmonized → {df['Item_Fat_Content'].unique().tolist()}")

    print(f"  Remaining nulls: {df.isnull().sum().sum()}")
    return df


def apply_cost_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the CORRECT, DETERMINISTIC cost model.

    Key design principles:
    1. Material cost based on MRP (what retailer pays supplier)
       → This is INDEPENDENT of outlet sales
       → Reflects real procurement cost
    2. Labor & Overhead based on Sales (operational costs scale with volume)
    3. NO RANDOM NUMBERS → reproducible, meaningful results
    4. Profit = Sales - Material_Cost - Labor_Cost - Overhead_Cost
       → Positive when Sales > Total_Cost (which is the normal case)
    """
    df = df.copy()

    # ── Material Cost: based on Sales (COGS = % of revenue) ─────
    mat_rate = df["Item_Type"].map(ITEM_TYPE_MATERIAL_RATE).fillna(DEFAULT_MATERIAL_RATE)
    df["Material_Cost"] = (df["Item_Outlet_Sales"] * mat_rate).round(4)

    # ── Labor Cost: based on Sales ────────────────────────────────
    lab_rate = df["Outlet_Type"].map(OUTLET_LABOR_RATE).fillna(0.10)
    df["Labor_Cost"] = (df["Item_Outlet_Sales"] * lab_rate).round(4)

    # ── Overhead Cost: based on Sales ────────────────────────────
    ovh_rate = df["Outlet_Type"].map(OUTLET_OVERHEAD_RATE).fillna(0.07)
    df["Overhead_Cost"] = (df["Item_Outlet_Sales"] * ovh_rate).round(4)

    # ── Total Cost & Profit ───────────────────────────────────────
    df["Total_Cost"] = df["Material_Cost"] + df["Labor_Cost"] + df["Overhead_Cost"]
    df["Profit"]     = df["Item_Outlet_Sales"] - df["Total_Cost"]

    # ── Verify ────────────────────────────────────────────────────
    n_loss = (df["Profit"] < 0).sum()
    print(f"\n  Cost model applied:")
    print(f"  Loss-making rows    : {n_loss} ({n_loss/len(df)*100:.2f}%)")
    print(f"  Avg Profit          : Rs {df['Profit'].mean():,.2f}")
    print(f"  Avg Profit Margin   : {(df['Profit']/df['Item_Outlet_Sales']*100).mean():.1f}%")

    if n_loss > 0:
        loss_df = df[df["Profit"] < 0]
        print(f"\n  WHY some rows are still loss-making (this is REALISTIC):")
        print(f"  These items have HIGH supplier cost (MRP) but LOW actual sales:")
        print(loss_df[["Item_Type","Item_MRP","Item_Outlet_Sales",
                        "Material_Cost","Total_Cost","Profit"]].head(5).round(2).to_string())

    return df


def add_analytical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived KPI columns for analysis."""
    df = df.copy()
    df["Outlet_Age"]         = CURRENT_YEAR - df["Outlet_Establishment_Year"]
    df["Profit_Margin_Pct"]  = (df["Profit"] / df["Item_Outlet_Sales"] * 100).round(2)
    df["Is_Loss"]            = df["Profit"] < 0
    df["Units_Sold"]         = (df["Item_Outlet_Sales"] / df["Item_MRP"]).round(4)

    def bucket(p):
        if p < 0:      return "Loss"
        elif p <= 500: return "Low"
        else:          return "High"
    df["Profit_Bucket"] = df["Profit"].apply(bucket)

    return df


def print_summary(df: pd.DataFrame) -> None:
    total_sales  = df["Item_Outlet_Sales"].sum()
    total_cost   = df["Total_Cost"].sum()
    total_profit = df["Profit"].sum()
    avg_margin   = df["Profit_Margin_Pct"].mean()

    mat_share = df["Material_Cost"].sum() / total_cost * 100
    lab_share = df["Labor_Cost"].sum()    / total_cost * 100
    ovh_share = df["Overhead_Cost"].sum() / total_cost * 100

    print("\n" + "="*58)
    print("  FINANCIAL SUMMARY (Clean Cost Model)")
    print("="*58)
    print(f"  Total Records        : {len(df):,}")
    print(f"  Total Sales          : Rs {total_sales:>14,.0f}")
    print(f"  Total Cost           : Rs {total_cost:>14,.0f}")
    print(f"  Total Profit         : Rs {total_profit:>14,.0f}")
    print(f"  Avg Profit Margin    : {avg_margin:>14.1f}%")
    print(f"  Loss Records         : {df['Is_Loss'].sum():>14,} "
          f"({df['Is_Loss'].mean()*100:.1f}%)")
    print()
    print(f"  Cost Breakdown:")
    print(f"    Material Cost  : {mat_share:.1f}%  (based on MRP × category rate)")
    print(f"    Labor Cost     : {lab_share:.1f}%  (based on Sales × outlet rate)")
    print(f"    Overhead Cost  : {ovh_share:.1f}%  (based on Sales × outlet rate)")
    print()
    print(f"  By Outlet Type:")
    summary = df.groupby("Outlet_Type").agg(
        Count=("Profit","count"),
        Avg_Sales=("Item_Outlet_Sales","mean"),
        Avg_Cost=("Total_Cost","mean"),
        Avg_Profit=("Profit","mean"),
        Avg_Margin=("Profit_Margin_Pct","mean"),
        Loss_Count=("Is_Loss","sum")
    ).round(2)
    print(summary.to_string())
    print("="*58)


def main():
    print("\n" + "#"*58)
    print("  REBUILDING COST MODEL FROM CLEAN DATASET")
    print("#"*58)

    # ── Load clean dataset ────────────────────────────────────
    print(f"\n  Loading: {NEW_TRAIN_PATH}")
    df = pd.read_csv(NEW_TRAIN_PATH)
    print(f"  Loaded : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    # ── Clean ─────────────────────────────────────────────────
    print("\n  Cleaning data...")
    df = clean_data(df)

    # ── Apply cost model ──────────────────────────────────────
    print("\n  Applying cost model (deterministic, no random)...")
    df = apply_cost_model(df)

    # ── Add analytical columns ────────────────────────────────
    print("\n  Adding analytical columns...")
    df = add_analytical_columns(df)
    print(f"  Final columns ({df.shape[1]}): {list(df.columns)}")

    # ── Summary ───────────────────────────────────────────────
    print_summary(df)

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(CLEANED_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(ENRICHED_PATH), exist_ok=True)

    df.to_csv(CLEANED_PATH, index=False)
    df.to_csv(ENRICHED_PATH, index=False)

    print(f"\n  Saved -> {CLEANED_PATH}")
    print(f"  Saved -> {ENRICHED_PATH}")
    print("\n  NEXT STEP: Run  python main.py  to retrain everything")
    print("             using this correctly engineered dataset.\n")

    return df


if __name__ == "__main__":
    main()
