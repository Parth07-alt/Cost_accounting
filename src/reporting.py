"""
reporting.py
============
Step 5 of the AI-Based Cost & Profit Optimization Pipeline.

Responsibilities:
    - Load all output CSVs (enriched data, metrics, feature importances,
      predictions).
    - Export a multi-sheet Excel workbook:
        Sheet 1: Enriched Dataset      (for Power BI main data source)
        Sheet 2: Outlet Summary        (aggregated KPIs per outlet)
        Sheet 3: Item Type Summary     (aggregated KPIs per item category)
        Sheet 4: Model Metrics         (MAE, RMSE, R²)
        Sheet 5: Feature Importances   (for Power BI bar chart)
        Sheet 6: Predictions           (actual vs predicted, for scatter)
    - Print a business-friendly final summary to console.

Usage:
    python src/reporting.py
"""

import os
import pandas as pd

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
OUTPUT_DIR        = "output"
ENRICHED_PATH     = os.path.join(OUTPUT_DIR, "enriched_bigmart.csv")
METRICS_PATH      = os.path.join(OUTPUT_DIR, "model_metrics.csv")
FEATURE_IMP_PATH  = os.path.join(OUTPUT_DIR, "feature_importance.csv")
PREDICTIONS_PATH  = os.path.join(OUTPUT_DIR, "predictions.csv")
EXCEL_PATH        = os.path.join(OUTPUT_DIR, "Retail_Cost_Profit_Report.xlsx")


# ─────────────────────────────────────────────
# AGGREGATION HELPERS
# ─────────────────────────────────────────────
def outlet_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate KPIs by Outlet_Type and Outlet_Identifier."""
    summary = df.groupby(
        ["Outlet_Type", "Outlet_Identifier", "Outlet_Location_Type",
         "Outlet_Size", "Outlet_Age"]
    ).agg(
        Records          = ("Profit", "count"),
        Total_Sales      = ("Item_Outlet_Sales", "sum"),
        Total_MaterialCost = ("Material_Cost", "sum"),
        Total_LaborCost  = ("Labor_Cost", "sum"),
        Total_Overhead   = ("Overhead_Cost", "sum"),
        Total_Cost       = ("Total_Cost", "sum"),
        Total_Profit     = ("Profit", "sum"),
        Avg_Profit_Margin= ("Profit_Margin_Pct", "mean"),
        Loss_Count       = ("Is_Loss", "sum"),
    ).reset_index()

    summary[["Total_Sales", "Total_MaterialCost", "Total_LaborCost",
             "Total_Overhead", "Total_Cost", "Total_Profit",
             "Avg_Profit_Margin"]] = summary[[
        "Total_Sales", "Total_MaterialCost", "Total_LaborCost",
        "Total_Overhead", "Total_Cost", "Total_Profit",
        "Avg_Profit_Margin"]].round(2)
    return summary.sort_values("Total_Profit", ascending=False)


def item_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate KPIs by Item_Type."""
    summary = df.groupby("Item_Type").agg(
        Records          = ("Profit", "count"),
        Avg_Sales        = ("Item_Outlet_Sales", "mean"),
        Avg_MaterialCost = ("Material_Cost", "mean"),
        Avg_LaborCost    = ("Labor_Cost", "mean"),
        Avg_Overhead     = ("Overhead_Cost", "mean"),
        Avg_TotalCost    = ("Total_Cost", "mean"),
        Avg_Profit       = ("Profit", "mean"),
        Avg_Margin_Pct   = ("Profit_Margin_Pct", "mean"),
        Loss_Count       = ("Is_Loss", "sum"),
    ).reset_index().round(2)
    return summary.sort_values("Avg_Profit", ascending=False)


# ─────────────────────────────────────────────
# EXCEL EXPORT
# ─────────────────────────────────────────────
def export_excel(df: pd.DataFrame,
                 metrics_df: pd.DataFrame,
                 fi_df: pd.DataFrame,
                 pred_df: pd.DataFrame) -> None:
    """
    Write a multi-sheet Excel workbook to EXCEL_PATH.

    Parameters
    ----------
    df         : enriched dataset
    metrics_df : model metrics (MAE, RMSE, R²)
    fi_df      : feature importances
    pred_df    : predictions vs actual (test set)
    """
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:

        # Sheet 1 — Enriched Dataset
        df.to_excel(writer, sheet_name="Enriched_Data", index=False)
        print(f"  ✔ Sheet 'Enriched_Data'       → {len(df):,} rows")

        # Sheet 2 — Outlet Summary
        out_sum = outlet_summary(df)
        out_sum.to_excel(writer, sheet_name="Outlet_Summary", index=False)
        print(f"  ✔ Sheet 'Outlet_Summary'      → {len(out_sum)} outlets")

        # Sheet 3 — Item Type Summary
        item_sum = item_type_summary(df)
        item_sum.to_excel(writer, sheet_name="Item_Type_Summary", index=False)
        print(f"  ✔ Sheet 'Item_Type_Summary'   → {len(item_sum)} categories")

        # Sheet 4 — Model Metrics
        metrics_df.to_excel(writer, sheet_name="Model_Metrics", index=False)
        print(f"  ✔ Sheet 'Model_Metrics'       → {len(metrics_df)} rows")

        # Sheet 5 — Feature Importances
        fi_df.to_excel(writer, sheet_name="Feature_Importances", index=False)
        print(f"  ✔ Sheet 'Feature_Importances' → {len(fi_df)} features")

        # Sheet 6 — Predictions
        pred_df.to_excel(writer, sheet_name="Predictions", index=False)
        print(f"  ✔ Sheet 'Predictions'         → {len(pred_df):,} rows")

    print(f"\n  ✔ Excel workbook saved → {EXCEL_PATH}")


# ─────────────────────────────────────────────
# BUSINESS SUMMARY PRINTOUT
# ─────────────────────────────────────────────
def print_business_summary(df: pd.DataFrame,
                            metrics_df: pd.DataFrame) -> None:
    """Print an executive-level business summary to the console."""
    total_records  = len(df)
    total_sales    = df["Item_Outlet_Sales"].sum()
    total_cost     = df["Total_Cost"].sum()
    total_profit   = df["Profit"].sum()
    avg_margin     = df["Profit_Margin_Pct"].mean()
    loss_count     = df["Is_Loss"].sum()
    loss_pct       = loss_count / total_records * 100

    mat_share  = df["Material_Cost"].sum() / total_cost * 100
    lab_share  = df["Labor_Cost"].sum()    / total_cost * 100
    ovh_share  = df["Overhead_Cost"].sum() / total_cost * 100

    r2  = metrics_df["R2"].iloc[0]
    mae = metrics_df["MAE"].iloc[0]

    print("\n" + "═"*58)
    print("  📊  FINAL BUSINESS INTELLIGENCE SUMMARY")
    print("═"*58)
    print(f"  Total Transactions   : {total_records:>12,}")
    print(f"  Total Sales          : ₹{total_sales:>14,.0f}")
    print(f"  Total Cost           : ₹{total_cost:>14,.0f}")
    print(f"  Total Profit         : ₹{total_profit:>14,.0f}")
    print(f"  Avg Profit Margin    : {avg_margin:>14.1f}%")
    print(f"  Loss-Making Records  : {loss_count:>12,}  ({loss_pct:.1f}%)")
    print()
    print("  Cost Component Breakdown:")
    print(f"    Material Cost  : {mat_share:>6.1f}% of Total Cost")
    print(f"    Labor Cost     : {lab_share:>6.1f}% of Total Cost")
    print(f"    Overhead Cost  : {ovh_share:>6.1f}% of Total Cost")
    print()
    print("  Machine Learning Results:")
    print(f"    R² Score       : {r2:>10.4f}  (variance explained)")
    print(f"    MAE            : ₹{mae:>10,.2f}")
    print()

    # Best outlet type
    best_outlet = (df.groupby("Outlet_Type")["Profit_Margin_Pct"]
                   .mean().idxmax())
    worst_outlet = (df.groupby("Outlet_Type")["Profit_Margin_Pct"]
                    .mean().idxmin())
    best_item = df.groupby("Item_Type")["Profit"].mean().idxmax()
    worst_item = df.groupby("Item_Type")["Profit"].mean().idxmin()

    print("  Key Insights:")
    print(f"    ✅ Most profitable outlet type  : {best_outlet}")
    print(f"    ⚠  Least profitable outlet type : {worst_outlet}")
    print(f"    ✅ Most profitable item category : {best_item}")
    print(f"    ⚠  Least profitable item category: {worst_item}")
    print("═"*58 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    print("\n" + "█"*55)
    print("  STEP 5 — REPORTING & EXPORT")
    print("█"*55)

    # Load all outputs
    print(f"\n  Loading output files...")
    df         = pd.read_csv(ENRICHED_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)
    fi_df      = pd.read_csv(FEATURE_IMP_PATH)

    # Predictions may not exist if model step was skipped
    if os.path.exists(PREDICTIONS_PATH):
        pred_df = pd.read_csv(PREDICTIONS_PATH)
    else:
        pred_df = pd.DataFrame(columns=["Item_Identifier", "Profit",
                                        "Predicted_Profit"])
        print("  ⚠ predictions.csv not found — Predictions sheet will be empty")

    print(f"  ✔ Loaded enriched data  : {df.shape[0]:,} rows")
    print(f"  ✔ Loaded model metrics  : {metrics_df.shape}")
    print(f"  ✔ Loaded feature imps   : {fi_df.shape}")

    # Export Excel
    print("\n── Exporting Excel Workbook ────────────────────────────────")
    export_excel(df, metrics_df, fi_df, pred_df)

    # Business summary
    print_business_summary(df, metrics_df)

    print("  STEP 5 COMPLETE ✓")
    print("  🎉 Full pipeline complete — check the 'output/' folder!\n")


if __name__ == "__main__":
    main()
