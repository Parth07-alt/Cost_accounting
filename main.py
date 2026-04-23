"""
main.py
=======
Master pipeline runner for the AI-Based Cost & Profit Optimization System.

This script chains all 5 pipeline steps in sequence:
    Step 1 → data_loader.py      (Load & Clean)
    Step 2 → feature_engineering (Cost Model + Profit)
    Step 3 → eda.py              (Exploratory Data Analysis)
    Step 4 → model.py            (ML Training & Evaluation)
    Step 5 → reporting.py        (Excel Export + Summary)

Usage:
    python main.py

All outputs are saved to the output/ directory.
"""

import os
import sys
import time
import io

# Fix Windows console encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   AI-Based Cost & Profit Optimization System                 ║
║   Retail Outlets  |  Big Mart Dataset  |  ML Pipeline        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_step(step_num: int, step_name: str, fn):
    """Run a pipeline step, timing it and catching any errors."""
    separator = "─" * 60
    print(f"\n{separator}")
    print(f"  ▶  RUNNING STEP {step_num}: {step_name}")
    print(separator)
    start = time.time()
    try:
        result = fn()
        elapsed = time.time() - start
        print(f"\n  ✅ Step {step_num} finished in {elapsed:.1f}s")
        return result
    except Exception as exc:
        elapsed = time.time() - start
        print(f"\n  ❌ Step {step_num} FAILED after {elapsed:.1f}s")
        print(f"     Error: {exc}")
        raise


def main():
    overall_start = time.time()
    print_banner()

    # ── Step 1: Load & Clean Data ─────────────────────────────
    from data_loader import main as step1
    df_clean = run_step(1, "Data Loading & Cleaning", step1)

    # ── Step 2: Feature Engineering (Cost Model) ──────────────
    from feature_engineering import main as step2
    df_enriched = run_step(2, "Feature Engineering (Cost Model + Profit)", step2)

    # ── Step 3: Exploratory Data Analysis ─────────────────────
    import pandas as pd
    # Pass the in-memory enriched df to avoid re-reading from disk
    enriched_path = os.path.join("output", "enriched_bigmart.csv")
    df_for_eda = pd.read_csv(enriched_path) if df_enriched is None else df_enriched

    from eda import main as step3
    run_step(3, "Exploratory Data Analysis (8 Charts)", lambda: step3(df_for_eda))

    # ── Step 4: Machine Learning Pipeline ─────────────────────
    from model import main as step4
    model_result = run_step(4, "Machine Learning (Random Forest + Evaluation)", step4)

    # ── Step 5: Reporting & Export ────────────────────────────
    from reporting import main as step5
    run_step(5, "Reporting & Excel Export", step5)

    # ── Final Summary ─────────────────────────────────────────
    total_time = time.time() - overall_start
    print("\n" + "═"*62)
    print("  🎉  FULL PIPELINE COMPLETE!")
    print(f"      Total time: {total_time:.1f} seconds")
    print()
    print("  📂  Outputs saved to:  output/")
    print("    ├── enriched_bigmart.csv          ← Main enriched dataset")
    print("    ├── model_metrics.csv             ← MAE, RMSE, R²")
    print("    ├── feature_importance.csv        ← Feature rankings")
    print("    ├── predictions.csv               ← Model predictions")
    print("    ├── Retail_Cost_Profit_Report.xlsx← Power BI source")
    print("    └── plots/")
    print("        ├── 01_profit_distribution.png")
    print("        ├── 02_cost_breakdown_by_outlet.png")
    print("        ├── 03_sales_vs_profit_scatter.png")
    print("        ├── 04_correlation_heatmap.png")
    print("        ├── 05_profit_margin_boxplot.png")
    print("        ├── 06_top_bottom_item_types.png")
    print("        ├── 07_profitable_vs_loss_pie.png")
    print("        ├── 08_outlet_age_vs_profit.png")
    print("        ├── 09_predicted_vs_actual.png")
    print("        ├── 10_feature_importance.png")
    print("        └── 11_residuals.png")
    print()
    print("  📊  Next step: Open Power BI Desktop and load")
    print("      output/Retail_Cost_Profit_Report.xlsx")
    print("═"*62 + "\n")


if __name__ == "__main__":
    main()
