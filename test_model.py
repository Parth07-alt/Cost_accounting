"""
test_model.py
=============
Comprehensive ML model testing & evaluation script.

3 Levels of Testing:
─────────────────────────────────────────────────────────
LEVEL 1 — Metrics on 20% held-out test split
           (Actual vs Predicted, MAE, RMSE, R²)

LEVEL 2 — Manual spot check
           (Pick 10 real records, predict, show side-by-side)

LEVEL 3 — Baseline comparison
           (Random Forest vs Dummy, Linear Regression, Decision Tree)
           Proves the model is actually learning something useful.
─────────────────────────────────────────────────────────

Usage:
    python test_model.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ─────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────
ENRICHED_PATH  = os.path.join("output", "enriched_bigmart.csv")
MODEL_PATH     = os.path.join("output", "trained_model.pkl")
PLOTS_DIR      = os.path.join("output", "plots")
REPORT_PATH    = os.path.join("output", "model_test_report.csv")

RANDOM_SEED = 42
TEST_SIZE   = 0.20

FEATURE_COLS = [
    "Item_MRP",
    "Item_Outlet_Sales",
    "Item_Visibility",
    "Item_Weight",
    "Outlet_Age",
    "Item_Fat_Content",
    "Item_Type",
    "Outlet_Identifier",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
]
TARGET_COL = "Profit"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _save_plot(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved -> {path}")

def metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"Model": label, "MAE": round(mae,2),
            "RMSE": round(rmse,2), "R2": round(r2,4)}

def encode(df_train_ref, df_target):
    """Label-encode categoricals using training data classes."""
    cat_cols = [c for c in FEATURE_COLS
                if df_train_ref[c].dtype == object]
    X = df_target[FEATURE_COLS].copy()
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df_train_ref[col].astype(str))
        def _t(v):
            v = str(v)
            return le.transform([v])[0] if v in le.classes_ else -1
        X[col] = X[col].astype(str).apply(_t)
    return X.fillna(X.median())


# ══════════════════════════════════════════════════════════════
#  LEVEL 1 — TEST SPLIT EVALUATION
# ══════════════════════════════════════════════════════════════
def level1_test_split(df, model):
    """
    Evaluate the model on the same 20% test split used during training.
    Since random_state=42 is fixed, we get the EXACT same split every time.
    Compare: Actual Profit (known) vs Predicted Profit (model output).
    """
    print("\n" + "="*58)
    print("  LEVEL 1 — TEST SPLIT EVALUATION (20% of Train.csv)")
    print("="*58)
    print("  We held out 20% of Train.csv during training.")
    print("  These rows have REAL profit values.")
    print("  We can measure exactly how wrong the model is.\n")

    X = encode(df, df)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    y_pred = pd.Series(model.predict(X_test), index=y_test.index)

    m = metrics(y_test, y_pred, "Random Forest (our model)")
    print(f"  Results on {len(y_test):,} test records:")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  R²   (variance explained)  : {m['R2']:>8.4f} │")
    print(f"  │  MAE  (avg absolute error)  : Rs{m['MAE']:>8,.2f} │")
    print(f"  │  RMSE (root mean sq error)  : Rs{m['RMSE']:>8,.2f} │")
    print(f"  └─────────────────────────────────────────┘")
    print(f"\n  Interpretation:")
    print(f"  - R²={m['R2']:.4f} means the model explains {m['R2']*100:.1f}% of")
    print(f"    profit variation using business features.")
    print(f"  - On average, predictions are off by Rs {m['MAE']:,.0f}.")
    print(f"  - Mean actual profit: Rs {y_test.mean():,.0f}")
    print(f"  - Error as % of mean: {m['MAE']/y_test.mean()*100:.1f}%")

    # Chart — Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter: actual vs predicted
    axes[0].scatter(y_test, y_pred, alpha=0.3, s=12,
                    color="#2E4057", label="Predictions")
    lims = [min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, color="#E63946", lw=2,
                 linestyle="--", label="Perfect (45° line)")
    axes[0].set_title(f"Actual vs Predicted Profit  (R²={m['R2']:.4f})\n"
                      "Closer to red line = better prediction",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Actual Profit (Rs)")
    axes[0].set_ylabel("Predicted Profit (Rs)")
    axes[0].legend()

    # Error distribution
    errors = y_test - y_pred
    sns.histplot(errors, bins=50, kde=True, color="#1B998B",
                 edgecolor="white", ax=axes[1])
    axes[1].axvline(0, color="#E63946", lw=2, linestyle="--",
                    label="Zero error")
    axes[1].axvline(errors.mean(), color="#F4A261", lw=1.5,
                    linestyle="-.", label=f"Mean error: Rs{errors.mean():.0f}")
    axes[1].set_title("Prediction Error Distribution\n"
                      "Centered at 0 = no systematic bias",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Error = Actual − Predicted (Rs)")
    axes[1].legend()

    plt.tight_layout()
    _save_plot(fig, "T1_actual_vs_predicted_test_split.png")

    return m, X_test, y_test, y_pred


# ══════════════════════════════════════════════════════════════
#  LEVEL 2 — MANUAL SPOT CHECK
# ══════════════════════════════════════════════════════════════
def level2_spot_check(df, model, X_test, y_test, y_pred):
    """
    Pick 10 specific records from the test split.
    Show the full feature row, the REAL profit, the PREDICTED profit,
    and the error. This is the simplest human-readable test.
    """
    print("\n" + "="*58)
    print("  LEVEL 2 — MANUAL SPOT CHECK (10 sample records)")
    print("="*58)
    print("  These are REAL rows from your data.")
    print("  Compare the 'Actual' column to 'Predicted'.\n")

    # Sample 10 rows: mix of high, low, and loss-making
    sample_idx = (
        list(y_test.nlargest(3).index) +    # top 3 highest profit
        list(y_test.nsmallest(3).index) +   # bottom 3 (likely losses)
        list(y_test.sample(4, random_state=42).index)  # 4 random
    )
    sample_idx = list(dict.fromkeys(sample_idx))[:10]  # deduplicate, keep order

    spot = df.loc[sample_idx, [
        "Item_Identifier", "Item_Type", "Outlet_Type",
        "Outlet_Location_Type", "Item_MRP", "Item_Outlet_Sales",
        "Profit"
    ]].copy()
    spot["Predicted_Profit"] = y_pred.loc[sample_idx].values
    spot["Error (Rs)"]       = (spot["Profit"] - spot["Predicted_Profit"]).round(2)
    spot["Error %"]          = (spot["Error (Rs)"] / spot["Profit"].abs() * 100).round(1)
    spot = spot.round(2)

    # Rename for clean display
    spot.columns = ["Item ID", "Item Type", "Outlet Type",
                    "Location", "MRP", "Sales",
                    "Actual Profit", "Predicted Profit",
                    "Error (Rs)", "Error %"]

    print(spot.to_string(index=False))

    # Also save as CSV for easy reference
    spot.to_csv(os.path.join("output", "spot_check_results.csv"), index=False)
    print(f"\n  Saved to output/spot_check_results.csv")

    # ── Chart: side-by-side bar ──────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))
    x = range(len(spot))
    width = 0.38
    ax.bar([i - width/2 for i in x], spot["Actual Profit"],
           width, label="Actual Profit", color="#048A81", edgecolor="white")
    ax.bar([i + width/2 for i in x], spot["Predicted Profit"],
           width, label="Predicted Profit", color="#2E4057",
           edgecolor="white", alpha=0.85)
    ax.axhline(0, color="#E63946", linestyle="--", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{row['Item Type'][:12]}\n{row['Outlet Type'][:12]}"
         for _, row in spot.iterrows()],
        fontsize=8, rotation=10
    )
    ax.set_title("Manual Spot Check — Actual vs Predicted Profit\n"
                 "Green = Actual  |  Dark Blue = Predicted",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Profit (Rs)")
    ax.legend()
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"Rs{x:,.0f}"))
    plt.tight_layout()
    _save_plot(fig, "T2_spot_check_actual_vs_predicted.png")


# ══════════════════════════════════════════════════════════════
#  LEVEL 3 — BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════
def level3_baseline_comparison(df, model):
    """
    Compare Random Forest against simpler models.
    If RF doesn't beat these baselines, it's not actually learning.

    Baselines:
      - Dummy (Mean)      — always predicts the average profit
      - Linear Regression — assumes linear relationship
      - Decision Tree     — single tree (what RF is made of, unpruned)
      - Gradient Boosting — a stronger ensemble (state-of-the-art)
    """
    print("\n" + "="*58)
    print("  LEVEL 3 — BASELINE COMPARISON")
    print("="*58)
    print("  Comparing Random Forest against simpler models.")
    print("  RF must beat the baselines to prove it's useful.\n")

    X = encode(df, df)
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    models_to_compare = {
        "Dummy Regressor\n(always predicts mean)":
            DummyRegressor(strategy="mean"),
        "Linear Regression\n(assumes linear fit)":
            LinearRegression(),
        "Decision Tree\n(single tree, depth=10)":
            DecisionTreeRegressor(max_depth=10, random_state=RANDOM_SEED),
        "Random Forest\n(OUR MODEL)":
            model,   # already trained — use directly
        "Gradient Boosting\n(stronger ensemble)":
            GradientBoostingRegressor(n_estimators=100,
                                      random_state=RANDOM_SEED),
    }

    results = []
    print(f"  {'Model':<40} {'R²':>8} {'MAE (Rs)':>12} {'RMSE (Rs)':>12}")
    print(f"  {'-'*40} {'-'*8} {'-'*12} {'-'*12}")

    for name, m in models_to_compare.items():
        display_name = name.replace("\n", " ")
        if m is not model:          # don't retrain our model
            m.fit(X_train, y_train)
        y_p = m.predict(X_test)
        r   = metrics(y_test, y_p, display_name)
        results.append(r)
        tag = "  <-- OUR MODEL" if "Random Forest" in name else ""
        print(f"  {display_name:<40} {r['R2']:>8.4f} "
              f"{r['MAE']:>12,.2f} {r['RMSE']:>12,.2f}{tag}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join("output", "model_comparison.csv"),
                      index=False)
    print(f"\n  Saved to output/model_comparison.csv")

    # ── Chart: grouped bar comparison ─────────────────────────
    short_names = [
        "Dummy\n(Mean)", "Linear\nRegression",
        "Decision\nTree", "Random\nForest\n(OURS)",
        "Gradient\nBoosting"
    ]
    r2_vals  = [r["R2"]   for r in results]
    mae_vals = [r["MAE"]  for r in results]
    colors   = ["#ADB5BD", "#6C757D", "#495057", "#048A81", "#1B998B"]
    rf_idx   = 3

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # R² bar
    bars = axes[0].bar(short_names, r2_vals, color=colors, edgecolor="white",
                       linewidth=0.8)
    bars[rf_idx].set_edgecolor("#E63946")
    bars[rf_idx].set_linewidth(3)
    for bar, val in zip(bars, r2_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    axes[0].set_title("R² Score Comparison\n(Higher is better)",
                      fontweight="bold")
    axes[0].set_ylabel("R² Score")
    axes[0].set_ylim(0, max(r2_vals) * 1.15)
    axes[0].axhline(0, color="black", lw=0.5)

    # MAE bar (lower is better)
    bars2 = axes[1].bar(short_names, mae_vals, color=colors,
                        edgecolor="white", linewidth=0.8)
    bars2[rf_idx].set_edgecolor("#E63946")
    bars2[rf_idx].set_linewidth(3)
    for bar, val in zip(bars2, mae_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 1,
                     f"Rs{val:,.0f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    axes[1].set_title("MAE Comparison\n(Lower is better)",
                      fontweight="bold")
    axes[1].set_ylabel("Mean Absolute Error (Rs)")
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"Rs{x:,.0f}"))

    plt.suptitle("Model Comparison: Random Forest vs Baselines\n"
                 "(Red border = Our Model)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_plot(fig, "T3_model_comparison_baselines.png")

    return results_df


# ══════════════════════════════════════════════════════════════
#  MAIN — RUN ALL 3 LEVELS
# ══════════════════════════════════════════════════════════════
def main():
    print("\n" + "#"*58)
    print("  ML MODEL TESTING & EVALUATION REPORT")
    print("  AI-Based Cost & Profit Optimization System")
    print("#"*58)

    # Load enriched training data & trained model
    print(f"\n  Loading data  : {ENRICHED_PATH}")
    df = pd.read_csv(ENRICHED_PATH)
    print(f"  Loaded        : {df.shape[0]:,} rows x {df.shape[1]} columns")

    print(f"\n  Loading model : {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("  Model loaded successfully.")

    # ── Run all 3 levels ──────────────────────────────────────
    m1, X_test, y_test, y_pred = level1_test_split(df, model)
    level2_spot_check(df, model, X_test, y_test, y_pred)
    results_df = level3_baseline_comparison(df, model)

    # ── Final verdict ─────────────────────────────────────────
    rf_row  = results_df[results_df["Model"].str.contains("Random Forest")].iloc[0]
    dum_row = results_df[results_df["Model"].str.contains("Dummy")].iloc[0]

    print("\n" + "="*58)
    print("  FINAL VERDICT")
    print("="*58)
    improvement = rf_row["R2"] - dum_row["R2"]
    mae_improvement = (dum_row["MAE"] - rf_row["MAE"]) / dum_row["MAE"] * 100

    print(f"\n  Random Forest R²  : {rf_row['R2']:.4f}")
    print(f"  Dummy (Mean) R²   : {dum_row['R2']:.4f}")
    print(f"  Improvement       : +{improvement:.4f} R² points")
    print(f"  MAE reduction     : {mae_improvement:.1f}% better than always")
    print(f"                      predicting the mean profit")

    print(f"\n  WHERE TO COMPARE:")
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  output/spot_check_results.csv  <- row-by-row table │")
    print(f"  │  output/model_comparison.csv    <- model vs baselines│")
    print(f"  │  output/predictions.csv         <- all 1,705 rows   │")
    print(f"  │  output/plots/T1_*.png          <- actual vs predict │")
    print(f"  │  output/plots/T2_*.png          <- spot check chart  │")
    print(f"  │  output/plots/T3_*.png          <- comparison chart  │")
    print(f"  └─────────────────────────────────────────────────────┘")

    print("\n  TESTING COMPLETE\n")


if __name__ == "__main__":
    main()
