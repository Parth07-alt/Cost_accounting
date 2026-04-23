"""
model.py
========
Step 4 of the AI-Based Cost & Profit Optimization Pipeline.

Responsibilities:
    - Prepare the feature matrix (encode categoricals, drop leaky columns).
    - Split data 80/20 (train/test) with a fixed random seed.
    - Train a Random Forest Regressor to predict Profit.
    - Evaluate: MAE, RMSE, R².
    - Generate Predicted vs. Actual and Feature Importance charts.
    - Save predictions, metrics, and importances to output/.

Usage:
    python src/model.py

Why Random Forest?
    - Handles mixed data types (numeric + encoded categoricals) well.
    - Robust to outliers — important since our cost model adds noise.
    - Provides feature importances natively.
    - No need for feature scaling (unlike linear regression or SVMs).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.20

# Features used as model INPUT — BUSINESS SIGNALS ONLY.
#
# ⚠ IMPORTANT — DATA LEAKAGE PREVENTION:
# The cost columns (Material_Cost, Labor_Cost, Overhead_Cost, Total_Cost)
# were created in Excel using formulas DERIVED FROM Sales:
#     Material_Cost = Sales × rate
#     Labor_Cost    = Sales × rate
#     Overhead_Cost = Sales × rate
#     Total_Cost    = sum of above
#     Profit        = Sales − Total_Cost   ← mathematical identity!
#
# If we include these as features, the model just learns the formula
# you already typed — not real business patterns. R²=0.97 would be fake.
#
# We use ONLY features a business analyst would know BEFORE calculating costs:
#     → Product characteristics (MRP, Visibility, Weight, Type, Fat Content)
#     → Outlet characteristics  (Type, Size, Location, Age, Identifier)
#     → Sales volume            (Item_Outlet_Sales)
#
# The model's job: "Given these business inputs, predict how profitable
# this product-outlet combination is."

FEATURE_COLS = [
    # --- Sales & Product ---
    "Item_MRP",              # max retail price — strong profitability signal
    "Item_Visibility",       # shelf space — marketing investment
    "Item_Weight",           # proxy for packaging/handling cost
    # --- Outlet Characteristics ---
    "Outlet_Age",            # older outlets may have loyal customer base
    # --- Categorical (will be label-encoded) ---
    "Item_Fat_Content",      # product health segment
    "Item_Type",             # product category
    "Outlet_Identifier",     # specific outlet — captures outlet-level patterns
    "Outlet_Size",           # small/medium/large
    "Outlet_Location_Type",  # tier 1/2/3 city
    "Outlet_Type",           # grocery vs supermarket type
]

TARGET_COL = "Profit"

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators":      300,
    "max_depth":         15,   # limit depth slightly to reduce overfitting
    "min_samples_split": 5,
    "min_samples_leaf":  3,
    "max_features":      "sqrt",
    "random_state":      RANDOM_SEED,
    "n_jobs":            -1,
}

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ENRICHED_PATH       = os.path.join("output", "enriched_bigmart.csv")
PLOTS_DIR           = os.path.join("output", "plots")
METRICS_PATH        = os.path.join("output", "model_metrics.csv")
FEATURE_IMP_PATH    = os.path.join("output", "feature_importance.csv")
PREDICTIONS_PATH    = os.path.join("output", "predictions.csv")
MODEL_PATH          = os.path.join("output", "trained_model.pkl")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _save_plot(fig, filename: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Plot saved → {path}")


# ─────────────────────────────────────────────
# STEP A — PREPARE DATA
# ─────────────────────────────────────────────
def prepare_features(df: pd.DataFrame):
    """
    Encode categorical columns and return X, y.

    Strategy: Label Encoding.
    Label encoding is used here (vs One-Hot Encoding) because:
      - Random Forest handles ordinal-ish numeric splits well.
      - OHE would add ~30+ dummy columns, increasing compute without
        significant accuracy gain for tree-based models.

    Returns
    -------
    X : pd.DataFrame  — feature matrix (numeric)
    y : pd.Series     — target vector (Profit)
    encoders : dict   — {column_name: fitted LabelEncoder}
    """
    print("\n── Feature Preparation ─────────────────────────────────────")
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    print(f"  Categorical columns to encode: {cat_cols}")

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"  ✔ Encoded '{col}' → {len(le.classes_)} classes")

    null_count = X.isnull().sum().sum()
    if null_count > 0:
        print(f"  ⚠ {null_count} nulls found in feature matrix — filling with median")
        X = X.fillna(X.median())

    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Target vector shape : {y.shape}")
    print(f"  Target range        : ₹{y.min():,.0f}  →  ₹{y.max():,.0f}")
    return X, y, encoders


# ─────────────────────────────────────────────
# STEP B — TRAIN MODEL
# ─────────────────────────────────────────────
def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Split data and train Random Forest Regressor.

    Returns
    -------
    model     : fitted RandomForestRegressor
    X_train, X_test, y_train, y_test
    """
    print("\n── Train / Test Split ──────────────────────────────────────")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )
    print(f"  Train set : {len(X_train):,} records  ({1-TEST_SIZE:.0%})")
    print(f"  Test set  : {len(X_test):,} records  ({TEST_SIZE:.0%})")

    print("\n── Training Random Forest Regressor ────────────────────────")
    print(f"  Parameters: {RF_PARAMS}")
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    print("  ✔ Model training complete")

    # 5-Fold Cross Validation on training data
    print("\n── Cross-Validation (5-Fold) ───────────────────────────────")
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=5, scoring="r2", n_jobs=-1)
    print(f"  CV R² scores  : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  CV R² mean    : {cv_scores.mean():.4f}")
    print(f"  CV R² std     : {cv_scores.std():.4f}")

    return model, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STEP C — EVALUATE
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, y_mean: float):
    """
    Compute and print MAE, RMSE, R² metrics.

    Returns
    -------
    dict of metrics
    pd.Series of predictions
    """
    print("\n── Model Evaluation ────────────────────────────────────────")
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mae_pct_of_mean = (mae / abs(y_mean)) * 100

    metrics = {
        "MAE":              round(mae, 2),
        "RMSE":             round(rmse, 2),
        "R2":               round(r2, 4),
        "MAE_pct_of_mean":  round(mae_pct_of_mean, 2),
    }

    print(f"\n  ┌──────────────────────────────────┐")
    print(f"  │  MAE              : ₹{mae:>10,.2f} │")
    print(f"  │  RMSE             : ₹{rmse:>10,.2f} │")
    print(f"  │  R²               : {r2:>13.4f} │")
    print(f"  │  MAE % of mean    : {mae_pct_of_mean:>12.2f}% │")
    print(f"  └──────────────────────────────────┘")

    # Acceptance check
    r2_pass  = r2 >= 0.80
    mae_pass = mae_pct_of_mean < 15.0
    print(f"\n  ✅ R² ≥ 0.80         : {'PASS ✓' if r2_pass  else 'FAIL ✗'}")
    print(f"  ✅ MAE < 15% of mean : {'PASS ✓' if mae_pass else 'FAIL ✗'}")

    return metrics, y_pred


# ─────────────────────────────────────────────
# STEP D — CHARTS
# ─────────────────────────────────────────────
def plot_predicted_vs_actual(y_test, y_pred, r2: float) -> None:
    """Chart: Actual Profit vs Predicted Profit — closer to the 45° line = better."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#2E4057", label="Predictions")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color="#E63946", linewidth=2,
            linestyle="--", label="Perfect Prediction (45°)")
    ax.set_title(f"Actual vs Predicted Profit  (R² = {r2:.4f})")
    ax.set_xlabel("Actual Profit (₹)")
    ax.set_ylabel("Predicted Profit (₹)")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    _save_plot(fig, "09_predicted_vs_actual.png")


def plot_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Chart: Horizontal bar of top feature importances (Gini impurity reduction)."""
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(fi_df["Feature"], fi_df["Importance"],
                   color="#1B998B", edgecolor="white")

    # Add value labels
    for bar, val in zip(bars, fi_df["Importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_title("Feature Importances (Random Forest — Gini Impurity Reduction)")
    ax.set_xlabel("Importance Score")
    ax.set_xlim(0, fi_df["Importance"].max() * 1.15)
    _save_plot(fig, "10_feature_importance.png")

    return fi_df.sort_values("Importance", ascending=False)


def plot_residuals(y_test, y_pred) -> None:
    """Chart: Residuals (actual - predicted) — checks for bias."""
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual histogram
    sns.histplot(residuals, bins=50, kde=True, color="#2E4057",
                 edgecolor="white", ax=axes[0])
    axes[0].axvline(0, color="#E63946", linestyle="--")
    axes[0].set_title("Residuals Distribution")
    axes[0].set_xlabel("Residual (Actual − Predicted)")

    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.2, s=8, color="#048A81")
    axes[1].axhline(0, color="#E63946", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs Predicted Profit")
    axes[1].set_xlabel("Predicted Profit (₹)")
    axes[1].set_ylabel("Residual (₹)")

    plt.tight_layout()
    _save_plot(fig, "11_residuals.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "█"*55)
    print("  STEP 4 — MACHINE LEARNING PIPELINE")
    print("█"*55)

    # Load
    print(f"\n  Loading enriched data from: {ENRICHED_PATH}")
    df = pd.read_csv(ENRICHED_PATH)
    print(f"  ✔ Loaded {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    # Prepare
    X, y, encoders = prepare_features(df)
    y_mean = y.mean()

    # Train
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    # Evaluate
    metrics, y_pred = evaluate_model(model, X_test, y_test, y_mean)

    # Charts
    print("\n── Generating Model Charts ─────────────────────────────────")
    plot_predicted_vs_actual(y_test, y_pred, metrics["R2"])
    fi_df = plot_feature_importance(model, FEATURE_COLS)
    plot_residuals(y_test, y_pred)

    # ── Save Outputs ──────────────────────────────────────────
    os.makedirs("output", exist_ok=True)

    # Metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"\n  ✔ Metrics saved     → {METRICS_PATH}")

    # Feature importances
    fi_df.to_csv(FEATURE_IMP_PATH, index=False)
    print(f"  ✔ Importances saved → {FEATURE_IMP_PATH}")

    # Predictions (test set only — for Power BI dashboard)
    pred_df = df.loc[X_test.index, ["Item_Identifier", "Outlet_Identifier",
                                     "Item_Type", "Outlet_Type",
                                     "Item_Outlet_Sales", "Profit"]].copy()
    pred_df["Predicted_Profit"] = y_pred.values
    pred_df["Residual"]         = pred_df["Profit"] - pred_df["Predicted_Profit"]
    pred_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"  ✔ Predictions saved → {PREDICTIONS_PATH}")

    # Save trained model object (for use in predict_test.py and Streamlit)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✔ Trained model saved → {MODEL_PATH}")

    # Save categorical encoders
    ENCODERS_PATH = os.path.join("output", "encoders.pkl")
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    print(f"  ✔ Encoders saved      → {ENCODERS_PATH}")

    print("\n  Feature Importance (Top 5):")
    print(fi_df.head(5).to_string(index=False))

    print("\n  STEP 4 COMPLETE ✓\n")
    return model, metrics, fi_df


if __name__ == "__main__":
    main()
