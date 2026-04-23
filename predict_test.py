"""
predict_test.py
===============
Apply the trained model to the Big Mart test set (Test_u94Q5KV.csv).

The test set has NO sales, cost, or profit columns.
This script:
    Step A — Cleans the test set (same transformations as train).
    Step B — Estimates Item_Outlet_Sales using Item_MRP as a proxy
             (since the model needs it as a feature and test.csv lacks it).
    Step C — Derives cost estimates using the same formula assumptions.
    Step D — Loads the trained model and generates profit predictions.
    Step E — Saves the prediction results to output/test_predictions.csv.

Usage:
    python predict_test.py

NOTE ON DESIGN DECISION:
    Test.csv is the original Big Mart COMPETITION test set — it was
    released WITHOUT sales or profit information (that is the label
    the competition participants had to predict).

    Since our project extends Big Mart with cost accounting, we must
    estimate missing values to apply our pipeline. We use:
      - Item_MRP as a proxy for estimating Sales (MRP × discount factor)
      - Then apply the same cost formula assumptions as the training set.
    This gives us a complete test set for end-to-end evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
TEST_RAW_PATH       = os.path.join("big mart dataset", "Test_u94Q5KV.csv")
TRAIN_ENRICHED_PATH = os.path.join("output", "enriched_bigmart.csv")
MODEL_PATH          = os.path.join("output", "trained_model.pkl")
OUTPUT_PATH         = os.path.join("output", "test_predictions.csv")
CURRENT_YEAR        = 2026

# These must match EXACTLY the features used when training the model
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

# Cost model assumptions (same as training set)
OVERHEAD_RATES = {
    "Grocery Store":      0.05,
    "Supermarket Type1":  0.08,
    "Supermarket Type2":  0.10,
    "Supermarket Type3":  0.12,
}
MATERIAL_COST_RATE = 0.55   # midpoint of 0.40–0.70
LABOR_COST_RATE    = 0.175  # midpoint of 0.10–0.25


def clean_test(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning as training data."""
    df = df.copy()
    df["Item_Weight"]     = df["Item_Weight"].fillna(df["Item_Weight"].mean())
    df["Outlet_Size"]     = df.groupby("Outlet_Type")["Outlet_Size"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Small")
    )
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({
        "LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"
    })
    df["Outlet_Age"] = CURRENT_YEAR - df["Outlet_Establishment_Year"]
    return df


def estimate_sales(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate Item_Outlet_Sales for the test set.

    Strategy: Use the average Sales/MRP ratio from the training set,
    segmented by Outlet_Type. This gives us a realistic sales estimate
    per outlet type without using any future/leaked data.

    Parameters
    ----------
    df       : test dataframe
    train_df : enriched training dataframe (has real sales)
    """
    # Compute avg (Sales / MRP) ratio per Outlet_Type from training data
    ratio_by_outlet = (
        train_df.groupby("Outlet_Type")
        .apply(lambda g: (g["Item_Outlet_Sales"] / g["Item_MRP"]).mean())
    )
    print("  Sales/MRP ratio by Outlet_Type (from training data):")
    for ot, ratio in ratio_by_outlet.items():
        print(f"    {ot:25s} : {ratio:.4f}x MRP")

    outlet_ratio = df["Outlet_Type"].map(ratio_by_outlet)
    # Fallback for any unseen outlet types
    global_ratio = train_df["Item_Outlet_Sales"].sum() / train_df["Item_MRP"].sum()
    outlet_ratio = outlet_ratio.fillna(global_ratio)

    df["Item_Outlet_Sales"] = (df["Item_MRP"] * outlet_ratio).round(4)
    print(f"\n  Estimated Sales | mean: Rs {df['Item_Outlet_Sales'].mean():,.0f} "
          f"| range: Rs {df['Item_Outlet_Sales'].min():,.0f} – "
          f"Rs {df['Item_Outlet_Sales'].max():,.0f}")
    return df


def estimate_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic cost formula (midpoint rates, no random noise).
    Using midpoints ensures reproducible, stable test predictions.
    """
    df = df.copy()
    df["Material_Cost"] = df["Item_Outlet_Sales"] * MATERIAL_COST_RATE
    df["Labor_Cost"]    = df["Item_Outlet_Sales"] * LABOR_COST_RATE
    overhead_rate       = df["Outlet_Type"].map(OVERHEAD_RATES).fillna(0.08)
    df["Overhead_Cost"] = df["Item_Outlet_Sales"] * overhead_rate
    df["Total_Cost"]    = df["Material_Cost"] + df["Labor_Cost"] + df["Overhead_Cost"]
    df["Profit"]        = df["Item_Outlet_Sales"] - df["Total_Cost"]
    df["Profit_Margin_Pct"] = (df["Profit"] / df["Item_Outlet_Sales"] * 100).round(2)
    print(f"  Estimated Profit | mean: Rs {df['Profit'].mean():,.0f} "
          f"| avg margin: {df['Profit_Margin_Pct'].mean():.1f}%")
    return df


def encode_features(test_df: pd.DataFrame,
                    train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode categorical columns using the SAME mapping as training.
    We fit encoders on the FULL training data so they know all categories.

    IMPORTANT: Never fit encoders on test data — that leaks test info.
    """
    cat_cols = ["Item_Fat_Content", "Item_Type", "Outlet_Identifier",
                "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"]

    X_test = test_df[FEATURE_COLS].copy()
    X_train_ref = train_df[FEATURE_COLS].copy()

    for col in cat_cols:
        le = LabelEncoder()
        le.fit(X_train_ref[col].astype(str))   # fit on TRAIN data only

        # Handle unseen categories in test (assign -1)
        def safe_transform(val):
            val = str(val)
            if val in le.classes_:
                return le.transform([val])[0]
            return -1

        X_test[col] = X_test[col].astype(str).apply(safe_transform)

    return X_test


def main():
    print("\n" + "="*58)
    print("  APPLYING MODEL TO TEST SET (Test_u94Q5KV.csv)")
    print("="*58)

    # ── Load data ─────────────────────────────────────────────
    print(f"\n  Loading test data  : {TEST_RAW_PATH}")
    test_df  = pd.read_csv(TEST_RAW_PATH)
    print(f"  Loaded : {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")

    print(f"\n  Loading train data : {TRAIN_ENRICHED_PATH}")
    train_df = pd.read_csv(TRAIN_ENRICHED_PATH)
    print(f"  Loaded : {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")

    # ── Step A: Clean ──────────────────────────────────────────
    print("\n[A] Cleaning test set...")
    test_df = clean_test(test_df)
    print(f"  Nulls remaining: {test_df.isnull().sum().sum()}")

    # ── Step B: Estimate Sales ─────────────────────────────────
    print("\n[B] Estimating Item_Outlet_Sales from MRP ratios...")
    test_df = estimate_sales(test_df, train_df)

    # ── Step C: Estimate Costs ─────────────────────────────────
    print("\n[C] Estimating costs using formula assumptions...")
    test_df = estimate_costs(test_df)

    # ── Step D: Load model & predict ──────────────────────────
    print(f"\n[D] Loading trained model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("  Model file not found! Re-training on train data...")
        # Re-train quickly
        from src.model import prepare_features, train_model
        X_train, y_train, _ = prepare_features(train_df)
        model, *_ = train_model(X_train, y_train)
    else:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("  Model loaded successfully.")

    print("\n[E] Encoding test features...")
    X_test_encoded = encode_features(test_df, train_df)
    print(f"  Feature matrix shape: {X_test_encoded.shape}")

    print("\n[F] Generating predictions...")
    predicted_profit = model.predict(X_test_encoded)

    # ── Step E: Save results ───────────────────────────────────
    result_df = test_df[["Item_Identifier", "Outlet_Identifier",
                          "Item_Type", "Outlet_Type", "Item_MRP",
                          "Item_Outlet_Sales", "Total_Cost",
                          "Profit_Margin_Pct"]].copy()
    result_df["Estimated_Profit"]   = test_df["Profit"].values
    result_df["Predicted_Profit"]   = predicted_profit.round(2)
    result_df["Prediction_vs_Est"]  = (
        result_df["Predicted_Profit"] - result_df["Estimated_Profit"]
    ).round(2)
    result_df["Is_Loss_Predicted"]  = result_df["Predicted_Profit"] < 0

    os.makedirs("output", exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n  Prediction Results:")
    print(f"  Records predicted         : {len(result_df):,}")
    print(f"  Avg Predicted Profit      : Rs {result_df['Predicted_Profit'].mean():,.0f}")
    print(f"  Avg Estimated Profit      : Rs {result_df['Estimated_Profit'].mean():,.0f}")
    print(f"  Predicted Loss records    : {result_df['Is_Loss_Predicted'].sum():,}")
    print(f"\n  Results saved -> {OUTPUT_PATH}")

    print("\n" + "="*58)
    print("  TEST PREDICTION COMPLETE")
    print("="*58 + "\n")

    return result_df


if __name__ == "__main__":
    main()
