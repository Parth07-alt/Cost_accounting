"""
eda.py
======
Step 3 of the AI-Based Cost & Profit Optimization Pipeline.

Responsibilities:
    - Generate & save all Exploratory Data Analysis charts:
        1. Profit Distribution (histogram + KDE)
        2. Cost Components by Outlet Type (stacked bar)
        3. Sales vs Profit scatter (coloured by Outlet_Type)
        4. Correlation Heatmap (numeric features)
        5. Profit Margin Box-plot by Outlet_Type
        6. Top-10 and Bottom-10 Item Types by average profit
        7. Loss vs Profitable records (pie chart)
        8. Outlet Age vs Profit (scatter)
    - Print segment-level summary tables.

Usage:
    python src/eda.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # non-interactive backend (safe for all environments)

# ─────────────────────────────────────────────
# STYLE CONFIG
# ─────────────────────────────────────────────
PALETTE_COST    = ["#2E4057", "#1B998B", "#E63946"]   # blue, teal, red
PALETTE_OUTLET  = sns.color_palette("Set2", 4)
PROFIT_COLOR    = "#048A81"
LOSS_COLOR      = "#E63946"
BG_COLOR        = "#F8F9FA"
GRID_COLOR      = "#DEE2E6"

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    BG_COLOR,
    "axes.edgecolor":    GRID_COLOR,
    "axes.grid":         True,
    "grid.color":        GRID_COLOR,
    "grid.linestyle":    "--",
    "grid.alpha":        0.7,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
})

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ENRICHED_PATH = os.path.join("output", "enriched_bigmart.csv")
PLOTS_DIR     = os.path.join("output", "plots")


def _save(fig, filename: str) -> None:
    """Save a matplotlib figure to the plots directory."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Saved → {path}")


# ─────────────────────────────────────────────
# CHART FUNCTIONS
# ─────────────────────────────────────────────

def plot_profit_distribution(df: pd.DataFrame) -> None:
    """
    Chart 1: Histogram + KDE of Profit.
    Shows the overall shape — are most transactions profitable?
    Where does the distribution center?
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(df["Profit"], bins=60, kde=True, color=PROFIT_COLOR,
                 edgecolor="white", linewidth=0.4, ax=ax)
    ax.axvline(0, color=LOSS_COLOR, linestyle="--", linewidth=1.5,
               label="Break-even (Profit = 0)")
    ax.axvline(df["Profit"].mean(), color="#F4A261", linestyle="-.",
               linewidth=1.5, label=f"Mean Profit = ₹{df['Profit'].mean():,.0f}")
    ax.set_title("Profit Distribution Across All Transactions")
    ax.set_xlabel("Profit (₹)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x:,.0f}"))
    _save(fig, "01_profit_distribution.png")


def plot_cost_breakdown_by_outlet(df: pd.DataFrame) -> None:
    """
    Chart 2: Stacked bar showing average cost components per Outlet_Type.
    Reveals which outlet type is the most / least cost-efficient.
    """
    cost_cols = ["Material_Cost", "Labor_Cost", "Overhead_Cost"]
    summary = df.groupby("Outlet_Type")[cost_cols].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    summary.plot(kind="bar", stacked=True, ax=ax,
                 color=PALETTE_COST, edgecolor="white", linewidth=0.5)
    ax.set_title("Average Cost Components by Outlet Type")
    ax.set_xlabel("Outlet Type")
    ax.set_ylabel("Average Cost (₹)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(["Material Cost", "Labor Cost", "Overhead Cost"],
              loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x:,.0f}"))
    _save(fig, "02_cost_breakdown_by_outlet.png")


def plot_sales_vs_profit(df: pd.DataFrame) -> None:
    """
    Chart 3: Scatter plot — Item_Outlet_Sales vs Profit, coloured by Outlet_Type.
    Confirms the linear relationship between sales and profit,
    and shows how spread varies by outlet format.
    """
    outlet_types = df["Outlet_Type"].unique()
    colors = dict(zip(outlet_types, PALETTE_OUTLET))

    fig, ax = plt.subplots(figsize=(12, 6))
    for ot in outlet_types:
        subset = df[df["Outlet_Type"] == ot]
        ax.scatter(subset["Item_Outlet_Sales"], subset["Profit"],
                   alpha=0.35, s=8, label=ot, color=colors[ot])
    ax.axhline(0, color=LOSS_COLOR, linestyle="--", linewidth=1,
               label="Break-even")
    ax.set_title("Sales vs Profit  (coloured by Outlet Type)")
    ax.set_xlabel("Item Outlet Sales (₹)")
    ax.set_ylabel("Profit (₹)")
    ax.legend(markerscale=3, fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x:,.0f}"))
    _save(fig, "03_sales_vs_profit_scatter.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Chart 4: Pearson correlation heatmap of all numeric features.
    Helps identify multicollinearity and feature relationships.
    """
    num_cols = [
        "Item_Weight", "Item_Visibility", "Item_MRP",
        "Item_Outlet_Sales", "Outlet_Age",
        "Material_Cost", "Labor_Cost", "Overhead_Cost",
        "Total_Cost", "Profit", "Profit_Margin_Pct"
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # show lower triangle only
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, square=True,
                linewidths=0.5, linecolor=GRID_COLOR,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Pearson Correlation Heatmap — All Numeric Features")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    _save(fig, "04_correlation_heatmap.png")


def plot_profit_margin_boxplot(df: pd.DataFrame) -> None:
    """
    Chart 5: Box-plot of Profit_Margin_Pct by Outlet_Type.
    Shows median, spread, and outliers in profit margins per outlet format.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="Outlet_Type", y="Profit_Margin_Pct",
                palette="Set2", ax=ax,
                flierprops=dict(marker="o", markersize=2, alpha=0.3))
    ax.axhline(0, color=LOSS_COLOR, linestyle="--", linewidth=1,
               label="Break-even (0%)")
    ax.set_title("Profit Margin (%) Distribution by Outlet Type")
    ax.set_xlabel("Outlet Type")
    ax.set_ylabel("Profit Margin (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend()
    _save(fig, "05_profit_margin_boxplot.png")


def plot_top_bottom_items(df: pd.DataFrame) -> None:
    """
    Chart 6: Horizontal bar — Top-10 and Bottom-10 Item Types by avg profit.
    Identifies which product categories are most / least profitable.
    """
    item_profit = (df.groupby("Item_Type")["Profit"]
                   .mean()
                   .sort_values(ascending=False))
    top10    = item_profit.head(10)
    bottom10 = item_profit.tail(10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top 10
    axes[0].barh(top10.index[::-1], top10.values[::-1],
                 color=PROFIT_COLOR, edgecolor="white")
    axes[0].set_title("Top 10 Item Types by Avg Profit")
    axes[0].set_xlabel("Average Profit (₹)")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x:,.0f}"))

    # Bottom 10
    axes[1].barh(bottom10.index, bottom10.values,
                 color=LOSS_COLOR, edgecolor="white")
    axes[1].set_title("Bottom 10 Item Types by Avg Profit")
    axes[1].set_xlabel("Average Profit (₹)")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x:,.0f}"))

    plt.suptitle("Item Type Profitability Analysis", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "06_top_bottom_item_types.png")


def plot_loss_pie(df: pd.DataFrame) -> None:
    """
    Chart 7: Pie chart — Profitable vs Loss-making records.
    Quick visual check on the proportion of loss-making transactions.
    """
    counts = df["Is_Loss"].value_counts()
    
    # Handle case where there are no loss rows
    if len(counts) == 1:
        val = counts.index[0]
        labels = ["Profitable"] if val == False else ["Loss-Making"]
        colors = [PROFIT_COLOR] if val == False else [LOSS_COLOR]
    else:
        # Match indices to labels dynamically just to be safe
        labels = ["Profitable" if not k else "Loss-Making" for k in counts.index]
        colors = [PROFIT_COLOR if not k else LOSS_COLOR for k in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Proportion of Loss-Making vs Profitable Transactions",
                 pad=20)
    _save(fig, "07_profitable_vs_loss_pie.png")


def plot_outlet_age_vs_profit(df: pd.DataFrame) -> None:
    """
    Chart 8: Scatter + regression line — Outlet_Age vs Profit.
    Tests whether older outlets tend to be more profitable.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.regplot(data=df, x="Outlet_Age", y="Profit",
                scatter_kws={"alpha": 0.2, "s": 8, "color": PROFIT_COLOR},
                line_kws={"color": "#E63946", "linewidth": 2},
                ax=ax)
    ax.set_title("Outlet Age vs Profit  (with Regression Trend Line)")
    ax.set_xlabel("Outlet Age (years)")
    ax.set_ylabel("Profit (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x:,.0f}"))
    _save(fig, "08_outlet_age_vs_profit.png")


# ─────────────────────────────────────────────
# SUMMARY TABLES
# ─────────────────────────────────────────────

def print_segment_summaries(df: pd.DataFrame) -> None:
    """Print outlet-wise and item-type-wise profit summaries."""

    print("\n── Outlet Type Summary ────────────────────────────────────────")
    outlet_summary = df.groupby("Outlet_Type").agg(
        Records       = ("Profit", "count"),
        Avg_Sales     = ("Item_Outlet_Sales", "mean"),
        Avg_TotalCost = ("Total_Cost", "mean"),
        Avg_Profit    = ("Profit", "mean"),
        Avg_Margin_Pct= ("Profit_Margin_Pct", "mean"),
        Loss_Count    = ("Is_Loss", "sum"),
    ).round(2)
    print(outlet_summary.to_string())

    print("\n── Outlet Location Summary ────────────────────────────────────")
    loc_summary = df.groupby("Outlet_Location_Type").agg(
        Records       = ("Profit", "count"),
        Avg_Profit    = ("Profit", "mean"),
        Avg_Margin_Pct= ("Profit_Margin_Pct", "mean"),
        Loss_Count    = ("Is_Loss", "sum"),
    ).round(2)
    print(loc_summary.to_string())

    print("\n── Profit Bucket Summary ──────────────────────────────────────")
    bucket_summary = df.groupby("Profit_Bucket").agg(
        Records    = ("Profit", "count"),
        Avg_Profit = ("Profit", "mean"),
        Avg_Sales  = ("Item_Outlet_Sales", "mean"),
    ).round(2)
    print(bucket_summary.to_string())
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(df: pd.DataFrame = None) -> None:
    print("\n" + "█"*55)
    print("  STEP 3 — EXPLORATORY DATA ANALYSIS")
    print("█"*55)

    if df is None:
        print(f"\n  Loading enriched data from: {ENRICHED_PATH}")
        df = pd.read_csv(ENRICHED_PATH)
        print(f"  ✔ Loaded {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    print(f"\n  Generating 8 charts → {PLOTS_DIR}/\n")

    plot_profit_distribution(df)
    plot_cost_breakdown_by_outlet(df)
    plot_sales_vs_profit(df)
    plot_correlation_heatmap(df)
    plot_profit_margin_boxplot(df)
    plot_top_bottom_items(df)
    plot_loss_pie(df)
    plot_outlet_age_vs_profit(df)

    print_segment_summaries(df)

    print("  STEP 3 COMPLETE ✓\n")


if __name__ == "__main__":
    main()
