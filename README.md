# 🧠 AI-Based Cost and Profit Optimization System for Retail Outlets

> **Integrating Cost Accounting with Machine Learning to Drive Retail Profitability**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi)](https://powerbi.microsoft.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Cost Model Explained](#-cost-model-explained)
- [Output Artifacts](#-output-artifacts)
- [Model Performance](#-model-performance)
- [Power BI Dashboard](#-power-bi-dashboard)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

This project builds an **end-to-end intelligent system** that applies machine learning to predict and optimize retail profitability. The foundation is the **Big Mart Sales Dataset**, which is augmented with engineered cost features — **material cost**, **labor cost**, and **overhead cost** — based on realistic business assumptions drawn from cost accounting principles.

Since the original dataset lacks any cost or profit data, a **structured cost model** is designed from scratch using industry benchmarks for FMCG retail. This enables the simulation of a complete financial ledger on top of the sales data.

The pipeline flows from **raw data → cost-enriched dataset → EDA → ML prediction → Power BI dashboard**, delivering business-ready insights into cost behavior and profit optimization.

### Why This Matters

| Challenge | This System's Answer |
|-----------|---------------------|
| No cost data in raw retail datasets | Engineer costs from business assumptions |
| Inability to predict profitability per SKU | Train a regression model on financial features |
| No visibility into loss-making segments | Automated loss detection and flagging |
| Data inaccessible to business users | Power BI dashboard with slicers and KPIs |

---

## ✨ Key Features

- 🏗️ **Cost Engineering** — Material, labor, and overhead costs derived from sales using industry-realistic rates.
- 💰 **Profit Calculation** — Per-record profit and profit margin computed from engineered cost components.
- 📉 **Loss Detection** — Automatic flagging of loss-making transactions (`Is_Loss` field).
- 📊 **Exploratory Data Analysis** — Distribution plots, heatmaps, and segment-level summaries.
- 🤖 **ML Profit Prediction** — Random Forest Regressor trained to predict profit from item and outlet features.
- 🏆 **Feature Importance** — Ranked list of features driving profit variability.
- 📈 **Power BI Dashboard** — 4-page interactive report covering overview, outlet performance, product analysis, and model evaluation.
- 📤 **Export Pipeline** — All outputs (enriched CSV, metrics, plots) saved to `output/`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Data Processing | Python 3.8+, Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest) |
| Visualization (Python) | Matplotlib, Seaborn |
| Visualization (Business) | Microsoft Power BI Desktop |
| Data Prep / Reporting | Microsoft Excel |
| Environment | Jupyter Notebook / `.py` scripts |

---

## 📁 Project Structure

```
Cost_Accounting/
│
├── data/
│   ├── raw/
│   │   └── bigmart_train.csv          # Original Big Mart dataset
│   └── processed/
│       └── bigmart_processed.csv      # Cleaned dataset (after imputation)
│
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
│   ├── 02_Cost_Engineering.ipynb      # Cost feature derivation
│   └── 03_ML_Pipeline.ipynb          # Model training & evaluation
│
├── src/
│   ├── data_loader.py                 # Load & validate raw dataset
│   ├── feature_engineering.py         # Cost features & profit derivation
│   ├── eda.py                         # EDA plots and summaries
│   ├── model.py                       # ML pipeline (encode → train → evaluate)
│   └── reporting.py                   # Export outputs to CSV and Excel
│
├── output/
│   ├── enriched_bigmart.csv           # Final enriched dataset with costs & profit
│   ├── model_metrics.csv              # MAE, RMSE, R² results
│   ├── feature_importance.csv         # Ranked feature importances
│   └── plots/                         # All EDA and model evaluation charts
│       ├── profit_distribution.png
│       ├── cost_breakdown_by_outlet.png
│       ├── correlation_heatmap.png
│       ├── predicted_vs_actual.png
│       └── feature_importance.png
│
├── dashboard/
│   └── retail_profit_dashboard.pbix   # Power BI dashboard file
│
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── PRD.md                             # Product Requirements Document
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Microsoft Power BI Desktop (for dashboard)
- The Big Mart Sales dataset (`bigmart_train.csv`)

### 1. Clone / Download the Project

```bash
git clone https://github.com/your-username/Cost_Accounting.git
cd Cost_Accounting
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the Dataset

Download the Big Mart dataset and place it at:

```
data/raw/bigmart_train.csv
```

> **Source**: [Kaggle — Big Mart Sales Prediction](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)

### 5. Run the Full Pipeline

```bash
python src/data_loader.py
python src/feature_engineering.py
python src/eda.py
python src/model.py
python src/reporting.py
```

Or run the Jupyter notebooks in order inside `notebooks/`.

---

## 📖 Usage Guide

### Step 1 — Data Loading & Cleaning

```python
# src/data_loader.py
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load and perform initial cleaning on the Big Mart dataset."""
    df = pd.read_csv(path)
    
    # Impute Item_Weight with mean
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    
    # Impute Outlet_Size with mode per Outlet_Type
    df['Outlet_Size'] = df.groupby('Outlet_Type')['Outlet_Size'].transform(
        lambda x: x.fillna(x.mode()[0])
    )
    
    # Harmonize fat content labels
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
        'LF': 'Low Fat', 'low fat': 'Low Fat',
        'reg': 'Regular'
    })
    
    return df
```

### Step 2 — Cost Feature Engineering

```python
# src/feature_engineering.py
import numpy as np

OVERHEAD_RATES = {
    'Grocery Store':       0.05,
    'Supermarket Type1':   0.08,
    'Supermarket Type2':   0.10,
    'Supermarket Type3':   0.12,
}

def engineer_costs(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Derive cost components and profit from sales data."""
    rng = np.random.default_rng(seed)
    n = len(df)
    
    # Variable costs (as % of sales)
    df['Material_Cost'] = df['Item_Outlet_Sales'] * rng.uniform(0.40, 0.70, n)
    df['Labor_Cost']    = df['Item_Outlet_Sales'] * rng.uniform(0.10, 0.25, n)
    
    # Overhead (fixed rate by outlet type + noise)
    base_rate = df['Outlet_Type'].map(OVERHEAD_RATES)
    noise     = rng.uniform(-0.01, 0.01, n)
    df['Overhead_Cost'] = df['Item_Outlet_Sales'] * (base_rate + noise)
    
    # Aggregate
    df['Total_Cost']        = df['Material_Cost'] + df['Labor_Cost'] + df['Overhead_Cost']
    df['Profit']            = df['Item_Outlet_Sales'] - df['Total_Cost']
    df['Profit_Margin_Pct'] = (df['Profit'] / df['Item_Outlet_Sales']) * 100
    df['Is_Loss']           = df['Profit'] < 0
    
    # Outlet age
    df['Outlet_Age'] = 2026 - df['Outlet_Establishment_Year']
    
    return df
```

### Step 3 — Exploratory Data Analysis

```python
# src/eda.py  (abbreviated — see full file for all plots)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_profit_distribution(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Profit'], bins=50, kde=True, color='#048A81')
    plt.title('Profit Distribution Across All Transactions')
    plt.xlabel('Profit')
    plt.savefig('output/plots/profit_distribution.png', dpi=150)
    plt.close()

def plot_cost_by_outlet(df):
    cost_cols = ['Material_Cost', 'Labor_Cost', 'Overhead_Cost']
    summary = df.groupby('Outlet_Type')[cost_cols].mean()
    summary.plot(kind='bar', figsize=(12, 6), colormap='viridis')
    plt.title('Average Cost Components by Outlet Type')
    plt.savefig('output/plots/cost_breakdown_by_outlet.png', dpi=150)
    plt.close()
```

### Step 4 — Machine Learning Pipeline

```python
# src/model.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

FEATURE_COLS = [
    'Item_MRP', 'Item_Outlet_Sales', 'Item_Visibility',
    'Outlet_Age', 'Item_Weight',
    'Item_Fat_Content', 'Item_Type',
    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
]
TARGET = 'Profit'

def train_model(df):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET]
    
    # Encode categoricals
    cat_cols = X.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")
    
    return rf, X_test, y_test, y_pred
```

---

## 💡 Cost Model Explained

The cost model is the core innovation of this project. Since no actual cost data exists, costs are **simulated using business logic** anchored to industry benchmarks.

```
┌──────────────────────────────────────────────────────┐
│            COST BREAKDOWN MODEL                      │
├─────────────────┬────────────────────────────────────┤
│ Material Cost   │ 40% – 70% of Sales (randomized)   │
│ Labor Cost      │ 10% – 25% of Sales (randomized)   │
│ Overhead Cost   │ Fixed % by Outlet Type + noise     │
├─────────────────┴────────────────────────────────────┤
│ Total Cost      = Material + Labor + Overhead        │
│ Profit          = Sales − Total Cost                 │
│ Profit Margin % = (Profit / Sales) × 100             │
└──────────────────────────────────────────────────────┘
```

### Overhead Rate by Outlet Type

| Outlet Type | Overhead Rate | Rationale |
|-------------|---------------|-----------|
| Grocery Store | 5% | Lowest overhead — simple operations |
| Supermarket Type1 | 8% | Moderate scale with more staff |
| Supermarket Type2 | 10% | Larger format with higher fixed costs |
| Supermarket Type3 | 12% | Highest scale — most overhead burden |

---

## 📤 Output Artifacts

| File | Description |
|------|-------------|
| `output/enriched_bigmart.csv` | Full dataset with all cost columns, profit, margin, loss flag |
| `output/model_metrics.csv` | MAE, RMSE, R² on test set |
| `output/feature_importance.csv` | Feature importance scores from Random Forest |
| `output/plots/profit_distribution.png` | Histogram of profit across all records |
| `output/plots/cost_breakdown_by_outlet.png` | Cost components grouped by outlet type |
| `output/plots/correlation_heatmap.png` | Pearson correlation matrix of numeric features |
| `output/plots/predicted_vs_actual.png` | Scatter plot of model predictions vs. actual profit |
| `output/plots/feature_importance.png` | Horizontal bar chart of top feature importances |

---

## 📊 Model Performance

Target thresholds for the Random Forest Regressor:

| Metric | Target |
|--------|--------|
| R² | ≥ 0.80 |
| MAE | < 15% of mean profit |
| RMSE | Reported (no hard threshold) |

> **Reproducibility Note**: All random states are fixed to `42`. Re-running any script will produce identical results.

### Potential Improvements

If baseline targets are not met, consider:

1. **Hyperparameter Tuning** — `GridSearchCV` on `n_estimators`, `max_depth`, `min_samples_leaf`
2. **Gradient Boosting** — Replace RF with `XGBRegressor` or `GradientBoostingRegressor`
3. **Additional Features** — Interaction terms (MRP × Outlet_Type), price bins

---

## 📈 Power BI Dashboard

The dashboard file `dashboard/retail_profit_dashboard.pbix` contains **4 interactive pages**:

| Page | Contents |
|------|----------|
| **1. Executive Overview** | KPI cards (Sales, Cost, Profit, Margin), Profit by Outlet, Cost breakdown donut |
| **2. Outlet Performance** | Bar chart by outlet ID, summary table, outlet size filter |
| **3. Product Analysis** | Top/bottom 10 items, MRP vs. Profit scatter, loss SKU count |
| **4. Predicted vs. Actual** | Scatter plot with 45° reference line, model metrics table, feature importances |

### Setup Instructions

1. Open **Power BI Desktop**.
2. Go to **Home → Get Data → Text/CSV**.
3. Load `output/enriched_bigmart.csv`.
4. Load `output/model_metrics.csv`.
5. Load `output/feature_importance.csv`.
6. Open `dashboard/retail_profit_dashboard.pbix` or rebuild visuals from scratch.
7. Refresh data and publish to Power BI Service if needed.

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m "Add: description of change"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a Pull Request.

### Code Standards

- Follow **PEP 8** for Python code style.
- Add **docstrings** to all functions.
- Keep notebooks clean — clear output before committing.
- Avoid committing large data files; use `.gitignore`.

### `.gitignore` Recommendation

```
data/raw/
output/
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.pbix
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Big Mart Dataset** — Originally from the Analytics Vidhya Big Mart Sales Prediction competition.
- **Scikit-learn** — For providing the ML toolkit used in this pipeline.
- **Microsoft Power BI** — For the business intelligence dashboard platform.
- **FMCG Cost Benchmarks** — Industry data used to calibrate the cost model assumptions.

---

> *"Accounting tells you where you've been. Machine Learning tells you where you're going."*

---

**Built with ❤️ | AI-Based Cost & Profit Optimization System | 2026**
