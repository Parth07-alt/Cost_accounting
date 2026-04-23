# Product Requirements Document (PRD)

## AI-Based Cost and Profit Optimization System for Retail Outlets using Machine Learning

---

> **Version**: 1.0.0  
> **Date**: April 2026  
> **Status**: Active Development  
> **Document Owner**: Project Team  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Scope](#4-scope)
5. [Stakeholders](#5-stakeholders)
6. [Dataset Overview](#6-dataset-overview)
7. [Cost Modeling Design](#7-cost-modeling-design)
8. [Functional Requirements](#8-functional-requirements)
9. [Non-Functional Requirements](#9-non-functional-requirements)
10. [System Architecture](#10-system-architecture)
11. [Data Pipeline](#11-data-pipeline)
12. [Machine Learning Requirements](#12-machine-learning-requirements)
13. [Dashboard Requirements](#13-dashboard-requirements)
14. [Assumptions & Constraints](#14-assumptions--constraints)
15. [Acceptance Criteria](#15-acceptance-criteria)
16. [Risks & Mitigations](#16-risks--mitigations)
17. [Timeline & Milestones](#17-timeline--milestones)
18. [Glossary](#18-glossary)

---

## 1. Executive Summary

This project delivers an **intelligent cost and profit optimization system** for retail outlets by integrating cost accounting principles with machine learning. Built on the Big Mart Sales dataset, the system engineers realistic financial features (material cost, labor cost, overhead cost) and trains a regression model to predict profit at the transaction level.

The outcome is a decision-support system that surfaces actionable insights — identifying high-cost segments, detecting loss-making scenarios, and enabling better budgeting — via a Power BI dashboard intended for retail managers and business analysts.

---

## 2. Problem Statement

Retail businesses often operate without granular, per-SKU cost visibility. Managers lack the tools to:

- Attribute costs (material, labor, overhead) at the product or outlet level.
- Predict profitability before stocking or pricing decisions.
- Detect underperforming segments before losses compound.

The Big Mart dataset provides rich sales and product metadata but **no cost or profit fields**, making it an ideal candidate for financial feature engineering combined with machine learning.

---

## 3. Project Objectives

| # | Objective | Priority |
|---|-----------|----------|
| O1 | Engineer cost features (material, labor, overhead) from raw sales data | Must Have |
| O2 | Calculate per-row profit as `Sales - Total Cost` | Must Have |
| O3 | Train a regression model to predict profit from product & outlet features | Must Have |
| O4 | Evaluate model performance using MAE and R² | Must Have |
| O5 | Identify high-cost, low-profit, and loss-making segments | Must Have |
| O6 | Build a Power BI dashboard for business insights | Must Have |
| O7 | Perform Exploratory Data Analysis (EDA) | Should Have |
| O8 | Export enriched dataset for Excel-based reporting | Should Have |
| O9 | Provide segment-level recommendations for cost reduction | Nice to Have |

---

## 4. Scope

### In Scope

- **Data Engineering**: Loading, cleaning, and enriching the Big Mart dataset.
- **Cost Modeling**: Rule-based financial feature engineering using business assumptions.
- **Profit Calculation**: Deriving profit and profit margin per record.
- **EDA**: Correlation analysis, distribution plots, outlet-wise and category-wise breakdowns.
- **ML Model**: Regression pipeline (feature encoding + model training + evaluation).
- **Visualization**: Power BI dashboard with KPIs, charts, and filters.
- **Output Files**: Enriched CSV/Excel for downstream use.

### Out of Scope

- Real-time data ingestion or live POS integration.
- Multi-user authentication or role-based access control.
- Cloud deployment or API endpoints.
- Inventory management or supply chain optimization.
- Time-series forecasting of future sales.

---

## 5. Stakeholders

| Role | Name / Team | Interest |
|------|-------------|----------|
| Project Sponsor | Academic Supervisor | Approves scope and deliverables |
| Developer / Data Scientist | Project Team | Builds and tests the system |
| Business Analyst | Project Team | Interprets insights and dashboard |
| End Users (Simulated) | Retail Managers | Consumes dashboard and recommendations |

---

## 6. Dataset Overview

### Source
**Big Mart Sales Dataset** — a publicly available retail dataset widely used in machine learning competitions.

### Key Columns

| Column | Description | Type |
|--------|-------------|------|
| `Item_Identifier` | Unique product code | Categorical |
| `Item_Weight` | Weight of the product | Numeric |
| `Item_Fat_Content` | Low Fat / Regular | Categorical |
| `Item_Visibility` | % of display area allocated | Numeric |
| `Item_Type` | Product category (e.g., Dairy, Snack) | Categorical |
| `Item_MRP` | Maximum Retail Price | Numeric |
| `Outlet_Identifier` | Unique outlet code | Categorical |
| `Outlet_Establishment_Year` | Year outlet was established | Numeric |
| `Outlet_Size` | Small / Medium / High | Categorical |
| `Outlet_Location_Type` | Tier 1 / Tier 2 / Tier 3 | Categorical |
| `Outlet_Type` | Grocery Store / Supermarket Type | Categorical |
| `Item_Outlet_Sales` | Actual sales value (target proxy) | Numeric |

### Dataset Size
- **Training Set**: ~8,523 records
- **Test Set**: ~5,681 records

---

## 7. Cost Modeling Design

Since the original dataset has no cost fields, cost components are engineered using rule-based business logic.

### 7.1 Material Cost

> **Material Cost = Item_Outlet_Sales × Material Cost Rate**

- Rate sampled uniformly from **[0.40, 0.70]** for each record.
- Rationale: Material cost in FMCG retail typically ranges 40%–70% of revenue.

### 7.2 Labor Cost

> **Labor Cost = Item_Outlet_Sales × Labor Cost Rate**

- Rate sampled uniformly from **[0.10, 0.25]** for each record.
- Rationale: Labor cost in retail typically ranges 10%–25% of revenue.

### 7.3 Overhead Cost

> **Overhead Cost = Base Overhead × Sales Scaling Factor**

Overhead base rates per outlet type:

| Outlet Type | Base Overhead Rate |
|-------------|-------------------|
| Grocery Store | 5% of sales |
| Supermarket Type1 | 8% of sales |
| Supermarket Type2 | 10% of sales |
| Supermarket Type3 | 12% of sales |

- A small random noise term `ε ~ Uniform(-0.01, +0.01)` is added to simulate real-world variability.

### 7.4 Total Cost & Profit

```
Total_Cost  = Material_Cost + Labor_Cost + Overhead_Cost
Profit      = Item_Outlet_Sales − Total_Cost
Profit_Margin (%) = (Profit / Item_Outlet_Sales) × 100
```

### 7.5 Loss Detection

A record is flagged as **loss-making** when `Profit < 0`.

---

## 8. Functional Requirements

### FR-01: Data Loading & Validation
- **FR-01.1**: System shall load the Big Mart CSV file using Pandas.
- **FR-01.2**: System shall report missing value counts per column.
- **FR-01.3**: System shall impute `Item_Weight` with column mean.
- **FR-01.4**: System shall impute `Outlet_Size` with mode grouped by `Outlet_Type`.
- **FR-01.5**: System shall standardize `Item_Fat_Content` labels (e.g., "LF" → "Low Fat").

### FR-02: Feature Engineering
- **FR-02.1**: System shall compute `Material_Cost`, `Labor_Cost`, `Overhead_Cost`.
- **FR-02.2**: System shall compute `Total_Cost` and `Profit`.
- **FR-02.3**: System shall compute `Profit_Margin_Pct`.
- **FR-02.4**: System shall flag records where `Profit < 0` as `Is_Loss`.
- **FR-02.5**: System shall derive `Outlet_Age = Current_Year − Outlet_Establishment_Year`.

### FR-03: Exploratory Data Analysis
- **FR-03.1**: System shall generate distribution plots for all numeric features.
- **FR-03.2**: System shall generate a correlation heatmap.
- **FR-03.3**: System shall produce outlet-wise and item-type-wise profit summaries.
- **FR-03.4**: System shall produce scatter plots of Sales vs. Profit colored by Outlet Type.
- **FR-03.5**: System shall identify top-10 and bottom-10 items by profit.

### FR-04: Machine Learning Pipeline
- **FR-04.1**: System shall encode categorical variables using Label Encoding or One-Hot Encoding.
- **FR-04.2**: System shall split data into 80% training and 20% testing sets (random state = 42).
- **FR-04.3**: System shall train a Random Forest Regressor as the baseline model.
- **FR-04.4**: System shall optionally train a Gradient Boosting or XGBoost model for comparison.
- **FR-04.5**: System shall output feature importances.
- **FR-04.6**: System shall evaluate predictions using MAE, RMSE, and R².

### FR-05: Segment Analysis
- **FR-05.1**: System shall segment records into High Profit, Low Profit, and Loss-Making buckets.
- **FR-05.2**: System shall aggregate cost components by Outlet Type, Location Type, and Item Type.
- **FR-05.3**: System shall output a summary table of segment-wise average profit margin.

### FR-06: Output Artifacts
- **FR-06.1**: System shall export the enriched dataset to `output/enriched_bigmart.csv`.
- **FR-06.2**: System shall export model performance metrics to `output/model_metrics.csv`.
- **FR-06.3**: System shall save EDA plots to `output/plots/`.
- **FR-06.4**: System shall export model feature importances to `output/feature_importance.csv`.

---

## 9. Non-Functional Requirements

| ID | Requirement | Metric |
|----|-------------|--------|
| NFR-01 | **Performance** | Full pipeline (data → model) completes in < 5 minutes on a standard laptop |
| NFR-02 | **Reproducibility** | All random states fixed; results are reproducible across runs |
| NFR-03 | **Modularity** | Code organized into distinct modules: `data_loader`, `feature_engineering`, `eda`, `model`, `reporting` |
| NFR-04 | **Documentation** | All functions have docstrings; notebooks have markdown explanations |
| NFR-05 | **Portability** | Runs on Python 3.8+ on Windows, macOS, and Linux |
| NFR-06 | **Data Integrity** | No data leakage between train and test sets |
| NFR-07 | **Accuracy** | Model MAE on profit prediction < 15% of mean profit |

---

## 10. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│          Big Mart Dataset (CSV / Excel)                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DATA ENGINEERING LAYER                         │
│   • Data Loading & Validation         (data_loader.py)          │
│   • Missing Value Imputation                                    │
│   • Label Standardization                                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING LAYER                       │
│   • Cost Component Calculation      (feature_engineering.py)    │
│   • Profit & Margin Derivation                                  │
│   • Loss Flagging                                               │
│   • Outlet Age Calculation                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
┌──────────────────────┐   ┌────────────────────────────────────┐
│     EDA LAYER        │   │       ML PIPELINE LAYER            │
│  • Distribution plots│   │  • Categorical Encoding            │
│  • Correlation maps  │   │  • Train/Test Split (80/20)        │
│  • Segment summaries │   │  • Random Forest Regressor         │
│  (eda.py)            │   │  • Model Evaluation (MAE/R²)       │
└──────────┬───────────┘   │  • Feature Importances             │
           │               │  (model.py)                        │
           │               └──────────────┬─────────────────────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                                │
│   • Enriched Dataset CSV        (output/enriched_bigmart.csv)  │
│   • EDA Plots                   (output/plots/)                │
│   • Model Metrics CSV           (output/model_metrics.csv)     │
│   • Feature Importance CSV      (output/feature_importance.csv)│
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DASHBOARD LAYER (Power BI)                     │
│   • KPI Cards (Total Profit, Avg Margin, Loss Count)           │
│   • Cost Breakdown by Component (Bar/Pie Charts)               │
│   • Outlet-wise Profit Performance                             │
│   • Predicted vs. Actual Profit Scatter                        │
│   • Segment Analysis Slicers                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Data Pipeline

```
Raw CSV
   │
   ├─ [Step 1] Load & Inspect
   │       pandas.read_csv()
   │       shape, dtypes, null counts
   │
   ├─ [Step 2] Clean & Impute
   │       Item_Weight  → mean imputation
   │       Outlet_Size  → mode by Outlet_Type
   │       Fat_Content  → label harmonization
   │
   ├─ [Step 3] Feature Engineering
   │       Material_Cost, Labor_Cost, Overhead_Cost
   │       Total_Cost = sum of above
   │       Profit = Sales − Total_Cost
   │       Profit_Margin_Pct, Is_Loss flag
   │       Outlet_Age
   │
   ├─ [Step 4] EDA
   │       Histograms, Box plots, Heatmap
   │       Segment summaries
   │
   ├─ [Step 5] Encode & Split
   │       Label / OHE Encoding
   │       Train 80% / Test 20%
   │
   ├─ [Step 6] Model Training
   │       Random Forest Regressor
   │       fit(X_train, y_train)
   │
   ├─ [Step 7] Evaluate
   │       predict(X_test) → MAE, RMSE, R²
   │       Feature Importances
   │
   └─ [Step 8] Export Outputs
           enriched_bigmart.csv
           model_metrics.csv
           feature_importance.csv
           EDA plots (PNG)
```

---

## 12. Machine Learning Requirements

### 12.1 Target Variable
- `Profit` (continuous, numeric)

### 12.2 Feature Set

| Feature | Type | Notes |
|---------|------|-------|
| `Item_MRP` | Numeric | Maximum retail price — strong predictor |
| `Item_Outlet_Sales` | Numeric | Revenue proxy |
| `Item_Visibility` | Numeric | Shelf exposure |
| `Outlet_Age` | Numeric | Derived from establishment year |
| `Item_Weight` | Numeric | Imputed |
| `Item_Fat_Content` | Categorical (encoded) | |
| `Item_Type` | Categorical (encoded) | |
| `Outlet_Size` | Categorical (encoded) | |
| `Outlet_Location_Type` | Categorical (encoded) | |
| `Outlet_Type` | Categorical (encoded) | |

> **Note**: `Material_Cost`, `Labor_Cost`, `Overhead_Cost`, and `Total_Cost` are **excluded** from features to prevent data leakage (they are derived from the target).

### 12.3 Model Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Regressor |
| `n_estimators` | 100 |
| `max_depth` | None (fully grown) |
| `random_state` | 42 |
| `test_size` | 0.20 |

### 12.4 Evaluation Metrics

| Metric | Formula | Acceptable Threshold |
|--------|---------|----------------------|
| MAE | Mean Absolute Error | < 15% of mean profit |
| RMSE | Root Mean Squared Error | Reported |
| R² | Coefficient of Determination | > 0.80 |

---

## 13. Dashboard Requirements

### 13.1 Power BI Dashboard Pages

#### Page 1 — Executive Overview
- **KPI Cards**: Total Sales, Total Cost, Total Profit, Avg. Profit Margin %
- **Bar Chart**: Profit by Outlet Type
- **Donut Chart**: Cost Component Breakdown (Material / Labor / Overhead)
- **Filter Slicers**: Outlet Type, Location Type, Item Type

#### Page 2 — Outlet Performance
- **Clustered Bar Chart**: Profit by Outlet Identifier
- **Map Visual** (if location data available): Profit by Tier
- **Table**: Outlet-wise summary (Sales, Total Cost, Profit, Margin %)
- **Filter**: Outlet Size

#### Page 3 — Product Analysis
- **Bar Chart**: Top 10 and Bottom 10 Item Types by Average Profit
- **Scatter Plot**: Item MRP vs. Profit (colored by Item Type)
- **Line Chart**: Item Visibility vs. Profit Margin
- **KPI**: Loss-making SKU count

#### Page 4 — Predicted vs. Actual
- **Scatter Plot**: Actual Profit vs. Predicted Profit (45° reference line)
- **Table**: MAE, RMSE, R² metrics
- **Bar Chart**: Feature Importances
- **Filter**: Profit Bucket (High / Low / Loss)

### 13.2 Design Standards
- Color palette: Blue (#2E4057) for costs, Green (#048A81) for profit, Red (#E63946) for losses.
- All visuals must have proper axis labels and titles.
- Dashboard must be filterable by Outlet Type and Item Type at minimum.

---

## 14. Assumptions & Constraints

### Assumptions

| ID | Assumption |
|----|------------|
| A1 | Material cost is 40%–70% of sales (FMCG benchmark) |
| A2 | Labor cost is 10%–25% of sales (retail benchmark) |
| A3 | Overhead rates are fixed by outlet type (see §7.3) |
| A4 | Random seeds are consistent for reproducibility |
| A5 | The Big Mart dataset is representative of a multi-outlet retail chain |
| A6 | Profit margin > 0% is considered viable |

### Constraints

| ID | Constraint |
|----|------------|
| C1 | No actual financial data available; cost model is simulated |
| C2 | Computations must run on a standard laptop (8 GB RAM, no GPU) |
| C3 | Dashboard built in Power BI Desktop (free version) |
| C4 | No real-time or streaming data |
| C5 | Dataset is static; no time-series component |

---

## 15. Acceptance Criteria

| ID | Criteria | Verification Method |
|----|----------|---------------------|
| AC1 | All cost features computed without NaN values | `df.isnull().sum()` on output |
| AC2 | `Profit = Sales − Total_Cost` holds for all rows | Assertion check in unit test |
| AC3 | ML model R² ≥ 0.80 on test set | Model evaluation output |
| AC4 | ML model MAE < 15% of mean profit | Model evaluation output |
| AC5 | EDA produces at least 5 distinct charts | File count in `output/plots/` |
| AC6 | Power BI dashboard has all 4 pages defined in §13.1 | Manual review |
| AC7 | Enriched CSV exported to `output/` directory | File existence check |
| AC8 | Loss-making records correctly flagged (`Is_Loss = True`) | Spot check on rows with negative profit |
| AC9 | No data leakage (cost features not in ML feature set) | Code review |
| AC10 | Pipeline runs end-to-end without errors | Full run log |

---

## 16. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Simulated costs don't reflect real business | High | Medium | Use industry benchmarks; document assumptions clearly |
| Model overfits on training data | Medium | High | Use cross-validation; tune `max_depth` |
| EDA reveals no significant patterns | Low | Medium | Enrich with derived features (e.g., Outlet Age) |
| Power BI import issues with CSV | Low | Low | Validate CSV schema before import |
| Missing value imputation introduces bias | Medium | Medium | Use grouped imputation; document methodology |

---

## 17. Timeline & Milestones

| Phase | Milestone | Target |
|-------|-----------|--------|
| Phase 1 | Dataset acquisition & EDA | Week 1 |
| Phase 2 | Cost modeling & feature engineering | Week 2 |
| Phase 3 | ML model training & evaluation | Week 3 |
| Phase 4 | Output export & validation | Week 4 |
| Phase 5 | Power BI dashboard development | Week 5 |
| Phase 6 | Documentation & final review | Week 6 |

---

## 18. Glossary

| Term | Definition |
|------|------------|
| **MAE** | Mean Absolute Error — average magnitude of prediction errors |
| **RMSE** | Root Mean Squared Error — square root of average squared errors |
| **R²** | Coefficient of Determination — proportion of variance explained by the model |
| **EDA** | Exploratory Data Analysis |
| **SKU** | Stock Keeping Unit — a unique product identifier |
| **MRP** | Maximum Retail Price |
| **FMCG** | Fast-Moving Consumer Goods |
| **Overhead** | Fixed or semi-fixed operational costs not directly tied to production |
| **Profit Margin** | Profit as a percentage of sales revenue |
| **Data Leakage** | Using information in model training that would not be available at prediction time |
| **Label Encoding** | Transforming categorical text values into numeric integers |
| **Feature Importance** | A model's ranking of input features by their predictive contribution |

---

*End of PRD — Version 1.0.0*
