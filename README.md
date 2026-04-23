# 🧠 AI-Based Cost and Profit Optimization System for Retail Outlets

> **Integrating Cost Accounting with Machine Learning to Drive Retail Profitability**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-blue?logo=plotly)](https://plotly.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Live-brightgreen)](https://parth07-alt-cost-accounting.streamlit.app)

---

## 🚀 Live Demo

> 🌐 **Deployed App**: [https://parth07-alt-cost-accounting.streamlit.app](https://parth07-alt-cost-accounting.streamlit.app)

The app includes **three powerful modules**:
| Tab | Feature |
|-----|---------|
| ⚡ Live Forecast | Input product & store parameters → get instant AI profit prediction |
| 📁 Batch Pipeline | Upload a CSV → AI predicts profit for all rows → export for Power BI |
| 📊 Analytics Studio | Upload any CSV → interactive charts, KPI cards, correlation heatmaps |

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [App Usage Guide](#-app-usage-guide)
- [Cost Model Explained](#-cost-model-explained)
- [Model Performance](#-model-performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project builds an **end-to-end intelligent system** that applies machine learning to predict and optimize retail profitability. The foundation is the **Big Mart Sales Dataset**, augmented with engineered cost features — **material cost**, **labor cost**, and **overhead cost** — based on realistic business assumptions from cost accounting principles.

Since the original dataset lacks any cost or profit data, a **structured cost model** is designed from scratch using industry benchmarks for FMCG retail. This enables the simulation of a complete financial ledger on top of the sales data.

The pipeline flows: **raw data → cost-enriched dataset → EDA → ML model → Streamlit Dashboard (deployed)**

### Why This Matters

| Challenge | This System's Answer |
|-----------|---------------------|
| No cost data in raw retail datasets | Engineer costs from business assumptions |
| Inability to predict profitability per SKU | Train a regression model on financial features |
| No visibility into loss-making segments | Automated loss detection and flagging |
| Data inaccessible to business users | Deployed Streamlit app + Power BI integration |
| Static dashboards | Interactive Analytics Studio with Plotly |

---

## ✨ Key Features

- 🏗️ **Cost Engineering** — Material, labor, and overhead costs derived from sales using industry-realistic rates.
- 💰 **Profit Prediction** — Random Forest model predicts per-record profit from product & outlet features.
- 📉 **Loss Detection** — Automatic flagging of loss-making transactions.
- 📊 **Analytics Studio** — Upload any CSV → instant interactive charts (bar, pie, scatter, heatmap, histogram).
- 📁 **Batch Pipeline** — Mass prediction for thousands of rows, export-ready for Power BI.
- ⚡ **Live Forecasting** — Real-time single-item profit simulation with revenue & unit estimates.
- 🌐 **Deployed** — Live on Streamlit Community Cloud, accessible by anyone with the link.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Web App | Streamlit |
| Interactive Charts | Plotly |
| Data Processing | Python 3.8+, Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest Regressor) |
| Static Visualization | Matplotlib, Seaborn |
| Business Intelligence | Microsoft Power BI |
| Deployment | Streamlit Community Cloud |
| Version Control | Git + GitHub |

---

## 📁 Project Structure

```
Cost_Accounting/
│
├── app.py                          # ⭐ Streamlit Web Application (3 tabs)
│
├── src/
│   ├── data_loader.py              # Load & validate raw dataset
│   ├── feature_engineering.py      # Cost features & profit derivation
│   ├── eda.py                      # EDA plots and summaries
│   ├── model.py                    # ML pipeline (encode → train → evaluate)
│   └── reporting.py                # Export outputs to CSV and Excel
│
├── data/
│   └── processed/
│       └── bigmart_cleaned.csv     # Cleaned dataset
│
├── output/
│   ├── trained_model.pkl           # Trained Random Forest model
│   ├── encoders.pkl                # Label encoders for categorical features
│   ├── model_metrics.csv           # MAE, RMSE, R² results
│   ├── feature_importance.csv      # Ranked feature importances
│   ├── model_comparison.csv        # Comparison across model variants
│   └── plots/                      # All EDA and model evaluation charts
│
├── main.py                         # Full ML pipeline runner
├── rebuild_cost_model.py           # Rebuild model from scratch
├── predict_test.py                 # Batch prediction script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── PRD.md                          # Product Requirements Document
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Parth07-alt/Cost_accounting.git
cd Cost_accounting
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App Locally

```bash
streamlit run app.py
# or
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`

### 4. Retrain the Model (Optional)

```bash
python main.py
```

---

## 📱 App Usage Guide

### ⚡ Tab 1 — Live Single Item Forecasting

1. Fill in **Product Economics** (category, MRP, weight, fat content)
2. Fill in **Store Architecture** (outlet type, city tier, size, age)
3. Set **Merchandising** (shelf visibility, outlet ID)
4. Click **"🚀 Execute AI Profit Simulation"**
5. Instantly see **estimated units sold**, **total revenue**, and **forecasted profit** with a color-coded status badge.

### 📁 Tab 2 — Batch Pipeline (Power BI Integration)

1. Download the **CSV template**
2. Populate it with your product/outlet data
3. Upload the completed file
4. Click **"⚙️ Execute Mass AI Forecasting"**
5. Download the output CSV with an `AI_Predicted_Profit_INR` column — ready to load into Power BI.

### 📊 Tab 3 — Analytics Studio

1. Upload **any CSV file** (batch output, raw data, custom dataset)
2. Instantly see **4 KPI cards** (total records, avg profit, top category, loss items)
3. Explore interactive **bar chart, pie chart, scatter plot, and correlation heatmap**
4. Use **filters** to slice data by any categorical column
5. View the **profit distribution histogram**
6. Download the **filtered dataset**

---

## 💡 Cost Model Explained

The cost model is the core innovation. Since no actual cost data exists, costs are **simulated using business logic** anchored to industry benchmarks:

```
┌──────────────────────────────────────────────────────┐
│            COST BREAKDOWN MODEL                       │
├─────────────────┬────────────────────────────────────┤
│ Material Cost   │ 50% – 68% of MRP (by item type)   │
│ Labor Cost      │ 8% – 12% (by outlet type)          │
│ Overhead Cost   │ 5% – 9% (by outlet type)           │
├─────────────────┴────────────────────────────────────┤
│ Total Cost Rate = Material + Labor + Overhead         │
│ Profit          = Predicted Sales − Total Cost        │
│ Profit Margin % = (1 − Total Cost Rate) × 100        │
└──────────────────────────────────────────────────────┘
```

### Overhead Rate by Outlet Type

| Outlet Type | Labor Rate | Overhead Rate |
|-------------|-----------|---------------|
| Grocery Store | 8% | 5% |
| Supermarket Type1 | 10% | 7% |
| Supermarket Type2 | 11% | 8% |
| Supermarket Type3 | 12% | 9% |

---

## 📊 Model Performance

| Metric | Description |
|--------|-------------|
| Algorithm | Random Forest Regressor |
| Target | Profit (₹) per product-outlet combination |
| Features | MRP, Visibility, Weight, Outlet Age, Fat Content, Item Type, Outlet Size, Location Tier, Outlet Type |
| Train/Test Split | 80% / 20% |

See `output/model_metrics.csv` for detailed R², MAE, and RMSE scores.

---

## 🤝 Contributing

Contributions and suggestions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add: description"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- **Big Mart Dataset** — Analytics Vidhya Big Mart Sales Prediction competition
- **Scikit-learn** — ML toolkit
- **Streamlit** — Web app framework & free deployment platform
- **Plotly** — Interactive chart library
- **FMCG Cost Benchmarks** — Industry data used to calibrate cost assumptions

---

> *"Accounting tells you where you've been. Machine Learning tells you where you're going."*

---

**Built with ❤️ | AI-Based Cost & Profit Optimization System | 2026**
