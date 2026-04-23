import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Profit Optimizer AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# PREMIUM CUSTOM CSS
# ─────────────────────────────────────────────
CSS = """
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #f4f6f9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Top Gradient Header */
    .premium-header {
        background: linear-gradient(135deg, #0b2e59 0%, #15559a 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .premium-header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .premium-header p {
        margin: 10px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Style the tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 8px;
        padding: 10px 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: #333;
        font-weight: 600;
        border: 1px solid #eee;
    }
    .stTabs [aria-selected="true"] {
        background-color: #15559a !important;
        color: white !important;
        border-bottom-color: transparent !important;
    }

    /* Custom Metric Cards */
    .metric-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 4px solid #2e8b57;
    }
    .metric-title {
        font-size: 1.1rem;
        color: #555;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #111;
        margin: 0;
    }
    .metric-subtitle {
        font-size: 1rem;
        color: #2e8b57;
        font-weight: bold;
        margin-top: 5px;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #15559a;
        color: white;
        font-weight: 600;
        padding: 10px 24px;
        border-radius: 6px;
        transition: all 0.3s;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #0b2e59;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Fix label text visibility */
    label, .stMarkdown p, .stSelectbox label {
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    
    /* Ensure Header and Buttons stay white */
    .premium-header p, .premium-header h1 {
        color: white !important;
    }
    .stButton>button p, .stButton>button span {
        color: white !important;
    }

    /* Input box styling */
    input, div[data-baseweb="select"] {
        border-radius: 6px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.3rem;
        color: #15559a !important;
        font-weight: 600;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 8px;
        margin-bottom: 15px;
        margin-top: 15px;
    }

    /* KPI Cards for Analytics */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 22px 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 16px;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #15559a;
        margin: 0;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #2e8b57;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Glass Card */
    .glass-card {
        background: white;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.07);
        margin-bottom: 18px;
    }

    /* Upload placeholder */
    .upload-placeholder {
        text-align: center;
        padding: 60px 20px;
        background: linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
        border: 2px dashed #c5cae9;
        border-radius: 16px;
        margin-top: 20px;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL & ENCODERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_ml_assets():
    model_path = os.path.join("output", "trained_model.pkl")
    encoder_path = os.path.join("output", "encoders.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

try:
    model, encoders = load_ml_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading models: {e}")

# ─────────────────────────────────────────────
# CONSTANTS & OPTIONS
# ─────────────────────────────────────────────
FEATURES = [
    "Item_MRP", "Item_Visibility", "Item_Weight", "Outlet_Age", 
    "Item_Fat_Content", "Item_Type", "Outlet_Identifier", 
    "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"
]
ITEM_TYPES = [
    "Health and Hygiene", "Hard Drinks", "Soft Drinks", "Household", 
    "Others", "Snack Foods", "Baking Goods", "Breakfast", "Starchy Foods", 
    "Breads", "Canned", "Frozen Foods", "Dairy", "Meat", "Seafood", "Fruits and Vegetables"
]
OUTLET_TYPES = ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
LOCATION_TYPES = ["Tier 1", "Tier 2", "Tier 3"]
OUTLET_SIZES = ["Small", "Medium", "High"]
FAT_CONTENTS = ["Low Fat", "Regular"]

ITEM_MAT = { "Health and Hygiene": 0.50, "Hard Drinks": 0.52, "Soft Drinks": 0.53, "Household": 0.55, "Others": 0.55, "Snack Foods": 0.58, "Baking Goods": 0.58, "Breakfast": 0.58, "Starchy Foods": 0.60, "Breads": 0.60, "Canned": 0.61, "Frozen Foods": 0.62, "Dairy": 0.63, "Meat": 0.65, "Seafood": 0.67, "Fruits and Vegetables": 0.68 }
OUT_LAB = { "Grocery Store": 0.08, "Supermarket Type1": 0.10, "Supermarket Type2": 0.11, "Supermarket Type3": 0.12 }
OUT_OVH = { "Grocery Store": 0.05, "Supermarket Type1": 0.07, "Supermarket Type2": 0.08, "Supermarket Type3": 0.09 }

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(color="#333", family="Inter, sans-serif"),
    colorway=["#15559a","#2e8b57","#e63946","#f4a261","#457b9d","#264653","#e76f51","#2a9d8f"],
    margin=dict(l=20, r=20, t=40, b=20),
)

# ─────────────────────────────────────────────
# UI - HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="premium-header">
    <h1>📈 Retail Financial & Profitability Hub</h1>
    <p>Powered by Artificial Intelligence | Pre-Sales Forecast Optimization Engine</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.stop()

# ─────────────────────────────────────────────
# MAIN DASHBOARD TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚡ Live Single Item Forecasting", "📁 Batch Pipeline (Power BI Integration)", "📊 Analytics Studio"])

# ==========================================
# TAB 1: SINGLE ITEM PREDICTOR
# ==========================================
with tab1:
    st.markdown("<p style='color: #666; font-size: 1.1rem; margin-bottom: 25px;'>Input product and placement parameters below to simulate the financial outcome before the product reaches the shelves.</p>", unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='section-header'>📦 Product Economics</div>", unsafe_allow_html=True)
            item_type = st.selectbox("Product Category", ITEM_TYPES)
            item_mrp = st.number_input("Maximum Retail Price (₹ MRP)", min_value=10.0, max_value=500.0, value=150.0, step=1.0)
            item_weight = st.number_input("Item Weight (kg)", min_value=0.0, max_value=30.0, value=12.5, step=0.1)
            item_fat = st.selectbox("Dietary/Fat Segment", FAT_CONTENTS)
        with col2:
            st.markdown("<div class='section-header'>🏢 Store Architecture</div>", unsafe_allow_html=True)
            outlet_type = st.selectbox("Operating Model (Outlet Type)", OUTLET_TYPES)
            outlet_loc = st.selectbox("City Classification (Tier)", LOCATION_TYPES)
            outlet_size = st.selectbox("Physical Footprint (Size)", OUTLET_SIZES)
            outlet_age = st.slider("Outlet Operational Age (Years)", min_value=1, max_value=50, value=25)
        with col3:
            st.markdown("<div class='section-header'>🎯 Merchandising Strategy</div>", unsafe_allow_html=True)
            item_visibility = st.slider("Shelf Visibility Score (%)", min_value=0.0, max_value=0.4, value=0.05, step=0.01)
            outlet_id = st.text_input("Internal Outlet Identifier", value="OUT027")

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if st.button("🚀 Execute AI Profit Simulation", use_container_width=True):
            input_data = {
                "Item_MRP": item_mrp, "Item_Visibility": item_visibility, "Item_Weight": item_weight,
                "Outlet_Age": outlet_age, "Item_Fat_Content": item_fat, "Item_Type": item_type,
                "Outlet_Identifier": outlet_id, "Outlet_Size": outlet_size,
                "Outlet_Location_Type": outlet_loc, "Outlet_Type": outlet_type
            }
            df_input = pd.DataFrame([input_data])
            for col in encoders.keys():
                if df_input[col].iloc[0] in encoders[col].classes_:
                    df_input[col] = encoders[col].transform(df_input[col].astype(str))
                else:
                    df_input[col] = 0

            prediction = model.predict(df_input)[0]
            mat_rate = ITEM_MAT.get(item_type, 0.60)
            lab_rate = OUT_LAB.get(outlet_type, 0.10)
            ovh_rate = OUT_OVH.get(outlet_type, 0.07)
            profit_margin_pct = 1.0 - mat_rate - lab_rate - ovh_rate
            expected_sales = prediction / profit_margin_pct if profit_margin_pct != 0 else 0
            expected_units = expected_sales / item_mrp if item_mrp > 0 else 0

            if prediction < 0:
                border_color, status, status_color = "#e63946", "Critical Loss Warning", "#e63946"
            elif prediction < 150:
                border_color, status, status_color = "#f4a261", "Moderate Margin Expected", "#f4a261"
            else:
                border_color, status, status_color = "#2a9d8f", "Strong Profitability Expected", "#2a9d8f"

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div style="background-color: white; padding: 25px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); text-align: center; border-top: 6px solid #15559a; margin-bottom: 20px;">
                    <div style="font-size: 0.9rem; color: #888; font-weight: 500; text-transform: uppercase;">Estimated Total Units Sold</div>
                    <div style="font-size: 2.8rem; font-weight: 800; color: #15559a; margin: 10px 0;">{int(expected_units)} units</div>
                    <div style="font-size: 0.9rem; color: #aaa;">Generating approx. ₹{expected_sales:,.0f} in Total Est. Revenue</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="background-color: white; padding: 25px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); text-align: center; border-top: 6px solid {border_color};">
                    <div style="font-size: 0.9rem; color: #888; font-weight: 500; text-transform: uppercase;">Forecasted Cumulative Profit</div>
                    <div style="font-size: 2.8rem; font-weight: 800; color: #1e293b; margin: 10px 0;">₹ {prediction:,.2f}</div>
                    <div style="font-size: 1.0rem; color: {status_color}; font-weight: 700; background-color: {status_color}20; display: inline-block; padding: 6px 15px; border-radius: 20px;">{status}</div>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# TAB 2: BATCH PREDICTION
# ==========================================
with tab2:
    st.markdown("<p style='color: #666; font-size: 1.1rem; margin-bottom: 25px;'>Upload an unlabelled dataset to execute mass forecasting. Results formatted for Power BI integration.</p>", unsafe_allow_html=True)
    st.info("💡 **Instructions**: Download the template, populate your data, and upload. The AI will attach a 'Predicted_Profit_Rs' column.")

    template_df = pd.DataFrame(columns=[
        "Item_Identifier", "Item_Weight", "Item_Fat_Content", "Item_Visibility", "Item_Type",
        "Item_MRP", "Outlet_Identifier", "Outlet_Establishment_Year", "Outlet_Size",
        "Outlet_Location_Type", "Outlet_Type"
    ])
    st.download_button("📄 Download Input CSV Template", template_df.to_csv(index=False).encode("utf-8"), "PowerBI_Forecast_Template.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload Populated CSV Data File", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully mounted {len(raw_df):,} records into memory.")
        if st.button("⚙️ Execute Mass AI Forecasting"):
            with st.spinner("Processing..."):
                try:
                    df = raw_df.copy()
                    if "Outlet_Age" not in df.columns and "Outlet_Establishment_Year" in df.columns:
                        df["Outlet_Age"] = 2026 - df["Outlet_Establishment_Year"]
                    if "Item_Weight" in df.columns:
                        df["Item_Weight"] = df["Item_Weight"].fillna(12.85)
                    if "Outlet_Size" in df.columns:
                        df["Outlet_Size"] = df["Outlet_Size"].fillna("Medium")
                    if "Item_Fat_Content" in df.columns:
                        df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"})
                    X_pred = df[FEATURES].copy()
                    for col in encoders.keys():
                        le = encoders[col]
                        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                        X_pred[col] = X_pred[col].astype(str).map(mapping).fillna(0).astype(int)
                    predictions = model.predict(X_pred)
                    final_df = raw_df.copy()
                    final_df["AI_Predicted_Profit_INR"] = np.round(predictions, 2)
                    st.success("✅ **Forecasting Complete!**")
                    st.dataframe(final_df.head(8), use_container_width=True)
                    st.download_button("📥 Download Export for Power BI", final_df.to_csv(index=False).encode("utf-8"), "AI_Profit_Forecast_Export.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# TAB 3: ANALYTICS STUDIO
# ==========================================
with tab3:
    st.markdown("<div class='section-header'>📊 Interactive Analytics Studio</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; font-size: 1.05rem;'>Upload any CSV to explore profit trends, distributions & patterns with interactive charts.</p>", unsafe_allow_html=True)

    ana_file = st.file_uploader("📂 Upload your dataset", type=["csv"], key="analytics_upload")

    if ana_file is None:
        st.markdown("""
        <div class="upload-placeholder">
            <div style="font-size: 3.5rem; margin-bottom: 16px;">📊</div>
            <div style="color: #15559a; font-size: 1.3rem; font-weight: 700; margin-bottom: 8px;">Upload a CSV to Begin Analysis</div>
            <div style="color: #888; font-size: 0.95rem;">Supports batch output, raw data, or any tabular CSV file</div>
        </div>""", unsafe_allow_html=True)
    else:
        df_ana = pd.read_csv(ana_file)
        num_cols = df_ana.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df_ana.select_dtypes(include="object").columns.tolist()
        profit_col = next((c for c in num_cols if "profit" in c.lower() or "sales" in c.lower()), num_cols[0] if num_cols else None)

        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Records</div><div class="kpi-value">{len(df_ana):,}</div><div class="kpi-sub">rows loaded</div></div>', unsafe_allow_html=True)
        with k2:
            avg_val = f"₹{df_ana[profit_col].mean():,.0f}" if profit_col else "N/A"
            st.markdown(f'<div class="kpi-card" style="border-top: 4px solid #2e8b57;"><div class="kpi-label">Avg Profit</div><div class="kpi-value" style="color:#2e8b57;">{avg_val}</div><div class="kpi-sub">per record</div></div>', unsafe_allow_html=True)
        with k3:
            top_cat = df_ana[cat_cols[0]].mode()[0] if cat_cols else "N/A"
            st.markdown(f'<div class="kpi-card" style="border-top: 4px solid #f4a261;"><div class="kpi-label">Top {cat_cols[0] if cat_cols else "Category"}</div><div class="kpi-value" style="color:#f4a261; font-size:1.3rem;">{str(top_cat)[:16]}</div><div class="kpi-sub">most frequent</div></div>', unsafe_allow_html=True)
        with k4:
            loss_count = int((df_ana[profit_col] < 0).sum()) if profit_col else 0
            st.markdown(f'<div class="kpi-card" style="border-top: 4px solid #e63946;"><div class="kpi-label">Loss Items</div><div class="kpi-value" style="color:#e63946;">{loss_count:,}</div><div class="kpi-sub">negative profit</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Filters
        with st.expander("🎛️ Filter Data", expanded=False):
            fcols = st.columns(min(len(cat_cols), 4)) if cat_cols else []
            mask = pd.Series([True] * len(df_ana))
            for i, cc in enumerate(cat_cols[:4]):
                opts = ["All"] + sorted(df_ana[cc].dropna().unique().tolist())
                sel = fcols[i].selectbox(cc, opts, key=f"filter_{cc}")
                if sel != "All":
                    mask &= df_ana[cc] == sel
            df_f = df_ana[mask]

        # Charts Row 1
        if profit_col and cat_cols:
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                grp = df_f.groupby(cat_cols[0])[profit_col].mean().nlargest(10).reset_index()
                fig1 = px.bar(grp, x=cat_cols[0], y=profit_col, title=f"Avg {profit_col} by {cat_cols[0]}", color=profit_col, color_continuous_scale=["#15559a","#2e8b57"])
                fig1.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with ch2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                pie_data = df_f[cat_cols[0]].value_counts().head(8)
                fig2 = px.pie(values=pie_data.values, names=pie_data.index, title=f"Distribution by {cat_cols[0]}")
                fig2.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Charts Row 2
        if len(num_cols) >= 2:
            ch3, ch4 = st.columns(2)
            with ch3:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                x_col = st.selectbox("X-Axis", num_cols, key="scatter_x")
                y_col = st.selectbox("Y-Axis", num_cols, index=min(1, len(num_cols)-1), key="scatter_y")
                fig3 = px.scatter(df_f, x=x_col, y=y_col, color=cat_cols[0] if cat_cols else None, title=f"{y_col} vs {x_col}", opacity=0.65)
                fig3.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig3, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with ch4:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                corr = df_f[num_cols[:8]].corr()
                fig4 = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale=[[0,"#f4f6f9"],[0.5,"#15559a"],[1,"#0b2e59"]], text=np.round(corr.values, 2), texttemplate="%{text}"))
                fig4.update_layout(title="Correlation Heatmap", **PLOTLY_LAYOUT)
                st.plotly_chart(fig4, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Profit Distribution
        if profit_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig5 = px.histogram(df_f, x=profit_col, nbins=40, title=f"{profit_col} Distribution", color_discrete_sequence=["#15559a"])
            fig5.update_layout(**PLOTLY_LAYOUT)
            fig5.add_vline(x=df_f[profit_col].mean(), line_dash="dash", line_color="#2e8b57", annotation_text="Mean", annotation_font_color="#2e8b57")
            st.plotly_chart(fig5, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Data Preview & Download
        with st.expander("📋 Raw Data Preview"):
            st.dataframe(df_f.head(100), use_container_width=True)
        st.download_button("📥 Download Filtered Data", df_f.to_csv(index=False).encode("utf-8"), "filtered_analysis.csv", "text/csv")
