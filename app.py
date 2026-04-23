import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

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
st.markdown("""
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
    
    /* Fix label text visibility if user has Dark Mode active */
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

    /* Input box dark styling for contrast */
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
</style>
""", unsafe_allow_html=True)

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
    st.error(f"Error loading models. Please ensure the backend ML pipeline has run. Details: {e}")

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
tab1, tab2 = st.tabs(["⚡ Live Single Item Forecasting", "📁 Batch Pipeline (Power BI Integration)"])

# ==========================================
# TAB 1: SINGLE ITEM PREDICTOR
# ==========================================
with tab1:
    st.markdown("<p style='color: #666; font-size: 1.1rem; margin-bottom: 25px;'>Input product and placement parameters below to simulate the financial outcome before the product reaches the shelves.</p>", unsafe_allow_html=True)
    
    # Use styled containers for form sections
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='section-header'>📦 Product Economics</div>", unsafe_allow_html=True)
            item_type = st.selectbox("Product Category", ITEM_TYPES)
            item_mrp = st.number_input("Maximum Retail Price (₹ MRP)", min_value=10.0, max_value=500.0, value=150.0, step=1.0)
            item_weight = st.number_input("Item Weight (kg) [Logistics Cost]", min_value=0.0, max_value=30.0, value=12.5, step=0.1)
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
            outlet_id = st.text_input("Internal Outlet Identifier", value="OUT027", help="e.g., OUT027")
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Center the button
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
            
            # Encode categorical inputs safely
            for col in encoders.keys():
                if df_input[col].iloc[0] in encoders[col].classes_:
                    df_input[col] = encoders[col].transform(df_input[col].astype(str))
                else:
                    df_input[col] = 0 
            
            # Predict
            prediction = model.predict(df_input)[0]
            
            # --- Back-calculate expected units sold based on our financial model ---
            ITEM_MAT = { "Health and Hygiene": 0.50, "Hard Drinks": 0.52, "Soft Drinks": 0.53, "Household": 0.55, "Others": 0.55, "Snack Foods": 0.58, "Baking Goods": 0.58, "Breakfast": 0.58, "Starchy Foods": 0.60, "Breads": 0.60, "Canned": 0.61, "Frozen Foods": 0.62, "Dairy": 0.63, "Meat": 0.65, "Seafood": 0.67, "Fruits and Vegetables": 0.68 }
            OUT_LAB = { "Grocery Store": 0.08, "Supermarket Type1": 0.10, "Supermarket Type2": 0.11, "Supermarket Type3": 0.12 }
            OUT_OVH = { "Grocery Store": 0.05, "Supermarket Type1": 0.07, "Supermarket Type2": 0.08, "Supermarket Type3": 0.09 }
            
            mat_rate = ITEM_MAT.get(item_type, 0.60)
            lab_rate = OUT_LAB.get(outlet_type, 0.10)
            ovh_rate = OUT_OVH.get(outlet_type, 0.07)
            total_cost_rate = mat_rate + lab_rate + ovh_rate
            profit_margin_pct = 1.0 - total_cost_rate
            
            # Expected cumulative sales
            expected_sales = prediction / profit_margin_pct if profit_margin_pct != 0 else 0
            expected_units = expected_sales / item_mrp if item_mrp > 0 else 0
            
            # Custom Outcome Logic
            if prediction < 0:
                border_color = "#e63946" # Red
                status = "Critical Loss Warning"
                status_color = "#e63946"
            elif prediction < 150:
                border_color = "#f4a261" # Orange
                status = "Moderate Margin Expected"
                status_color = "#f4a261"
            else:
                border_color = "#2a9d8f" # Green
                status = "Strong Profitability Expected"
                status_color = "#2a9d8f"
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Beautiful Metric Cards Side-by-Side
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
    st.markdown("<p style='color: #666; font-size: 1.1rem; margin-bottom: 25px;'>Upload an unlabelled dataset (e.g., historical records or future pipeline) to execute mass forecasting. Results will be formatted automatically for Microsoft Power BI integration.</p>", unsafe_allow_html=True)
    
    st.info("💡 **Instructions**: Download the template below, populate your item and outlet data, and upload the completed file. The AI will attach a 'Predicted_Profit_Rs' column.")
    
    template_df = pd.DataFrame(columns=[
        "Item_Identifier", "Item_Weight", "Item_Fat_Content", "Item_Visibility", "Item_Type", 
        "Item_MRP", "Outlet_Identifier", "Outlet_Establishment_Year", "Outlet_Size", 
        "Outlet_Location_Type", "Outlet_Type"
    ])
    
    st.download_button(
        label="📄 Download Input CSV Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="PowerBI_Forecast_Template.csv",
        mime="text/csv"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Populated CSV Data File", type=["csv"])
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully mounted {len(raw_df):,} records into memory.")
        
        if st.button("⚙️ Execute Mass AI Forecasting"):
            with st.spinner("Initializing AI Engine... Processing matrix..."):
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
                    
                    csv = final_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Export for Power BI",
                        data=csv,
                        file_name="AI_Profit_Forecast_Export.csv",
                        mime="text/csv",
                    )
                    
                except Exception as e:
                    st.error(f"Computation Error: {e}. Please ensure your dataset matches the template columns.")

