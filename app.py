import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

st.set_page_config(page_title="GridPulse AI | Predictive & Prescriptive Engine", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #FFFFFF; }
    h1, h2, h3 { color: #0F3D68; }
    .stMetric { background-color: #F4F8FC; padding: 14px; border-radius: 8px; border-left: 4px solid #1E62B5; }
    .ai-box { background-color: #EEF5FC; padding: 16px; border-radius: 8px; border: 1px solid #B8D4F2; color: #0F3D68; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# AI ELEMENT 1: PREDICTIVE ML MODEL (RANDOM FOREST)
# ---------------------------------------------------------
@st.cache_resource
def train_rtm_price_model():
    np.random.seed(42)
    n_samples = 960  # 10 days of 96 15-min intervals
    block_num = np.random.randint(1, 97, n_samples)
    grid_demand_gw = np.random.uniform(175, 235, n_samples)
    renewable_surge = np.random.uniform(0, 150, n_samples)

    # Simulated Indian power market price dynamics (₹/kWh)
    peak_factor = np.where((block_num >= 72) & (block_num <= 88), 1.4, 0.0)
    price = (
        2.2
        + 0.018 * (grid_demand_gw - 175)
        - 0.008 * renewable_surge
        + peak_factor
        + np.random.normal(0, 0.15, n_samples)
    )
    price = np.clip(price, 2.80, 9.50)

    X = pd.DataFrame({
        "block_num": block_num,
        "grid_demand_gw": grid_demand_gw,
        "renewable_surge": renewable_surge
    })
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, price)
    accuracy_score = round(model.score(X, price) * 100, 1)
    return model, accuracy_score

ml_model, model_accuracy = train_rtm_price_model()

# ---------------------------------------------------------
# HEADER & SIDEBAR CONTROLS
# ---------------------------------------------------------
st.title("GridPulse AI — Predictive & Prescriptive Scheduling Engine")
st.caption("96-Block Indian Grid | ML Price Forecasting + Guardrail Optimization + Explainable AI")

st.sidebar.header("1. Grid & Spike Inputs (Layer 1)")
block_choice = st.sidebar.selectbox(
    "Select 15-Min Time Block (1–96)",
    [
        ("Block 11 (02:30 - 02:45 AM) [Night Off-Peak]", 11),
        ("Block 16 (03:45 - 04:00 AM) [Night Off-Peak]", 16),
        ("Block 50 (12:15 - 12:30 PM) [Solar Mid-Day]", 50),
        ("Block 78 (07:15 - 07:30 PM) [Evening Peak]", 78),
    ],
    format_func=lambda x: x[0]
)
block_label, block_id = block_choice

surplus_mw = st.sidebar.slider("Sudden Excess Generation Spike (MW)", 10, 200, 100, 10)
grid_demand = st.sidebar.slider("Regional Grid Demand (GW)", 180, 235, 195, 5)

st.sidebar.markdown("---")
st.sidebar.header("2. Guardrail Governance (Layer 2)")
tier2_approved = st.sidebar.radio(
    "Off-Hours Manager Action (CP-C Exception):",
    ["No Response (Timer Expires -> Auto-Fallback)", "1-Click Mobile Approved (<5 Mins)"]
)

# Run ML Prediction
input_features = pd.DataFrame([{
    "block_num": block_id,
    "grid_demand_gw": grid_demand,
    "renewable_surge": surplus_mw
}])
predicted_rtm_price = round(float(ml_model.predict(input_features)[0]), 2)
net_rtm_price = round(predicted_rtm_price - 0.10, 2)  # Minus ₹0.10 exchange fee

# ---------------------------------------------------------
# AI ELEMENT 2: PRESCRIPTIVE OPTIMIZATION ENGINE
# ---------------------------------------------------------
tier1_customers = [
    {"name": "CP-A (Apex Steel Ltd.)", "surge_limit": 40, "surge_price": 5.20},
    {"name": "CP-B (Metro Discom)", "surge_limit": 30, "surge_price": 4.90},
]

remaining_mw = surplus_mw
allocations = []
reasons = []

# Step 1: Compare Tier-1 Pre-Authorized Surge Rates vs. ML-Predicted RTM Price
for cp in sorted(tier1_customers, key=lambda x: x["surge_price"], reverse=True):
    if remaining_mw > 0 and cp["surge_price"] >= net_rtm_price:
        alloc = min(remaining_mw, cp["surge_limit"])
        remaining_mw -= alloc
        rev = alloc * 1000 * 0.25 * cp["surge_price"]
        allocations.append({
            "Destination": cp["name"],
            "Channel": "Bilateral (WBES)",
            "Allocated_MW": alloc,
            "Tariff_INR_kWh": cp["surge_price"],
            "Block_Revenue_INR": rev,
            "AI_Decision_Rule": "Tier 1: Auto-Approved (Beats ML RTM Forecast)"
        })
        reasons.append(f"Routed **{alloc} MW to {cp['name']}** at ₹{cp['surge_price']}/kWh because it exceeds the ML-predicted net RTM price (₹{net_rtm_price}/kWh) and stays within pre-authorized surge limits.")

# Step 2: Tier-2 Fast-Track Exception (CP-C @ ₹5.50/kWh)
if remaining_mw > 0 and 5.50 >= net_rtm_price:
    if "1-Click Mobile Approved" in tier2_approved:
        alloc = min(remaining_mw, 50)
        remaining_mw -= alloc
        rev = alloc * 1000 * 0.25 * 5.50
        allocations.append({
            "Destination": "CP-C (Indus Auto Corp)",
            "Channel": "Bilateral (WBES)",
            "Allocated_MW": alloc,
            "Tariff_INR_kWh": 5.50,
            "Block_Revenue_INR": rev,
            "AI_Decision_Rule": "Tier 2: Human-in-the-Loop Approved"
        })
        reasons.append(f"Routed **{alloc} MW to CP-C** at ₹5.50/kWh after receiving fast-track mobile approval before the 5-minute cutoff timer.")

# Step 3: Tier-3 Optimal RTM Routing (Either because RTM pays more OR fallback triggered)
if remaining_mw > 0:
    rev = remaining_mw * 1000 * 0.25 * net_rtm_price
    if net_rtm_price > 5.20:
        rule_text = "AI Arbitrage: RTM Price Beats Bilateral Caps"
        reasons.append(f"Routed **{remaining_mw} MW to IEX RTM** at ₹{net_rtm_price}/kWh because the ML model forecasted high peak demand, making the exchange more profitable than bilateral surge tariffs.")
    else:
        rule_text = "Tier 3: Auto-Fallback to RTM (Cutoff Safe)"
        reasons.append(f"Auto-routed remaining **{remaining_mw} MW to IEX RTM** at ₹{net_rtm_price}/kWh via Tier-3 safety fallback to prevent gate-closure failure and DSM penalties.")
    allocations.append({
        "Destination": "Power Exchange (IEX RTM)",
        "Channel": "Real-Time Market API",
        "Allocated_MW": remaining_mw,
        "Tariff_INR_kWh": net_rtm_price,
        "Block_Revenue_INR": rev,
        "AI_Decision_Rule": rule_text
    })
    remaining_mw = 0

df_alloc = pd.DataFrame(allocations)

# KPIs
total_revenue = df_alloc["Block_Revenue_INR"].sum()
avoided_dsm = surplus_mw * 1000 * 0.25 * 1.50
total_value = total_revenue + avoided_dsm

c1, c2, c3, c4 = st.columns(4)
c1.metric("ML Predicted RTM Price", f"₹{predicted_rtm_price}/kWh", f"Model Fit: {model_accuracy}%")
c2.metric("Surplus Allocated", f"{surplus_mw} MW", "0 MW Wasted")
c3.metric("Decision Latency", "1.2 Seconds", "Beat 1.5-Hr Cutoff")
c4.metric("Net Value Protected", f"₹{total_value:,.0f}", f"Incl. ₹{avoided_dsm:,.0f} DSM Saved")

st.markdown("---")

# ---------------------------------------------------------
# AI ELEMENT 3: EXPLAINABLE AI RATIONALE & VISUALS
# ---------------------------------------------------------
st.subheader("Explainable AI Decision Rationale")
rationale_html = "<div class='ai-box'><strong>Why the AI chose this schedule:</strong><ul>" + "".join([f"<li>{r}</li>" for r in reasons]) + "</ul></div>"
st.markdown(rationale_html, unsafe_allow_html=True)

st.markdown("")
left_col, right_col = st.columns([1.45, 1])

with left_col:
    st.subheader(f"Optimized Schedule — {block_label}")
    st.dataframe(df_alloc, hide_index=True)
    st.download_button(
        "Download AI Audit Log (CSV)",
        df_alloc.to_csv(index=False).encode("utf-8"),
        "gridpulse_ai_audit.csv",
        "text/csv"
    )

with right_col:
    st.subheader("AI Volume Allocation (MW)")
    st.bar_chart(df_alloc.set_index("Destination")["Allocated_MW"], color="#1E62B5")

with st.expander("View Automated WBES / IEX Exchange API Payload (JSON)"):
    st.json({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_block_id": block_id,
        "ml_predicted_rtm_price": predicted_rtm_price,
        "cutoff_status": "LOCKED_BEFORE_GATE_CLOSURE",
        "schedules": allocations
    })