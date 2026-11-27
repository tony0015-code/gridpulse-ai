import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

st.set_page_config(page_title="GridPulse AI | Predictive & Prescriptive Engine", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #FFFFFF; }
    h1, h2, h3 { color: #0F3D68; }
    .stMetric { background-color: #F4F8FC; padding: 14px; border-radius: 8px; border-left: 4px solid #1E62B5; }
    .ai-box { background-color: #EEF5FC; padding: 16px; border-radius: 8px; border: 1px solid #B8D4F2; color: #0F3D68; }
    .sync-banner { background-color: #0F3D68; color: #FFFFFF; padding: 12px 18px; border-radius: 8px; margin-bottom: 14px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. TRAIN & VALIDATE RANDOM FOREST ML MODEL
# ---------------------------------------------------------
@st.cache_resource
def build_and_validate_ml_model():
    np.random.seed(42)
    n_samples = 960  # 10 days of 96 15-min intervals
    block_num = np.random.randint(1, 97, n_samples)
    grid_demand_gw = np.random.uniform(175, 235, n_samples)
    renewable_surge = np.random.uniform(0, 150, n_samples)

    # Simulated Indian power market price dynamics (₹/kWh)
    peak_factor = np.where((block_num >= 72) & (block_num <= 88), 1.4, 0.0)
    actual_price = (
        2.2
        + 0.018 * (grid_demand_gw - 175)
        - 0.008 * renewable_surge
        + peak_factor
        + np.random.normal(0, 0.15, n_samples)
    )
    actual_price = np.round(np.clip(actual_price, 2.80, 9.50), 2)

    df_history = pd.DataFrame({
        "Time_Block_ID": block_num,
        "Grid_Demand_GW": np.round(grid_demand_gw, 1),
        "Surplus_Spike_MW": np.round(renewable_surge, 1),
        "Historical_RTM_Price_INR": actual_price
    })

    X = df_history[["Time_Block_ID", "Grid_Demand_GW", "Surplus_Spike_MW"]]
    y = df_history["Historical_RTM_Price_INR"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    r2 = round(r2_score(y_test, y_pred) * 100, 1)

    df_importance = pd.DataFrame({
        "Market Driver": ["Time Block (Peak/Off-Peak)", "Regional Grid Demand (GW)", "Excess Generation Spike (MW)"],
        "Model Weight (%)": np.round(model.feature_importances_ * 100, 1)
    }).set_index("Market Driver")

    return model, mae, r2, df_history, df_importance

ml_model, model_mae, model_r2, df_history, df_importance = build_and_validate_ml_model()

# ---------------------------------------------------------
# 2. SHARED SIDEBAR CONTROLS (CONNECTED VIA SESSION STATE)
# ---------------------------------------------------------
st.title("GridPulse AI — Predictive & Prescriptive Scheduling Engine")
st.caption("96-Block Indian Grid | Live ML Price Forecasting + Guardrail Optimization + Explainable AI")

st.sidebar.header("1. Live Grid Inputs (Updates Both Views)")
block_choice = st.sidebar.selectbox(
    "Select 15-Min Time Block (1–96)",
    [
        ("Block 11 (02:30 - 02:45 AM) [Night Off-Peak]", 11),
        ("Block 16 (03:45 - 04:00 AM) [Night Off-Peak]", 16),
        ("Block 50 (12:15 - 12:30 PM) [Solar Mid-Day]", 50),
        ("Block 78 (07:15 - 07:30 PM) [Evening Peak]", 78),
    ],
    format_func=lambda x: x[0],
    key="sb_block"
)
block_label, block_id = block_choice

surplus_mw = st.sidebar.slider("Sudden Excess Generation Spike (MW)", 10, 200, 100, 10, key="sb_surplus")
grid_demand = st.sidebar.slider("Regional Grid Demand (GW)", 175, 235, 195, 5, key="sb_demand")

st.sidebar.markdown("---")
st.sidebar.header("2. Guardrail Governance (Layer 2)")
tier2_approved = st.sidebar.radio(
    "Off-Hours Manager Action (CP-C Exception):",
    ["No Response (Timer Expires -> Auto-Fallback)", "1-Click Mobile Approved (<5 Mins)"],
    key="sb_tier2"
)

# ---------------------------------------------------------
# 3. LIVE ML INFERENCE & SESSION STATE SYNC
# ---------------------------------------------------------
input_features = pd.DataFrame([{
    "Time_Block_ID": block_id,
    "Grid_Demand_GW": grid_demand,
    "Surplus_Spike_MW": surplus_mw
}])

# Individual predictions from all 50 Decision Trees for the CURRENT sliders
tree_votes = [round(float(tree.predict(input_features.values)[0]), 2) for tree in ml_model.estimators_]
final_predicted_price = round(float(np.mean(tree_votes)), 2)
net_rtm_price = round(final_predicted_price - 0.10, 2)

# Save to st.session_state so both pages share the exact same live values
st.session_state["final_predicted_price"] = final_predicted_price
st.session_state["net_rtm_price"] = net_rtm_price
st.session_state["tree_votes"] = tree_votes

# ---------------------------------------------------------
# 4. TWO CONNECTED VIEWS (MAIN CONTROL ROOM & LIVE ML ENGINE)
# ---------------------------------------------------------
tab_main, tab_ml = st.tabs([
    "1. Main Scheduling Control Room",
    "2. Live ML Price Prediction Engine"
])

# ==================== VIEW 1: MAIN SCHEDULING PAGE ====================
with tab_main:
    tier1_customers = [
        {"name": "CP-A (Apex Steel Ltd.)", "surge_limit": 40, "surge_price": 5.20},
        {"name": "CP-B (Metro Discom)", "surge_limit": 30, "surge_price": 4.90},
    ]

    remaining_mw = surplus_mw
    allocations = []
    reasons = []

    for cp in sorted(tier1_customers, key=lambda x: x["surge_price"], reverse=True):
        if remaining_mw > 0 and cp["surge_price"] >= st.session_state["net_rtm_price"]:
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
            reasons.append(f"Routed <b>{alloc} MW to {cp['name']}</b> at ₹{cp['surge_price']}/kWh because it exceeds the ML-predicted net RTM price (₹{st.session_state['net_rtm_price']}/kWh) and stays within pre-authorized surge limits.")

    if remaining_mw > 0 and 5.50 >= st.session_state["net_rtm_price"]:
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
            reasons.append(f"Routed <b>{alloc} MW to CP-C</b> at ₹5.50/kWh after receiving fast-track mobile approval before the 5-minute cutoff timer.")

    if remaining_mw > 0:
        rev = remaining_mw * 1000 * 0.25 * st.session_state["net_rtm_price"]
        if st.session_state["net_rtm_price"] > 5.20:
            rule_text = "AI Arbitrage: RTM Price Beats Bilateral Caps"
            reasons.append(f"Routed <b>{remaining_mw} MW to IEX RTM</b> at ₹{st.session_state['net_rtm_price']}/kWh because the ML model forecasted high peak demand, making the exchange more profitable than bilateral surge tariffs.")
        else:
            rule_text = "Tier 3: Auto-Fallback to RTM (Cutoff Safe)"
            reasons.append(f"Auto-routed remaining <b>{remaining_mw} MW to IEX RTM</b> at ₹{st.session_state['net_rtm_price']}/kWh via Tier-3 safety fallback to prevent gate-closure failure and DSM penalties.")
        allocations.append({
            "Destination": "Power Exchange (IEX RTM)",
            "Channel": "Real-Time Market API",
            "Allocated_MW": remaining_mw,
            "Tariff_INR_kWh": st.session_state["net_rtm_price"],
            "Block_Revenue_INR": rev,
            "AI_Decision_Rule": rule_text
        })
        remaining_mw = 0

    df_alloc = pd.DataFrame(allocations)
    total_revenue = df_alloc["Block_Revenue_INR"].sum()
    avoided_dsm = surplus_mw * 1000 * 0.25 * 1.50
    total_value = total_revenue + avoided_dsm

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final ML Predicted Price", f"₹{st.session_state['final_predicted_price']}/kWh", f"Net RTM: ₹{st.session_state['net_rtm_price']}/kWh")
    c2.metric("Surplus Allocated", f"{surplus_mw} MW", "0 MW Wasted")
    c3.metric("Decision Latency", "1.2 Seconds", "Beat 1.5-Hr Cutoff")
    c4.metric("Net Value Protected", f"₹{total_value:,.0f}", f"Incl. ₹{avoided_dsm:,.0f} DSM Saved")

    st.markdown("---")
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
            "final_ml_predicted_price": st.session_state["final_predicted_price"],
            "cutoff_status": "LOCKED_BEFORE_GATE_CLOSURE",
            "schedules": allocations
        })

# ==================== VIEW 2: DYNAMIC ML MODEL ENGINE ====================
with tab_ml:
    st.markdown(
        f"<div class='sync-banner'><strong>LIVE SYNCED SCENARIO:</strong> {block_label} &nbsp;|&nbsp; "
        f"Grid Demand: <b>{grid_demand} GW</b> &nbsp;|&nbsp; Surplus Spike: <b>{surplus_mw} MW</b> &nbsp;→&nbsp; "
        f"<strong>Final Predicted Price: ₹{st.session_state['final_predicted_price']}/kWh</strong></div>",
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Predicted Price (Synced)", f"₹{st.session_state['final_predicted_price']}/kWh", "Sent to Main Control Room")
    m2.metric("50-Tree Vote Range", f"₹{min(tree_votes)} – ₹{max(tree_votes)}", "Live Ensemble Consensus")
    m3.metric("Model Accuracy (R²)", f"{model_r2}%", "Tested on Unseen Blocks")
    m4.metric("Average Error (MAE)", f"±₹{model_mae}/kWh", "50 Decision Trees")

    st.markdown("---")
    col_live1, col_live2 = st.columns(2)

    with col_live1:
        st.subheader("1. Live 50-Tree Ensemble Votes (Updates with Sliders)")
        st.caption(f"Each bar is one Decision Tree's price vote for Block {block_id}, {grid_demand} GW demand, and {surplus_mw} MW spike:")
        df_trees = pd.DataFrame({
            "Tree": [f"T{i+1}" for i in range(50)],
            "Predicted Price (₹/kWh)": st.session_state["tree_votes"]
        }).set_index("Tree")
        st.bar_chart(df_trees, color="#1E62B5")

    with col_live2:
        st.subheader("2. Live Price Sensitivity Curve (Demand vs. Price)")
        st.caption(f"How the ML model shifts predicted price across demand levels for Block {block_id} & {surplus_mw} MW spike:")
        demand_axis = list(range(175, 240, 5))
        curve_inputs = pd.DataFrame({
            "Time_Block_ID": [block_id] * len(demand_axis),
            "Grid_Demand_GW": demand_axis,
            "Surplus_Spike_MW": [surplus_mw] * len(demand_axis)
        })
        curve_preds = np.round(ml_model.predict(curve_inputs), 2)
        df_curve = pd.DataFrame({
            "Grid Demand (GW)": demand_axis,
            "ML Predicted Price (₹/kWh)": curve_preds
        }).set_index("Grid Demand (GW)")
        st.line_chart(df_curve, color="#0F3D68")

    st.markdown("---")
    col_bot1, col_bot2 = st.columns([1, 1.3])

    with col_bot1:
        st.subheader("3. Feature Importance (Model Weights)")
        st.caption("Relative weight assigned by the Random Forest to each input:")
        st.bar_chart(df_importance, color="#38A3A5")

    with col_bot2:
        st.subheader("4. Historical 96-Block Training Data Sample")
        st.caption("Sample records from the 960 15-minute intervals used to train the model:")
        st.dataframe(df_history.head(6), hide_index=True)
