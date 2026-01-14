import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # Updated for better stability
import plotly.graph_objects as go
from engine import generate_data
from analysis import DEASolver, calculate_ahp_weights, AllocationSolver, SimulationEngine

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SupplyChain-AI Optimizer",
    page_icon="🛡️",
    layout="wide"
)

# --- DATA INITIALIZATION ---
# Session state ensures data doesn't reset when you move the slider
if 'df' not in st.session_state:
    st.session_state.df = generate_data()
df = st.session_state.df

# --- SIDEBAR: GLOBAL CONTROLS ---
st.sidebar.header("🕹️ Your Requirements")

total_demand = st.sidebar.number_input(
    "How many units do you need?", 
    min_value=1000, max_value=20000, value=5000
)

st.sidebar.divider()
st.sidebar.subheader("Strategic Priorities")
st.sidebar.write("What matters most to your business right now?")

cost_vs_time = st.sidebar.slider(
    "Cheaper Price ← Balance → Faster Speed", 
    0.1, 9.0, 1.0,
    help="Move left to save money. Move right for speed/reliability."
)

# Calculate weights from the slider (AHP)
try:
    matrix = [[1, cost_vs_time], [1/cost_vs_time, 1]]
    weights, cr = calculate_ahp_weights(matrix)
except Exception as e:
    st.error(f"AHP Calculation Error: {e}")
    weights = [0.5, 0.5]

# Visualizing the Weights in the Sidebar
st.sidebar.write("### 🎯 Current Strategy")
st.sidebar.progress(weights[0], text=f"💰 Cost Focus: {weights[0]*100:.1f}%")
st.sidebar.progress(weights[1], text=f"⚡ Speed Focus: {weights[1]*100:.1f}%")

# --- APP STYLING & INTRO ---
st.title("🛡️ Resilient Sourcing & Risk Optimizer")
st.markdown("""
This tool uses **Dual-Metric Ranking**:
1.  **Efficiency (DEA):** An objective math score (who is the best 'all-rounder'?).
2.  **Strategic Fit:** A score based on *your* specific preference for cost vs. speed.
""")

# --- TABBED INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📊 1. Supplier Scorecard", "🎯 2. The Smart Order Plan", "🌪️ 3. Crisis Simulation"])

with tab1:
    st.header("Step 1: Supplier Evaluation")
    
    # 1. CALCULATE OBJECTIVE EFFICIENCY (DEA)
    solver = DEASolver(df, inputs=['Unit_Cost', 'Lead_Time'], outputs=['Reliability'])
    df['Efficiency_Score'] = solver.get_all_scores()
    
    # 2. CALCULATE STRATEGIC FIT
    c_min, c_max = df['Unit_Cost'].min(), df['Unit_Cost'].max()
    t_min, t_max = df['Lead_Time'].min(), df['Lead_Time'].max()
    norm_cost = 1 - (df['Unit_Cost'] - c_min) / (c_max - c_min)
    norm_time = 1 - (df['Lead_Time'] - t_min) / (t_max - t_min)
    df['Strategic_Fit'] = (norm_cost * weights[0]) + (norm_time * weights[1])
    
    st.write("### Comparison Table")
    st.info("💡 **Efficiency** is the objective math score. **Strategic Fit** changes as you move the sidebar slider.")
    
    st.dataframe(
        df[['Supplier', 'Unit_Cost', 'Lead_Time', 'Reliability', 'Efficiency_Score', 'Strategic_Fit']]
        .sort_values('Strategic_Fit', ascending=False)
        .style.background_gradient(subset=['Efficiency_Score'], cmap="Blues")
        .background_gradient(subset=['Strategic_Fit'], cmap="YlGn")
        .format(precision=2),
        use_container_width=True
    )

with tab2:
    st.header("Step 2: Optimal Purchase Plan")
    st.markdown("We use the **Efficiency Score** as the driver to ensure you buy from the best overall performers.")
    
    eff_importance = st.select_slider(
        "Optimization Strategy:",
        options=[0, 25, 50, 75, 100],
        value=50,
        help="Higher values focus more on efficient suppliers than just raw price."
    )
    
    if st.button("🚀 Find the Best Order Split"):
        alloc_solver = AllocationSolver(df, total_demand)
        allocations, status = alloc_solver.solve_allocation(lambda_weight=eff_importance)
        
        if status == "Optimal":
            df['Allocated_Qty'] = df.index.map(allocations)
            st.session_state.allocations = allocations 
            st.session_state.optimized = True
            
            # Metrics
            total_cost = (df['Allocated_Qty'] * df['Unit_Cost']).sum()
            avg_eff = (df['Allocated_Qty'] * df['Efficiency_Score']).sum() / total_demand
            avg_fit = (df['Allocated_Qty'] * df['Strategic_Fit']).sum() / total_demand
            
            st.subheader("Plan Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Order Cost", f"${total_cost:,.2f}")
            m2.metric("Plan Efficiency", f"{avg_eff:.2f}/1.00")
            m3.metric("Strategic Alignment", f"{avg_fit:.2f}/1.00")
            
            st.bar_chart(df.set_index('Supplier')['Allocated_Qty'])
            st.write("### Recommended Purchase Order")
            st.table(df[df['Allocated_Qty'] > 0][['Supplier', 'Allocated_Qty', 'Unit_Cost', 'Efficiency_Score', 'Strategic_Fit']])
        else:
            st.error("❌ Demand too high for available capacity.")

with tab3:
    st.header("Step 3: Risk Stress Test")
    if 'optimized' not in st.session_state:
        st.warning("⚠️ Please run Step 2 first to create a plan.")
    else:
        if st.button("🎲 Run 1,000 Crisis Scenarios"):
            sim_engine = SimulationEngine(df, st.session_state.allocations)
            sim_results = sim_engine.run_risk_simulation()
            
            risk = (np.sum(sim_results < total_demand) / 1000) * 100
            
            st.subheader("Resilience Report")
            if risk > 15:
                st.error(f"⚠️ **High Risk!** {risk:.1f}% chance of supply failure.")
            else:
                st.success(f"✅ **Safe!** Only {risk:.1f}% chance of supply failure.")
            
            # Updated Histogram using Plotly Express for better stability
            fig = px.histogram(
                x=sim_results, 
                nbins=30,
                labels={'x': 'Total Units Delivered'},
                title="Distribution of Possible Outcomes",
                color_discrete_sequence=['#636EFA']
            )
            fig.add_vline(x=total_demand, line_dash="dash", line_color="red", 
                          annotation_text="Your Requirement", annotation_position="top left")
            st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.divider()
with st.expander("🛠️ Technical Details"):
    st.write("Calculations use Data Envelopment Analysis (DEA) for objective efficiency and AHP for subjective strategic weighting.")