# 🛡️ Resilient Sourcing & Risk Optimizer
**Optimization-as-a-Service (OaaS) for Global Procurement**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://resilient-sourcing-risk-optimizer-fxn6g63uy8o6btxf4hhmgj.streamlit.app/)

## 📌 Project Overview
In a world of volatile supply chains, picking the "cheapest" supplier is no longer enough. This Decision Support System (DSS) helps procurement managers balance **Cost**, **Operational Efficiency**, and **Risk Resilience**.

The application uses a tri-phase mathematical approach to move from raw supplier data to a "Stress-Tested" purchase plan.

## 🏗️ System Architecture
The following diagram illustrates the decision-making pipeline implemented in this tool:

```mermaid
graph TD
    A[Raw Supplier Data] --> B{Phase 1: Evaluation}
    B -->|DEA| C[Objective Efficiency]
    B -->|AHP| D[Strategic Fit Weights]
    
    C & D --> E{Phase 2: Execution}
    E -->|Simplex Algorithm| F[Optimal Order Allocation]
    
    F --> G{Phase 3: Resilience}
    G -->|Monte Carlo Simulation| H[Risk & Shortfall Analysis]
    
    H --> I[Executive Decision Support]
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style G fill:#dfd,stroke:#333,stroke-width:2px