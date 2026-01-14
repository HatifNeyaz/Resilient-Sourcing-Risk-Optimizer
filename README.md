# 🛡️ Resilient Sourcing & Risk Optimizer
**Optimization-as-a-Service (OaaS) for Global Procurement**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://resilient-sourcing-risk-optimizer-fxn6g63uy8o6btxf4hhmgj.streamlit.app/)

## 📌 Project Overview
In a world of volatile supply chains, picking the "cheapest" supplier is no longer enough. This Decision Support System (DSS) helps procurement managers balance **Cost**, **Operational Efficiency**, and **Risk Resilience**.

The application uses a tri-phase mathematical approach to move from raw supplier data to a "Stress-Tested" purchase plan.

## 🏗️ System Architecture
The following diagram illustrates the decision-making pipeline implemented in this tool:

graph TD
    %% Node Definitions
    A[Raw Supplier Data]
    B{Phase 1: Evaluation}
    C[Objective Efficiency]
    D[Strategic Fit Weights]
    E{Phase 2: Execution}
    F[Optimal Order Allocation]
    G{Phase 3: Resilience}
    H[Risk & Shortfall Analysis]
    I[Executive Decision Support]

    %% Flow
    A --> B
    B -->|DEA| C
    B -->|AHP| D
    C & D --> E
    E -->|Simplex Algorithm| F
    F --> G
    G -->|Monte Carlo Simulation| H
    H --> I

    %% Professional Styling
    style A font-weight:bold,color:#000,font-size:16px
    style B fill:#f9f,stroke:#333,stroke-width:2px,font-weight:bold,color:#000,font-size:16px
    style C font-weight:bold,color:#000,font-size:16px
    style D font-weight:bold,color:#000,font-size:16px
    style E fill:#bbf,stroke:#333,stroke-width:2px,font-weight:bold,color:#000,font-size:16px
    style F font-weight:bold,color:#000,font-size:16px
    style G fill:#dfd,stroke:#333,stroke-width:2px,font-weight:bold,color:#000,font-size:16px
    style H font-weight:bold,color:#000,font-size:16px
    style I font-weight:bold,color:#000,font-size:16px