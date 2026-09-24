import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Profit Analytics",
    page_icon="💰",
    layout="wide"
)

# ============================================================
# Load data
# ============================================================

profit_summary = joblib.load(
    "models/profit_analysis.pkl"
)

profit_department = pd.read_csv(
    "data/profit_by_department.csv"
)

# ============================================================
# Header
# ============================================================

st.title("💰 Profit Analytics")

st.markdown(
    """
    ### Financial Intelligence

    Explore historical profitability, loss rates, and department-level
    financial performance across the supply chain.
    """
)

st.info(
    "This module provides descriptive historical analytics. "
    "It does not forecast future profit or establish causal relationships."
)

st.divider()

# ============================================================
# KPIs
# ============================================================

st.subheader("📌 Financial Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average Profit / Order",
        f"${profit_summary['mean_profit']:.2f}"
    )

with col2:
    st.metric(
        "Median Profit / Order",
        f"${profit_summary['median_profit']:.2f}"
    )

with col3:
    st.metric(
        "Loss Rate",
        f"{profit_summary['loss_rate']:.1f}%"
    )

with col4:
    st.metric(
        "Total Observed Profit",
        f"${profit_summary['total_profit']:,.0f}"
    )

st.divider()

# ============================================================
# Department Analysis
# ============================================================

st.subheader("🏢 Department Performance")

col1, col2 = st.columns(2)

with col1:

    st.markdown("#### Average Profit per Order")

    profit_chart = (
        profit_department
        .set_index("department_name")["mean_profit"]
        .sort_values()
    )

    st.bar_chart(
        profit_chart,
        use_container_width=True
    )

with col2:

    st.markdown("#### Loss Rate")

    loss_chart = (
        profit_department
        .set_index("department_name")["loss_rate"]
        .sort_values()
    )

    st.bar_chart(
        loss_chart,
        use_container_width=True
    )

st.divider()

# ============================================================
# Detailed table
# ============================================================

st.subheader("📋 Department Financial Summary")

display_df = profit_department.copy()

display_df["mean_profit"] = display_df["mean_profit"].round(2)
display_df["loss_rate"] = display_df["loss_rate"].round(2)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ============================================================
# Profit range
# ============================================================

st.subheader("📊 Observed Profit Range")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Minimum Profit",
        f"${profit_summary['min_profit']:.2f}"
    )

with col2:
    st.metric(
        "Median Profit",
        f"${profit_summary['median_profit']:.2f}"
    )

with col3:
    st.metric(
        "Maximum Profit",
        f"${profit_summary['max_profit']:.2f}"
    )

st.divider()

# ============================================================
# Insights
# ============================================================

st.subheader("💡 Financial Insights")

st.markdown(
    f"""
    - The average observed profit per order is
      **${profit_summary['mean_profit']:.2f}**.
    - The median observed profit is
      **${profit_summary['median_profit']:.2f}**.
    - Approximately **{profit_summary['loss_rate']:.1f}%**
      of orders are loss-making.
    - Total observed profit across the dataset is approximately
      **${profit_summary['total_profit']:,.0f}**.
    - Profitability differs across departments, providing areas
      for further operational investigation.
    """
)

st.caption(
    "Source: DataCo Smart Supply Chain dataset. "
    "All figures represent observed historical data."
)