import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Supply Chain Dashboard",
    page_icon="📊",
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

st.title("📊 Supply Chain Intelligence Dashboard")

st.markdown(
    """
    ### Executive Overview

    A high-level view of historical supply-chain profitability
    and financial performance.
    """
)

st.divider()

# ============================================================
# KPI Cards
# ============================================================

st.subheader("📌 Key Performance Indicators")

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
        "Loss-Making Orders",
        f"{profit_summary['loss_rate']:.1f}%"
    )

with col4:
    st.metric(
        "Total Observed Profit",
        f"${profit_summary['total_profit']:,.0f}"
    )

st.divider()

# ============================================================
# Profitability by Department
# ============================================================

st.subheader("🏢 Profitability by Department")

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

    st.markdown("#### Loss Rate by Department")

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
# Department Table
# ============================================================

st.subheader("📋 Department Performance")

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
# Financial Insights
# ============================================================

st.subheader("💡 Financial Insights")

col1, col2 = st.columns(2)

with col1:

    st.markdown("#### Observed Financial Performance")

    st.write(
        f"""
        The dataset contains an average observed profit of
        **${profit_summary['mean_profit']:.2f} per order**.

        The median profit is **${profit_summary['median_profit']:.2f}**,
        while approximately **{profit_summary['loss_rate']:.1f}%**
        of orders are loss-making.
        """
    )

with col2:

    st.markdown("#### Business Interpretation")

    st.write(
        """
        Profitability varies across departments. These differences
        can be used to identify areas that deserve further
        operational investigation.
        """
    )

st.info(
    "These results are descriptive historical analytics. "
    "They do not establish causal relationships or predict future profit."
)

st.caption(
    "Source: DataCo Smart Supply Chain dataset"
)