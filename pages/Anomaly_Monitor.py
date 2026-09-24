import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Anomaly Monitor",
    page_icon="🚨",
    layout="wide"
)

# ============================================================
# Load models and reference statistics
# ============================================================

model = joblib.load(
    "models/isolation_forest.pkl"
)

scaler = joblib.load(
    "models/scaler_anomaly.pkl"
)

reference_stats = pd.read_csv(
    "data/anomaly_reference_stats.csv",
    index_col=0
).T

# ============================================================
# Header
# ============================================================

st.title("🚨 Supply Chain Anomaly Monitor")

st.markdown(
    """
    ### Post-Transaction Anomaly Detection

    Identify unusual financial and operational transaction patterns
    using an **Isolation Forest** model.
    """
)

st.info(
    "This module analyzes post-transaction information. "
    "A detected anomaly indicates an unusual pattern and does not "
    "automatically mean fraud or an error."
)

st.divider()

# ============================================================
# Transaction Information
# ============================================================

st.subheader("📦 Transaction Information")

col1, col2 = st.columns(2)

with col1:

    st.markdown("#### Transaction Values")

    sales = st.number_input(
        "Sales",
        min_value=0.0,
        value=200.0
    )

    order_total = st.number_input(
        "Order Total",
        min_value=0.0,
        value=180.0
    )

    product_price = st.number_input(
        "Product Price",
        min_value=0.0,
        value=150.0
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=1,
        step=1
    )

    discount = st.number_input(
        "Discount",
        min_value=0.0,
        value=20.0
    )

with col2:

    st.markdown("#### Financial & Location Information")

    discount_rate = st.number_input(
        "Discount Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.10
    )

    profit = st.number_input(
        "Order Profit",
        value=25.0
    )

    profit_ratio = st.number_input(
        "Profit Ratio",
        value=0.13
    )

    latitude = st.number_input(
        "Latitude",
        value=34.05
    )

    longitude = st.number_input(
        "Longitude",
        value=-118.24
    )

st.divider()

# ============================================================
# Detection
# ============================================================

if st.button(
    "🔎 Detect Anomaly",
    use_container_width=True,
    type="primary"
):

    input_data = pd.DataFrame([{
        "sales": sales,
        "order_item_total": order_total,
        "order_item_product_price": product_price,
        "order_item_quantity": quantity,
        "order_item_discount": discount,
        "order_item_discount_rate": discount_rate,
        "order_profit_per_order": profit,
        "order_item_profit_ratio": profit_ratio,
        "latitude": latitude,
        "longitude": longitude
    }])

    # ========================================================
    # Model prediction
    # ========================================================

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(
        input_scaled
    )[0]

    score = model.decision_function(
        input_scaled
    )[0]

    st.divider()

    # ========================================================
    # Result
    # ========================================================

    st.subheader("🎯 Detection Result")

    if prediction == -1:

        st.error(
            "🚨 ANOMALOUS TRANSACTION"
        )

        st.metric(
            "Anomaly Score",
            f"{score:.4f}"
        )

        st.warning(
            "This transaction contains characteristics that are "
            "unusual compared with patterns learned by the "
            "anomaly detector."
        )

    else:

        st.success(
            "✅ NORMAL TRANSACTION"
        )

        st.metric(
            "Anomaly Score",
            f"{score:.4f}"
        )

        st.success(
            "The transaction is consistent with the patterns "
            "learned by the anomaly detector."
        )

    st.divider()

    # ========================================================
    # Reference comparison
    # ========================================================

    st.subheader("📊 Reference Distribution")

    st.write(
        """
        Compare the transaction values with the statistical
        distribution of the data used to train the anomaly detector.
        """
    )

    comparison = []

    for feature in input_data.columns:

        value = input_data[feature].iloc[0]

        mean = reference_stats.loc[
            feature, "mean"
        ]

        std = reference_stats.loc[
            feature, "std"
        ]

        minimum = reference_stats.loc[
            feature, "min"
        ]

        maximum = reference_stats.loc[
            feature, "max"
        ]

        if std != 0:
            z_score = (value - mean) / std
        else:
            z_score = 0

        comparison.append({
            "Feature": feature,
            "Input": value,
            "Training Mean": mean,
            "Training Std": std,
            "Training Min": minimum,
            "Training Max": maximum,
            "Standardized Deviation": z_score
        })

    comparison_df = pd.DataFrame(
        comparison
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The reference statistics provide contextual information "
        "about the training distribution. They do not represent "
        "feature importance or explain the Isolation Forest decision."
    )

    st.divider()

    # ========================================================
    # Interpretation
    # ========================================================

    st.subheader("📖 How to Interpret the Result")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### 🚨 Anomaly Detection")

        st.write(
            """
            Isolation Forest identifies observations that are
            unusual relative to the patterns learned from the
            training data.
            """
        )

    with col2:

        st.markdown("#### 📊 Reference Distribution")

        st.write(
            """
            The reference statistics show whether individual
            transaction values are close to or far from the
            historical training distribution.
            """
        )

    st.warning(
        "An anomaly should be investigated using additional "
        "business context. It should not automatically be "
        "interpreted as fraud, an error, or misconduct."
    )

st.caption(
    "Model: Isolation Forest • Analysis type: Unsupervised ML • "
    "Use case: Post-transaction monitoring"
)