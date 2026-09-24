import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Delivery Risk",
    page_icon="🚚",
    layout="wide"
)

# ============================================================
# Load model
# ============================================================

model = joblib.load(
    "models/xgb_delivery_risk.pkl"
)

# ============================================================
# Header
# ============================================================

st.title("🚚 Delivery Risk Prediction")

st.markdown(
    """
    ### Predictive Delivery Risk Monitoring

    Estimate the probability that an order will be associated
    with **late-delivery risk** using the trained XGBoost model.
    """
)

st.info(
    "The probability is a model estimate based on patterns learned "
    "from historical data. It is not a guarantee that an order will "
    "actually be late."
)

st.divider()

# ============================================================
# Order Information
# ============================================================

st.subheader("📦 Order Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("#### Product")

    product_name = st.selectbox(
        "Product",
        [
            "Nike Men's Dri-FIT Victory Golf Polo",
            "Perfect Fitness Perfect Rip Deck",
            "Under Armour Girls' Toddler Playtime Set",
            "Other"
        ]
    )

    category_name = st.selectbox(
        "Category",
        [
            "Men's Footwear",
            "Women's Apparel",
            "Cleats",
            "Other"
        ]
    )

    department_name = st.selectbox(
        "Department",
        [
            "Apparel",
            "Footwear",
            "Fan Shop",
            "Other"
        ]
    )

with col2:

    st.markdown("#### Customer")

    customer_segment = st.selectbox(
        "Customer Segment",
        [
            "Consumer",
            "Corporate",
            "Home Office"
        ]
    )

    customer_country = st.text_input(
        "Customer Country",
        value="Estados Unidos"
    )

    order_country = st.text_input(
        "Order Country",
        value="Estados Unidos"
    )

with col3:

    st.markdown("#### Location")

    order_region = st.text_input(
        "Order Region",
        value="West of USA"
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
# Shipping & Transaction
# ============================================================

st.subheader("🚚 Shipping & Transaction")

col1, col2, col3 = st.columns(3)

with col1:

    shipping_mode = st.selectbox(
        "Shipping Mode",
        [
            "Standard Class",
            "Second Class",
            "First Class",
            "Same Day"
        ]
    )

    market = st.selectbox(
        "Market",
        [
            "USCA",
            "LATAM",
            "Europe",
            "Pacific Asia",
            "Africa"
        ]
    )

    order_month = st.number_input(
        "Order Month",
        min_value=1,
        max_value=12,
        value=6
    )

with col2:

    order_item_product_price = st.number_input(
        "Product Price",
        min_value=0.0,
        value=50.0
    )

    order_item_quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=1
    )

    order_item_discount = st.number_input(
        "Discount",
        min_value=0.0,
        value=5.0
    )

with col3:

    order_item_discount_rate = st.number_input(
        "Discount Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.10
    )

    sales = st.number_input(
        "Sales",
        min_value=0.0,
        value=50.0
    )

    order_item_total = st.number_input(
        "Order Total",
        min_value=0.0,
        value=45.0
    )

st.divider()

# ============================================================
# Transaction Type
# ============================================================

st.subheader("💳 Transaction")

type_value = st.selectbox(
    "Payment / Transaction Type",
    [
        "DEBIT",
        "TRANSFER",
        "PAYMENT",
        "CASH"
    ]
)

st.divider()

# ============================================================
# Prediction
# ============================================================

if st.button(
    "🚀 Analyze Delivery Risk",
    use_container_width=True,
    type="primary"
):

    order = {
        "type": type_value,
        "category_name": category_name,
        "customer_country": customer_country,
        "customer_segment": customer_segment,
        "department_name": department_name,
        "latitude": latitude,
        "longitude": longitude,
        "market": market,
        "order_country": order_country,
        "order_item_discount": order_item_discount,
        "order_item_discount_rate": order_item_discount_rate,
        "order_item_product_price": order_item_product_price,
        "order_item_quantity": order_item_quantity,
        "sales": sales,
        "order_item_total": order_item_total,
        "order_region": order_region,
        "product_name": product_name,
        "shipping_mode": shipping_mode,
        "order_month": order_month
    }

    probability = model.predict_proba(
        pd.DataFrame([order])
    )[0][1]

    prediction = int(probability >= 0.5)

    st.divider()

    # ========================================================
    # Result
    # ========================================================

    st.subheader("🎯 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Late-Delivery Risk Probability",
            f"{probability:.1%}"
        )

    with col2:

        if prediction == 1:

            st.error(
                "⚠️ Higher predicted late-delivery risk"
            )

        else:

            st.success(
                "✅ Lower predicted late-delivery risk"
            )

    st.divider()

    # ========================================================
    # Interpretation
    # ========================================================

    st.subheader("📖 Interpretation")

    st.write(
        f"""
        The model estimates a **{probability:.1%} probability**
        for the late-delivery-risk class.

        This probability represents the model's estimate based on
        patterns learned from historical observations.
        """
    )

    st.caption(
        "A model probability should not be interpreted as certainty "
        "about the actual delivery outcome."
    )