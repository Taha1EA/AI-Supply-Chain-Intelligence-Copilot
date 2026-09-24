import streamlit as st

from rag_engine import build_index, search
from llm import generate_answer, explain_delivery_risk_with_shap
from ml_tools import predict_delivery_risk
from explainability import explain_delivery_risk as get_shap_explanation


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="AI Supply Chain Copilot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# Header
# ============================================================

st.title("🤖 AI Supply Chain Copilot")

st.markdown(
    """
    ### AI-Powered Supply Chain Intelligence

    Ask questions about supply-chain operations or analyze a specific
    order using the delivery-risk model and explainable AI.
    """
)

st.info(
    "The Copilot uses Retrieval-Augmented Generation (RAG) to ground "
    "answers in the project's knowledge base. ML predictions are "
    "estimates and should not be interpreted as certainty."
)


# ============================================================
# Load RAG system
# ============================================================

@st.cache_resource
def load_rag():

    model, index, metadata = build_index()

    return model, index, metadata


model, index, metadata = load_rag()


# ============================================================
# Section 1 — Knowledge Copilot
# ============================================================

st.divider()

st.header("💬 Knowledge Copilot")

st.write(
    "Ask a question about shipping, delivery risk, financial monitoring, "
    "or other topics covered by the knowledge base."
)

question = st.text_input(
    "Your question",
    placeholder="Example: What shipping modes are used?"
)

if st.button(
    "🔎 Ask Copilot",
    use_container_width=True,
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching the knowledge base..."
        ):

            results = search(
                question,
                model,
                index,
                metadata,
                top_k=3
            )

        with st.spinner(
            "Generating grounded answer..."
        ):

            answer = generate_answer(
                question,
                results
            )

        st.subheader("💡 Copilot Answer")

        st.markdown(answer)

        st.divider()

        st.subheader("📚 Retrieved Sources")

        for i, result in enumerate(
            results,
            start=1
        ):

            with st.expander(
                f"Source {i}: {result['source']}"
            ):

                st.write(
                    result["text"]
                )

                st.caption(
                    f"FAISS distance: "
                    f"{result['distance']:.4f}"
                )


# ============================================================
# Section 2 — Delivery Risk Intelligence
# ============================================================

st.divider()

st.header("🚚 Delivery Risk Intelligence")

st.write(
    """
    Analyze an order using the trained XGBoost model and obtain
    an explainable AI interpretation using SHAP.
    """
)


# ============================================================
# Order inputs
# ============================================================

st.subheader("📦 Order Information")

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


with col2:

    order_item_discount = st.number_input(
        "Discount",
        min_value=0.0,
        value=5.0
    )

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


with col3:

    order_item_total = st.number_input(
        "Order Total",
        min_value=0.0,
        value=45.0
    )

    order_month = st.number_input(
        "Order Month",
        min_value=1,
        max_value=12,
        value=6
    )

    customer_segment = st.selectbox(
        "Customer Segment",
        [
            "Consumer",
            "Corporate",
            "Home Office"
        ]
    )


# ============================================================
# Delivery risk analysis
# ============================================================

if st.button(
    "🚀 Analyze Delivery Risk",
    use_container_width=True,
    type="primary"
):

    order = {

        "type": "DEBIT",

        "category_name": "Men's Footwear",

        "customer_country": "Estados Unidos",

        "customer_segment": customer_segment,

        "department_name": "Apparel",

        "latitude": 34.05,

        "longitude": -118.24,

        "market": "USCA",

        "order_country": "Estados Unidos",

        "order_item_discount": order_item_discount,

        "order_item_discount_rate":
            order_item_discount_rate,

        "order_item_product_price":
            order_item_product_price,

        "order_item_quantity":
            order_item_quantity,

        "sales":
            sales,

        "order_item_total":
            order_item_total,

        "order_region": "West of USA",

        "product_name":
            "Nike Men's Dri-FIT Victory Golf Polo",

        "shipping_mode":
            shipping_mode,

        "order_month":
            order_month
    }


    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    result = predict_delivery_risk(
        order
    )

    probability = result[
        "probability"
    ]

    prediction = result[
        "prediction"
    ]


    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------

    shap_explanation = get_shap_explanation(
        order
    )


    # --------------------------------------------------------
    # Retrieve knowledge
    # --------------------------------------------------------

    with st.spinner(
        "Retrieving supply-chain knowledge..."
    ):

        results = search(
            "delivery risk probability and late delivery",
            model,
            index,
            metadata,
            top_k=3
        )


    # --------------------------------------------------------
    # Generate AI explanation
    # --------------------------------------------------------

    with st.spinner(
        "Generating explainable AI analysis..."
    ):

        explanation = (
            explain_delivery_risk_with_shap(
                question=
                    "Analyze this delivery-risk result.",
                prediction=prediction,
                probability=probability,
                shap_explanation=
                    shap_explanation,
                results=results
            )
        )


    # ========================================================
    # Prediction result
    # ========================================================

    st.divider()

    st.subheader("🎯 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Late-Delivery Probability",
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


    # ========================================================
    # AI explanation
    # ========================================================

    st.divider()

    st.subheader("🤖 AI Analysis")

    st.markdown(
        explanation
    )


    # ========================================================
    # SHAP explanation
    # ========================================================

    st.divider()

    st.subheader("🔎 Model Explanation")

    st.write(
        """
        SHAP values show how individual features contributed
        to this specific model prediction.
        """
    )

    st.dataframe(
        shap_explanation[
            [
                "feature",
                "value",
                "shap_value",
                "effect"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "SHAP values describe model behavior. They do not "
        "establish causal relationships."
    )


    # ========================================================
    # Retrieved evidence
    # ========================================================

    st.divider()

    st.subheader("📚 Knowledge Base Evidence")

    for i, result in enumerate(
        results,
        start=1
    ):

        with st.expander(
            f"Evidence {i}: {result['source']}"
        ):

            st.write(
                result["text"]
            )

            st.caption(
                f"FAISS distance: "
                f"{result['distance']:.4f}"
            )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "AI Supply Chain Intelligence Copilot • "
    "RAG + XGBoost + SHAP + FAISS"
)