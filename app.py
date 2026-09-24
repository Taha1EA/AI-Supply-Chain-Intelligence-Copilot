import streamlit as st

st.set_page_config(
    page_title="AI Supply Chain Intelligence Copilot",
    page_icon="📦",
    layout="wide"
)

# ============================================================
# Header
# ============================================================

st.title("📦 AI Supply Chain Intelligence Copilot")

st.markdown(
    """
    ### Intelligent Supply Chain Monitoring & Decision Support

    An AI-powered platform combining **machine learning, anomaly
    detection, financial analytics, explainable AI, and RAG-based
    document intelligence** to support supply-chain operations.
    """
)

st.divider()

# ============================================================
# Main capabilities
# ============================================================

st.subheader("🚀 Intelligence Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🚚 Delivery Risk")
    st.write(
        "Predict the probability of late-delivery risk "
        "using an XGBoost classification model."
    )
    st.caption("XGBoost • SHAP • Classification")

with col2:
    st.markdown("### 💰 Profit Analytics")
    st.write(
        "Analyze profitability, loss rates, total profit "
        "and financial performance across departments."
    )
    st.caption("Financial Analytics • Historical Data")

with col3:
    st.markdown("### 🚨 Anomaly Detection")
    st.write(
        "Identify unusual post-transaction financial and "
        "operational patterns using Isolation Forest."
    )
    st.caption("Isolation Forest • Unsupervised ML")

st.divider()

# ============================================================
# AI Copilot
# ============================================================

st.subheader("🤖 AI Supply Chain Copilot")

st.write(
    """
    The Copilot combines **Retrieval-Augmented Generation (RAG)**
    with the project's supply-chain knowledge base to provide
    evidence-grounded answers.
    """
)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        **Capabilities**

        - 📚 Search the supply-chain knowledge base
        - 🔎 Retrieve relevant document context
        - 🤖 Generate evidence-grounded answers
        - 🚚 Explain delivery-risk predictions
        - 📊 Combine ML predictions with model explanations
        """
    )

with col2:
    st.info(
        "Use the **AI Copilot** page from the sidebar "
        "to interact with the system."
    )

st.divider()

# ============================================================
# Architecture
# ============================================================

st.subheader("🏗️ System Architecture")

st.code(
    """
DataCo Supply Chain Dataset
            │
            ├───────────────┐
            │               │
            ▼               ▼
   Delivery Risk       Financial Data
   XGBoost + SHAP           │
            │        ┌──────┴──────┐
            │        │             │
            │        ▼             ▼
            │   Profit Analytics  Anomaly Detection
            │                      Isolation Forest
            │
            └──────────┬───────────┘
                       │
                       ▼
              AI Supply Chain
                  Copilot
                       │
                 RAG + FAISS
                       │
                       ▼
                  LLM Answer
    """,
    language="text"
)

st.divider()

# ============================================================
# Technology Stack
# ============================================================

st.subheader("🛠️ Technology Stack")

tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)

with tech_col1:
    st.markdown("**Machine Learning**")
    st.write("XGBoost\nIsolation Forest")

with tech_col2:
    st.markdown("**Explainability**")
    st.write("SHAP")

with tech_col3:
    st.markdown("**RAG**")
    st.write("Sentence Transformers\nFAISS")

with tech_col4:
    st.markdown("**Application**")
    st.write("Python\nStreamlit")

st.divider()

# ============================================================
# Model limitations
# ============================================================

st.subheader("⚠️ Important Notes")

st.markdown(
    """
    - Delivery-risk probabilities are **model estimates**, not guarantees.
    - SHAP values describe **model behavior**, not causal relationships.
    - Profit Analytics is **descriptive** and is not a profit forecast.
    - Anomaly detection identifies unusual patterns but does not prove fraud.
    - AI Copilot answers depend on the quality of the retrieved knowledge base.
    """
)

st.caption(
    "AI Supply Chain Intelligence Copilot • "
    "Machine Learning & Data Science Project"
)