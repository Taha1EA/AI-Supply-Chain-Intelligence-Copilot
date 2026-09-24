import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Delivery Risk Model Performance")

st.write(
    "Evaluation of the XGBoost delivery-risk model "
    "on the held-out test set."
)

# Load performance results
performance = joblib.load(
    "models/delivery_risk_performance.pkl"
)

# Metrics
accuracy = performance["accuracy"]
precision = performance["precision"]
recall = performance["recall"]
f1 = performance["f1"]
roc_auc = performance["roc_auc"]
test_samples = performance["test_samples"]

# KPI cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Accuracy", f"{accuracy:.1%}")

with col2:
    st.metric("Precision", f"{precision:.1%}")

with col3:
    st.metric("Recall", f"{recall:.1%}")

with col4:
    st.metric("F1-score", f"{f1:.1%}")

with col5:
    st.metric("ROC-AUC", f"{roc_auc:.1%}")

st.caption(
    f"Evaluation performed on {test_samples:,} held-out test observations."
)

st.divider()

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = np.array(
    performance["confusion_matrix"]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual: No Risk", "Actual: Risk"],
    columns=["Predicted: No Risk", "Predicted: Risk"]
)

st.dataframe(
    cm_df,
    use_container_width=True
)

st.divider()

st.divider()

st.subheader("📈 ROC Curve")

y_test = np.array(performance["y_test"])
y_prob = np.array(performance["y_prob"])

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

roc_data = pd.DataFrame({
    "False Positive Rate": fpr,
    "True Positive Rate": tpr
})

st.line_chart(
    roc_data,
    x="False Positive Rate",
    y="True Positive Rate"
)

st.caption(
    f"ROC-AUC = {roc_auc:.3f}. "
    "The curve shows the trade-off between detecting high-risk "
    "orders and generating false alarms as the classification "
    "threshold changes."
)


# Interpretation
st.subheader("📖 Metric Interpretation")

st.markdown("""
- **Accuracy** measures the proportion of all predictions that are correct.
- **Precision** measures how often orders predicted as high risk are actually high risk.
- **Recall** measures how many actual high-risk orders are detected.
- **F1-score** balances precision and recall.
- **ROC-AUC** measures the model's ability to distinguish between the two classes across classification thresholds.
""")

st.info(
    "The model shows relatively high precision but lower recall. "
    "This means that when the model flags an order as high risk, "
    "the prediction is relatively precise, while some actual high-risk "
    "orders are not detected at the default 0.5 threshold."
)

st.caption(
    "These metrics describe performance on the held-out test set and "
    "should not be interpreted as guaranteed performance on future data."
)