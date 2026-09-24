# 📦 AI Supply Chain Intelligence Copilot

An AI-powered supply chain intelligence platform combining **machine learning, explainable AI, anomaly detection, financial analytics, and Retrieval-Augmented Generation (RAG)** to support supply-chain monitoring and decision-making.

## 🚀 Overview

The **AI Supply Chain Intelligence Copilot** transforms historical supply-chain data and operational knowledge into an interactive intelligence platform.

The application combines:

* 🚚 Delivery risk prediction
* 🔎 Explainable AI with SHAP
* 💰 Profitability analytics
* 🚨 Unsupervised anomaly detection
* 📚 Retrieval-Augmented Generation (RAG)
* 🤖 LLM-powered supply-chain assistance
* 📊 Interactive Streamlit dashboards

The project is designed as a practical **Machine Learning / Data Science application**, connecting predictive models with an AI-powered user interface.

---

## 🏗️ System Architecture

```text
                    DataCo Supply Chain Dataset
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
      Delivery Risk     Profit Analytics   Anomaly Detection
        XGBoost            Historical       Isolation Forest
             │               Analysis              │
             │                                      │
             ▼                                      ▼
          SHAP                              Reference Statistics
      Explainability
             │
             └──────────────┐
                            │
                            ▼
                  AI Supply Chain Copilot
                            │
                     RAG + FAISS
                            │
                            ▼
                    LLM-generated Answer
                            │
                            ▼
                     Streamlit Interface
```

---

## 🤖 Main Features

### 🚚 1. Delivery Risk Prediction

The application uses an **XGBoost Classifier** to estimate the probability of late-delivery risk for an order.

The model uses order-level information such as:

* Shipping mode
* Product category
* Product
* Customer segment
* Customer country
* Order country
* Order region
* Market
* Product price
* Quantity
* Discount
* Discount rate
* Sales
* Order total
* Geographic coordinates
* Order month
* Transaction type

The model outputs:

* Predicted class
* Late-delivery probability

### 🔎 Explainable AI

**SHAP (SHapley Additive exPlanations)** is used to explain individual delivery-risk predictions.

The explanation identifies features that contributed positively or negatively to the specific model prediction.

> SHAP values describe model behavior and should not be interpreted as causal effects.

---

### 💰 2. Profit Analytics

The platform provides descriptive financial intelligence based on historical order profitability.

The dashboard includes:

* Average profit per order
* Median profit per order
* Loss rate
* Total observed profit
* Minimum profit
* Maximum profit
* Profitability by department
* Department-level loss rates

Profit is treated as a **descriptive analytics problem** rather than a core prediction task because the tested regression models provided limited predictive value.

---

### 🚨 3. Anomaly Detection

The Anomaly Monitor uses an **Isolation Forest** to identify unusual post-transaction patterns.

The detector analyzes:

* Sales
* Order total
* Product price
* Quantity
* Discount
* Discount rate
* Realized profit
* Profit ratio
* Latitude
* Longitude

The application also provides a **reference distribution comparison**, showing how an input transaction compares with the statistical distribution of the training data.

An anomaly does not automatically indicate fraud or an error. It indicates an unusual pattern that should be investigated using additional business context.

---

### 🤖 4. AI Supply Chain Copilot

The AI Copilot combines **Retrieval-Augmented Generation (RAG)** with the project's supply-chain knowledge base.

The pipeline is:

```text
User Question
      │
      ▼
Sentence Transformer
      │
      ▼
Vector Embedding
      │
      ▼
FAISS Similarity Search
      │
      ▼
Relevant Document Chunks
      │
      ▼
LLM
      │
      ▼
Grounded Answer
```

The Copilot can answer questions using the project's supply-chain documentation and provide the retrieved source context.

It can also combine:

* Delivery-risk predictions
* SHAP explanations
* Knowledge-base information
* LLM-generated explanations

---

## 📊 Model Performance

The delivery-risk XGBoost model was evaluated on a held-out test set of **27,078 observations**.

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 70.32% |
| Precision | 84.29% |
| Recall    | 56.38% |
| F1-score  | 67.57% |
| ROC-AUC   | 77.02% |

The model shows relatively high precision with lower recall at the default 0.5 classification threshold.

These metrics describe performance on the held-out test set and should not be interpreted as guaranteed performance on future data.

---

## 🧰 Technology Stack

### Machine Learning

* Python
* Scikit-learn
* XGBoost
* Isolation Forest
* Joblib

### Explainability

* SHAP

### RAG / NLP

* Sentence Transformers
* FAISS
* OpenRouter LLM API

### Application

* Streamlit
* Pandas
* NumPy

---

## 📁 Project Structure

```text
AI-Supply-Chain-Intelligence-Copilot/
│
├── app.py
├── rag_engine.py
├── llm.py
├── ml_tools.py
├── explainability.py
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── xgb_delivery_risk.pkl
│   ├── delivery_risk_performance.pkl
│   ├── isolation_forest.pkl
│   ├── scaler_anomaly.pkl
│   └── profit_analysis.pkl
│
├── data/
│   ├── profit_by_department.csv
│   └── anomaly_reference_stats.csv
│
├── documents/
│   ├── shipping_policy.md
│   └── ml_model_guide.md
│
└── pages/
    ├── Dashboard.py
    ├── Delivery_Risk.py
    ├── Profit_Analytics.py
    ├── Anomaly_Monitor.py
    ├── Model_Performance.py
    └── AI_Copilot.py
```

---

## 📚 Dataset

This project uses the **DataCo Smart Supply Chain for Big Data Analysis** dataset from Kaggle.

The raw dataset is not included in this repository.

The repository contains the trained machine-learning models and derived analytical artifacts required by the Streamlit application.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Supply-Chain-Intelligence-Copilot.git
```

Move into the project:

```bash
cd AI-Supply-Chain-Intelligence-Copilot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

The `.env` file must **never be committed to GitHub**.

It is already excluded through `.gitignore`.

---

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## ⚠️ Limitations

* Delivery-risk probabilities are model estimates, not guarantees.
* SHAP explanations describe model behavior and do not establish causality.
* Profit analytics are descriptive and are not future profit forecasts.
* Isolation Forest detects unusual patterns but does not prove fraud.
* RAG answers depend on the quality and coverage of the knowledge base.
* LLM responses may vary depending on the selected model and API availability.
* Model performance on historical test data may differ from performance on future operational data.

---

## 🎯 Project Objective

The goal of this project is to demonstrate how **Machine Learning, Explainable AI, anomaly detection, RAG, and LLMs can be integrated into a practical business intelligence application**.

Rather than building a standalone prediction model, the project focuses on the complete path from:

```text
Data
 ↓
Machine Learning
 ↓
Model Evaluation
 ↓
Explainability
 ↓
Knowledge Retrieval
 ↓
LLM
 ↓
Interactive Application
```

---

## 👨‍💻 Author

**Taha El Ansari**

Master's student in **Applied Mathematics and Machine Intelligence (MAIM)**.

Interested in:

* Machine Learning
* Data Science
* Artificial Intelligence
* Explainable AI
* Applied Mathematics
* Intelligent Decision Support Systems
