# AI Supply Chain Intelligence Copilot — ML Model Guide

## Delivery Risk Prediction

The Delivery Risk module predicts whether an order is likely to have a late delivery risk.

The target variable is `late_delivery_risk`.

The model used is XGBoost Classifier.

The model uses information available around the order level, including:

- shipping mode
- product category
- product name
- customer country
- customer segment
- department
- market
- order country
- order region
- latitude
- longitude
- order item discount
- discount rate
- product price
- quantity
- sales
- order item total
- order month
- order type

The model outputs a probability for the late-delivery-risk class.

For example, a predicted probability of 0.998 means that the model estimates a 99.8% probability for the positive class under the model's learned patterns.

This probability is not a guarantee that the order will actually be late.

The model is intended to support operational risk monitoring and prioritization.

## Profit Analytics

The Profit Analytics module provides descriptive financial analysis.

The main financial variable is `order_profit_per_order`.

The dashboard reports:

- average profit per order
- median profit per order
- loss rate
- total observed profit
- minimum profit
- maximum profit
- profitability by department

These results describe observed historical data.

They should not be interpreted as causal effects or guaranteed future profit forecasts.

## Anomaly Detection

The Anomaly Monitor uses Isolation Forest.

It is an unsupervised machine-learning method used to identify unusual transactions.

The model analyzes:

- sales
- order item total
- product price
- quantity
- discount
- discount rate
- realized profit
- profit ratio
- latitude
- longitude

The anomaly detector is designed for post-transaction monitoring because realized profit and profit ratio are available after the transaction.

An anomaly does not automatically mean fraud or an error.

It means that the transaction has characteristics that are unusual compared with the data used to train the detector.

Flagged transactions should therefore be reviewed using additional business context.

## RAG and AI Copilot

The AI Copilot uses Retrieval-Augmented Generation (RAG).

The pipeline is:

1. The user asks a question.
2. Sentence Transformers converts the question into an embedding.
3. FAISS searches for similar document chunks.
4. The most relevant chunks are retrieved.
5. The retrieved context is provided to an LLM.
6. The LLM generates an answer using the retrieved information.

The Copilot should not invent information that is not supported by the retrieved context.

## Model Limitations

Machine-learning predictions are estimates and should not be treated as certainty.

Delivery-risk predictions depend on the quality and distribution of the training data.

Profit analytics are descriptive and do not establish causality.

Anomaly detection identifies unusual patterns but does not prove fraud, errors, or misconduct.

The AI Copilot depends on the quality of the retrieved documents and the language model.