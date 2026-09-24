import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

def generate_answer(question, results):

    context = "\n\n".join(
        f"Source: {result['source']}\n{result['text']}"
        for result in results
    )

    prompt = f"""
You are an AI Supply Chain Intelligence Copilot.

Answer the user's question using ONLY the information provided
in the context below.

If the answer cannot be found in the context, say:
"I don't have enough information in the knowledge base to answer this."

Do not invent facts.

Context:
{context}

User question:
{question}

Give a clear and concise answer.
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openrouter/free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    print("\n===== RAW OPENROUTER RESPONSE =====")
    print(data)

    return data["choices"][0]["message"]["content"]


def explain_delivery_risk(question, prediction, probability, results):

    risk_label = (
        "higher predicted late-delivery risk"
        if prediction == 1
        else "lower predicted late-delivery risk"
    )

    context = "\n\n".join(
        f"Source: {result['source']}\n{result['text']}"
        for result in results
    )

    prompt = f"""
You are an AI Supply Chain Intelligence Copilot.

The delivery-risk model analyzed an order.

Model result:
- Prediction: {prediction}
- Late-delivery probability: {probability:.2%}
- Interpretation: {risk_label}

Use the knowledge base below to explain the result.

Important rules:
- Do not change or recalculate the model probability.
- Do not claim that the order will definitely be late.
- Explain that the probability is a model estimate.
- Use only information supported by the context.
- If the context does not contain enough information, say so.

Knowledge base:
{context}

User question:
{question}

Give a concise business-oriented explanation.
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openrouter/free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]





def explain_delivery_risk_with_shap(
    question,
    prediction,
    probability,
    shap_explanation,
    results
):

    risk_label = (
        "higher predicted late-delivery risk"
        if prediction == 1
        else "lower predicted late-delivery risk"
    )

    shap_context = "\n".join(
        f"- Feature: {row['feature']}; "
        f"Value: {row['value']}; "
        f"SHAP contribution: {row['shap_value']:.3f}; "
        f"Effect: {row['effect']}"
        for _, row in shap_explanation.iterrows()
    )

    context = "\n\n".join(
        f"Source: {result['source']}\n{result['text']}"
        for result in results
    )

    prompt = f"""
You are an AI Supply Chain Intelligence Copilot.

Analyze the delivery-risk prediction using the model result,
SHAP feature contributions, and the retrieved knowledge base.

MODEL RESULT:
- Prediction: {prediction}
- Late-delivery probability: {probability:.2%}
- Interpretation: {risk_label}

SHAP EXPLANATION:
{shap_context}

KNOWLEDGE BASE:
{context}
IMPORTANT RULES:

1. Report the model probability exactly as provided.
2. Explain prediction = 0 as "lower predicted late-delivery risk"
   and prediction = 1 as "higher predicted late-delivery risk".
3. Do not create additional risk categories such as:
   "low", "moderate", "high", or "low-to-moderate"
   unless the application explicitly provides such thresholds.
4. SHAP values describe how features contributed to the model's
   prediction. They do NOT establish causality.
5. Do not invent explanations for why a feature has its SHAP value.
6. Do not infer real-world logistics facts from latitude/longitude
   unless those facts are explicitly present in the knowledge base.
7. Do not claim that a shipping mode is operationally safer,
   faster, slower, or more reliable unless the knowledge base
   explicitly states this.
8. Do not recommend changing the shipping mode, carrier, destination,
   or operational process based solely on SHAP.
9. Do not invent missing factors.
10. Clearly separate:
    - Model prediction
    - SHAP model contributions
    - Knowledge-base information
11. If the knowledge base does not support an interpretation,
    explicitly say that the information is not available.
12. Keep the answer concise and evidence-based.

USER QUESTION:
{question}
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openrouter/free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]