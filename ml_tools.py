import joblib
import pandas as pd


MODEL_PATH = "models/xgb_delivery_risk.pkl"


model = joblib.load(MODEL_PATH)


def predict_delivery_risk(order_data):

    df = pd.DataFrame([order_data])

    probability = model.predict_proba(df)[0][1]

    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "probability": probability
    }