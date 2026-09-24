import joblib
import pandas as pd
import shap

MODEL_PATH = "models/xgb_delivery_risk.pkl"

model = joblib.load(MODEL_PATH)


def explain_delivery_risk(order_data, top_n=5):

    df = pd.DataFrame([order_data])

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    X_transformed = preprocessor.transform(df)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    feature_names = preprocessor.get_feature_names_out()

    explanation = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values,
        "input_value": X_transformed[0]
    })

    explanation["importance"] = explanation["shap_value"].abs()

    filtered_rows = []

    for _, row in explanation.iterrows():

        feature = row["feature"]
        value = row["input_value"]

        if feature.startswith("cat__"):

            # Only keep the active category
            if value == 0:
                continue

            clean_feature = feature.replace("cat__", "")

            # Extract original feature name
            original_feature = clean_feature.split("_", 1)[0]

            original_value = order_data.get(
                original_feature,
                clean_feature
            )

        else:

            clean_feature = feature.replace("num__", "")
            original_feature = clean_feature

            original_value = order_data.get(
                original_feature,
                None
            )

        effect = (
            "increases late-delivery risk"
            if row["shap_value"] > 0
            else "decreases late-delivery risk"
        )

        filtered_rows.append({
            "feature": original_feature,
            "value": original_value,
            "shap_value": row["shap_value"],
            "effect": effect,
            "importance": row["importance"]
        })

    explanation = pd.DataFrame(filtered_rows)

    explanation = (
        explanation
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return explanation