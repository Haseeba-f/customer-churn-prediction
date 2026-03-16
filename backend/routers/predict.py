"""
Predict router — POST /predict endpoint.
Accepts customer features, applies preprocessing, runs model, returns churn prediction.
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from schemas import PredictionRequest, PredictionResponse

router = APIRouter()

# Feature order must match training pipeline exactly
NUMERICAL_FEATURES = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                       "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CATEGORICAL_FEATURES = ["Geography", "Gender"]


def preprocess_input(data: PredictionRequest, scaler, label_encoders) -> np.ndarray:
    """
    Transform raw input into model-ready feature array.
    Applies label encoding for categoricals and standard scaling for numericals.
    Must replicate the exact preprocessing from the training notebook.
    """
    input_dict = data.model_dump()

    # Encode categorical features
    if label_encoders:
        for col in CATEGORICAL_FEATURES:
            if col in label_encoders:
                try:
                    input_dict[col] = label_encoders[col].transform([input_dict[col]])[0]
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown value '{input_dict[col]}' for feature '{col}'. "
                               f"Expected one of: {list(label_encoders[col].classes_)}"
                    )
    else:
        # Manual encoding fallback (must match training)
        geography_map = {"France": 0, "Germany": 1, "Spain": 2}
        gender_map = {"Female": 0, "Male": 1}

        if input_dict["Geography"] not in geography_map:
            raise HTTPException(status_code=400, detail=f"Unknown Geography: {input_dict['Geography']}")
        if input_dict["Gender"] not in gender_map:
            raise HTTPException(status_code=400, detail=f"Unknown Gender: {input_dict['Gender']}")

        input_dict["Geography"] = geography_map[input_dict["Geography"]]
        input_dict["Gender"] = gender_map[input_dict["Gender"]]

    # Build feature array in the correct order
    feature_order = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
                     "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
                     "EstimatedSalary"]
    features = np.array([[input_dict[f] for f in feature_order]], dtype=np.float64)

    # Scale features
    if scaler is not None:
        features = scaler.transform(features)

    return features


@router.post("/predict", response_model=PredictionResponse)
async def predict_churn(data: PredictionRequest, request: Request):
    """
    Predict whether a customer will churn.

    Returns probability, label (Churn/Retained), and confidence level.
    """
    model = request.app.state.model
    scaler = request.app.state.scaler
    label_encoders = getattr(request.app.state, "label_encoders", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first by running the notebook."
        )

    # Preprocess
    features = preprocess_input(data, scaler, label_encoders)

    # Predict
    try:
        probability = float(model.predict(features, verbose=0)[0][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Interpret result
    prediction = "Churn" if probability >= 0.5 else "Retained"

    if probability >= 0.8 or probability <= 0.2:
        confidence = "High"
    elif probability >= 0.6 or probability <= 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"

    return PredictionResponse(
        churn_probability=round(probability, 4),
        prediction=prediction,
        confidence=confidence,
    )
