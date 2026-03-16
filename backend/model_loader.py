"""
Model loader — loads trained Keras model, StandardScaler, and LabelEncoders at startup.
All artifacts are expected in the ../models/ directory.
"""

import os
import pickle
import logging

logger = logging.getLogger(__name__)

# Paths relative to this file's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")


def load_model_artifacts():
    """
    Load the trained ANN model, scaler, and label encoders.

    Returns:
        tuple: (model, scaler, label_encoders)
            - model: Keras Sequential model or None if file not found
            - scaler: fitted StandardScaler or None
            - label_encoders: dict of fitted LabelEncoders or None
    """
    model = None
    scaler = None
    label_encoders = None

    # Load Keras model
    if os.path.exists(MODEL_PATH):
        try:
            from tensorflow.keras.models import load_model
            model = load_model(MODEL_PATH)
            logger.info("✅ Keras model loaded from %s", MODEL_PATH)
        except Exception as e:
            logger.error("❌ Failed to load Keras model: %s", e)
    else:
        logger.warning("⚠️ Model file not found at %s — run the training notebook first", MODEL_PATH)

    # Load StandardScaler
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            logger.info("✅ Scaler loaded from %s", SCALER_PATH)
        except Exception as e:
            logger.error("❌ Failed to load scaler: %s", e)
    else:
        logger.warning("⚠️ Scaler file not found at %s", SCALER_PATH)

    # Load LabelEncoders (optional)
    if os.path.exists(ENCODERS_PATH):
        try:
            with open(ENCODERS_PATH, "rb") as f:
                label_encoders = pickle.load(f)
            logger.info("✅ Label encoders loaded from %s", ENCODERS_PATH)
        except Exception as e:
            logger.error("❌ Failed to load label encoders: %s", e)
    else:
        logger.info("ℹ️ No label encoders file at %s (may not be needed)", ENCODERS_PATH)

    return model, scaler, label_encoders
