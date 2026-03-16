"""
FastAPI backend for Salesforce Customer Churn Prediction.
Loads trained ANN model and serves predictions via REST API.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model_loader import load_model_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model artifacts on startup."""
    app.state.model, app.state.scaler, app.state.label_encoders = load_model_artifacts()
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn using a trained ANN model",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from routers.predict import router as predict_router
app.include_router(predict_router)


@app.get("/")
async def root():
    """Root endpoint — API info."""
    return {
        "service": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predict customer churn",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation (Swagger UI)",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    return {"status": "ok", "model_loaded": model_loaded}
