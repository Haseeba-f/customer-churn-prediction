"""
Pydantic schemas for prediction request and response.
Field names match the dataset columns exactly.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Customer features for churn prediction."""
    CreditScore: int = Field(..., ge=300, le=900, description="Customer credit score")
    Geography: str = Field(..., description="Country: France, Spain, or Germany")
    Gender: str = Field(..., description="Male or Female")
    Age: int = Field(..., ge=18, le=100, description="Customer age")
    Tenure: int = Field(..., ge=0, le=10, description="Years as bank customer")
    Balance: float = Field(..., ge=0, description="Account balance")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of bank products used")
    HasCrCard: int = Field(..., ge=0, le=1, description="Has credit card (0 or 1)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Is active member (0 or 1)")
    EstimatedSalary: float = Field(..., ge=0, description="Estimated annual salary")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "CreditScore": 600,
                    "Geography": "France",
                    "Gender": "Female",
                    "Age": 40,
                    "Tenure": 3,
                    "Balance": 60000.0,
                    "NumOfProducts": 2,
                    "HasCrCard": 1,
                    "IsActiveMember": 1,
                    "EstimatedSalary": 50000.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Churn prediction result."""
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    prediction: str = Field(..., description="Churn or Retained")
    confidence: str = Field(..., description="High, Medium, or Low")
