# CLAUDE.md — Salesforce Churn Prediction

> Project constitution for all Claude Code agents. Read this before touching any file.

---

## 📌 Project Overview

**Project:** Salesforce Customer Churn Prediction  
**Type:** ML Web App (ANN-based)  
**Developer:** Haseeba | MLRIT Hyderabad | B.Tech CSE (Data Science)  
**Status:** In Progress — Building from dataset CSV  
**Goal:** Train ANN churn model → expose via FastAPI → visualize via Streamlit

---

## 🗂️ Folder Structure

```
churn-prediction/
├── CLAUDE.md                  ← you are here
├── .claude/
│   └── agents/                ← subagents
├── scripts/                   ← hook scripts
│
├── data/
│   └── churn.csv              ← raw Salesforce dataset (DO NOT modify)
│
├── notebooks/
│   └── churn_eda_training.ipynb  ← EDA + model training notebook
│
├── models/
│   └── churn_model.h5         ← saved trained ANN
│   └── scaler.pkl             ← saved StandardScaler
│   └── label_encoders.pkl     ← saved encoders (if any)
│
├── backend/
│   ├── main.py                ← FastAPI entry point
│   ├── routers/
│   │   └── predict.py         ← /predict endpoint
│   ├── schemas.py             ← Pydantic request/response models
│   ├── model_loader.py        ← loads .h5 and .pkl at startup
│   └── requirements.txt
│
├── frontend/
│   └── app.py                 ← Streamlit UI
│
└── .env                       ← secrets (NEVER read or print)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | TensorFlow / Keras (ANN) |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Serialization | `.h5` for Keras model, `pickle` for scaler/encoders |
| Dataset | Salesforce Churn CSV |

---

## 📊 Dataset Info

- **File:** `data/churn.csv`
- **Target column:** `Churn` (binary: 0 = retained, 1 = churned)
- **Typical features:** CustomerID, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- **DO NOT** modify the raw CSV — always work on a copy inside the notebook

---

## 🧠 ML Pipeline

### Training Flow (notebook)
```
Load CSV
  → EDA (check nulls, distributions, correlations)
  → Preprocessing (encode categoricals, scale numericals)
  → Train/Test Split (80/20)
  → Build ANN (Input → Dense → Dropout → Dense → Output sigmoid)
  → Train (binary_crossentropy, adam, epochs=100)
  → Evaluate (accuracy, confusion matrix, ROC-AUC)
  → Save model → models/churn_model.h5
  → Save scaler → models/scaler.pkl
```

### ANN Architecture Convention
```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## ⚙️ Commands

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# Swagger docs at: http://localhost:8000/docs
```

### Frontend (Streamlit)
```bash
cd frontend
streamlit run app.py
# Opens at: http://localhost:8501
```

### Notebook
```bash
jupyter notebook notebooks/churn_eda_training.ipynb
```

---

## 🔌 API Design

### POST `/predict`
```json
Request:
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
  "EstimatedSalary": 50000.0
}

Response:
{
  "churn_probability": 0.82,
  "prediction": "Churn",
  "confidence": "High"
}
```

### GET `/health`
```json
{ "status": "ok", "model_loaded": true }
```

---

## 🎨 Streamlit UI Guidelines

- **Title:** "Customer Churn Predictor"
- **Input:** Sidebar sliders/dropdowns for each feature
- **Output:** Big colored metric (🔴 Churn / 🟢 Retained) + probability gauge
- **Extras:** Show feature importance bar chart if possible
- **API call:** Streamlit calls FastAPI at `http://localhost:8000/predict`
- Keep it clean — no clutter, recruiter-friendly layout

---

## 🔒 Security Rules

- **NEVER** read, print, or log `.env` contents
- **NEVER** hardcode any credentials
- **NEVER** modify `data/churn.csv` directly
- All model files in `/models/` are read-only after training

---

## 🤖 Subagent Guidelines

### `code-reviewer`
- Check: are preprocessing steps consistent between training and inference?
- Check: is the scaler applied in the same order in both notebook and API?
- Check: Pydantic schema matches actual CSV feature names exactly

### `debugger`
- Most common bug: scaler/encoder not applied before model.predict() in API
- Second most common: feature order mismatch between training and inference
- Always check `model_loader.py` first when prediction errors occur

### `data-scientist`
- Use `data/churn.csv` as source — never modify it
- Always `df.copy()` before any transformations
- Report class imbalance in target column first thing

---

## 📌 Definition of Done

- [ ] EDA notebook complete with visualizations
- [ ] Model trained with >85% accuracy
- [ ] `churn_model.h5` + `scaler.pkl` saved in `/models/`
- [ ] FastAPI `/predict` endpoint working (tested via Swagger)
- [ ] Streamlit UI connects to API and shows prediction
- [ ] Deployed on Streamlit Cloud or local demo ready

---

*Last updated: March 2026 | Haseeba — MLRIT Hyderabad*
