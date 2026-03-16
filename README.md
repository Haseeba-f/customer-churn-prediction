# 🏦 Bank Customer Churn Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An end-to-end Machine Learning web app that predicts whether a bank customer will churn using an Artificial Neural Network.**

[🚀 Live Demo](https://customer-churn-prediction-nmpej4eqzanretozy3cqgg.streamlit.app/) · [📊 Dataset](#dataset) · [🛠️ Installation](#installation)

</div>

---

## 📌 Overview

Customer churn is one of the biggest challenges in banking — losing a customer costs far more than retaining one. This project uses a trained **Artificial Neural Network (ANN)** to predict which customers are likely to leave, giving banks the ability to intervene proactively.

> Built with TensorFlow/Keras for the model, FastAPI for the backend REST API, and Streamlit for the interactive web UI.

---

## 🎯 Features

- 🤖 **ANN Model** — Deep learning model trained on 10,000 customer records
- ⚡ **FastAPI Backend** — REST API serving real-time predictions with Swagger docs
- 🎨 **Streamlit Frontend** — Clean, interactive UI with sliders and probability gauge
- 📊 **EDA Notebook** — Full exploratory data analysis with visualizations
- 🔁 **End-to-End Pipeline** — From raw CSV to deployed web app

---

## 🖥️ Demo

<div align="center">

> 🔗 **Live App:** [customer-churn-prediction.streamlit.app](https://customer-churn-prediction-nmpej4eqzanretozy3cqgg.streamlit.app/)

</div>

Adjust customer features in the sidebar → click **Predict Churn** → get instant prediction with probability score.

---

## 🧠 Model Architecture

```
Input Layer  (10 features)
      ↓
Dense(64, relu)
      ↓
Dropout(0.3)
      ↓
Dense(32, relu)
      ↓
Dropout(0.2)
      ↓
Output(1, sigmoid)  →  Churn Probability
```

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Epochs | 100 |
| Train/Test Split | 80/20 |
| Target Metric | Accuracy + ROC-AUC |

---

## 📊 Dataset

- **Source:** Bank Customer Churn Dataset (Kaggle)
- **Rows:** 10,000 customers
- **Target:** `Exited` (0 = Retained, 1 = Churned)

| Feature | Type | Description |
|---|---|---|
| CreditScore | Numerical | Customer credit score |
| Geography | Categorical | France / Spain / Germany |
| Gender | Categorical | Male / Female |
| Age | Numerical | Customer age |
| Tenure | Numerical | Years with the bank |
| Balance | Numerical | Account balance |
| NumOfProducts | Numerical | Number of bank products |
| HasCrCard | Binary | Has credit card (0/1) |
| IsActiveMember | Binary | Active member (0/1) |
| EstimatedSalary | Numerical | Estimated annual salary |

---

## 🗂️ Project Structure

```
bank-churn-prediction/
├── data/
│   └── churn.csv                  # Raw dataset
├── notebooks/
│   └── churn_eda_training.ipynb   # EDA + model training
├── models/
│   ├── churn_model.h5             # Trained ANN model
│   └── scaler.pkl                 # Fitted StandardScaler
├── backend/
│   ├── main.py                    # FastAPI entry point
│   ├── routers/
│   │   └── predict.py             # /predict endpoint
│   ├── schemas.py                 # Pydantic request/response models
│   ├── model_loader.py            # Loads model at startup
│   └── requirements.txt
├── frontend/
│   └── app.py                     # Streamlit UI
└── README.md
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repo
```bash
git clone https://github.com/Haseeba-f/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Train the model (first time only)
```bash
jupyter notebook notebooks/churn_eda_training.ipynb
# Run all cells — saves churn_model.h5 and scaler.pkl to /models
```

### 4. Start the FastAPI backend
```bash
cd backend
uvicorn main:app --reload --port 8000
# API docs → http://localhost:8000/docs
```

### 5. Launch the Streamlit UI
```bash
cd frontend
streamlit run app.py
# Opens → http://localhost:8501
```

---

## 🔌 API Reference

### `POST /predict`

Predicts churn probability for a given customer.

**Request Body:**
```json
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
```

**Response:**
```json
{
  "churn_probability": 0.82,
  "prediction": "Churn",
  "confidence": "High"
}
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

---

## 🚀 Tech Stack

| Layer | Technology |
|---|---|
| ML Model | TensorFlow 2.x / Keras |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## 👩‍💻 Author

**Haseeba**  
B.Tech CSE (Data Science) — MLR Institute of Technology, Hyderabad  
[![GitHub](https://img.shields.io/badge/GitHub-Haseeba--f-black?style=flat&logo=github)](https://github.com/Haseeba-f)

---

## 📄 License

This project is licensed under the MIT License.
