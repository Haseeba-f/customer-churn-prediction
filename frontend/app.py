"""
Streamlit frontend for Salesforce Customer Churn Prediction.
Loads the trained TensorFlow model directly — no FastAPI backend required.
"""

import os
import pickle

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-churn {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b11);
        border-left: 4px solid #ff4b4b;
        padding: 20px;
        border-radius: 10px;
    }
    .metric-retained {
        background: linear-gradient(135deg, #00c85322, #00c85311);
        border-left: 4px solid #00c853;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🔮 Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by Artificial Neural Network — Salesforce Dataset</p>', unsafe_allow_html=True)


# ── Model Loading (cached) ──────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    """Load the trained Keras model, scaler, and label encoders from the models/ directory."""
    from tensorflow.keras.models import load_model  # noqa: delayed import to keep startup fast

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    model_path = os.path.join(models_dir, "churn_model.h5")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    encoders_path = os.path.join(models_dir, "label_encoders.pkl")

    # Load Keras model
    model = None
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        st.error(f"⚠️ Model file not found at `{model_path}`. Run the training notebook first.")

    # Load StandardScaler
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # Load LabelEncoders
    label_encoders = None
    if os.path.exists(encoders_path):
        with open(encoders_path, "rb") as f:
            label_encoders = pickle.load(f)

    return model, scaler, label_encoders


model, scaler, label_encoders = load_model_artifacts()

# Feature definitions (must match training pipeline)
FEATURE_ORDER = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary",
]
CATEGORICAL_FEATURES = ["Geography", "Gender"]


def preprocess_and_predict(input_dict: dict) -> dict:
    """
    Encode categoricals, scale features, run model prediction, and return result dict.
    Replicates the exact preprocessing from the training notebook / FastAPI backend.
    """
    # ── Encode categoricals ──────────────────────────────────
    if label_encoders:
        for col in CATEGORICAL_FEATURES:
            if col in label_encoders:
                input_dict[col] = label_encoders[col].transform([input_dict[col]])[0]
    else:
        # Manual fallback (must match training)
        geography_map = {"France": 0, "Germany": 1, "Spain": 2}
        gender_map = {"Female": 0, "Male": 1}
        input_dict["Geography"] = geography_map[input_dict["Geography"]]
        input_dict["Gender"] = gender_map[input_dict["Gender"]]

    # ── Build feature array in correct order & scale ─────────
    features = np.array([[input_dict[f] for f in FEATURE_ORDER]], dtype=np.float64)
    if scaler is not None:
        features = scaler.transform(features)

    # ── Predict ──────────────────────────────────────────────
    probability = float(model.predict(features, verbose=0)[0][0])
    prediction = "Churn" if probability >= 0.5 else "Retained"

    if probability >= 0.8 or probability <= 0.2:
        confidence = "High"
    elif probability >= 0.6 or probability <= 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "confidence": confidence,
    }


# ── Sidebar — Customer Features ──────────────────────────────
st.sidebar.header("📋 Customer Features")
st.sidebar.markdown("---")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 600, step=10)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
age = st.sidebar.slider("Age", 18, 100, 40)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 3)
balance = st.sidebar.number_input("Account Balance ($)", min_value=0.0, value=60000.0, step=1000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.sidebar.selectbox("Has Credit Card?", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
is_active = st.sidebar.selectbox("Active Member?", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
estimated_salary = st.sidebar.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=5000.0)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 Predict Churn", use_container_width=True, type="primary")


def create_gauge(probability: float) -> go.Figure:
    """Create a Plotly gauge chart for churn probability."""
    color = "#ff4b4b" if probability >= 0.5 else "#00c853"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 48, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 2},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 30], "color": "rgba(0,200,83,0.15)"},
                {"range": [30, 60], "color": "rgba(255,193,7,0.15)"},
                {"range": [60, 100], "color": "rgba(255,75,75,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
        title={"text": "Churn Probability", "font": {"size": 18}},
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Main Content ─────────────────────────────────────────────
if predict_btn:
    if model is None:
        st.error("⚠️ Model not loaded. Please run the training notebook first.")
    else:
        payload = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card[1],
            "IsActiveMember": is_active[1],
            "EstimatedSalary": estimated_salary,
        }

        try:
            with st.spinner("Analyzing customer profile..."):
                result = preprocess_and_predict(payload)

            prob = result["churn_probability"]
            prediction = result["prediction"]
            confidence = result["confidence"]

            # Result columns
            col1, col2 = st.columns([1, 1])

            with col1:
                if prediction == "Churn":
                    st.markdown('<div class="metric-churn">', unsafe_allow_html=True)
                    st.markdown(f"## 🔴 {prediction}")
                    st.markdown(f"**Confidence:** {confidence}")
                    st.markdown(f"**Probability:** {prob:.1%}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-retained">', unsafe_allow_html=True)
                    st.markdown(f"## 🟢 {prediction}")
                    st.markdown(f"**Confidence:** {confidence}")
                    st.markdown(f"**Probability:** {1 - prob:.1%} retention")
                    st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.plotly_chart(create_gauge(prob), use_container_width=True)

            # Feature summary
            st.markdown("---")
            st.subheader("📊 Customer Profile Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.metric("Credit Score", credit_score)
                st.metric("Age", age)
                st.metric("Tenure", f"{tenure} yrs")

            with summary_col2:
                st.metric("Balance", f"${balance:,.2f}")
                st.metric("Salary", f"${estimated_salary:,.2f}")
                st.metric("Products", num_products)

            with summary_col3:
                st.metric("Geography", geography)
                st.metric("Gender", gender)
                st.metric("Active Member", "Yes" if is_active[1] else "No")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

else:
    # Landing state
    st.info("👈 Adjust customer features in the sidebar and click **Predict Churn** to get started.")

    st.markdown("---")
    st.subheader("ℹ️ How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Input")
        st.markdown("Enter customer features like credit score, geography, balance, etc.")
    with col2:
        st.markdown("### 2️⃣ Predict")
        st.markdown("Our trained ANN model analyzes the profile for churn risk.")
    with col3:
        st.markdown("### 3️⃣ Result")
        st.markdown("Get a clear Churn/Retained prediction with probability gauge.")

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Built by Haseeba | MLRIT Hyderabad | B.Tech CSE (Data Science)"
    "</p>",
    unsafe_allow_html=True,
)
