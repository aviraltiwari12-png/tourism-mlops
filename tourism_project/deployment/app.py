"""Streamlit app: predict whether a customer will buy the Wellness Tourism Package.

The trained model (best_tourism_model.joblib) is committed to this folder by the
GitHub Actions pipeline. The form below is built from the features the model was
trained on, so the app always stays consistent with the deployed model.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "best_tourism_model.joblib"
METRICS_PATH = Path(__file__).parent / "model_metrics.json"   # written by train.py
CLASSIFICATION_THRESHOLD = 0.5  # keep identical to train.py

POST_PITCH_HELP = "Known only after the sales pitch has been delivered."

# One input widget per model feature. Categorical options match the cleaned training data.
INPUT_BUILDERS = {
    "Age": lambda: st.number_input("Age", 18, 80, 35),
    "TypeofContact": lambda: st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"]),
    "CityTier": lambda: st.selectbox("City tier (1 = most developed)", [1, 2, 3]),
    "Occupation": lambda: st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]),
    "Gender": lambda: st.selectbox("Gender", ["Male", "Female"]),
    "NumberOfPersonVisiting": lambda: st.number_input("Number of persons visiting", 1, 10, 3),
    "PreferredPropertyStar": lambda: st.selectbox("Preferred hotel star rating", [3.0, 4.0, 5.0]),
    "MaritalStatus": lambda: st.selectbox(
        "Marital status", ["Married", "Single", "Divorced", "Unmarried"]),
    "NumberOfTrips": lambda: st.number_input("Average trips per year", 0, 30, 3),
    "Passport": lambda: 1 if st.selectbox("Valid passport", ["No", "Yes"]) == "Yes" else 0,
    "OwnCar": lambda: 1 if st.selectbox("Owns a car", ["No", "Yes"]) == "Yes" else 0,
    "NumberOfChildrenVisiting": lambda: st.number_input("Children (under 5) visiting", 0, 5, 1),
    "Designation": lambda: st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]),
    "MonthlyIncome": lambda: st.number_input("Monthly income", 1000, 200000, 23000, step=500),
    "PitchSatisfactionScore": lambda: st.slider(
        "Pitch satisfaction score", 1, 5, 3, help=POST_PITCH_HELP),
    "NumberOfFollowups": lambda: st.number_input(
        "Number of follow-ups", 0, 10, 4, help=POST_PITCH_HELP),
    "DurationOfPitch": lambda: st.number_input(
        "Duration of pitch (minutes)", 1, 130, 15, help=POST_PITCH_HELP),
}


@st.cache_resource
def load_model():
    """Load the trained pipeline once and reuse it across reruns."""
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="wide")
st.title("Wellness Tourism Package - Purchase Predictor")
st.write(
    "Enter the customer details below to estimate the likelihood that the customer "
    "will purchase the Wellness Tourism Package."
)

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Run the GitHub Actions pipeline so that "
        "best_tourism_model.joblib is committed to tourism_project/deployment/."
    )
    st.stop()

model = load_model()
features = list(model.feature_names_in_)

missing_inputs = [f for f in features if f not in INPUT_BUILDERS]
if missing_inputs:
    st.error(f"The model expects features without an input widget: {missing_inputs}")
    st.stop()

# --- Collect the inputs ---
with st.form("customer_form"):
    columns = st.columns(3)
    inputs = {}
    for i, feature in enumerate(features):
        with columns[i % 3]:
            inputs[feature] = INPUT_BUILDERS[feature]()
    submitted = st.form_submit_button("Predict purchase likelihood")

# --- Predict ---
if submitted:
    input_df = pd.DataFrame([inputs])[features]  # inputs saved into a DataFrame
    probability = float(model.predict_proba(input_df)[0, 1])
    is_likely_buyer = probability >= CLASSIFICATION_THRESHOLD

    st.subheader("Customer details (model input)")
    st.dataframe(input_df, hide_index=True)

    st.subheader("Prediction")
    st.metric("Purchase probability", f"{probability:.1%}")
    if is_likely_buyer:
        st.success("Likely buyer - prioritise this customer for the Wellness Tourism Package campaign.")
    else:
        st.info("Unlikely buyer at the current threshold - consider a lower-cost outreach channel.")
    st.caption(f"Decision threshold: {CLASSIFICATION_THRESHOLD:.0%} purchase probability.")

# --- Model information (read from the metrics report written during training) ---
if METRICS_PATH.exists():
    info = json.loads(METRICS_PATH.read_text())
    test = info["test_metrics"]
    with st.expander("About this model"):
        st.write(
            f"Trained on {info['n_train']:,} customers ({info['trained_at_utc']}). "
            f"Performance on {info['n_test']:,} unseen test customers: "
            f"recall {test['recall']:.0%} (share of real buyers found), "
            f"precision {test['precision']:.0%} (share of flagged customers who buy), "
            f"F1 {test['f1']:.2f}, ROC-AUC {test['roc_auc']:.2f}."
        )
