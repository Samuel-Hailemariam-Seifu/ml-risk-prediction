# advanced_app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# ===========================
# Load trained model
# ===========================
model = joblib.load(BASE_DIR / "models" / "credit_risk_model.pkl")


# ===========================
# Load dataset for summaries
# ===========================
df = pd.read_csv(BASE_DIR / "data" / "german_credit_data.csv")
df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
df["Saving accounts"] = df["Saving accounts"].fillna("unknown")
df["Checking account"] = df["Checking account"].fillna("unknown")
df["Risk"] = df["Risk"].map({"good": 0, "bad": 1})

x = df.drop(columns=["Risk"])
y = df["Risk"]

# ===========================
# Sidebar
# ===========================
st.sidebar.title("Dataset Overview")
st.sidebar.write("Number of records:", df.shape[0])
st.sidebar.write("Number of features:", df.shape[1]-1)
st.sidebar.write("Class distribution:")
st.sidebar.bar_chart(df["Risk"].value_counts())

# ===========================
# Tabs
# ===========================
tab1, tab2, tab3 = st.tabs(["Predict Risk", "Feature Importance", "SHAP Explanation"])

# ===========================
# Tab 1: Predict Risk
# ===========================
with tab1:
    st.header("Customer Credit Risk Prediction")
    st.write("Enter customer details:")

    # Inputs
    age = st.number_input("Age", 18, 100, 30)
    sex = st.selectbox("Sex", df["Sex"].unique())
    job = st.selectbox("Job", df["Job"].unique())
    housing = st.selectbox("Housing", df["Housing"].unique())
    saving_accounts = st.selectbox("Saving Accounts", df["Saving accounts"].unique())
    checking_account = st.selectbox("Checking Account", df["Checking account"].unique())
    credit_amount = st.number_input("Credit Amount", 0, 100000, 1000)
    duration = st.number_input("Duration (Months)", 1, 100, 12)
    purpose = st.selectbox("Purpose", df["Purpose"].unique())

    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }])

    if st.button("Predict Risk"):
        # Preprocess the input
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        input_processed = preprocessor.transform(input_df)

        # Predict
        pred = classifier.predict(input_processed)[0]
        prob = classifier.predict_proba(input_processed)[0][1]

        risk_label = "High Risk (Bad)" if pred == 1 else "Low Risk (Good)"
        st.success(f"Prediction: {risk_label}")
        st.info(f"Probability of Risk: {prob:.2f}")

# ===========================
# Tab 2: Feature Importance
# ===========================
with tab2:
    st.header("Overall Feature Importance")
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    st.bar_chart(fi_df.set_index("feature")["importance"].head(10))


x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
# ===========================
# Tab 3: SHAP Explanation
# ===========================
with tab3:
    st.header("SHAP Explanation (Local Prediction)")

    import shap
    import matplotlib.pyplot as plt

    # ============================
    # Preprocess input and background
    # ============================
    X_train_proc = preprocessor.transform(x_train)
    input_proc = preprocessor.transform(input_df)

    # ============================
    # Define prediction function on numeric data
    # ============================
    def predict_proc(X):
        return classifier.predict_proba(X)[:, 1]

    # ============================
    # KernelExplainer
    # ============================
    explainer = shap.KernelExplainer(predict_proc, X_train_proc)

    # Compute SHAP values for the single input
    shap_values = explainer.shap_values(input_proc)

    # ----------------------------
    # Local waterfall plot
    # ----------------------------
    st.subheader("Local Explanation (Waterfall Plot)")
    fig, ax = plt.subplots(figsize=(10,4))
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=preprocessor.get_feature_names_out(), max_display=10)
    st.pyplot(fig)

    # ----------------------------
    # Global explanation (summary plot)
    # ----------------------------
    st.subheader("Global Feature Importance (Summary Plot)")
    # Use a subset of training set for speed
    shap_values_global = explainer.shap_values(X_train_proc[:100])

    fig2, ax2 = plt.subplots(figsize=(10,6))
    shap.summary_plot(
        shap_values_global,
        X_train_proc[:100],
        feature_names=preprocessor.get_feature_names_out(),
        plot_type="bar",
        show=False
    )
    st.pyplot(fig2)

