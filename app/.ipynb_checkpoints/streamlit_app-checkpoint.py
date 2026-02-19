import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ===============================
# Load Model
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")

model = joblib.load(MODEL_PATH)

# ===============================
# UI
# ===============================

st.title("⚡ AI-Based Energy Consumption Fraud Detection System")

st.write("""
Upload energy consumption data to detect fraudulent behavior using Machine Learning.
""")

# Model info (portfolio value)
st.info("Model Used: Random Forest Classifier")
st.info("Model Accuracy: 99%")

# ===============================
# Upload Data
# ===============================

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(data.head())

    if st.button("🔍 Detect Fraud"):

        try:
            input_data = data.copy()

            # ===============================
            # Remove columns not used in training
            # ===============================

            drop_columns = ["UserID", "MeterID", "FraudReported"]

            for col in drop_columns:
                if col in input_data.columns:
                    input_data = input_data.drop(col, axis=1)

            # ===============================
            # Encode categorical columns
            # ===============================

            categorical_cols = [
                "UsageType",
                "TariffPlan",
                "Location",
                "TimeOfDay",
                "MeterStatus",
                "PaymentHistory"
            ]

            for col in categorical_cols:
                if col in input_data.columns:
                    le = LabelEncoder()
                    input_data[col] = le.fit_transform(input_data[col])

            # ===============================
            # Make prediction
            # ===============================

            predictions = model.predict(input_data)

            # Add results
            data["Prediction"] = [
                "Fraud" if p == 1 else "No Fraud"
                for p in predictions
            ]

            st.subheader("✅ Prediction Results")
            st.dataframe(data)

            # ===============================
            # Summary Dashboard
            # ===============================

            fraud_count = (predictions == 1).sum()
            total_records = len(data)
            fraud_percentage = (fraud_count / total_records) * 100

            st.write("## 📊 Summary Dashboard")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Records", total_records)
            col2.metric("Fraud Cases", fraud_count)
            col3.metric("Fraud %", f"{fraud_percentage:.2f}%")

            # ===============================
            # Fraud vs Normal Chart
            # ===============================

            st.subheader("📈 Fraud vs Non-Fraud Distribution")

            fraud_vs_normal = data["Prediction"].value_counts()

            fig, ax = plt.subplots()
            fraud_vs_normal.plot(kind="bar", ax=ax)
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            ax.set_title("Fraud vs Non-Fraud")

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
