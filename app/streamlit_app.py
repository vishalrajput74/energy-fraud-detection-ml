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

# Model info
st.info("Model Used: Random Forest Classifier")
st.info("Model Accuracy: 99%")

# ===============================
# Upload Data
# ===============================

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
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
            # Make Prediction
            # ===============================

            predictions = model.predict(input_data)

            data["Prediction"] = [
                "Fraud" if p == 1 else "No Fraud"
                for p in predictions
            ]

            # ===============================
            # Prediction Results
            # ===============================

            st.subheader("✅ Prediction Results")
            st.dataframe(data)

            # ===============================
            # Summary Dashboard (FIRST INSIGHT)
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
            # Fraud vs Non-Fraud Chart
            # ===============================

            st.subheader("📈 Fraud vs Non-Fraud Distribution")

            fraud_vs_normal = data["Prediction"].value_counts()

            fig, ax = plt.subplots()
            fraud_vs_normal.plot(kind="bar", ax=ax)
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            ax.set_title("Fraud vs Non-Fraud")

            st.pyplot(fig)

            # ===============================
            # Top 10 Fraud Cases
            # ===============================

            st.subheader("🔴 Top 10 Fraud Cases")

            fraud_cases = data[data["Prediction"] == "Fraud"].head(10)

            if len(fraud_cases) > 0:
                st.dataframe(fraud_cases)
            else:
                st.write("No fraud cases detected.")

            # ===============================
            # Feature Importance (Model Explainability)
            # ===============================

            st.subheader("⭐ Feature Importance (What causes fraud?)")

            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                feature_names = input_data.columns

                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance
                }).sort_values("Importance", ascending=False)

                fig2, ax2 = plt.subplots()
                ax2.barh(importance_df["Feature"], importance_df["Importance"])
                ax2.set_xlabel("Importance Score")
                ax2.set_title("Feature Importance")
                ax2.invert_yaxis()

                st.pyplot(fig2)
            else:
                st.write("Feature importance not available for this model.")

            # ===============================
            # Download Results Button
            # ===============================

            st.subheader("⬇ Download Results")

            csv = data.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")
