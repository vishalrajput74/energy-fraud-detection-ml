import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# ===============================
# Page Config (Better Layout)
# ===============================

st.set_page_config(
    page_title="Energy Fraud Detection",
    page_icon="⚡",
    layout="centered"
)


# ===============================
# Load Model (Robust + Cached)
# ===============================

@st.cache_resource
def load_model():
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
        return joblib.load(MODEL_PATH)
    except:
        try:
            return joblib.load("models/fraud_model.pkl")
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            return None


model = load_model()

if model is None:
    st.stop()


# ===============================
# Header UI
# ===============================

st.title("⚡ AI-Based Energy Consumption Fraud Detection System")

st.markdown("""
Detect fraudulent electricity and gas consumption using Machine Learning.
Upload your dataset and get instant predictions with analytics.
""")

st.divider()

st.info("Model: Random Forest Classifier | Accuracy: 99%")


# ===============================
# Upload Section
# ===============================

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(data.head(10), use_container_width=True)

    st.divider()

    if st.button("🔍 Detect Fraud", use_container_width=True):

        try:
            input_data = data.copy()

            # ===============================
            # Remove unused columns
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
            # Prediction
            # ===============================

            predictions = model.predict(input_data)

            data["Prediction"] = [
                "Fraud" if p == 1 else "No Fraud"
                for p in predictions
            ]

            st.success("Fraud detection completed successfully")

            # ===============================
            # Summary Dashboard
            # ===============================

            fraud_count = (predictions == 1).sum()
            total_records = len(data)
            fraud_percentage = (fraud_count / total_records) * 100

            st.subheader("📊 Summary Dashboard")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Records", total_records)
            col2.metric("Fraud Cases", fraud_count)
            col3.metric("Fraud %", f"{fraud_percentage:.2f}%")

            st.divider()

            # ===============================
            # Fraud Distribution (Better Chart)
            # ===============================

            st.subheader("📈 Fraud Distribution")

            fraud_counts = data["Prediction"].value_counts()

            fig, ax = plt.subplots()
            ax.pie(
                fraud_counts.values,
                labels=fraud_counts.index,
                autopct="%1.1f%%",
                startangle=90
            )
            ax.set_title("Fraud vs Non-Fraud")

            st.pyplot(fig)

            st.divider()

            # ===============================
            # Top 10 Fraud Cases
            # ===============================

            st.subheader("🔴 Top 10 Fraud Cases")

            fraud_cases = data[data["Prediction"] == "Fraud"].head(10)

            if len(fraud_cases) > 0:
                st.dataframe(fraud_cases, use_container_width=True)
            else:
                st.write("No fraud cases detected.")

            st.divider()

            # ===============================
            # Feature Importance (Improved)
            # ===============================

            st.subheader("⭐ Feature Importance")

            if hasattr(model, "feature_importances_"):

                importance_df = pd.DataFrame({
                    "Feature": input_data.columns,
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=True)

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.barh(importance_df["Feature"], importance_df["Importance"])
                ax2.set_xlabel("Importance Score")
                ax2.set_title("Most Important Features")

                st.pyplot(fig2)

            else:
                st.write("Feature importance not available.")

            st.divider()

            # ===============================
            # Prediction Results Table
            # ===============================

            st.subheader("✅ Prediction Results")
            st.dataframe(data, use_container_width=True)

            st.divider()

            # ===============================
            # Download Results
            # ===============================

            st.subheader("⬇ Download Results")

            csv = data.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="fraud_detection_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")