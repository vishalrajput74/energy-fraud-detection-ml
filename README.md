# ⚡ AI-Based Energy Consumption Fraud Detection System

## 📌 Project Overview

This project detects fraudulent electricity and gas consumption using Machine Learning. The system analyzes energy usage patterns and predicts suspicious behavior using a trained classification model.

It helps energy companies identify fraud cases quickly, monitor usage behavior, and generate business insights through an interactive dashboard.

---

## 🎯 Problem Statement

Energy companies face financial losses due to fraudulent energy consumption. Manual detection is slow and inefficient.

This project uses Machine Learning to automatically detect fraud based on:

- Energy usage patterns
- Payment history
- Meter status
- Location and consumption behavior

The system provides automated predictions, analytics, and visual insights.

---

## ⭐ Key Features

### 📊 Interactive Dashboard

- Download sample dataset for testing
- Upload CSV dataset
- Fraud prediction using Machine Learning
- Business summary KPI metrics
- Fraud percentage calculation
- Professional data visualization

### 📈 Analytics & Visualization

- Fraud vs Non-Fraud distribution (donut chart)
- Top 10 high-risk fraud cases
- Feature importance visualization
- Business insights for decision-making

### 📄 Data Management

- Dataset preview and overview
- Dataset shape and column details
- Download prediction results as CSV

### 🤖 Model Information

- Model details page
- Accuracy information
- Key fraud indicators display

---

## 🧠 Machine Learning Model

- **Model:** Random Forest Classifier
- **Accuracy:** 99%
- **Type:** Classification
- **Purpose:** Detect fraudulent energy consumption behavior
- **Explainability:** Feature importance analysis

---

### 📊 Model Performance

- Model Type: Classification
- Algorithm: Random Forest Classifier
- Accuracy: 99%
- Evaluation Metrics:
  - Precision
  - Recall
  - F1-score
- Feature Importance Analysis performed

---

## 📊 Dataset

- **Source:** Kaggle Energy Consumption Dataset
- Contains energy usage patterns, meter readings, payment data, and fraud labels.
- Used for training and testing fraud detection model.

---

### 📁 Dataset Information

- Dataset: Energy Consumption Dataset
- Format: CSV
- Contains features like:
  - Energy usage
  - Payment history
  - Meter status
  - Location data
- Used for training and fraud prediction.

Users can upload their own dataset in the same format for prediction.

---

## ⚙ Model Training Steps

The machine learning model was developed using the following pipeline:

### Data Preprocessing

- Handling missing values
- Data cleaning
- Feature selection

### Feature Encoding

- Convert categorical variables into numerical format

### Model Training

- Random Forest Classifier training

### Model Evaluation

- Accuracy measurement
- Performance validation

### Model Saving

- Saved using Joblib for deployment

---

## 🛠 Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

---

## 📂 Project Structure

```
energy-fraud-detection-ml/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── energy_fraud_dataset.csv
│
├── models/
│   └── fraud_model.pkl
│
├── notebook/
│   └── fraud_detection.ipynb
│
├── src/
│   └── train_model.py
│
├── assets/
│   ├── dashboard.png
│   ├── business_summary.png
│   ├── fraud_chart.png
│   ├── top_fraud_cases.png
│   ├── feature_importance.png
│   ├── model_summary.png
│   └── prediction_results.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 1️⃣ Clone Repository

```
git clone https://github.com/vishalrajput74/energy-fraud-detection-ml
cd energy-fraud-detection-ml
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶ How to Run the Project

### Train the Model (Optional)

```
python src/train_model.py
```

### Run Streamlit Application

```
streamlit run app/streamlit_app.py
```

The application will open automatically in your browser.

---

## 📊 Application Output

The system provides:

- Fraud prediction results
- Business summary dashboard with KPIs
- Fraud distribution visualization
- Top fraud records table
- Feature importance chart
- Dataset overview and statistics
- Model information display
- Downloadable prediction results

---

## 📸 Application Preview

### 📊 Dashboard Interface

![Dashboard](assets/dashboard.png)

### 📈 Fraud Detection Results / Business Summary

![Business Summary](assets/business_summary.png)

### 📊 Fraud Distribution Chart

This chart shows fraud vs non-fraud transactions.  
![Fraud Chart](assets/fraud_chart.png)

### 🔴 Top Fraud Cases

![Top Fraud Cases](assets/top_fraud_cases.png)

### ⭐ Feature Importance

![Feature Importance](assets/feature_importance.png)

### 🤖 Model Information

![Model Information](assets/model_summary.png)

### ✅ Prediction Results Table

![Prediction Results](assets/prediction_results.png)

---

## 🚀 Future Improvements

- Real-time fraud detection system
- Database integration
- REST API deployment
- Deep learning models
- Model monitoring system
- Cloud deployment (AWS/GCP/Azure)

---

## 👨‍💻 Author

**Vishal Rajput**
