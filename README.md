# ⚡ AI-Based Energy Consumption Fraud Detection System

## 📌 Project Overview

This project detects fraudulent electricity and gas consumption using Machine Learning.  
The system analyzes energy usage patterns and predicts suspicious behavior.

It helps energy companies identify fraud cases quickly and improve monitoring systems.

---

## 🎯 Problem Statement

Energy companies face losses due to fraudulent consumption.  
This project uses Machine Learning to detect fraud based on usage patterns, payment history, and meter status.

---

## ⭐ Features

- Upload CSV dataset
- Fraud prediction using Random Forest
- Summary dashboard with metrics
- Fraud vs Non-Fraud visualization
- Top 10 fraud cases display
- Feature importance chart (model explainability)
- Download prediction results

---

## 🧠 Machine Learning Model

- Random Forest Classifier
- Model Accuracy: 99%
- Handles classification of fraudulent and non-fraudulent behavior

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
├── requirements.txt
└── README.md
```

---

## ⚙ Installation

Clone the repository:

```
git clone https://github.com/yourusername/energy-fraud-detection-ml.git
cd energy-fraud-detection-ml
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶ How to Run Project

Train the model:

```
python src/train_model.py
```

Run Streamlit app:

```
streamlit run app/streamlit_app.py
```

---

## 📊 Output

- Fraud prediction results
- Summary dashboard
- Fraud distribution chart
- Top fraud records
- Feature importance visualization
- Downloadable results file

---

## 🚀 Future Improvements

- Real-time fraud detection system
- Database integration
- Deep learning models
- API deployment
- Model monitoring system

---

## 👨‍💻 Author

Vishal Rajput
