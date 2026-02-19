import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

data = pd.read_csv(r"C:\Users\Vishal Rajput\OneDrive\Desktop\energy-fraud-detection-ml\energy-fraud-detection-ml\data\energy_fraud_dataset.csv")

# Encode categorical columns
categorical_cols = ['UsageType','TariffPlan','Location','TimeOfDay','MeterStatus','PaymentHistory']

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features and target
X = data.drop(columns=['UserID','MeterID','FraudReported'])
y = data['FraudReported']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_model.pkl")

print("Model saved successfully!")
