import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

# ===============================
# Project Root Path (Professional)
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "energy_fraud_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")

# Load dataset
data = pd.read_csv(DATA_PATH)

# ===============================
# Encode categorical columns
# ===============================

categorical_cols = [
    'UsageType',
    'TariffPlan',
    'Location',
    'TimeOfDay',
    'MeterStatus',
    'PaymentHistory'
]

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# ===============================
# Features and target
# ===============================

X = data.drop(columns=['UserID','MeterID','FraudReported'])
y = data['FraudReported']

# ===============================
# Train test split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Train model
# ===============================

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ===============================
# Evaluate model
# ===============================

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# ===============================
# Save model (always to main models/)
# ===============================

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("Model saved to:", MODEL_PATH)