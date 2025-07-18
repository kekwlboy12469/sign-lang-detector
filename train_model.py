
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_CSV = "sign_data/signs_63.csv"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/sign_model_63.pkl"
ENC_PATH = f"{MODEL_DIR}/label_encoder.pkl"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"

import os
os.makedirs(MODEL_DIR, exist_ok=True)


df = pd.read_csv(DATA_CSV)

X = df.drop("label", axis=1).values.astype(float)
y = df["label"].astype(str)


le = LabelEncoder()
y_enc = le.fit_transform(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))


joblib.dump(model, MODEL_PATH)
joblib.dump(le, ENC_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"📦 Saved model:  {MODEL_PATH}")
print(f"📦 Saved labels: {ENC_PATH}")
print(f"📦 Saved scaler: {SCALER_PATH}")




