import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "DATA"
_ARTIFACTS_DIR = _REPO_ROOT / "ARTIFACTS"
os.makedirs(_ARTIFACTS_DIR, exist_ok=True)

#Loading Data
X_test=pd.read_csv(_DATA_DIR / "X_test.csv")
X_train=pd.read_csv(_DATA_DIR / "X_train.csv")
y_test=pd.read_csv(_DATA_DIR / "y_test.csv")
y_train=pd.read_csv(_DATA_DIR / "y_train.csv")

# Random Forest
rfModel = RandomForestClassifier(random_state=42)
rfModel.fit(X_train, y_train)
rfPred = rfModel.predict(X_test)
rfAccuracy = accuracy_score(y_test, rfPred)

# Decision Tree
dtModel = DecisionTreeClassifier(random_state=42)
dtModel.fit(X_train, y_train)
dtPred = dtModel.predict(X_test)
dtAccuracy = accuracy_score(y_test, dtPred)

#XGBoost 
xgbModel = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
xgbModel.fit(X_train, y_train)
xgbPred = xgbModel.predict(X_test)
xgbAccuracy = accuracy_score(y_test,xgbPred)

#return accuracy
print("Random Forest Accuracy:", rfAccuracy)
print("Decision Tree Accuracy:", dtAccuracy)
print("XGBoost accuracy", xgbAccuracy)
print(classification_report(y_test, rfPred))

# Save models only (no prediction arrays — smaller files, fine for Git/deploy).
# Dash loads only rfModelBundle["model"].
_dump_kw = dict(compress=3)
joblib.dump({"model": rfModel, "accuracy": rfAccuracy}, _ARTIFACTS_DIR / "Diabetes_rfModel.pkl", **_dump_kw)
joblib.dump({"model": dtModel, "accuracy": dtAccuracy}, _ARTIFACTS_DIR / "Diabetes_dtModel.pkl", **_dump_kw)
joblib.dump({"model": xgbModel, "accuracy": xgbAccuracy}, _ARTIFACTS_DIR / "Diabetes_xgbModel.pkl", **_dump_kw)