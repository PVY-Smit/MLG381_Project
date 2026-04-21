from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "DATA"
_ARTIFACTS_DIR = _REPO_ROOT / "ARTIFACTS"

X_train = pd.read_csv(_DATA_DIR / "X_train_hd.csv")
X_test = pd.read_csv(_DATA_DIR / "X_test_hd.csv")
y_train = pd.read_csv(_DATA_DIR / "y_train_hd.csv").squeeze("columns")
y_test = pd.read_csv(_DATA_DIR / "y_test_hd.csv").squeeze("columns")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

rfModelBundle = {
    "model": model
}

joblib.dump(rfModelBundle, _ARTIFACTS_DIR / "Heart_rfModel.pkl")
print("\nSaved model to ARTIFACTS/Heart_rfModel.pkl")