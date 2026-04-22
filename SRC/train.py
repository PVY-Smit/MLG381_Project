import os
import time
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


def _atomic_joblib_dump(obj, path: Path, **kwargs) -> None:
    """Write to a temp file, then replace the artifact (handles Windows locks / OneDrive)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp, **kwargs)

    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
        except OSError:
            break

    alt = path.with_name(path.stem + "_alt" + path.suffix)
    try:
        os.replace(tmp, alt)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    print(
        f"WARNING: Could not replace {path.name} (still open or locked — often OneDrive/antivirus). "
        f"New model saved as {alt.name}. The app prefers the newer of these two files. "
        f"When safe, delete the lock on {path.name} and re-run train to consolidate."
    )


# Loading Data (y as 1d — avoids sklearn column-vector warnings)
X_test = pd.read_csv(_DATA_DIR / "X_test_db.csv")
X_train = pd.read_csv(_DATA_DIR / "X_train_db.csv")
y_test = pd.read_csv(_DATA_DIR / "y_test_db.csv").squeeze("columns")
y_train = pd.read_csv(_DATA_DIR / "y_train_db.csv").squeeze("columns")
if hasattr(y_test, "to_numpy"):
    y_test = y_test.to_numpy().ravel()
if hasattr(y_train, "to_numpy"):
    y_train = y_train.to_numpy().ravel()

# Random Forest
rfModel = RandomForestClassifier(random_state=42, n_jobs=1)
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
_atomic_joblib_dump({"model": rfModel, "accuracy": rfAccuracy}, _ARTIFACTS_DIR / "Diabetes_rfModel.pkl", **_dump_kw)
_atomic_joblib_dump({"model": dtModel, "accuracy": dtAccuracy}, _ARTIFACTS_DIR / "Diabetes_dtModel.pkl", **_dump_kw)
_atomic_joblib_dump({"model": xgbModel, "accuracy": xgbAccuracy}, _ARTIFACTS_DIR / "Diabetes_xgbModel.pkl", **_dump_kw)

