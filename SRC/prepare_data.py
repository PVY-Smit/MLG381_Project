import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "DATA"
_ARTIFACTS_DIR = _REPO_ROOT / "ARTIFACTS"
os.makedirs(_ARTIFACTS_DIR, exist_ok=True)

# loading dataset
df = pd.read_csv(_DATA_DIR / "Diabetes_and_LifeStyle_Dataset.csv")

# cleaning column names
df.columns = df.columns.str.strip()

# defining the target
targetColumn = "diabetes_stage"

# dropping leakage columns if they exist
columnsToDrop = [targetColumn]
for col in ["diagnosed_diabetes", "diabetes_risk_score"]:
    if col in df.columns:
        columnsToDrop.append(col)

# save original categorical options before encoding
categoricalColumns = [col for col in df.select_dtypes(include=["object"]).columns if col != targetColumn]
categoryMaps = {}

for col in categoricalColumns:
    df[col] = df[col].astype("category")
    categoryMaps[col] = list(df[col].cat.categories)
    df[col] = df[col].cat.codes

# encoding the target
df[targetColumn] = df[targetColumn].astype("category")
targetMap = list(df[targetColumn].cat.categories)
df[targetColumn] = df[targetColumn].cat.codes

# features and target
X = df.drop(columns=columnsToDrop)
y = df[targetColumn]

# separate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Prepairing UI Model Bundle

def worst_stage_index(labels: list) -> int:
    """Pick class index used as 'high risk' for SHAP and messaging."""
    lowered = [str(x).lower() for x in labels]
    for i, name in enumerate(lowered):
        if any(
            k in name
            for k in ("type 2", "type2", "t2dm", "stage 3", "severe")
        ):
            return i
    for i, name in enumerate(lowered):
        if "type 1" in name or "type1" in name or "t1dm" in name:
            return i
    for i, name in enumerate(lowered):
        if "pre" in name or "prediabetes" in name or "borderline" in name:
            return i
    return len(labels) - 1
worstStageIndex = worst_stage_index(targetMap)

SEMANTIC_MAX = {
    "alcohol_consumption_per_week": 20.0,
    "sleep_hours_per_day": 24.0,
    "screen_time_hours_per_day": 24.0,
    "physical_activity_minutes_per_week": 3000.0,
    "age": 120.0,
    "heart_rate": 220.0,
    "bmi": 54.0,
    "waist_to_hip_ratio": 1.0,
    "systolic_bp": 250.0,
    "diastolic_bp": 180.0,
    "cholesterol_total": 600.0,
    "hdl_cholesterol": 150.0,
    "ldl_cholesterol": 400.0,
    "triglycerides": 2000.0,
    "glucose_fasting": 500.0,
    "glucose_postprandial": 600.0,
    "insulin_level": 200.0,
    "hba1c": 20.0,
    "diet_score": 100.0,
}

SEMANTIC_MIN = {
    "age": 0.0,
    "sleep_hours_per_day": 0.0,
    "screen_time_hours_per_day": 0.0,
    "physical_activity_minutes_per_week": 0.0,
    "alcohol_consumption_per_week": 0.0,
    "bmi": 19.0,
    "heart_rate": 30.0,
}


def build_slider_bounds(feature_name: str, series: pd.Series) -> dict:
    key = feature_name.lower()
    if key == "diet_score":
        return {"min": 0.0, "max": 100.0}
    p1 = float(series.quantile(0.01))
    p99 = float(series.quantile(0.99))
    data_min = float(series.min())
    data_max = float(series.max())
    lo = max(p1, data_min)
    hi = min(p99, data_max)
    if hi <= lo:
        hi = lo + 1.0
    smin = SEMANTIC_MIN.get(key)
    smax = SEMANTIC_MAX.get(key)
    if smin is not None:
        lo = max(lo, smin)
    if smax is not None:
        hi = min(hi, smax)
    if hi <= lo:
        hi = lo + 1.0
    return {"min": lo, "max": hi}


def build_feature_quantiles(series: pd.Series) -> dict:
    return {
        "p1": float(series.quantile(0.01)),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "p99": float(series.quantile(0.99)),
    }


numericColumns = [c for c in X_train.columns if c not in categoricalColumns]

sliderBounds = {c: build_slider_bounds(c, X_train[c]) for c in numericColumns}

featureQuantiles = {c: build_feature_quantiles(X_train[c]) for c in numericColumns}

rng = np.random.default_rng(42)
bg_n = min(200, len(X_train))
bg_idx = rng.choice(X_train.index, size=bg_n, replace=False)
shapBackground = X_train.loc[bg_idx].to_numpy(dtype=np.float64)

#CAPTURING MODELS
UIModelBundle = {
    "worstStageIndex": worstStageIndex,
    "shapBackground": shapBackground,
    "sliderBounds": sliderBounds,
    "featureQuantiles": featureQuantiles
}

joblib.dump(UIModelBundle, _ARTIFACTS_DIR / "UIModel.pkl")

DataModelBundle={
    "featureColumns": list(X.columns),
    "categoricalColumns": categoricalColumns,
    "categoryMaps": categoryMaps,
    "targetMap": targetMap,
}
joblib.dump(DataModelBundle, _ARTIFACTS_DIR / "DataModel.pkl")

#Capturing Formatted Data
X_test.to_csv(_DATA_DIR / "X_test.csv", index=False)
y_test.to_csv(_DATA_DIR / "y_test.csv", index=False)
X_train.to_csv(_DATA_DIR / "X_train.csv", index=False)
y_train.to_csv(_DATA_DIR / "y_train.csv", index=False)
