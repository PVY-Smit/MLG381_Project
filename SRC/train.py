import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

#Loading Data
X_test=pd.read_csv("DATA/X_test.csv")
X_train=pd.read_csv("DATA/X_train.csv")
y_test=pd.read_csv("DATA/y_test.csv")
y_train=pd.read_csv("DATA/y_train.csv")

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

#Save all models
rfModelBundel ={
    "model":rfModel,
    "predictions":rfPred,
    "accuracy":rfAccuracy
}
joblib.dump(rfModelBundel,"ARTIFACTS/Diabetes_rfModel.pkl")

dtModelBundel ={
    "model":dtModel,
    "predictions":dtPred,
    "accuracy":dtAccuracy
}
joblib.dump(dtModelBundel,"ARTIFACTS/Diabetes_dtModel.pkl")

xgbModelBundel ={
    "model":xgbModel,
    "predictions":xgbPred,
    "accuracy":xgbAccuracy
}
joblib.dump(xgbModelBundel,"ARTIFACTS/Diabetes_xgbModel.pkl")