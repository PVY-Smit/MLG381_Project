import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

#loading dataset
df = pd.read_csv("Diabetes_and_LifeStyle_Dataset.csv")

#cleaning column names
df.columns = df.columns.str.strip()

#defining the target
targetColumn = "diabetes_stage"

#dropping leakage columns if they exist
columnsToDrop = [targetColumn]
for col in ["diagnosed_diabetes", "diabetes_risk_score"]:
    if col in df.columns:
        columnsToDrop.append(col)

#save original categorical options before encoding
categoricalColumns = [col for col in df.select_dtypes(include=["object"]).columns if col != targetColumn]
categoryMaps = {}

for col in categoricalColumns:
    df[col] = df[col].astype("category")
    categoryMaps[col] = list(df[col].cat.categories)
    df[col] = df[col].cat.codes

#encoding the target
df[targetColumn] = df[targetColumn].astype("category")
targetMap = list(df[targetColumn].cat.categories)
df[targetColumn] = df[targetColumn].cat.codes

#features and target
X = df.drop(columns=columnsToDrop)
y = df[targetColumn]

#separate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Random Forest
rfModel = RandomForestClassifier(random_state=42)
rfModel.fit(X_train, y_train)
rfPred = rfModel.predict(X_test)
rfAccuracy = accuracy_score(y_test, rfPred)

#Decision Tree
dtModel = DecisionTreeClassifier(random_state=42)
dtModel.fit(X_train, y_train)
dtPred = dtModel.predict(X_test)
dtAccuracy = accuracy_score(y_test, dtPred)

print("Random Forest Accuracy:", rfAccuracy)
print("Decision Tree Accuracy:", dtAccuracy)
print(classification_report(y_test, rfPred))

#saving the best model and metadata
modelBundle = {
    "model": rfModel,
    "featureColumns": list(X.columns),
    "categoricalColumns": categoricalColumns,
    "categoryMaps": categoryMaps,
    "targetMap": targetMap
}

joblib.dump(modelBundle, "diabetesModel.pkl")
print("Model saved successfully!")