import pandas as pd
import joblib
import numpy as np
import dash
from dash import html, dcc, Input, Output, State

# Load saved model bundle
modelBundle = joblib.load("diabetesModel.pkl")
model = modelBundle["model"]
featureColumns = modelBundle["featureColumns"]
categoricalColumns = modelBundle["categoricalColumns"]
categoryMaps = modelBundle["categoryMaps"]
targetMap = modelBundle["targetMap"]

# Load dataset for simple defaults
df = pd.read_csv("Diabetes_and_LifeStyle_Dataset.csv")
df.columns = df.columns.str.strip()

app = dash.Dash(__name__)

inputComponents = []

for col in featureColumns:
    label = html.Label(col, style={"fontWeight": "bold", "marginTop": "12px"})

    if col in categoricalColumns:
        options = [{"label": value, "value": value} for value in categoryMaps[col]]

        dropdown = dcc.Dropdown(
            id=f"{col}Input",
            options=options,
            value=categoryMaps[col][0] if len(categoryMaps[col]) > 0 else None,
            placeholder=f"Select {col}",
            style={"marginBottom": "15px"}
        )

        inputComponents.extend([label, dropdown])

    else:   
        defaultValue = float(df[col].median()) if col in df.columns else 0

        slider = dcc.Slider(
            id=f"{col}Input",
            min=0,
            max=100,
            step=1,
            value=int(defaultValue),
            tooltip={"placement": "bottom", "always_visible": False}
        )

        inputComponents.extend([label, slider])

app.layout = html.Div(
    style={
        "backgroundColor": "lightgrey",
        "minHeight": "100%",
        "padding": "30px",
        "fontFamily": "Aptos"
    },
    children=[
        html.Div(
            style={
                "maxWidth": "800px",
                "margin": "0 auto",
                "backgroundColor": "white",
                "padding": "30px",
                "borderRadius": "16px",
                "boxShadow": "0 4px 12px"
            },
            children=[
                html.H1(
                    "Diabetes Risk Decision Support System",
                    style={"textAlign": "center", "color": "black", "marginBottom": "10px"}
                ),

                html.P(
                    "Enter patient's information to predict their diabetes risk.",
                    style={"textAlign": "center", "marginBottom": "25px", "fontWeight": "bold"}
                ),

                *inputComponents,

                html.Button(
                    "Predict",
                    id="predictButton",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "padding": "12px",
                        "marginTop": "20px",
                        "backgroundColor": "purple",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "8px",
                        "fontSize": "16px",
                        "cursor": "pointer"
                    }
                ),

                html.Div(
                    id="resultOutput",
                    style={
                        "marginTop": "25px",
                        "fontSize": "20px",
                        "fontWeight": "bold",
                        "textAlign": "center"
                    }
                )
            ]
        )
    ]
)

stateInputs = [State(f"{col}Input", "value") for col in featureColumns]

@app.callback(
    Output("resultOutput", "children"),
    Input("predictButton", "n_clicks"),
    stateInputs
)
def makePrediction(clicks, *values):
    if clicks == 0:
        return ""

    inputData = {}

    for col, value in zip(featureColumns, values):
        if col in categoricalColumns:
            categories = categoryMaps[col]
            if value in categories:
                inputData[col] = categories.index(value)
            else:
                inputData[col] = 0
        else:
            inputData[col] = float(value) if value is not None else 0

    inputFrame = pd.DataFrame([inputData])
    predictionCode = model.predict(inputFrame)[0]
    predictionLabel = targetMap[predictionCode]

    return html.Div(
        f"Predicted Diabetes Stage: {predictionLabel}",
        style={"color": "green"}
    )

if __name__ == "__main__":
    app.run(debug=True)