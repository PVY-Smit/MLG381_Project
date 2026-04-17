from typing import Optional
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import dash
from dash import html, dcc, Input, Output, State, no_update
from dash.exceptions import PreventUpdate


def friendly_feature_label(column: str) -> str:
    text = column.replace("_", " ").strip().title()
    for wrong, right in (
        ("Bmi", "BMI"),
        ("Hdl", "HDL"),
        ("Ldl", "LDL"),
        ("Bp", "BP"),
        ("Hba1C", "HbA1c"),
        ("Hba1c", "HbA1c"),
    ):
        text = text.replace(wrong, right)
    return text


def col_key(name: str) -> str:
    return str(name).strip().lower()


FEATURE_HELP = {
    "age": "Patient age in years.",
    "gender": "Sex recorded for the patient.",
    "ethnicity": "Self-reported ethnic group used for population context.",
    "education_level": "Highest completed education level.",
    "income_level": "Household income band.",
    "employment_status": "Current work situation.",
    "smoking_status": "Whether and how often the patient smokes.",
    "alcohol_consumption_per_week": "Typical number of alcoholic drinks consumed per week.",
    "physical_activity_minutes_per_week": "Minutes per week spent in moderate or vigorous activity.",
    "diet_score": "Summary score of diet quality (higher usually means healthier eating patterns).",
    "sleep_hours_per_day": "Average hours of sleep in a 24-hour day.",
    "screen_time_hours_per_day": "Average leisure screen time per day.",
    "family_history_diabetes": "Whether close relatives have diabetes.",
    "hypertension_history": "History of high blood pressure.",
    "cardiovascular_history": "History of heart or stroke-related conditions.",
    "bmi": "Body mass index; weight relative to height.",
    "waist_to_hip_ratio": "Waist measurement divided by hip measurement; reflects body fat distribution.",
    "systolic_bp": "Top number of blood pressure, pressure when the heart beats.",
    "diastolic_bp": "Bottom number of blood pressure, pressure between beats.",
    "heart_rate": "Resting heart rate in beats per minute.",
    "cholesterol_total": "Total blood cholesterol.",
    "hdl_cholesterol": "HDL is often called 'good' cholesterol.",
    "ldl_cholesterol": "LDL is often called 'bad' cholesterol.",
    "triglycerides": "A type of fat (lipid) in the blood.",
    "glucose_fasting": "Blood sugar measured after not eating (fasting).",
    "glucose_postprandial": "Blood sugar measured after a meal (post-meal / postprandial).",
    "insulin_level": "Blood insulin concentration.",
    "hba1c": "Average blood sugar over roughly the past 3 months (glycated haemoglobin).",
}

FEATURE_UNITS = {
    "age": "years",
    "alcohol_consumption_per_week": "drinks / week",
    "physical_activity_minutes_per_week": "minutes / week",
    "diet_score": "score 0–100",
    "sleep_hours_per_day": "hours / day",
    "screen_time_hours_per_day": "hours / day",
    "bmi": "kg / m²",
    "waist_to_hip_ratio": "ratio",
    "systolic_bp": "mmHg",
    "diastolic_bp": "mmHg",
    "heart_rate": "bpm",
    "cholesterol_total": "mg/dL",
    "hdl_cholesterol": "mg/dL",
    "ldl_cholesterol": "mg/dL",
    "triglycerides": "mg/dL",
    "glucose_fasting": "mg/dL",
    "glucose_postprandial": "mg/dL",
    "insulin_level": "µIU/mL",
    "hba1c": "%",
    "family_history_diabetes": "0–1",
    "hypertension_history": "0–1",
    "cardiovascular_history": "0–1",
}

SECTION_HELP = {
    "Demographics": "Who the patient is: basic traits and social context used for risk context.",
    "Positive influences": "Behaviours and markers that usually improve when raised (activity, sleep, HDL, diet score).",
    "Negative influences": "Behaviours and lab markers that usually worsen metabolic risk when out of range.",
    "Other clinical indicators": "Remaining risk-related fields from the dataset.",
}

DEMOGRAPHICS = {
    "age",
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "employment_status",
}
POSITIVE_INFLUENCES = {
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "hdl_cholesterol",
}
NEGATIVE_INFLUENCES = {
    "smoking_status",
    "alcohol_consumption_per_week",
    "screen_time_hours_per_day",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "ldl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
}

HIGHER_BETTER = {
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "hdl_cholesterol",
}


def feature_direction(feature: str) -> str:
    k = col_key(feature)
    if k in HIGHER_BETTER:
        return "higher_better"
    return "higher_worse"


SHARP_CARD = {
    "maxWidth": "960px",
    "margin": "0 auto",
    "backgroundColor": "white",
    "padding": "30px",
    "borderRadius": "0",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.12)",
    "border": "1px solid #ccc",
}

SECTION_TITLE = {
    "fontSize": "18px",
    "fontWeight": "bold",
    "marginTop": "28px",
    "marginBottom": "6px",
    "borderBottom": "2px solid #333",
    "paddingBottom": "6px",
}

SECTION_SUB = {
    "fontSize": "13px",
    "color": "#444",
    "marginBottom": "12px",
}

GRID = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
    "columnGap": "28px",
    "rowGap": "24px",
    "alignItems": "start",
}

MODAL_BACKDROP_BASE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "right": 0,
    "bottom": 0,
    "backgroundColor": "rgba(0,0,0,0.5)",
    "zIndex": 1000,
    "justifyContent": "center",
    "alignItems": "center",
    "padding": "24px",
}

MODAL_PANEL = {
    "backgroundColor": "white",
    "maxWidth": "640px",
    "width": "100%",
    "maxHeight": "90vh",
    "overflowY": "auto",
    "padding": "28px",
    "borderRadius": "0",
    "border": "1px solid #222",
    "boxShadow": "0 8px 32px rgba(0,0,0,0.25)",
}

# Repo root (parent of SRC/) so ARTIFACTS/DATA resolve on Render regardless of process cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACTS_DIR = _REPO_ROOT / "ARTIFACTS"

_rf_path = _ARTIFACTS_DIR / "Diabetes_rfModel.pkl"
if not _rf_path.is_file():
    raise FileNotFoundError(
        f"Missing {_rf_path}. From the project root run: "
        "python SRC/prepare_data.py && python SRC/train.py "
        "(needs DATA/Diabetes_and_LifeStyle_Dataset.csv)."
    )

rfModelBundle = joblib.load(_rf_path)
model = rfModelBundle["model"]

dataModelBundle =joblib.load(_ARTIFACTS_DIR / "DataModel.pkl")
featureColumns = dataModelBundle["featureColumns"]
categoricalColumns = list(dataModelBundle["categoricalColumns"])
categoryMaps = dataModelBundle["categoryMaps"]
targetMap = list(dataModelBundle["targetMap"])

uiModelBundle =joblib.load(_ARTIFACTS_DIR / "UIModel.pkl")
worstStageIndex = int(uiModelBundle.get("worstStageIndex", len(targetMap) - 1))
shapBackground = uiModelBundle.get("shapBackground")
sliderBounds = dict(uiModelBundle.get("sliderBounds") or {})
featureQuantiles = dict(uiModelBundle.get("featureQuantiles") or {})


def _numeric_median_default(col: str, lo: float, hi: float) -> float:
    """Median for slider defaults from training quantiles (avoids loading full CSV at import)."""
    q = featureQuantiles.get(col)
    if q:
        if "p50" in q:
            try:
                return float(q["p50"])
            except (TypeError, ValueError):
                pass
        if "p25" in q and "p75" in q:
            try:
                return (float(q["p25"]) + float(q["p75"])) / 2.0
            except (TypeError, ValueError):
                pass
    return float(lo + hi) / 2.0


# UI bounds (extra changes; also applied in diabetesModel for new bundles)
sliderBounds["diet_score"] = {"min": 0.0, "max": 100.0}
for _bmi_col in featureColumns:
    if col_key(_bmi_col) == "bmi":
        sliderBounds[_bmi_col] = {"min": 19.0, "max": 54.0}
        break
for _w in featureColumns:
    if col_key(_w) == "waist_to_hip_ratio":
        b = sliderBounds.get(_w, {"min": 0.0, "max": 1.0})
        sliderBounds[_w] = {"min": float(b["min"]), "max": min(1.0, float(b["max"]))}
        break
for _a in featureColumns:
    if col_key(_a) == "alcohol_consumption_per_week":
        b = sliderBounds.get(_a, {"min": 0.0, "max": 20.0})
        sliderBounds[_a] = {"min": float(b["min"]), "max": min(20.0, float(b["max"]))}
        break

_explainer = None
_ENABLE_SHAP = os.getenv("ENABLE_SHAP", "false").strip().lower() in ("1", "true", "yes", "on")


def get_explainer():
    global _explainer
    if not _ENABLE_SHAP:
        return None
    if _explainer is None and shapBackground is not None:
        import shap

        try:
            _explainer = shap.TreeExplainer(model, data=shapBackground)
        except Exception:
            _explainer = shap.TreeExplainer(model)
    return _explainer


def assign_section(col: str) -> str:
    k = col_key(col)
    if k in DEMOGRAPHICS:
        return "Demographics"
    if k in POSITIVE_INFLUENCES:
        return "Positive influences"
    if k in NEGATIVE_INFLUENCES:
        return "Negative influences"
    return "Other clinical indicators"


def build_slider_step(col: str, lo: float, hi: float) -> float:
    span = hi - lo
    if span <= 0:
        return 1.0
    k = col_key(col)
    if k == "sleep_hours_per_day":
        return 0.5
    if k == "alcohol_consumption_per_week":
        return 1.0
    if k == "diet_score":
        return 1.0
    if k == "bmi":
        return 0.1
    if k in ("waist_to_hip_ratio", "hba1c"):
        return round(min(0.05, span / 40), 2) or 0.01
    if span <= 30:
        return 1.0
    return max(1.0, round(span / 100))


def is_binary_numeric_column(col: str) -> bool:
    if col in categoricalColumns:
        return False
    b = sliderBounds.get(col)
    if not b:
        return False
    lo, hi = float(b["min"]), float(b["max"])
    return lo <= 0.01 and 0.99 <= hi <= 1.01


def numeric_columns_with_slider_input() -> list:
    out = []
    for c in featureColumns:
        if c in categoricalColumns:
            continue
        if is_binary_numeric_column(c):
            continue
        out.append(c)
    return out


def slider_value_as_display_text(sv) -> str:
    """String for text inputs when syncing from slider (avoid browser int coercion)."""
    try:
        x = float(sv)
    except (TypeError, ValueError):
        return ""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.8f}".rstrip("0").rstrip(".")
    return s if s else str(x)


def numeric_text_input_initial_value(med: float) -> str:
    return slider_value_as_display_text(med)


def build_field(col: str) -> html.Div:
    friendly = friendly_feature_label(col)
    k = col_key(col)
    help_text = FEATURE_HELP.get(k, "")
    unit = FEATURE_UNITS.get(k, "")
    unit_display = f" ({unit})" if unit else ""

    label = html.Span(
        [
            html.Span(friendly, style={"fontWeight": "bold"}),
            html.Span(
                unit_display,
                style={"fontSize": "12px", "color": "#555", "fontWeight": "normal"},
            ),
        ],
        title=help_text or None,
        style={
            "display": "block",
            "cursor": "help" if help_text else "default",
        },
    )

    if col in categoricalColumns:
        options = [{"label": v, "value": v} for v in categoryMaps[col]]
        control = dcc.Dropdown(
            id=f"{col}Input",
            options=options,
            value=categoryMaps[col][0] if categoryMaps[col] else None,
            placeholder=f"Select {friendly}",
            clearable=False,
            searchable=False,
            style={"width": "100%"},
        )
    elif is_binary_numeric_column(col):
        bounds = sliderBounds.get(col, {"min": 0.0, "max": 1.0})
        lo_b, hi_b = float(bounds["min"]), float(bounds["max"])
        raw_bin = _numeric_median_default(col, lo_b, hi_b)
        try:
            bin_v = int(round(float(raw_bin)))
        except (TypeError, ValueError):
            bin_v = 0
        if bin_v not in (0, 1):
            bin_v = 0
        control = dcc.RadioItems(
            id=f"{col}Input",
            options=[
                {"label": "No", "value": 0},
                {"label": "Yes", "value": 1},
            ],
            value=bin_v,
            inline=True,
            style={"marginTop": "4px"},
        )
    elif k == "bmi":
        bounds = sliderBounds.get(col, {"min": 19.0, "max": 54.0})
        lo, hi = float(bounds["min"]), float(bounds["max"])
        raw_med = _numeric_median_default(col, lo, hi)
        try:
            med = float(raw_med)
        except (TypeError, ValueError):
            med = (lo + hi) / 2
        med = max(lo, min(hi, med))
        step = build_slider_step(col, lo, hi)
        tooltip = {"placement": "bottom", "always_visible": False}
        if unit:
            tooltip["template"] = "{value:.2f} " + unit.replace(" / ", "/")
        control = html.Div(
            [
                dcc.RadioItems(
                    id="bmiEntryModeInput",
                    options=[
                        {"label": "Slider", "value": "slider"},
                        {"label": "Calculator (kg & m)", "value": "calculator"},
                    ],
                    value="slider",
                    inline=True,
                    style={"marginBottom": "10px"},
                ),
                html.Div(
                    id="bmiCalcRow",
                    style={"display": "none", "marginTop": "8px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "alignItems": "end"},
                            children=[
                                html.Div(
                                    [
                                        html.Label("Weight (kg)", style={"fontSize": "12px"}),
                                        dcc.Input(
                                            id="bmiCalcWeightKg",
                                            type="number",
                                            step="any",
                                            placeholder="e.g. 80",
                                            style={"width": "120px", "padding": "6px"},
                                        ),
                                    ],
                                    style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Height (m)", style={"fontSize": "12px"}),
                                        dcc.Input(
                                            id="bmiCalcHeightM",
                                            type="number",
                                            step="any",
                                            placeholder="e.g. 1.75",
                                            style={"width": "120px", "padding": "6px"},
                                        ),
                                    ],
                                    style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                                ),
                                html.Button(
                                    "Apply BMI",
                                    id="bmiCalcApply",
                                    n_clicks=0,
                                    type="button",
                                    style={
                                        "padding": "8px 14px",
                                        "border": "1px solid #333",
                                        "background": "#f5f5f5",
                                        "cursor": "pointer",
                                        "borderRadius": "0",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="bmiSliderRow",
                    style={"display": "block"},
                    children=[
                        html.P(
                            "Value used for prediction (decimals with . or , ; slider updates this when moved):",
                            style={"fontSize": "12px", "color": "#444", "margin": "10px 0 4px 0"},
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "alignItems": "center",
                                "gap": "14px",
                                "width": "100%",
                            },
                            children=[
                                dcc.Input(
                                    id=f"{col}Input",
                                    type="text",
                                    inputMode="numeric",
                                    debounce=True,
                                    value=numeric_text_input_initial_value(med),
                                    style={
                                        "width": "6.5rem",
                                        "minWidth": "6.5rem",
                                        "flexShrink": "0",
                                        "padding": "8px",
                                        "boxSizing": "border-box",
                                    },
                                ),
                                html.Div(
                                    style={"flex": "1", "minWidth": "120px"},
                                    children=[
                                        dcc.Slider(
                                            id="bmiSlider",
                                            min=lo,
                                            max=hi,
                                            step=step,
                                            value=med,
                                            tooltip=tooltip,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            style={"display": "flex", "flexDirection": "column", "gap": "4px"},
        )
    else:
        bounds = sliderBounds.get(col, {"min": 0.0, "max": 100.0})
        lo, hi = float(bounds["min"]), float(bounds["max"])
        raw_med = _numeric_median_default(col, lo, hi)
        try:
            med = float(raw_med)
        except (TypeError, ValueError):
            med = (lo + hi) / 2
        med = max(lo, min(hi, med))
        step = build_slider_step(col, lo, hi)
        tooltip = {"placement": "bottom", "always_visible": False}
        if unit:
            tooltip["template"] = "{value:.2f} " + unit.replace(" / ", "/")

        control = html.Div(
            style={
                "display": "flex",
                "flexDirection": "row",
                "alignItems": "center",
                "gap": "14px",
                "width": "100%",
            },
            children=[
                dcc.Input(
                    id=f"{col}Input",
                    type="text",
                    inputMode="numeric",
                    debounce=True,
                    value=numeric_text_input_initial_value(med),
                    style={
                        "width": "6.5rem",
                        "minWidth": "6.5rem",
                        "flexShrink": "0",
                        "padding": "8px",
                        "boxSizing": "border-box",
                    },
                ),
                html.Div(
                    style={"flex": "1", "minWidth": "120px"},
                    children=[
                        dcc.Slider(
                            id=f"{col}Slider",
                            min=lo,
                            max=hi,
                            step=step,
                            value=med,
                            tooltip=tooltip,
                        ),
                    ],
                ),
            ],
        )

    return html.Div(
        [label, control],
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "8px",
            "minWidth": "0",
            "width": "100%",
        },
    )


def section_block(title: str, subtitle: str, columns: list) -> html.Div:
    if not columns:
        return html.Div()
    cells = [build_field(c) for c in columns]
    help_line = SECTION_HELP.get(title, "")
    return html.Div(
        [
            html.H2(title, style=SECTION_TITLE, title=help_line or None),
            html.P(subtitle, style=SECTION_SUB),
            html.Div(cells, style=GRID),
        ]
    )


def order_columns_for_sections() -> dict:
    buckets = {
        "Demographics": [],
        "Positive influences": [],
        "Negative influences": [],
        "Other clinical indicators": [],
    }
    for col in featureColumns:
        buckets[assign_section(col)].append(col)
    return buckets


buckets = order_columns_for_sections()

section_layout = [
    section_block(
        "Demographics",
        "Basic information and background (including age and profile dropdowns).",
        buckets["Demographics"],
    ),
    section_block(
        "Positive influences",
        "Factors that typically support lower risk when they are in a healthy range.",
        buckets["Positive influences"],
    ),
    section_block(
        "Negative influences",
        "Lifestyle and clinical markers that often track with higher risk when out of range.",
        buckets["Negative influences"],
    ),
    section_block(
        "Other clinical indicators",
        "Additional fields used by the model.",
        buckets["Other clinical indicators"],
    ),
]


def _notebook_gallery_items() -> list[dict]:
    manifest_path = (
        Path(__file__).resolve().parent / "assets" / "notebook_figures" / "manifest.json"
    )
    if not manifest_path.is_file():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


dash_app = dash.Dash(__name__)
server = dash_app.server

_decision_support_children = [
    html.H1(
        "Diabetes Risk Decision Support System",
        style={
            "textAlign": "center",
            "color": "#111",
            "marginBottom": "10px",
            "fontSize": "26px",
        },
    ),
    html.P(
        "Enter patient information, then press Predict to see stage estimate and guidance. "
        "Tip: hover field labels for more information.",
        style={"textAlign": "center", "marginBottom": "20px", "fontWeight": "600"},
    ),
    *section_layout,
    html.Button(
        "Predict",
        id="predictButton",
        n_clicks=0,
        style={
            "width": "100%",
            "padding": "14px",
            "marginTop": "28px",
            "backgroundColor": "#4a148c",
            "color": "white",
            "border": "none",
            "borderRadius": "0",
            "fontSize": "16px",
            "cursor": "pointer",
            "fontWeight": "bold",
        },
    ),
]

# Modal lives outside dcc.Tabs so it stays mounted and callbacks work when switching tabs.
_modal_layer = html.Div(
    id="modalBackdrop",
    style={**MODAL_BACKDROP_BASE, "display": "none"},
    children=[
        html.Div(
            style=MODAL_PANEL,
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "marginBottom": "16px",
                    },
                    children=[
                        html.H3(
                            "Results",
                            style={"margin": 0, "fontSize": "20px"},
                        ),
                        html.Button(
                            "Close",
                            id="modalClose",
                            n_clicks=0,
                            style={
                                "padding": "8px 16px",
                                "border": "1px solid #333",
                                "background": "#fff",
                                "cursor": "pointer",
                                "borderRadius": "0",
                                "fontWeight": "600",
                            },
                        ),
                    ],
                ),
                html.Div(id="resultsModalBody"),
            ],
        )
    ],
)

_gallery_items = _notebook_gallery_items()
_notebook_gallery_children: list = [
    html.H1(
        "Notebook analysis figures",
        style={
            "textAlign": "center",
            "color": "#111",
            "marginBottom": "10px",
            "fontSize": "26px",
        },
    ),
    html.P(
        "Plots saved from the project Jupyter notebooks (embedded cell outputs). "
        "To refresh after you change a notebook, run: python SRC/extract_notebook_figures.py "
        "from the project root.",
        style={"textAlign": "center", "marginBottom": "8px", "fontWeight": "600"},
    ),
    html.P(
        "K-Means.ipynb is included in the extractor when it contains figure outputs; "
        "the current checked-in file has none.",
        style={"textAlign": "center", "marginBottom": "24px", "color": "#555", "fontSize": "14px"},
    ),
]
if not _gallery_items:
    _notebook_gallery_children.append(
        html.P(
            "No figures found. Add manifest.json and PNGs under SRC/assets/notebook_figures/, "
            "or run python SRC/extract_notebook_figures.py.",
            style={"textAlign": "center", "color": "#b71c1c"},
        )
    )
else:
    for idx, item in enumerate(_gallery_items, start=1):
        nb_label = str(item.get("notebook", "Notebook")).replace("_", " ")
        cap = str(item.get("caption", "Figure"))
        fn = item["file"]
        _notebook_gallery_children.append(
            html.Div(
                style={"marginBottom": "36px"},
                children=[
                    html.H4(
                        f"{nb_label} — {cap} (figure {idx})",
                        style={"marginBottom": "12px", "color": "#222"},
                    ),
                    html.Img(
                        src=dash_app.get_asset_url(f"notebook_figures/{fn}"),
                        alt=f"{nb_label} figure {idx}",
                        style={
                            "maxWidth": "100%",
                            "height": "auto",
                            "display": "block",
                            "border": "1px solid #ccc",
                            "backgroundColor": "#fff",
                        },
                    ),
                ],
            )
        )

dash_app.layout = html.Div(
    style={
        "backgroundColor": "#e8e8e8",
        "minHeight": "100%",
        "padding": "30px",
        "fontFamily": "system-ui, Segoe UI, Roboto, sans-serif",
    },
    children=[
        html.Div(
            style={"maxWidth": "960px", "margin": "0 auto"},
            children=[
        dcc.Tabs(
            id="appMainTabs",
            value="tab-decision",
            persistence=True,
            persistence_type="session",
            colors={
                "border": "#222",
                "primary": "#4a148c",
                "background": "#f5f5f5",
            },
            style={"marginBottom": "4px"},
            children=[
                dcc.Tab(
                    label="Decision support",
                    value="tab-decision",
                    style={"padding": "10px 14px", "fontWeight": "600"},
                    selected_style={"padding": "10px 14px", "fontWeight": "700"},
                    children=[
                        html.Div(style=SHARP_CARD, children=_decision_support_children),
                    ],
                ),
                dcc.Tab(
                    label="Notebook figures",
                    value="tab-notebook",
                    style={"padding": "10px 14px", "fontWeight": "600"},
                    selected_style={"padding": "10px 14px", "fontWeight": "700"},
                    children=[
                        html.Div(style=SHARP_CARD, children=_notebook_gallery_children),
                    ],
                ),
            ],
        ),
        _modal_layer,
            ],
        ),
    ],
)

stateInputs = [State(f"{col}Input", "value") for col in featureColumns]


def _parse_clamped_numeric(value, lo: float, hi: float) -> float:
    if value is None:
        return float(lo)
    if isinstance(value, (int, float, np.floating)):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return float(lo)
        return max(float(lo), min(float(hi), v))
    s = str(value).strip().replace(",", ".")
    if s == "":
        return float(lo)
    try:
        v = float(s)
    except ValueError:
        return float(lo)
    return max(float(lo), min(float(hi), v))


def try_parse_optional_float(text) -> Optional[float]:
    """Parse user-typed number; None if empty or not a number (do not coerce to min)."""
    if text is None:
        return None
    s = str(text).strip().replace(",", ".")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def collect_input_frame(values):
    input_data = {}
    for col, value in zip(featureColumns, values):
        if col in categoricalColumns:
            cats = categoryMaps[col]
            if value in cats:
                input_data[col] = cats.index(value)
            else:
                input_data[col] = 0
        elif is_binary_numeric_column(col):
            try:
                iv = int(value)
            except (TypeError, ValueError):
                iv = 0
            input_data[col] = 0 if iv not in (0, 1) else iv
        else:
            bounds = sliderBounds.get(col, {"min": 0.0, "max": 100.0})
            lo, hi = float(bounds["min"]), float(bounds["max"])
            input_data[col] = _parse_clamped_numeric(value, lo, hi)
    return pd.DataFrame([input_data], columns=featureColumns)


BMI_COL = next((c for c in featureColumns if col_key(c) == "bmi"), None)
NUMERIC_SLIDER_SYNC_COLS = numeric_columns_with_slider_input()

for _sync_col in NUMERIC_SLIDER_SYNC_COLS:
    if col_key(_sync_col) == "bmi":
        continue

    def _make_slider_pusher(c):
        lo = float(sliderBounds[c]["min"])
        hi = float(sliderBounds[c]["max"])

        def _slider_to_text(sv, text_in):
            try:
                s_num = float(sv)
            except (TypeError, ValueError):
                return no_update
            try:
                t_num = _parse_clamped_numeric(text_in, lo, hi)
                if abs(t_num - s_num) < 1e-8:
                    return no_update
            except (TypeError, ValueError):
                pass
            return slider_value_as_display_text(sv)

        dash_app.callback(
            Output(f"{c}Input", "value", allow_duplicate=True),
            Input(f"{c}Slider", "value"),
            State(f"{c}Input", "value"),
            prevent_initial_call="initial_duplicate",
        )(_slider_to_text)

    def _make_input_puller(c):
        lo = float(sliderBounds[c]["min"])
        hi = float(sliderBounds[c]["max"])

        def _text_to_slider(txt, sl_cur):
            raw = try_parse_optional_float(txt)
            if raw is None:
                raise PreventUpdate
            v = max(lo, min(hi, raw))
            try:
                sc = float(sl_cur)
            except (TypeError, ValueError):
                sc = None
            if sc is not None and abs(float(v) - sc) < 1e-8:
                raise PreventUpdate
            return float(v)

        dash_app.callback(
            Output(f"{c}Slider", "value", allow_duplicate=True),
            Input(f"{c}Input", "value"),
            State(f"{c}Slider", "value"),
            prevent_initial_call=True,
        )(_text_to_slider)

    _make_slider_pusher(_sync_col)
    _make_input_puller(_sync_col)

if BMI_COL:
    _bmi_lo = float(sliderBounds.get(BMI_COL, {"min": 19.0, "max": 54.0})["min"])
    _bmi_hi = float(sliderBounds.get(BMI_COL, {"min": 19.0, "max": 54.0})["max"])

    @dash_app.callback(
        Output(f"{BMI_COL}Input", "value", allow_duplicate=True),
        Input("bmiSlider", "value"),
        State(f"{BMI_COL}Input", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def _push_bmi_slider(sv, text_in):
        try:
            s_num = float(sv)
        except (TypeError, ValueError):
            return no_update
        try:
            t_num = _parse_clamped_numeric(text_in, _bmi_lo, _bmi_hi)
            if abs(t_num - s_num) < 1e-8:
                return no_update
        except (TypeError, ValueError):
            pass
        return slider_value_as_display_text(sv)

    @dash_app.callback(
        Output("bmiSlider", "value", allow_duplicate=True),
        Input(f"{BMI_COL}Input", "value"),
        State("bmiSlider", "value"),
        prevent_initial_call=True,
    )
    def _pull_bmi_text_to_slider(txt, sl_cur):
        raw = try_parse_optional_float(txt)
        if raw is None:
            raise PreventUpdate
        v = max(_bmi_lo, min(_bmi_hi, raw))
        try:
            sc = float(sl_cur)
        except (TypeError, ValueError):
            sc = None
        if sc is not None and abs(float(v) - sc) < 1e-8:
            raise PreventUpdate
        return float(v)

    @dash_app.callback(
        Output("bmiSliderRow", "style"),
        Output("bmiCalcRow", "style"),
        Input("bmiEntryModeInput", "value"),
    )
    def _toggle_bmi_rows(mode):
        if mode == "calculator":
            return {"display": "none"}, {"display": "block", "marginTop": "8px"}
        return {"display": "block"}, {"display": "none", "marginTop": "8px"}

    @dash_app.callback(
        Output(f"{BMI_COL}Input", "value", allow_duplicate=True),
        Input("bmiCalcApply", "n_clicks"),
        State("bmiCalcWeightKg", "value"),
        State("bmiCalcHeightM", "value"),
        prevent_initial_call=True,
    )
    def _apply_bmi_from_calc(n_clicks, w_kg, h_m):
        if not n_clicks:
            raise PreventUpdate
        try:
            w = float(w_kg)
            h = float(h_m)
        except (TypeError, ValueError):
            raise PreventUpdate
        if h <= 0 or w <= 0:
            raise PreventUpdate
        bmi = w / (h**2)
        lo = float(sliderBounds.get(BMI_COL, {"min": 19.0, "max": 54.0})["min"])
        hi = float(sliderBounds.get(BMI_COL, {"min": 19.0, "max": 54.0})["max"])
        bmi = max(lo, min(hi, bmi))
        return slider_value_as_display_text(bmi)


def shap_top_features(input_frame: pd.DataFrame, k: int = 3):
    if not _ENABLE_SHAP:
        # Low-memory fallback: approximate "drivers" from model feature importance
        # weighted by how far the current input is from training median.
        try:
            importances = np.asarray(getattr(model, "feature_importances_", []), dtype=np.float64)
        except Exception:
            importances = np.array([], dtype=np.float64)
        if importances.size != len(featureColumns):
            return None, []

        scores = np.zeros(len(featureColumns), dtype=np.float64)
        row = input_frame.iloc[0]
        for i, col in enumerate(featureColumns):
            q = featureQuantiles.get(col)
            if not q:
                continue
            try:
                value = float(row[col])
                p50 = float(q.get("p50", value))
                p25 = float(q.get("p25", p50))
                p75 = float(q.get("p75", p50))
            except (TypeError, ValueError):
                continue
            scale = max(abs(p75 - p25), 1e-6)
            z = abs(value - p50) / scale
            scores[i] = float(importances[i]) * z

        order = np.argsort(-scores)
        top = [featureColumns[int(idx)] for idx in order if scores[int(idx)] > 0][:k]
        if len(top) < k:
            for idx in np.argsort(-importances):
                name = featureColumns[int(idx)]
                if name in top:
                    continue
                top.append(name)
                if len(top) >= k:
                    break
        return scores, top[:k]
    explainer = get_explainer()
    if explainer is None:
        return None, []
    import shap

    x = input_frame[featureColumns].to_numpy(dtype=np.float64)
    try:
        sv = explainer.shap_values(x)
    except Exception:
        return None, []
    arr = np.asarray(sv)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        phi = arr[0, :, worstStageIndex].ravel()
    elif isinstance(sv, list):
        arr2 = np.asarray(sv[worstStageIndex])
        phi = arr2[0].ravel() if arr2.ndim == 2 else arr2.ravel()
    else:
        phi = arr[0].ravel() if arr.ndim == 2 else arr.ravel()
    if phi.size != len(featureColumns):
        return None, []
    order = np.argsort(-phi)
    top_positive = []
    for idx in order:
        if phi[int(idx)] > 0:
            top_positive.append(featureColumns[int(idx)])
        if len(top_positive) >= k:
            break
    if len(top_positive) < k:
        for idx in order:
            name = featureColumns[int(idx)]
            if name in top_positive:
                continue
            top_positive.append(name)
            if len(top_positive) >= k:
                break
    return phi, top_positive[:k]


def render_emphasis_paragraph(text: str, extra_style=None):
    parts = text.split("**")
    style = {"marginBottom": "8px"}
    if extra_style:
        style = {**style, **extra_style}
    if len(parts) == 3:
        return html.P([parts[0], html.Strong(parts[1]), parts[2]], style=style)
    return html.P(text, style=style)


def advice_lines_for_features(features: list) -> list:
    lines = []
    for name in features:
        friendly = friendly_feature_label(name)
        d = feature_direction(name)
        if name in categoricalColumns or is_binary_numeric_column(name):
            lines.append(
                f"Reviewing **{friendly}** with your care team can help reduce your risk."
            )
        elif d == "higher_better":
            lines.append(f"Increasing **{friendly}** can help reduce your risk.")
        else:
            lines.append(f"Reducing **{friendly}** can help reduce your risk.")
    return lines


def weakest_healthy_feature(input_data: dict):
    best_col = None
    best_margin = None
    for col in featureColumns:
        if col in categoricalColumns:
            continue
        q = featureQuantiles.get(col)
        if not q:
            continue
        v = float(input_data[col])
        d = feature_direction(col)
        if d == "higher_worse":
            if v <= q["p25"]:
                margin = float(q["p25"] - v)
                if best_margin is None or margin < best_margin:
                    best_margin = margin
                    best_col = col
        else:
            if v >= q["p75"]:
                margin = float(v - q["p75"])
                if best_margin is None or margin < best_margin:
                    best_margin = margin
                    best_col = col
    return best_col


@dash_app.callback(
    Output("modalBackdrop", "style"),
    Output("resultsModalBody", "children"),
    Input("predictButton", "n_clicks"),
    stateInputs,
)
def on_predict(n_clicks, *values):
    hidden = {**MODAL_BACKDROP_BASE, "display": "none"}
    if n_clicks is None or n_clicks == 0:
        return hidden, None

    try:
        input_frame = collect_input_frame(values)
        input_dict = input_frame.iloc[0].to_dict()
        pred_code = int(model.predict(input_frame)[0])
        pred_label = targetMap[pred_code]
    except Exception as exc:
        err_visible = {**MODAL_BACKDROP_BASE, "display": "flex"}
        return err_visible, html.Div(
            [
                html.P(
                    "Prediction could not be completed. Check inputs and try again.",
                    style={"color": "#b71c1c", "marginBottom": "8px"},
                ),
                html.P(str(exc), style={"fontSize": "12px", "color": "#555"}),
            ]
        )
    proba = None
    try:
        proba_row = model.predict_proba(input_frame)[0]
        proba = float(proba_row[worstStageIndex])
    except Exception:
        pass

    worst_label = targetMap[worstStageIndex]
    phi, top_names = shap_top_features(input_frame)
    top_names = list(top_names or [])
    shap_failed = phi is None or len(top_names) == 0

    bullets = []
    if not shap_failed:
        for fname in top_names[:3]:
            bullets.append(
                html.Li(friendly_feature_label(fname), style={"marginBottom": "6px"})
            )

    advice = advice_lines_for_features(top_names[:3]) if top_names else []
    weakest = weakest_healthy_feature(input_dict)
    weak_sentence = ""
    if weakest:
        wf = friendly_feature_label(weakest)
        if feature_direction(weakest) == "higher_better":
            weak_sentence = f"Improving on **{wf}** could help reduce your risk."
        else:
            weak_sentence = (
                f"Improving **{wf}** (keeping it well inside the healthy range) "
                "could help reduce your risk."
            )

    body_children = [
        html.P(
            [
                html.Strong("Predicted diabetes stage: "),
                html.Span(pred_label, style={"color": "#1b5e20"}),
            ],
            style={"fontSize": "18px", "marginBottom": "12px"},
        ),
    ]
    if proba is not None:
        body_children.append(
            html.P(
                f"Estimated probability of the highest-risk stage in this model "
                f"({worst_label}): {proba * 100:.1f}%.",
                style={"fontSize": "14px", "color": "#333", "marginBottom": "16px"},
            )
        )

    if shap_failed:
        shap_msg = (
            "Detailed drivers are temporarily unavailable. "
            "If this persists, verify artifacts were rebuilt with "
            "`python SRC/prepare_data.py && python SRC/train.py`."
        )
        body_children.append(
            html.P(
                shap_msg,
                style={"color": "#b71c1c"},
            )
        )
    else:
        body_children.extend(
            [
                html.H4(
                    "Highest-impact inputs to review",
                    style={"marginTop": "8px", "marginBottom": "8px"},
                ),
                html.Ul(bullets, style={"paddingLeft": "20px"}),
                html.H4(
                    "What you can do",
                    style={"marginTop": "16px", "marginBottom": "8px"},
                ),
            ]
        )
        for line in advice:
            body_children.append(render_emphasis_paragraph(line))

    if weak_sentence:
        body_children.append(render_emphasis_paragraph(weak_sentence, {"marginTop": "12px", "marginBottom": "0"}))

    body_children.append(
        html.P(
            "This tool supports decisions and does not replace medical advice.",
            style={"fontSize": "12px", "color": "#666", "marginTop": "20px"},
        )
    )

    visible = {**MODAL_BACKDROP_BASE, "display": "flex"}
    return visible, html.Div(body_children)


@dash_app.callback(
    Output("modalBackdrop", "style", allow_duplicate=True),
    Input("modalClose", "n_clicks"),
    prevent_initial_call=True,
)
def close_modal(n):
    if not n:
        raise PreventUpdate
    return {**MODAL_BACKDROP_BASE, "display": "none"}


if __name__ == "__main__":
    # use_reloader=False avoids a second Python process (Windows) and duplicate callback issues.
    dash_app.run(debug=True, use_reloader=False)
