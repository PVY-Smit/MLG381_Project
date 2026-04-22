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
    k = col_key(column)
    heart_labels = {
        "cp": "Chest pain type",
        "trestbps": "Resting blood pressure",
        "chol": "Serum cholesterol",
        "fbs": "Fasting blood sugar high",
        "restecg": "Resting ECG",
        "thalach": "Max heart rate (stress test)",
        "exang": "Exercise-induced angina",
        "oldpeak": "ST depression (exercise)",
        "slope": "ST segment slope",
        "ca": "Major vessels (fluoroscopy)",
        "thal": "Thallium scan",
    }
    if k in heart_labels:
        return heart_labels[k]
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


def is_forced_binary_column(col: str) -> bool:
    return col_key(col) in FORCED_BINARY_FIELDS


def is_forced_dropdown_column(col: str) -> bool:
    return col_key(col) in FORCED_DROPDOWN_FIELDS


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
    # Cleveland / Statlog heart disease attributes (same keys as prepared CSV columns)
    "sex": "Sex encoded as in the training data (often 0 = female, 1 = male).",
    "cp": "Chest pain category (typical angina, atypical, non-anginal, or asymptomatic).",
    "trestbps": "Resting blood pressure on admission (mmHg).",
    "chol": "Serum cholesterol (mg/dL).",
    "fbs": "Whether fasting blood sugar exceeds ~120 mg/dL (1 = yes).",
    "restecg": "Resting electrocardiogram pattern (normal, ST-T changes, or LV hypertrophy).",
    "thalach": "Maximum heart rate achieved during exercise stress testing (bpm).",
    "exang": "Exercise-induced angina (1 = yes).",
    "oldpeak": "ST depression induced by exercise relative to rest (ST segment shift).",
    "slope": "Slope of the peak exercise ST segment (upsloping, flat, downsloping).",
    "ca": "Number of major vessels coloured by fluoroscopy (0–3).",
    "thal": "Thalassemia / perfusion defect category from scintigraphy (training encoding).",
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
    "sex": "category",
    "cp": "category",
    "trestbps": "mmHg",
    "chol": "mg/dL",
    "fbs": "0–1",
    "restecg": "category",
    "thalach": "bpm",
    "exang": "0–1",
    "oldpeak": "mm",
    "slope": "category",
    "ca": "0–3",
    "thal": "category",
}

FORCED_BINARY_FIELDS = {"sex", "fbs", "exang"}

FORCED_DROPDOWN_FIELDS = {
    "cp": {
        1: "1 - Typical angina",
        2: "2 - Atypical angina",
        3: "3 - Non-anginal pain",
        4: "4 - Asymptomatic",
    },
    "restecg": {
        0: "0 - Normal",
        1: "1 - ST-T abnormality",
        2: "2 - Left ventricular hypertrophy",
    },
    "slope": {
        1: "1 - Upsloping",
        2: "2 - Flat",
        3: "3 - Downsloping",
    },
    "thal": {
        3: "3 - Normal",
        6: "6 - Fixed defect",
        7: "7 - Reversible defect",
    },
}

SECTION_HELP = {
    "Demographics": "Who the patient is: basic traits and social context used for risk context.",
    "Positive influences": "Behaviours and markers that usually improve when raised (activity, sleep, HDL, diet score).",
    "Negative influences": "Behaviours and lab markers that usually worsen metabolic risk when out of range.",
    "Other clinical indicators": "Remaining risk-related fields from the dataset.",
    "Resting presentation": "Chest pain type, resting BP, cholesterol, fasting glucose, and resting ECG.",
    "Exercise stress": "Stress-test exercise capacity, angina, ST depression, and ST slope.",
    "Catheterisation / imaging": "Angiographic vessel count and thallium categories used in this dataset.",
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
    "thalach",  # higher achieved HR on stress test often reflects better exercise capacity
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

try:
    from .dataset_runtime import load_panels
except ImportError:
    from dataset_runtime import load_panels

PANELS = load_panels(_ARTIFACTS_DIR)
if not PANELS:
    raise FileNotFoundError(
        "No trained models in ARTIFACTS/. From the project root build at least one pipeline:\n"
        "  Diabetes: python SRC/prepare_diabetes_data.py then python SRC/train.py\n"
        "  Heart (Statlog): python SRC/prepare_heart_disease_data.py then python SRC/train_hd.py"
    )


def _numeric_median_default(col: str, lo: float, hi: float, feature_quantiles: dict) -> float:
    """Median for slider defaults from training quantiles (avoids loading full CSV at import)."""
    q = feature_quantiles.get(col)
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


_ENABLE_SHAP = os.getenv("ENABLE_SHAP", "false").strip().lower() in ("1", "true", "yes", "on")
_explainers: dict = {}


def get_explainer(panel: dict):
    pref = panel["prefix"]
    if pref in _explainers:
        return _explainers[pref]
    if not _ENABLE_SHAP:
        _explainers[pref] = None
        return None
    bg = panel.get("shapBackground")
    if bg is None:
        _explainers[pref] = None
        return None
    import shap

    try:
        _explainers[pref] = shap.TreeExplainer(panel["model"], data=bg)
    except Exception:
        _explainers[pref] = shap.TreeExplainer(panel["model"])
    return _explainers[pref]


def assign_section(col: str, panel: dict) -> str:
    k = col_key(col)
    if panel["prefix"] == "hd":
        if k in ("age", "sex"):
            return "Demographics"
        if k in ("cp", "trestbps", "chol", "fbs", "restecg"):
            return "Resting presentation"
        if k in ("thalach", "exang", "oldpeak", "slope"):
            return "Exercise stress"
        return "Catheterisation / imaging"
    if k in DEMOGRAPHICS:
        return "Demographics"
    if k in POSITIVE_INFLUENCES:
        return "Positive influences"
    if k in NEGATIVE_INFLUENCES:
        return "Negative influences"
    return "Other clinical indicators"


_HEART_SECTION_TITLE_HELP = {
    "Demographics": "Patient age and sex used as baseline attributes (Statlog-style encoding).",
    "Resting presentation": SECTION_HELP["Resting presentation"],
    "Exercise stress": SECTION_HELP["Exercise stress"],
    "Catheterisation / imaging": SECTION_HELP["Catheterisation / imaging"],
}

_DIABETES_SECTION_DEFS = [
    ("Demographics", "Basic information and background (including age and profile dropdowns)."),
    ("Positive influences", "Factors that typically support lower risk when they are in a healthy range."),
    ("Negative influences", "Lifestyle and clinical markers that often track with higher risk when out of range."),
    ("Other clinical indicators", "Additional fields used by the model."),
]

_HEART_SECTION_DEFS = [
    ("Demographics", "Patient age and sex (encoding matches the original Statlog/Cleveland schema)."),
    ("Resting presentation", "Chest pain type, resting BP, cholesterol, fasting glucose, resting ECG."),
    ("Exercise stress", "Stress-test heart rate, exercise angina, ST depression and slope."),
    ("Catheterisation / imaging", "Major vessels visualised on fluoroscopy and thallium scan category."),
]


def section_help_line(title: str, panel: dict) -> str:
    if panel["prefix"] == "hd":
        return _HEART_SECTION_TITLE_HELP.get(title, "")
    return SECTION_HELP.get(title, "")


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


def is_binary_numeric_column(col: str, categorical_columns: list, slider_bounds: dict) -> bool:
    if col in categorical_columns:
        return False
    b = slider_bounds.get(col)
    if not b:
        return False
    lo, hi = float(b["min"]), float(b["max"])
    return lo <= 0.01 and 0.99 <= hi <= 1.01


def numeric_columns_with_slider_input(panel: dict) -> list:
    fc = panel["featureColumns"]
    cc = panel["categoricalColumns"]
    sb = panel["sliderBounds"]
    out = []
    for c in fc:
        if c in cc:
            continue
        if is_forced_dropdown_column(c):
            continue
        if is_forced_binary_column(c):
            continue
        if is_binary_numeric_column(c, cc, sb):
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


def build_field(col: str, panel: dict) -> html.Div:
    px = panel["prefix"]
    sliderBounds = panel["sliderBounds"]
    featureQuantiles = panel["featureQuantiles"]
    categoricalColumns = panel["categoricalColumns"]
    categoryMaps = panel["categoryMaps"]
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

    if is_forced_dropdown_column(col):
        option_map = FORCED_DROPDOWN_FIELDS[col_key(col)]
        options = [{"label": label, "value": value} for value, label in option_map.items()]
        first_value = list(option_map.keys())[0]
        control = dcc.Dropdown(
            id=f"{px}_{col}Input",
            options=options,
            value=first_value,
            clearable=False,
            searchable=False,
            style={"width": "100%"},
        )

    elif col in categoricalColumns:
        options = [{"label": v, "value": v} for v in categoryMaps[col]]
        control = dcc.Dropdown(
            id=f"{px}_{col}Input",
            options=options,
            value=categoryMaps[col][0] if categoryMaps[col] else None,
            clearable=False,
            searchable=False,
            style={"width": "100%"},
        )

    elif is_forced_binary_column(col):
        default_value = 0
        q = featureQuantiles.get(col)
        if q and "p50" in q:
            try:
                default_value = int(round(float(q["p50"])))
            except (TypeError, ValueError):
                default_value = 0

        binary_options = [
            {"label": "No", "value": 0},
            {"label": "Yes", "value": 1},
        ]
        if col_key(col) == "sex":
            binary_options = [
                {"label": "Female", "value": 0},
                {"label": "Male", "value": 1},
            ]

        control = dcc.RadioItems(
            id=f"{px}_{col}Input",
            options=binary_options,
            value=default_value if default_value in (0, 1) else 0,
            inline=True,
            style={"marginTop": "4px"},
        )

    elif is_binary_numeric_column(col, categoricalColumns, sliderBounds):
        bounds = sliderBounds.get(col, {"min": 0.0, "max": 1.0})
        lo_b, hi_b = float(bounds["min"]), float(bounds["max"])
        raw_bin = _numeric_median_default(col, lo_b, hi_b, featureQuantiles)
        try:
            bin_v = int(round(float(raw_bin)))
        except (TypeError, ValueError):
            bin_v = 0
        if bin_v not in (0, 1):
            bin_v = 0
        control = dcc.RadioItems(
            id=f"{px}_{col}Input",
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
        raw_med = _numeric_median_default(col, lo, hi, featureQuantiles)
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
                    id=f"{px}_bmiEntryModeInput",
                    options=[
                        {"label": "Slider", "value": "slider"},
                        {"label": "Calculator (kg & m)", "value": "calculator"},
                    ],
                    value="slider",
                    inline=True,
                    style={"marginBottom": "10px"},
                ),
                html.Div(
                    id=f"{px}_bmiCalcRow",
                    style={"display": "none", "marginTop": "8px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "alignItems": "end"},
                            children=[
                                html.Div(
                                    [
                                        html.Label("Weight (kg)", style={"fontSize": "12px"}),
                                        dcc.Input(
                                            id=f"{px}_bmiCalcWeightKg",
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
                                            id=f"{px}_bmiCalcHeightM",
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
                                    id=f"{px}_bmiCalcApply",
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
                    id=f"{px}_bmiSliderRow",
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
                                    id=f"{px}_{col}Input",
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
                                            id=f"{px}_bmiSlider",
                                            min=lo,
                                            max=hi,
                                            step=step,
                                            value=med,
                                            tooltip=tooltip,
                                            marks=None,
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
        raw_med = _numeric_median_default(col, lo, hi, featureQuantiles)
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
                    id=f"{px}_{col}Input",
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
                            id=f"{px}_{col}Slider",
                            min=lo,
                            max=hi,
                            step=step,
                            value=med,
                            tooltip=tooltip,
                            marks=None,
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


def section_block(title: str, subtitle: str, columns: list, panel: dict) -> html.Div:
    if not columns:
        return html.Div()
    cells = [build_field(c, panel) for c in columns]
    help_line = section_help_line(title, panel)
    return html.Div(
        [
            html.H2(title, style=SECTION_TITLE, title=help_line or None),
            html.P(subtitle, style=SECTION_SUB),
            html.Div(cells, style=GRID),
        ]
    )


def order_columns_for_sections(panel: dict) -> dict:
    if panel["prefix"] == "hd":
        buckets = {
            "Demographics": [],
            "Resting presentation": [],
            "Exercise stress": [],
            "Catheterisation / imaging": [],
        }
    else:
        buckets = {
            "Demographics": [],
            "Positive influences": [],
            "Negative influences": [],
            "Other clinical indicators": [],
        }
    for col in panel["featureColumns"]:
        buckets[assign_section(col, panel)].append(col)
    return buckets


def make_section_layout(panel: dict) -> list:
    buckets = order_columns_for_sections(panel)
    defs = _HEART_SECTION_DEFS if panel["prefix"] == "hd" else _DIABETES_SECTION_DEFS
    return [
        section_block(title, subtitle, buckets[title], panel)
        for title, subtitle in defs
        if buckets.get(title)
    ]


def build_decision_tab_children(panel: dict) -> list:
    px = panel["prefix"]
    btn_style = {
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
    }
    return [
        html.H1(
            panel["page_title"],
            style={
                "textAlign": "center",
                "color": "#111",
                "marginBottom": "10px",
                "fontSize": "26px",
            },
        ),
        html.P(
            panel["page_intro"],
            style={"textAlign": "center", "marginBottom": "20px", "fontWeight": "600"},
        ),
        *make_section_layout(panel),
        html.Button(
            "Predict",
            id=f"{px}_predictButton",
            n_clicks=0,
            style=btn_style,
        ),
    ]


def build_modal(panel: dict) -> html.Div:
    px = panel["prefix"]
    return html.Div(
        id=f"{px}_modalBackdrop",
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
                                id=f"{px}_modalClose",
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
                    html.Div(id=f"{px}_resultsModalBody"),
                ],
            )
        ],
    )


def _notebook_gallery_items() -> list[dict]:
    manifest_path = (
        Path(__file__).resolve().parent / "assets" / "notebook_figures" / "manifest.json"
    )
    if not manifest_path.is_file():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


dash_app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = dash_app.server

_default_main_tab = (
    "tab-db"
    if any(p["prefix"] == "db" for p in PANELS)
    else f"tab-{PANELS[0]['prefix']}"
)

_decision_model_tabs = []
for p in PANELS:
    _tab_label = "Diabetes lifestyle" if p["prefix"] == "db" else "Heart disease (Statlog)"
    _decision_model_tabs.append(
        dcc.Tab(
            label=_tab_label,
            value=f"tab-{p['prefix']}",
            style={"padding": "10px 14px", "fontWeight": "600"},
            selected_style={"padding": "10px 14px", "fontWeight": "700"},
            children=[html.Div(style=SHARP_CARD, children=build_decision_tab_children(p))],
        )
    )

_modal_layer = html.Div(children=[build_modal(p) for p in PANELS])

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
        "Plots saved from the Jupyter notebooks under NOTEBOOKS/ (embedded cell outputs), "
        "including Diabetes_Lifestyle and Heart_Disease. "
        "After editing or re-running cells, run python SRC/extract_notebook_figures.py from the project root.",
        style={"textAlign": "center", "marginBottom": "8px", "fontWeight": "600"},
    ),
    html.P(
        "K-Means.ipynb is picked up automatically if a cell outputs a figure; add outputs in the notebook, then re-run the extractor.",
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
            value=_default_main_tab,
            persistence=True,
            persistence_type="session",
            colors={
                "border": "#222",
                "primary": "#4a148c",
                "background": "#f5f5f5",
            },
            style={"marginBottom": "4px"},
            children=[
                *_decision_model_tabs,
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


def collect_input_frame(values, panel: dict):
    input_data = {}
    featureColumns = panel["featureColumns"]
    categoricalColumns = panel["categoricalColumns"]
    categoryMaps = panel["categoryMaps"]
    sliderBounds = panel["sliderBounds"]

    for col, value in zip(featureColumns, values):
        if is_forced_dropdown_column(col):
            try:
                input_data[col] = int(value)
            except (TypeError, ValueError):
                input_data[col] = list(FORCED_DROPDOWN_FIELDS[col_key(col)].keys())[0]

        elif col in categoricalColumns:
            cats = categoryMaps[col]
            if value in cats:
                input_data[col] = cats.index(value)
            else:
                input_data[col] = 0

        elif is_forced_binary_column(col):
            try:
                iv = int(value)
            except (TypeError, ValueError):
                iv = 0
            input_data[col] = 0 if iv not in (0, 1) else iv

        elif is_binary_numeric_column(col, categoricalColumns, sliderBounds):
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


for _panel in PANELS:
    _px = _panel["prefix"]
    _slider_bounds = _panel["sliderBounds"]
    for _sync_col in numeric_columns_with_slider_input(_panel):
        if col_key(_sync_col) == "bmi":
            continue

        def _make_slider_pusher(c, px=_px, sb=_slider_bounds):
            lo = float(sb[c]["min"])
            hi = float(sb[c]["max"])

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
                Output(f"{px}_{c}Input", "value", allow_duplicate=True),
                Input(f"{px}_{c}Slider", "value"),
                State(f"{px}_{c}Input", "value"),
                prevent_initial_call="initial_duplicate",
            )(_slider_to_text)

        def _make_input_puller(c, px=_px, sb=_slider_bounds):
            lo = float(sb[c]["min"])
            hi = float(sb[c]["max"])

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
                Output(f"{px}_{c}Slider", "value", allow_duplicate=True),
                Input(f"{px}_{c}Input", "value"),
                State(f"{px}_{c}Slider", "value"),
                prevent_initial_call=True,
            )(_text_to_slider)

        _make_slider_pusher(_sync_col)
        _make_input_puller(_sync_col)

for _panel in PANELS:
    _px = _panel["prefix"]
    _sb = _panel["sliderBounds"]
    _bmi_col = next((c for c in _panel["featureColumns"] if col_key(c) == "bmi"), None)
    if not _bmi_col:
        continue
    _bmi_lo = float(_sb.get(_bmi_col, {"min": 19.0, "max": 54.0})["min"])
    _bmi_hi = float(_sb.get(_bmi_col, {"min": 19.0, "max": 54.0})["max"])

    @dash_app.callback(
        Output(f"{_px}_{_bmi_col}Input", "value", allow_duplicate=True),
        Input(f"{_px}_bmiSlider", "value"),
        State(f"{_px}_{_bmi_col}Input", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def _push_bmi_slider(sv, text_in, lo=_bmi_lo, hi=_bmi_hi):
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

    @dash_app.callback(
        Output(f"{_px}_bmiSlider", "value", allow_duplicate=True),
        Input(f"{_px}_{_bmi_col}Input", "value"),
        State(f"{_px}_bmiSlider", "value"),
        prevent_initial_call=True,
    )
    def _pull_bmi_text_to_slider(txt, sl_cur, lo=_bmi_lo, hi=_bmi_hi):
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

    @dash_app.callback(
        Output(f"{_px}_bmiSliderRow", "style"),
        Output(f"{_px}_bmiCalcRow", "style"),
        Input(f"{_px}_bmiEntryModeInput", "value"),
    )
    def _toggle_bmi_rows(mode):
        if mode == "calculator":
            return {"display": "none"}, {"display": "block", "marginTop": "8px"}
        return {"display": "block"}, {"display": "none", "marginTop": "8px"}

    @dash_app.callback(
        Output(f"{_px}_{_bmi_col}Input", "value", allow_duplicate=True),
        Input(f"{_px}_bmiCalcApply", "n_clicks"),
        State(f"{_px}_bmiCalcWeightKg", "value"),
        State(f"{_px}_bmiCalcHeightM", "value"),
        prevent_initial_call=True,
    )
    def _apply_bmi_from_calc(n_clicks, w_kg, h_m, sb=_sb, col=_bmi_col):
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
        lo = float(sb.get(col, {"min": 19.0, "max": 54.0})["min"])
        hi = float(sb.get(col, {"min": 19.0, "max": 54.0})["max"])
        bmi = max(lo, min(hi, bmi))
        return slider_value_as_display_text(bmi)


def shap_top_features(input_frame: pd.DataFrame, panel: dict, k: int = 3):
    model = panel["model"]
    featureColumns = panel["featureColumns"]
    featureQuantiles = panel["featureQuantiles"]
    worstStageIndex = panel["worstStageIndex"]

    if not _ENABLE_SHAP:
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
    explainer = get_explainer(panel)
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


def advice_lines_for_features(features: list, panel: dict) -> list:
    cc = panel["categoricalColumns"]
    sb = panel["sliderBounds"]
    lines = []
    for name in features:
        friendly = friendly_feature_label(name)
        d = feature_direction(name)
        if name in cc or is_binary_numeric_column(name, cc, sb):
            lines.append(
                f"Reviewing **{friendly}** with your care team can help reduce your risk."
            )
        elif d == "higher_better":
            lines.append(f"Increasing **{friendly}** can help reduce your risk.")
        else:
            lines.append(f"Reducing **{friendly}** can help reduce your risk.")
    return lines


def weakest_healthy_feature(input_data: dict, panel: dict):
    featureColumns = panel["featureColumns"]
    categoricalColumns = panel["categoricalColumns"]
    featureQuantiles = panel["featureQuantiles"]
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


def _register_predict_callback(panel: dict):
    px = panel["prefix"]
    states = [State(f"{px}_{c}Input", "value") for c in panel["featureColumns"]]

    @dash_app.callback(
        Output(f"{px}_modalBackdrop", "style"),
        Output(f"{px}_resultsModalBody", "children"),
        Input(f"{px}_predictButton", "n_clicks"),
        states,
    )
    def on_predict(n_clicks, *values):
        hidden = {**MODAL_BACKDROP_BASE, "display": "none"}
        if n_clicks is None or n_clicks == 0:
            return hidden, None

        try:
            input_frame = collect_input_frame(values, panel)
            input_dict = input_frame.iloc[0].to_dict()
            pred_code = int(panel["model"].predict(input_frame)[0])
            pred_label = panel["targetMap"][pred_code]
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
            proba_row = panel["model"].predict_proba(input_frame)[0]
            proba = float(proba_row[panel["worstStageIndex"]])
        except Exception:
            pass

        worst_label = panel["targetMap"][panel["worstStageIndex"]]
        phi, top_names = shap_top_features(input_frame, panel)
        top_names = list(top_names or [])
        shap_failed = phi is None or len(top_names) == 0

        bullets = []
        if not shap_failed:
            for fname in top_names[:3]:
                bullets.append(
                    html.Li(friendly_feature_label(fname), style={"marginBottom": "6px"})
                )

        advice = advice_lines_for_features(top_names[:3], panel) if top_names else []
        weakest = weakest_healthy_feature(input_dict, panel)
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
                    html.Strong(f"{panel['pred_label']}: "),
                    html.Span(str(pred_label), style={"color": "#1b5e20"}),
                ],
                style={"fontSize": "18px", "marginBottom": "12px"},
            ),
        ]
        if proba is not None:
            body_children.append(
                html.P(
                    f"Estimated probability of the highest-risk class in this model "
                    f"({worst_label}): {proba * 100:.1f}%.",
                    style={"fontSize": "14px", "color": "#333", "marginBottom": "16px"},
                )
            )

        if shap_failed:
            shap_msg = (
                "Detailed drivers are temporarily unavailable. "
                "Rebuild artifacts with prepare/train scripts if needed."
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
            body_children.append(
                render_emphasis_paragraph(weak_sentence, {"marginTop": "12px", "marginBottom": "0"})
            )

        body_children.append(
            html.P(
                "This tool supports decisions and does not replace medical advice.",
                style={"fontSize": "12px", "color": "#666", "marginTop": "20px"},
            )
        )

        visible = {**MODAL_BACKDROP_BASE, "display": "flex"}
        return visible, html.Div(body_children)


def _register_close_callback(panel: dict):
    px = panel["prefix"]

    @dash_app.callback(
        Output(f"{px}_modalBackdrop", "style", allow_duplicate=True),
        Input(f"{px}_modalClose", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_modal(n):
        if not n:
            raise PreventUpdate
        return {**MODAL_BACKDROP_BASE, "display": "none"}


for _p in PANELS:
    _register_predict_callback(_p)
    _register_close_callback(_p)


if __name__ == "__main__":
    # use_reloader=False avoids a second Python process (Windows) and duplicate callback issues.
    # threaded=True keeps the dev server responsive while a callback runs (e.g. SHAP).
    dash_app.run(debug=True, use_reloader=False, threaded=True, port=8051)
