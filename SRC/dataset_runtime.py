"""Load diabetes + heart disease artifact bundles for the Dash app."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib


def _resolve_rf_path(artifacts_dir: Path, names: tuple[str, ...]) -> Optional[Path]:
    """Pick newest among existing candidates so a fallback `*_alt.pkl` after a failed replace still wins."""
    found = [artifacts_dir / n for n in names if (artifacts_dir / n).is_file()]
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    return max(found, key=lambda p: p.stat().st_mtime)


def col_key(name: str) -> str:
    return str(name).strip().lower()


def finalize_slider_bounds(panel: dict) -> None:
    """Apply semantic clamps when those columns exist (diabetes-centric; safe no-ops for heart-only)."""
    sb = panel["sliderBounds"]
    fc = panel["featureColumns"]
    keys = {col_key(c) for c in fc}
    if "diet_score" in keys:
        sb["diet_score"] = {"min": 0.0, "max": 100.0}
    for col in fc:
        if col_key(col) == "bmi":
            sb[col] = {"min": 19.0, "max": 54.0}
            break
    for col in fc:
        if col_key(col) == "waist_to_hip_ratio":
            b = sb.get(col, {"min": 0.0, "max": 1.0})
            sb[col] = {"min": float(b["min"]), "max": min(1.0, float(b["max"]))}
            break
    for col in fc:
        if col_key(col) == "alcohol_consumption_per_week":
            b = sb.get(col, {"min": 0.0, "max": 20.0})
            sb[col] = {"min": float(b["min"]), "max": min(20.0, float(b["max"]))}
            break


def load_panel(
    artifacts_dir: Path,
    prefix: str,
    rf_name: str | tuple[str, ...],
    dm_name: str,
    ui_name: str,
    *,
    page_title: str,
    page_intro: str,
    pred_label: str,
) -> Optional[dict]:
    rf_names = (rf_name,) if isinstance(rf_name, str) else tuple(rf_name)
    rf_path = _resolve_rf_path(artifacts_dir, rf_names)
    dm_path = artifacts_dir / dm_name
    ui_path = artifacts_dir / ui_name
    if rf_path is None or not dm_path.is_file() or not ui_path.is_file():
        return None
    model = joblib.load(rf_path)["model"]
    dm = joblib.load(dm_path)
    ui = joblib.load(ui_path)
    sb = dict(ui.get("sliderBounds") or {})
    fq = dict(ui.get("featureQuantiles") or {})
    fc = list(dm["featureColumns"])
    tm = list(dm["targetMap"])
    panel: dict[str, Any] = {
        "prefix": prefix,
        "model": model,
        "featureColumns": fc,
        "categoricalColumns": list(dm["categoricalColumns"]),
        "categoryMaps": dm["categoryMaps"],
        "targetMap": tm,
        "worstStageIndex": int(ui.get("worstStageIndex", len(tm) - 1)),
        "shapBackground": ui.get("shapBackground"),
        "sliderBounds": sb,
        "featureQuantiles": fq,
        "page_title": page_title,
        "page_intro": page_intro,
        "pred_label": pred_label,
    }
    finalize_slider_bounds(panel)
    return panel


def load_panels(artifacts_dir: Path) -> list[dict]:
    out: list[dict] = []
    db = load_panel(
        artifacts_dir,
        "db",
        ("Diabetes_rfModel.pkl", "Diabetes_rfModel_alt.pkl"),
        "DataModel_db.pkl",
        "UIModel_db.pkl",
        page_title="Diabetes Lifestyle Risk",
        page_intro=(
            "Enter lifestyle and clinical information, then press Predict for a diabetes-stage estimate "
            "and guidance. Tip: hover field labels for more information."
        ),
        pred_label="Predicted diabetes stage",
    )
    if db:
        out.append(db)
    hd = load_panel(
        artifacts_dir,
        "hd",
        "Heart_rfModel.pkl",
        "DataModel_hd.pkl",
        "UIModel_hd.pkl",
        page_title="Heart Disease Risk (Statlog)",
        page_intro=(
            "Enter Cleveland / Statlog heart attributes, then press Predict for heart disease presence "
            "versus the training labels. Tip: hover field labels for more information."
        ),
        pred_label="Predicted heart disease class",
    )
    if hd:
        out.append(hd)
    return out
