"""Flask application exposing the ANN weather condition predictor."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "weather_ann.keras"
TRANSFORMER_PATH = BASE_DIR / "models" / "feature_transformer.joblib"
LABEL_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.joblib"

REGIONS = ["North", "South", "East", "West", "Central"]
MONTHS = list(range(1, 13))

app = Flask(__name__)


def load_artifacts() -> Tuple[keras.Model, object, object]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train.py` to generate the model before starting the API."
        )
    model = keras.models.load_model(MODEL_PATH)
    transformer = joblib.load(TRANSFORMER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, transformer, label_encoder


MODEL, TRANSFORMER, LABEL_ENCODER = load_artifacts()


def prepare_dataframe(payload: Dict[str, object]) -> pd.DataFrame:
    """Validate and convert inbound payload to a single-row DataFrame."""
    required_fields = {
        "region": str,
        "month": (int, float),
        "temperature_c": (int, float),
        "humidity_pct": (int, float),
        "pressure_hpa": (int, float),
        "wind_speed_kph": (int, float),
        "precip_mm": (int, float),
        "cloud_cover_pct": (int, float),
    }

    data = {}
    for field, expected_type in required_fields.items():
        if field not in payload:
            raise ValueError(f"Missing field: {field}")
        value = payload[field]
        if isinstance(expected_type, tuple):
            valid = isinstance(value, expected_type)
        else:
            valid = isinstance(value, expected_type)
        if not valid:
            try:
                # Attempt coercion to float/int as appropriate
                if expected_type is str:
                    value = str(value)
                else:
                    value = float(value)
            except Exception as exc:  # pylint: disable=broad-except
                raise ValueError(f"Invalid value for {field}: {payload[field]}") from exc
        data[field] = value

    region = str(data["region"]).title()
    if region not in REGIONS:
        raise ValueError(f"Unsupported region '{data['region']}'. Expected one of {', '.join(REGIONS)}")
    data["region"] = region

    month = int(round(float(data["month"])))
    if month < 1 or month > 12:
        raise ValueError("Month must be between 1 and 12")
    data["month"] = month

    df = pd.DataFrame([data])
    return df


def predict_condition(payload: Dict[str, object]) -> Tuple[str, float, Dict[str, float]]:
    df = prepare_dataframe(payload)
    features = TRANSFORMER.transform(df)
    if hasattr(features, "toarray"):
        features = features.toarray()
    features = features.astype(np.float32)

    probabilities = MODEL.predict(features, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    condition = LABEL_ENCODER.inverse_transform([predicted_idx])[0]

    probability_map = {
        LABEL_ENCODER.inverse_transform([idx])[0]: float(round(prob, 4))
        for idx, prob in enumerate(probabilities)
    }
    confidence = float(round(probabilities[predicted_idx], 4))
    return condition, confidence, probability_map


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_data = {
        "region": REGIONS[0],
        "month": 6,
        "temperature_c": 30,
        "humidity_pct": 70,
        "pressure_hpa": 1005,
        "wind_speed_kph": 12,
        "precip_mm": 10,
        "cloud_cover_pct": 65,
    }

    if request.method == "POST":
        raw_form = request.form.to_dict()
        form_data.update(raw_form)
        try:
            payload = {
                "region": raw_form.get("region", ""),
                "month": raw_form.get("month", 0),
                "temperature_c": raw_form.get("temperature_c", 0),
                "humidity_pct": raw_form.get("humidity_pct", 0),
                "pressure_hpa": raw_form.get("pressure_hpa", 0),
                "wind_speed_kph": raw_form.get("wind_speed_kph", 0),
                "precip_mm": raw_form.get("precip_mm", 0),
                "cloud_cover_pct": raw_form.get("cloud_cover_pct", 0),
            }
            condition, confidence, probability_map = predict_condition(payload)
            prediction = {
                "condition": condition,
                "confidence": confidence,
                "probabilities": probability_map,
            }
        except ValueError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        regions=REGIONS,
        months=MONTHS,
        form_data=form_data,
        prediction=prediction,
        error=error,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True)
        condition, confidence, probability_map = predict_condition(payload)
        return jsonify(
            {
                "condition": condition,
                "confidence": confidence,
                "probabilities": probability_map,
            }
        )
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True)
