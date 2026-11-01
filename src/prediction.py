"""Strict input contract with range and physical-output warnings."""
import numpy as np
import pandas as pd
from config import FEATURES


def validate_inputs(values):
    if set(values) != set(FEATURES):
        raise ValueError("Supply exactly the eight concrete mixture inputs")
    clean = {}
    for name in FEATURES:
        if isinstance(values[name], (bool, str)):
            raise ValueError(f"{name} must be a numeric value")
        try:
            value = float(values[name])
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a numeric value") from None
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        if name in ["Cement", "Water", "Age"] and value <= 0:
            raise ValueError(f"{name} must be greater than zero")
        if name == "Age" and value != int(value):
            raise ValueError("Age must be a whole number of days")
        clean[name] = value
    return pd.DataFrame([clean], columns=FEATURES)


def predict_strength(bundle, values):
    row = validate_inputs(values)
    selected = bundle["metadata"]["selected"]
    training = bundle["data"].iloc[bundle["train_indices"]]
    outside = [f for f in FEATURES if row[f].iloc[0] < training[f].min() or row[f].iloc[0] > training[f].max()]
    prediction = float(bundle["models"][selected].predict(row)[0])
    if not np.isfinite(prediction):
        raise ValueError("Model returned a non-finite prediction; choose values within training ranges")
    warnings = []
    if outside:
        warnings.append("Outside training range: " + ", ".join(outside) + ". Polynomial extrapolation can be unstable.")
    if prediction < 0:
        warnings.append("The model returned a physically impossible negative strength. This prediction is unreliable; it has not been clipped.")
    if prediction > training.Strength.max():
        warnings.append("Predicted strength exceeds the observed training target range; interpret cautiously.")
    return {"Prediction (MPa)": prediction, "Water/Cement": float(row.Water.iloc[0] / row.Cement.iloc[0]),
            "Model": selected, "Experiment": bundle["metadata"]["experiment_id"], "warnings": warnings}
