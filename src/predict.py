"""
Prediction module for UAE Rent Prediction.

Loads the saved Ridge pipeline and exposes a simple function that takes a
property description as a dict and returns the predicted annual rent in AED.

The pipeline already encapsulates all preprocessing (imputation, encoding)
and produces predictions in log-space which are back-transformed via expm1.
"""

import pathlib
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "ridge_model.pkl"

_model = None  # lazy-loaded singleton


def _load_model():
    """Load the trained pipeline once and cache it."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_rent(input_dict: dict) -> float:
    """
    Predict the annual rent (AED) for a single property.

    Parameters
    ----------
    input_dict : dict
        A dictionary whose keys match the feature column names expected by the
        pipeline (e.g. ``{"City": "Dubai", "Beds": 2, "Baths": 2, ...}``).
        Missing keys will be handled by the imputation step inside the pipeline.

    Returns
    -------
    float
        Predicted annual rent in AED.
    """
    model = _load_model()
    input_df = pd.DataFrame([input_dict])
    log_pred = model.predict(input_df)[0]
    return float(np.expm1(log_pred))


if __name__ == "__main__":
    # Quick smoke test
    sample = {
        "City": "Dubai",
        "Location": "Downtown Dubai",
        "Beds": 2,
        "Baths": 2,
        "Type": "Apartment",
        "Area_in_sqft": 1200,
        "Latitude": 25.1972,
        "Longitude": 55.2744,
        "Furnishing": "Furnished",
    }
    predicted = predict_rent(sample)
    print(f"Predicted annual rent: AED {predicted:,.2f}")
