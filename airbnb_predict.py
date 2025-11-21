# airbnb_predict.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Load artifacts once at import time
ARTIFACT_PATH = Path(__file__).parent / "airbnb_ensemble_artifacts.pkl"
artifacts = joblib.load(ARTIFACT_PATH)

city_models = artifacts["city_models"]
feature_columns = artifacts["feature_columns"]
feature_medians = artifacts["feature_medians"]

def normalize_city_name(city: str) -> str:
    """
    Map city name from human-readable (e.g. 'San Francisco')
    to the key used in training (e.g. 'San_Francisco').
    """
    return city.strip().replace(" ", "_")


def predict_price(city: str, features: dict) -> float:
    """
    Predict price for a single listing.

    Parameters
    ----------
    city : str
        City name, e.g. 'San Francisco' or 'San_Francisco'.
    features : dict
        Mapping from feature name -> numeric value, e.g.
        {
          "bedrooms": 2,
          "bathrooms": 1,
          "accommodates": 3,
          ...
        }

        Feature names MUST match the columns in feature_columns[city_key].
        Any missing feature will be filled with the city median.
    """
    city_key = normalize_city_name(city)

    if city_key not in city_models:
        raise ValueError(f"Unsupported city: {city_key}")

    # Get per-city configs
    cols = feature_columns[city_key]
    med = feature_medians[city_key]
    models = city_models[city_key]

    # Build a one-row DataFrame from input features
    X = pd.DataFrame([features])

    # Reindex to match training column order
    X = X.reindex(columns=cols)

    # Coerce to numeric and fill NA with training medians
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(med)

    # Base models
    xgb = models["xgb"]
    rf = models["rf"]
    meta = models["meta"]

    # Base predictions (on log-price scale)
    base_preds = np.column_stack([
        xgb.predict(X),
        rf.predict(X),
    ])

    # Meta model prediction (still log-price)
    log_price_pred = meta.predict(base_preds)[0]

    # Inverse transform (because you trained on log1p(price))
    price_pred = float(np.expm1(log_price_pred))

    return price_pred


if __name__ == "__main__":
    # Quick manual test
    example_features = {
        # Fill with real feature names from feature_columns["San_Francisco"]
        "bedrooms": 2,
        "bathrooms": 1,
        "accommodates": 3,
        # ... etc
    }

    print(predict_price("San Francisco", example_features))

