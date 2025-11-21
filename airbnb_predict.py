# airbnb_predict.py

from pathlib import Path
from functools import lru_cache
import os
import requests
import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Model location & auto-download from Google Drive
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "airbnb_ensemble_artifacts.pkl"

# 🔴 IMPORTANT: put YOUR real Google Drive FILE ID here
# Example Drive link:
#   https://drive.google.com/file/d/1AbCDEFGHIJKLMNOP/view?usp=sharing
# Then FILE_ID = "1AbCDEFGHIJKLMNOP"
FILE_ID = "1XYA9KZr6Ghs2eLZdnzVUuRSy2PbuTDJS"

MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"


def download_if_missing() -> None:
    """
    Download the big model file from Google Drive if it does not exist yet.
    This runs at import time on Render so we don't need the .pkl in the repo.
    """
    if MODEL_PATH.exists():
        print("[airbnb_predict] Model file already exists, skipping download.")
        return

    print("[airbnb_predict] Model file missing. Downloading from Google Drive...")
    resp = requests.get(MODEL_URL, stream=True)
    resp.raise_for_status()

    # Write in chunks so we don't blow memory
    with open(MODEL_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print("[airbnb_predict] Download complete.")


# Run download before loading
download_if_missing()

# -----------------------------------------------------------------------------
# 2. Load artifacts (city models, feature columns & medians)
# -----------------------------------------------------------------------------

print("[airbnb_predict] Loading artifacts from:", MODEL_PATH)
artifacts = joblib.load(MODEL_PATH)

city_models = artifacts["city_models"]
feature_columns = artifacts["feature_columns"]
feature_medians = artifacts["feature_medians"]


# -----------------------------------------------------------------------------
# 3. Helper: normalize city input to match artifact keys
# -----------------------------------------------------------------------------

def normalize_city_name(city: str) -> str:
    """
    Take a user-entered city string and map it to the key used in artifacts.

    We try to be robust:
    - ignore case
    - ignore spaces vs underscores
    - e.g. "San Francisco" -> "San_Francisco"
    """
    if city is None:
        raise ValueError("City must not be None")

    s = city.strip().lower()
    s_nospace = s.replace(" ", "")
    s_us = s.replace(" ", "_")

    # Try direct matches to the artifact keys
    for key in city_models.keys():
        k = key.lower()
        k_nospace = k.replace("_", "")
        if s == k or s_nospace == k_nospace or s_us == k:
            return key

    raise ValueError(
        f"City '{city}' is not supported. "
        f"Available cities: {list(city_models.keys())}"
    )


# -----------------------------------------------------------------------------
# 4. Helper: load per-city model if needed & predict
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_city_model(city_key: str):
    """
    Return the model object for a given city key.
    Cached so we don't repeatedly deserialize.
    """
    model_obj = city_models[city_key]
    return model_obj


def _predict_with_model(model_obj, X: np.ndarray) -> float:
    """
    Try to handle a few possible ways the model might be stored in artifacts.
    """
    # Simple case: model itself has .predict
    if hasattr(model_obj, "predict"):
        return float(model_obj.predict(X)[0])

    # Dictionary-style container
    if isinstance(model_obj, dict):
        # Try typical keys
        for key in ["stacked_model", "meta_model", "model", "estimator"]:
            if key in model_obj and hasattr(model_obj[key], "predict"):
                return float(model_obj[key].predict(X)[0])

    raise ValueError("Unsupported model object structure for prediction.")


# -----------------------------------------------------------------------------
# 5. Public function: predict_price
# -----------------------------------------------------------------------------

def predict_price(city: str, features: dict) -> float:
    """
    Main entry point used by the FastAPI backend.

    Parameters
    ----------
    city : str
        City name as entered by user ("San Francisco", "LosAngeles", etc.)
    features : dict
        Mapping from feature name -> value, coming from the frontend.

    Returns
    -------
    float
        Predicted price.
    """
    if not isinstance(features, dict):
        raise ValueError("features must be a dict of {feature_name: value}.")

    city_key = normalize_city_name(city)

    # Feature config for this city
    cols = feature_columns[city_key]
    med = feature_medians[city_key]

    # Start with a one-row DataFrame from the provided features
    # We will later add any missing columns
    row = {}

    # Use only known columns; ignore extra keys from the frontend
    for col in cols:
        if col in features:
            row[col] = features[col]
        else:
            # If the frontend didn't provide this feature, use median
            row[col] = med.get(col, 0)

    X_df = pd.DataFrame([row], columns=cols)

    # Convert to numeric and fill remaining NaNs with medians
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(pd.Series(med)).fillna(0)

    X = X_df.to_numpy()

    model_obj = get_city_model(city_key)
    y = _predict_with_model(model_obj, X)

    return float(y)


# -----------------------------------------------------------------------------
# 6. Manual test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example manual test (you can run this locally, not on Render)
    example_city = "San Francisco"
    example_features = {
        # You should adjust these to your actual feature names
        # (they must match feature_columns[city_key])
        # e.g. "Bedrooms", "Bathrooms", "Accommodates", etc.
    }

    try:
        print("Predicted price:", predict_price(example_city, example_features))
    except Exception as e:
        print("Error during test prediction:", e)
