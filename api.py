# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from airbnb_predict import predict_price, normalize_city_name, artifacts

import os
import requests
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "airbnb_ensemble_artifacts.pkl"

# YOUR GOOGLE DRIVE DIRECT LINK:
MODEL_URL = "https://drive.google.com/uc?export=download&id=1XYA9KZr6Ghs2eLZdnzVUuRSy2PbuTDJS"

def download_if_missing():
    """Download the model from Google Drive if not present."""
    if MODEL_PATH.exists():
        print("Model file already exists.")
        return

    print("Downloading model from Google Drive...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Download complete.")

download_if_missing()

# Now load the model as usual
artifacts = joblib.load(MODEL_PATH)



app = FastAPI(
    title="Airbnb Price Prediction API",
    description="STA160 Team 12 Airbnb model",
    version="1.0.0",
)

# -------- CORS: allow browser requests (IMPORTANT) --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ok for class demo; can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    city: str
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    city: str
    city_key: str
    prediction: float


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    city_key = normalize_city_name(req.city)

    if city_key not in artifacts["city_models"]:
        raise HTTPException(status_code=400, detail=f"Unsupported city: {city_key}")

    try:
        price = predict_price(req.city, req.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictResponse(
        city=req.city,
        city_key=city_key,
        prediction=price,
    )
