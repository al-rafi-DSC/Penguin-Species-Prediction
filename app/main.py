# Build the Prediction API

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH} run 'uv run train.py' first")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Penguin Species Prediction API",
    description="Predicting penguin species using physical features",
    version="1.0.0",
)


class PenguinFeatures(BaseModel):
    island: Literal["Biscoe", "Dream", "Torgersen"]
    bill_length_mm: float = Field(gt=0)
    bill_depth_mm: float = Field(gt=0)
    flipper_length_mm: float = Field(gt=0)
    body_mass_g: float = Field(gt=0)
    sex: Literal["male", "female"]


class PredictionResponse(BaseModel):
    predicted_species: str
    probabilities: dict[str, float]


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "I am alive Baby"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PenguinFeatures) -> PredictionResponse:
    input_data = pd.DataFrame([features.model_dump()])

    predicted_species = model.predict(input_data)[0]
    predicted_probabilities = model.predict_proba(input_data)[0]

    classes = model.named_steps["classifier"].classes_

    probabilities = {
        species: round(float(probability), 4)
        for species, probability in zip(
            classes,
            predicted_probabilities,
            strict=True,
        )
    }

    return PredictionResponse(
        predicted_species=str(predicted_species),
        probabilities=probabilities,
    )
