# Penguin Species Prediction

[![CI](https://github.com/al-rafi-DSC/Penguin-Species-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/al-rafi-DSC/Penguin-Species-Prediction/actions/workflows/ci.yml)
[![CD](https://github.com/al-rafi-DSC/Penguin-Species-Prediction/actions/workflows/cd.yml/badge.svg)](https://github.com/al-rafi-DSC/Penguin-Species-Prediction/actions/workflows/cd.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED)

A compact machine-learning project built to demonstrate a complete CI/CD workflow. It trains a penguin species classifier, serves predictions through FastAPI, validates the application with automated tests, builds and tests a Docker image, and publishes validated images to GitHub Container Registry.

## What the project demonstrates

- Reproducible dependency management with `uv` and `uv.lock`
- A reusable scikit-learn preprocessing and classification pipeline
- Model-quality enforcement with a minimum accuracy gate
- FastAPI request validation and prediction endpoints
- Automated testing and coverage reporting with pytest
- Linting and formatting checks with Ruff
- Docker image build and container health testing
- Continuous delivery to GitHub Container Registry

## CI/CD pipeline

```mermaid
flowchart LR
    A["Push or pull request"] --> B["Lint and format checks"]
    B --> C["Train model"]
    C --> D["Accuracy quality gate"]
    D --> E["Automated tests"]
    E --> F["Build Docker image"]
    F --> G["Container health check"]
    G --> H["Publish versioned image to GHCR"]
```

The CI workflow runs on pushes and pull requests targeting `main`. CD starts only after CI succeeds on `main`.

Published images receive two tags:

- `latest` for the newest validated release
- The full Git commit SHA for traceability and rollback

## Model

The model predicts one of three species:

- Adelie
- Chinstrap
- Gentoo

It uses these features:

- Island
- Bill length
- Bill depth
- Flipper length
- Body mass
- Sex

The saved scikit-learn pipeline contains the complete preprocessing and classification workflow:

1. Missing numeric values are replaced with the median.
2. Numeric features are standardized.
3. Missing categorical values are replaced with the most frequent value.
4. Categorical features are one-hot encoded.
5. Logistic regression predicts the species and class probabilities.

The deterministic test split currently produces 100% accuracy. This is a result on a small educational dataset, not a guarantee of real-world performance. CI rejects a model if its measured accuracy drops below 90%.

## Dataset

The project uses the simplified [Palmer Penguins dataset](https://github.com/allisonhorst/palmerpenguins), containing 344 observations from three penguin species. The data were collected near Palmer Station, Antarctica, and are available under CC0.

## Project structure

```text
Penguin-Species-Prediction/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── app/
│   ├── __init__.py
│   └── main.py
├── artifacts/
│   ├── model.joblib
│   └── metadata.json
├── data/
│   └── penguins.csv
├── tests/
│   ├── test_api.py
│   └── test_training.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── train.py
└── uv.lock
```

Model artifacts are generated locally or during CI and are not committed to Git.

## Local setup

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)

Clone the repository and install the locked dependencies:

```bash
git clone https://github.com/al-rafi-DSC/Penguin-Species-Prediction.git
cd Penguin-Species-Prediction
uv sync --locked
```

Train and validate the model:

```bash
uv run python train.py
```

This creates:

```text
artifacts/model.joblib
artifacts/metadata.json
```

## Run the API

```bash
uv run uvicorn app.main:app --reload
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

### Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Container and service health check |
| `POST` | `/predict` | Predict a species and return class probabilities |

Example prediction request:

```json
{
  "island": "Biscoe",
  "bill_length_mm": 46.1,
  "bill_depth_mm": 14.8,
  "flipper_length_mm": 211,
  "body_mass_g": 4800,
  "sex": "female"
}
```

Example response:

```json
{
  "predicted_species": "Gentoo",
  "probabilities": {
    "Adelie": 0.0025,
    "Chinstrap": 0.0081,
    "Gentoo": 0.9894
  }
}
```

Probabilities are returned as decimal fractions, so `0.9894` is equivalent to `98.94%`.

## Quality checks

Run the same checks used by CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run python train.py
uv run pytest
```

The test suite covers dataset integrity, model training, API health, successful predictions, probability output, and invalid request handling.

## Docker

Generate the model before building locally:

```bash
uv run python train.py
docker build --tag penguin-api .
docker run --rm --publish 8000:8000 penguin-api
```

The validated image is also published by CD:

```text
ghcr.io/al-rafi-dsc/penguin-species-prediction:latest
```

If the package is private, authenticate to GHCR before pulling it.

## Workflow files

- `.github/workflows/ci.yml` performs linting, formatting, training, testing, Docker building, and a live container health check.
- `.github/workflows/cd.yml` checks out the exact validated commit, retrains the model, builds the image, and publishes immutable and `latest` tags to GHCR.

This repository is designed as a small, understandable foundation for learning practical ML delivery rather than maximizing model complexity.
