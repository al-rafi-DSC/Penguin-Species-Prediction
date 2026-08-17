from pandas._libs import missing
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Setting up project paths
project_root=Path(__file__).parents[0]
DATA_PATH=Path("data/penguins.csv")
ARTIFACTS_PATH=Path("artifacts")
MODEL_PATH=Path("artifacts/model.joblib")
PREPROCESSOR_PATH=Path("artifacts/preprocessor.joblib")


# Defining feature columns
numeric_features=[
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

categorical_features=[
    "island",
    "sex"
]


features=numeric_features + categorical_features
target="species"
min_accuracy=0.90

# Defing the Functions
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and split it into features and target."""
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    data=pd.read_csv(DATA_PATH)

    required_columns=set(features+[target])
    missing_columns=required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"Dataset is Missing required columns:  {sorted(missing_columns)}")
    
    features=data[features]
    target=data[target]