"""
Training script for UAE Rent Prediction — Ridge Regression.

Reproduces exactly the modeling decisions from the exploration notebook:
  1. Load dataset
  2. Drop leakage / high-cardinality columns
  3. Apply log1p transform to target
  4. Build Ridge pipeline (preprocessing + Ridge alpha=1.0)
  5. Perform 5-fold cross-validation (log-RMSE)
  6. Train final model on the full dataset
  7. Save trained pipeline to models/ridge_model.pkl
"""

import os
import sys
import logging
import pathlib

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

# ---------------------------------------------------------------------------
# Allow imports from src/ regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import prepare_features, build_preprocessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "dubai_properties.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "ridge_model.pkl"

RIDGE_ALPHA = 1.0
CV_SPLITS = 5
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def train() -> Pipeline:
    """
    End-to-end training routine.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Fitted Ridge pipeline (preprocessor + Ridge).
    """

    # 1. Load data ---------------------------------------------------------------
    logger.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("Dataset shape: %s", df.shape)

    # 2. Prepare features (drop leakage, high-cardinality, separate X/y) ---------
    X, y, num_features, cat_features = prepare_features(df)
    logger.info("Features  — numerical: %d, categorical: %d", len(num_features), len(cat_features))

    # 3. Log-transform the target ------------------------------------------------
    y_log = np.log1p(y)

    # 4. Build pipeline ----------------------------------------------------------
    preprocessor = build_preprocessor(num_features, cat_features)
    ridge_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", Ridge(alpha=RIDGE_ALPHA))
    ])

    # 5. 5-fold cross-validation -------------------------------------------------
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        ridge_pipeline,
        X,
        y_log,
        scoring="neg_root_mean_squared_error",
        cv=kf,  
        n_jobs=-1,
    )
    cv_rmse_log = -cv_scores
    logger.info(
        "Cross-validation log-RMSE — mean: %.4f, std: %.4f",
        cv_rmse_log.mean(),
        cv_rmse_log.std(),
    )

    # 6. Train final model on full dataset ----------------------------------------
    logger.info("Training final model on full dataset …")
    ridge_pipeline.fit(X, y_log)

    # 7. Save model ---------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_pipeline, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    return ridge_pipeline


if __name__ == "__main__":
    train()
