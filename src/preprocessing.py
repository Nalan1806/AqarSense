"""
Preprocessing module for UAE Rent Prediction.

Extracts feature type identification and ColumnTransformer definition
from the exploration notebook to provide a single reusable preprocessing
pipeline used by both training and inference.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# Columns that leak target information or are not useful for prediction
LEAKAGE_AND_USELESS_COLS = ["Rent_per_sqft", "Rent_category", "Frequency", "Purpose"]

# High-cardinality text feature excluded from baseline
HIGH_CARDINALITY_COLS = ["Address"]

# Target column
TARGET_COL = "Rent"


def identify_feature_types(X: pd.DataFrame):
    """
    Identify numerical and categorical feature columns from the feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (target and leakage columns already removed).

    Returns
    -------
    num_features : pd.Index
        Numerical column names.
    cat_features : pd.Index
        Categorical column names.
    """
    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns
    return num_features, cat_features


def build_preprocessor(num_features, cat_features) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies:
      - Median imputation to numerical features
      - Most-frequent imputation + one-hot encoding to categorical features

    Parameters
    ----------
    num_features : array-like
        Names of numerical columns.
    cat_features : array-like
        Names of categorical columns.

    Returns
    -------
    preprocessor : ColumnTransformer
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    return preprocessor


def prepare_features(df: pd.DataFrame):
    """
    Full feature preparation: drop leakage/useless columns, drop high-cardinality
    text columns, separate features and target, and identify feature types.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as loaded from CSV.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target series (Rent).
    num_features : pd.Index
    cat_features : pd.Index
    """
    df = df.drop(columns=LEAKAGE_AND_USELESS_COLS, errors="ignore")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X = X.drop(columns=HIGH_CARDINALITY_COLS, errors="ignore")

    num_features, cat_features = identify_feature_types(X)

    return X, y, num_features, cat_features
