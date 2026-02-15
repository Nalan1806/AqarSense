# UAE Rental Price Prediction

## Problem Statement

Predict **annual rental prices (AED)** for properties in the UAE using real-world real estate listing data (73,742 rows, 17 columns). This is a supervised regression problem evaluated using RMSE.

## Methodology

| Step | Detail |
|---|---|
| **Data cleaning** | Drop leakage columns (`Rent_per_sqft`, `Rent_category`, `Frequency`, `Purpose`) and high-cardinality text (`Address`) |
| **Preprocessing** | `ColumnTransformer` — median imputation for numerics; most-frequent imputation + one-hot encoding for categoricals |
| **Target transform** | `log1p` applied to `Rent` to stabilise variance under extreme right-skew; predictions are back-transformed with `expm1` |
| **Model** | Ridge Regression (`alpha=1.0`) chosen after comparing Linear Regression, Ridge, Random Forest, and Gradient Boosting |
| **Validation** | 5-fold cross-validation (`KFold`, `shuffle=True`, `random_state=42`) — mean log-RMSE ≈ 0.480, std ≈ 0.014 |

## Why Ridge Regression?

Tree-based models (Random Forest, Gradient Boosting) suffered instability after exponential back-transformation of log-space predictions, producing worse AED-scale RMSE than Ridge. Ridge's L2 regularisation handles the high-dimensional one-hot encoded feature space gracefully and provides the best balance of stability, generalizability, and interpretability.

## Project Structure

```
rent-prediction-uae/
├── data/
│   └── dubai_properties.csv
├── notebooks/
│   └── exploration.ipynb       # Full experiment notebook 
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Feature prep & ColumnTransformer
│   ├── train.py                # Training + CV + model saving
│   └── predict.py              # Load model & predict single property
├── models/
│   └── ridge_model.pkl         # Saved pipeline (created by train.py)
├── app.py                      # FastAPI serving endpoint
├── requirements.txt
└── README.md
```

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python -m src.train
```

This will run 5-fold cross-validation, train the final Ridge pipeline on the full dataset, and save it to `models/ridge_model.pkl`.

### 3. Predict from Python

```python
from src.predict import predict_rent

rent = predict_rent({
    "City": "Dubai",
    "Location": "Downtown Dubai",
    "Beds": 2,
    "Baths": 2,
    "Type": "Apartment",
    "Area_in_sqft": 1200,
    "Latitude": 25.1972,
    "Longitude": 55.2744,
    "Furnishing": "Furnished",
})
print(f"Predicted annual rent: AED {rent:,.2f}")
```

### 4. Serve via API

```bash
uvicorn app:app --reload
```

Then POST to `http://127.0.0.1:8000/predict` with a JSON body matching the property schema.

## Cross-Validation Results

| Metric | Value |
|---|---|
| Mean log-RMSE | 0.480 |
| Std log-RMSE | 0.014 |
| Approx multiplicative error | ×1.62 (predictions within ±62% of actual) |

## License

MIT
