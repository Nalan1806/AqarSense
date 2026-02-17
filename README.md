# AqarSense — UAE Rental Price Prediction

**Intelligent Rental Estimation for UAE Real Estate**

AqarSense is an end-to-end machine learning system designed to estimate annual rental prices (AED) for UAE properties using real-world listing data.

Built with a focus on clean preprocessing pipelines, leakage prevention, reproducible validation, and practical deployment through a Streamlit interface.

---

## Problem Statement

Predict **annual rental prices (AED)** for properties across the UAE using structured listing data (73,742 rows, 17 columns).

This is a supervised regression problem evaluated using RMSE.

---

## Dataset Characteristics

- Real-world listing data  
- Highly right-skewed rent distribution  
- Extreme luxury outliers (max ≈ 55 million AED)  
- No transactional history  
- No amenity-level or building-quality features  

These characteristics significantly influence model error behavior.

---

## Methodology

| Step | Detail |
|------|--------|
| **Leakage prevention** | Dropped `Rent_per_sqft`, `Rent_category`, `Frequency`, `Purpose` |
| **High-cardinality removal** | Dropped raw `Address` field |
| **Preprocessing** | `ColumnTransformer` with median imputation (numeric) and most-frequent + OneHotEncoder (categorical) |
| **Target transformation** | `log1p(Rent)` to stabilize variance under heavy right-skew |
| **Model comparison** | Linear Regression, Ridge, Random Forest, Gradient Boosting |
| **Final model** | Ridge Regression (`alpha=1.0`) |
| **Validation** | 5-fold cross-validation (`KFold`, shuffle=True, random_state=42) |

---

## Why Log Transformation?

Rental prices exhibit extreme right-skew due to luxury listings.

Training directly on raw AED values causes:

- Instability  
- High sensitivity to outliers  
- Poor generalization  

Applying `log1p`:

- Stabilizes variance  
- Reduces impact of extreme values  
- Improves model stability  

Predictions are converted back using `expm1()`.

---

## Why Ridge Regression?

Tree-based models (Random Forest, Gradient Boosting) showed instability after exponential back-transformation from log space.

Ridge Regression:

- Handles high-dimensional one-hot encoded features well  
- Provides stable generalization  
- Prevents coefficient explosion  
- Delivered the best cross-validated performance  

---

## Cross-Validation Results

| Metric | Value |
|--------|--------|
| Mean log-RMSE | 0.480 |
| Std log-RMSE | 0.014 |
| Approx multiplicative error | ×1.62 |

### Interpretation

A multiplicative error of ×1.62 means predictions are typically within **±62% of the true rent**.

Given the dataset's characteristics — highly skewed listings, absence of transaction-level data, and extreme luxury outliers — this error range is consistent with real-world scraped real estate datasets.

Importantly:

- The low standard deviation (0.014) across folds indicates **stable generalization**
- Performance is consistent across multiple data splits
- The model is not dependent on a single validation split

---

## Engineering Highlights

- Full preprocessing encapsulated inside pipeline  
- No data leakage  
- Reproducible cross-validation  
- Modular training and inference separation  
- Production-ready Streamlit interface  
- Clean project structure  

---

## Project Structure

rent-prediction-uae/
├── data/
│ └── dubai_properties.csv
├── notebooks/
│ └── exploration.ipynb
├── src/
│ ├── init.py
│ ├── preprocessing.py
│ ├── train.py
│ └── predict.py
├── models/
│ └── ridge_model.pkl
├── app.py
├── requirements.txt
└── README.md


---

## Running AqarSense (Streamlit App)

### 1. Install dependencies

```bash
pip install -r requirements.txt
2. Train the model
python -m src.train
This will:

Run 5-fold cross-validation

Train the final Ridge pipeline on full dataset

Save model to models/ridge_model.pkl

3. Launch the Streamlit app
streamlit run app.py
The application will open in your browser, allowing you to:

Select city and location

Enter property details

View predicted rent

See confidence range

Explore rent distribution

Visualize property location and rent heatmap

Limitations
Based on listing prices, not finalized transactions

No building age, view quality, or amenity data

No macroeconomic or time-based features

Luxury outliers inflate RMSE

Future improvements could include:

Target encoding for high-cardinality features

Spatial clustering features

Hyperparameter tuning

Transaction-level datasets

Author
AqarSense
A project by Nalan Baburajan

License
MIT