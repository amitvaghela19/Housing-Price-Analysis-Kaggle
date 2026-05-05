# House Price Prediction – End-to-End ML with Scikit‑Learn

End‑to‑end regression project to predict house sale prices from basic property characteristics using a clean preprocessing pipeline and multiple scikit‑learn models.

---

## 1. Project overview

This project predicts house prices from a tabular dataset using a modern scikit‑learn workflow:

- Exploratory data analysis (EDA) to understand distributions and correlations.
- A reusable preprocessing pipeline with numeric scaling and categorical one‑hot encoding.
- Several regression models (linear, tree‑based, and regularized) with cross‑validation.
- A final, data‑driven comparison of model performance.

The notebook is written to be interview‑ and portfolio‑ready: the steps are explicit and the trade‑offs between models are clearly evaluated.

---

## 2. Dataset

The data comes from a synthetic house price dataset with 2,000 rows and 10 columns:

- **Id**: Row identifier (dropped before modeling).
- **Area**: Floor area in square feet (approximately 500–5,000).
- **Bedrooms**: Number of bedrooms (1–5).
- **Bathrooms**: Number of bathrooms (1–4).
- **Floors**: Number of floors (1–3).
- **YearBuilt**: Construction year (1900–2023).
- **Location**: Categorical, e.g. `Downtown`, `Suburban`, `Rural`, etc.
- **Condition**: Categorical, e.g. `Excellent`, `Good`, `Fair`, `Poor`.
- **Garage**: Categorical, `Yes`/`No`.
- **Price**: Target variable, sale price (≈ 50,000–1,000,000).

Basic checks in the notebook show:

- No missing values in any column.
- No duplicate rows.
- Reasonable numeric ranges and a wide spread in prices.

---

## 3. Exploratory data analysis

The notebook performs targeted EDA to understand the problem:

- **Summary statistics** (`df.describe`) to check ranges and potential outliers.
- **Correlation heatmap** on numeric features to see how `Area`, `Bedrooms`, `Bathrooms`, etc. relate to `Price`.
- **Boxplots** of `Price` by `Location`, `Condition`, and `Garage` to see how price varies across segments.

These plots highlight intuitive patterns:

- Larger area, more bedrooms/bathrooms, and newer houses tend to correlate positively with higher prices.
- Downtown or “Excellent” condition homes sit at higher price ranges than rural or “Poor” condition homes.
- Having a garage shifts the price distribution upward.

---

## 4. Feature engineering and preprocessing

The project uses a clean, production‑style preprocessing pipeline built with `ColumnTransformer` and `Pipeline`.

### 4.1 Train/test split

- `X` = all features except `Price` (and dropping `Id`).
- `y` = `Price`.
- Split: `train_test_split(X, y, test_size=0.2, random_state=42)`.

### 4.2 Feature groups

Features are split into numeric and categorical sets:

- **Numeric**: `Area`, `Bedrooms`, `Bathrooms`, `Floors`, `YearBuilt`
- **Categorical**: `Location`, `Condition`, `Garage`

### 4.3 Preprocessing pipeline

- **Numeric transformer**:
  - `SimpleImputer(strategy="median")`
  - `StandardScaler`
- **Categorical transformer**:
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`

These are combined in a `ColumnTransformer` and used both for model training and evaluation, ensuring no data leakage from the test set.

---

## 5. Models

Several baseline and improved models are trained and compared.

### 5.1 Baseline models

After preprocessing, the notebook trains:

- **Linear Regression** (`LinearRegression`)
- **Decision Tree Regressor** (`DecisionTreeRegressor`)
- **Random Forest Regressor** (`RandomForestRegressor`)

Each model is evaluated using:

- RMSE (root mean squared error)
- MAE (mean absolute error)
- R² (coefficient of determination)

Metrics are computed for both train and test sets, and there is an additional cross‑validation run using `cross_validate` to get more stable estimates.

### 5.2 Tuned Ridge Regression

To explore regularization, a **Ridge Regression** model is tuned via `GridSearchCV`:

- Model: `Ridge`
- Hyperparameter grid: `alpha` ∈ {0.1, 1.0, 10.0, 100.0}
- Scoring: negative RMSE (`neg_root_mean_squared_error`)
- CV: 5‑fold cross‑validation

The best configuration is:

- **Best alpha**: `100.0`
- **Best CV RMSE**: ≈ 276,000

The tuned Ridge model is then evaluated on the train and test sets, with metrics captured for comparison.

---

## 6. Model performance

All models are evaluated on the same train/test split after identical preprocessing.

Approximate performance (test set):

| Model                     | Train RMSE | Test RMSE | Train MAE | Test MAE | Train R² | Test R² |
|---------------------------|-----------:|----------:|----------:|---------:|---------:|--------:|
| Linear Regression         |   274,300  |  279,900  |  237,100  | 243,200  |   0.01   | -0.01   |
| Decision Tree Regressor   |        ~0  |  405,000  |        ~0 | 332,000  |   1.00   | -1.11   |
| Random Forest Regressor   |   107,600  |  285,000–295,000  |  91,000  | 245,000–255,000 | 0.85 | ≈ -0.07 to -0.11 |
| Ridge Regression (tuned)  |   274,400  |  279,800  |  237,200  | 243,300  |   0.01   | -0.01   |

Key observations:

- **Decision Tree** massively overfits: near‑zero train error but very poor generalization, with large negative R² on the test set.
- **Random Forest** achieves very low train error but still fails to generalize well on this small, low‑dimensional dataset, ending up slightly worse than linear models on the test set.
- **Linear Regression vs. Ridge**: tuned Ridge (alpha=100) performs essentially the same as plain Linear Regression, improving test RMSE by only about 50 units, which is negligible relative to overall price scale.
- Overall, the dataset is simple enough—and the chosen features basic enough—that **a well‑regularized linear model is as good as more complex tree models**.

---

## 7. What this project demonstrates

This notebook is a compact but solid example of end‑to‑end tabular regression:

- **Proper preprocessing** with train/test separation and scikit‑learn pipelines.
- **Multiple model families** evaluated under a common pipeline (linear, tree, ensemble, regularized linear).
- **Honest generalization checks** showing:
  - How trees and forests can overfit small, low‑dimensional data.
  - Why a simple linear model is sometimes the right choice.
- **Metric‑driven model selection**: Ridge is selected as the “best” model by lowest test RMSE, but the narrative explains that gains over vanilla Linear Regression are minimal.

This kind of analysis reads well in interviews and shows maturity in model choice and evaluation.

---

## 8. Suggested repository structure

A clean structure for a GitHub repo based on this notebook:

```text
house-price-ml/
├─ data/
│  └─ house_price_dataset.csv      # Original dataset (or a sample)
├─ notebooks/
│  └─ House_Price_Prediction_ML.ipynb
├─ src/
│  ├─ preprocessing.py             # ColumnTransformer & pipeline builders
│  ├─ models.py                    # Model training & evaluation functions
│  └─ utils.py                     # Metrics, plotting helpers
├─ README.md
└─ requirements.txt
```

---

## 9. How to run

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/house-price-ml.git
   cd house-price-ml
   ```

2. **Set up the environment**

   ```bash
   pip install -r requirements.txt
   ```

   Core libraries:

   - `pandas`
   - `numpy`
   - `matplotlib`
   - `seaborn`
   - `scikit-learn`

3. **Place the data**

   - Save the CSV as `data/house_price_dataset.csv` or update the path in the notebook.

4. **Run the notebook**

   - Open `notebooks/House_Price_Prediction_ML.ipynb` in Jupyter or Google Colab.
   - Run all cells to reproduce the preprocessing, model training, and performance comparison.

---

## 10. Possible extensions

If you want to extend this project further:

- Add **more features** (e.g., neighborhood, distance to amenities, lot size).
- Try **non‑linear models** like Gradient Boosting, XGBoost, or LightGBM with proper hyperparameter tuning.
- Use **log‑transformed price** as the target to stabilize variance.
- Add **k‑fold cross‑validation plots** for different models to better visualize variance.
- Build a small **Streamlit app** to let users interactively enter features and get predicted prices.
