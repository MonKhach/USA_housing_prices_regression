# USA Housing Price Prediction (Linear Regression)

## Goal
Predict house sale price using a linear regression baseline. Because raw prices are highly right-skewed, the model is trained on **log(price)**.

## Dataset
Kaggle: USA Housing Dataset  
Rows: ~4,140  
Target: `price` (USD)  
Key features: bedrooms, bathrooms, sqft_living, sqft_lot, floors, view, condition, waterfront, yr_built, yr_renovated, city, statezip, date.

Preprocessing note: rows with `price <= 0` were removed to allow log transform.

## Method
- Feature engineering:
  - Date → `SaleYear`, `SaleMonth`, `SaleQuarter`
  - Renovation → `IsRenovated`, `AgeRenovated`
  - Target transform → `log_price = log(price)`
- Split strategy:
  - Chronological split (no shuffle): Train 70% / Val 15% / Test 15%
- Encoding:
  - One-hot encode: `city`, `statezip`
- Model:
  - `LinearRegression` (scikit-learn)
- Metrics:
  - Evaluated in log space and converted back to USD for interpretability

## EDA Highlights
- Price distribution is strongly right-skewed; log transform makes it close to normal.
- `sqft_living` has the strongest positive relationship with log_price.
- Bedroom/bathroom counts show increasing price trend with saturation at high values.
- Correlation ranking confirms sqft and bathrooms as top predictors.

## Results
Validation (log): RMSE = 0.249, MAE = 0.183  
Test (log): RMSE = 0.471, MAE = 0.272  

Validation (USD): RMSE = $201,732, MAE = $113,171  
Test (USD): RMSE = $1,205,749, MAE = $202,697  

Notes:
- Test RMSE is much higher due to sensitivity to extreme outliers.
- MAE is more stable and reflects typical error magnitude.

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Open and run: `notebooks/housing_regression.ipynb`

## Next Steps
- Try Ridge/Lasso regression to reduce overfitting with many one-hot features
- Investigate and handle outliers (cap/winsorize or robust regression)
- Use cross-validation (time-series aware)
- Compare with tree-based models as a benchmark
