# 🔋 Electric Vehicle Forecasting by County

**Project:** TANMAY-GUHA-WEEK2.ipynb
**Author:** Tanmay Guha

This project performs advanced forecasting of Electric Vehicle (EV) adoption trends across various counties using time-series feature engineering and a tuned Random Forest Regressor. It includes historical analysis, feature extraction, lag-based forecasting, and county-wise 3-year prediction.

---

## 🎯 Objectives

* Clean and preprocess EV population data
* Engineer time-series features (lags, rolling averages, slopes)
* Train a Random Forest model for predicting EV counts
* Generate 3-year monthly forecasts for all counties
* Visualize top 5 counties by cumulative EV adoption

---

## 🧰 Tech Stack & Libraries

| Tool                            | Use                                 |
| ------------------------------- | ----------------------------------- |
| Python                          | Programming Language                |
| Pandas, NumPy                   | Data handling & numerical computing |
| Seaborn, Matplotlib             | Data visualization                  |
| Scikit-learn                    | Machine learning                    |
| Joblib                          | Model persistence                   |
| Jupyter Notebook / Google Colab | Development Environment             |

---

## 📊 Dataset

**File Used:** `Electric_Vehicle_Population_By_County.csv`
Key columns:

* `Date`
* `County`, `State`
* `Electric Vehicle (EV) Total`
* `Battery Electric Vehicles (BEVs)`, `PHEVs`, `Non-Electric Vehicle Total`
* `Percent Electric Vehicles`

---

## 🔧 Preprocessing & Feature Engineering

* Converted `Date` column to datetime
* Filled nulls in `County`/`State`, removed rows with missing target
* Detected and capped outliers using IQR method
* Converted numeric columns to correct types
* Extracted date-based features: year, month, numeric\_date
* Created lag features (`lag1`, `lag2`, `lag3`)
* Rolling mean (3-month)
* Percent change over 1 and 3 months
* 6-month EV growth slope (via linear regression)
* Cumulative EV counts

---

## 🤖 Model Training

* **Model:** RandomForestRegressor

* **Tuning:** RandomizedSearchCV (30 iterations with CV=3)

* **Target:** `Electric Vehicle (EV) Total`

* **Features Used:**

  ```text
  months_since_start, county_encoded, ev_total_lag1,
  ev_total_lag2, ev_total_lag3, ev_total_roll_mean_3,
  ev_total_pct_change_1, ev_total_pct_change_3, ev_growth_slope
  ```

* **Evaluation Metrics:**

  * MAE
  * RMSE
  * R² Score
  * Visual: Actual vs Predicted Plot

---

## 📈 Forecasting

* Forecasts generated for **next 36 months (3 years)** per county
* Predicts future EV adoption using latest historical trends
* Combines both:

  * **Individual Forecast:** e.g., Kings County
  * **Multi-County Forecast:** All counties in dataset

---

## 📊 Visualizations

* Stacked column chart (BEV vs PHEV vs Non-EV)
* Actual vs Predicted line plot
* Feature importance bar chart
* County-wise forecast line plot
* Top 5 counties: cumulative EV growth forecast

---

## 💾 Model Saving

* Trained model saved as:

  ```bash
  forecasting_ev_model.pkl
  ```
* Test predictions with saved model loaded using `joblib`

---

## ✨ Improvisations Done by Me

1. **Outlier Handling with Capping** using IQR
2. **Lag, rolling average, and percent change features** for time-series learning
3. **Growth slope calculation** using rolling linear regression
4. **County-wise forecasting loop** for all regions with cumulative trends
5. **Interactive and informative visualizations** for interpretability
6. **Hyperparameter tuning** via RandomizedSearchCV for better accuracy
7. **Model serialization** for deployment or reuse
8. **Forecast validation** using real vs predicted comparison

---

## 📂 How to Run

### Setup Environment

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Run Notebook

Launch Jupyter Notebook or open in Google Colab:

```bash
jupyter notebook TANMAY-GUHA-WEEK2.ipynb
```

---

## 📈 Sample Output

* Best Parameters of Random Forest printed
* Metrics: `MAE`, `RMSE`, `R²`
* Predicted vs Actual line graph
* Feature importances
* 3-year forecast for top 5 counties

---

## 📇 Author Info

**Tanmay Guha**

*BTech CSE - Data Science Student*

---
