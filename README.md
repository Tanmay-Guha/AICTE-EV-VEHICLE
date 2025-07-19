# 🔋 Electric Vehicle Population Analysis & Prediction

**Project Name:** `TANMAY_GUHA_MONTH1ipynb`
**Author:** Tanmay Guha

This project analyzes and predicts electric vehicle adoption patterns using a dataset titled `Electric_Vehicle_Population_By_County.csv`. It performs thorough preprocessing, outlier handling, and builds a machine learning model using Random Forest Regressor.

---

## 📌 Objectives

* Analyze electric vehicle distribution across counties
* Handle missing data and outliers
* Apply Random Forest Regression to predict EV adoption
* Evaluate model performance using error metrics (MAE, RMSE, R²)

---

## 📂 Dataset

**File Used:** `Electric_Vehicle_Population_By_County.csv`
**Key Columns:**

* `Date` — Date of record
* `County`, `State` — Geographic location
* `Electric Vehicle (EV) Total` — Number of EVs
* `Percent Electric Vehicles` — % of total vehicles that are EVs

---

## 🛠️ Libraries Used

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
```

---

## 🔍 Workflow Summary

### ✅ Data Loading & Exploration

* Load CSV with `pandas`
* View initial data, types, shape
* Identify missing values and data issues

### 🔧 Data Cleaning

* Convert `Date` to datetime
* Drop or fill missing values in key columns (`County`, `State`, `EV Total`)
* Remove rows with invalid dates or null target values

### 🚨 Outlier Detection & Treatment

* Use IQR to identify and cap outliers in `Percent Electric Vehicles`

```python
Q1 = df['Percent Electric Vehicles'].quantile(0.25)
Q3 = df['Percent Electric Vehicles'].quantile(0.75)
IQR = Q3 - Q1
df['Percent Electric Vehicles'] = np.where(df['Percent Electric Vehicles'] > upper_bound, upper_bound,
      np.where(df['Percent Electric Vehicles'] < lower_bound, lower_bound, df['Percent Electric Vehicles']))
```

### 📊 Feature Engineering & Model Training

* Label encoding if necessary
* Train/Test split
* Model: `RandomForestRegressor`
* Tuning: `RandomizedSearchCV` (if applied)

### 📈 Model Evaluation

* MAE, RMSE, R² Score
* Save model with `joblib` for future use

---

## 📊 Output

* Cleaned and processed EV dataset
* Outlier-handled data
* RandomForest prediction model
* Evaluation metrics with visualizations

---
