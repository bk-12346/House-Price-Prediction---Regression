# 🏘️ House Price Prediction using Regression Models

This project implements and compares three different regression models---**Linear Regression**, **Ridge Regression**, and **RandomForest Regressor**---to predict house sale prices using the advanced "House Prices - Advanced Regression Techniques" dataset from Kaggle.

The target variable (`SalePrice`) is transformed using a logarithmic function ($\log(SalePrice)$) before training, and predictions are inverse-transformed ($\exp(\text{predictions})$) back to USD for final analysis.

## 🚀 Project Structure

* `house_price_prediction_local.py`: The main Python script for data processing, model training, evaluation, and visualization.

* `requirements.txt`: Lists all necessary Python dependencies.

* `random_forest_regressor_model.joblib`: The saved trained Random Forest model (created after running the script).

* `house_price_predictions.csv`: The predicted sale prices for the test set (created after running the script).

## ⚙️ Setup and Installation

### 1. Clone the Repository

```
git clone https://github.com/bk-12346/House-Price-Prediction---Regression
cd House-Price-Prediction---Regression
```

### 2. Install Dependencies

Install the required Python packages using pip:

```bash pip install -r requirements.txt```

### 3. Configure Kaggle API (Data Source)

The script automatically downloads the dataset using `kagglehub`. You must configure your Kaggle API key:

1.  Go to your Kaggle account settings and generate an **API Token** (`kaggle.json`).

2.  Place the `kaggle.json` file in the **`.kaggle`** folder in your home directory (`~/.kaggle/`).


## ▶️ How to Run the Script
------------------------

Execute the main Python file from your terminal:

Bash

```
python main.py

```

The script will:

1.  Download and preprocess the data (imputation, one-hot encoding).

2.  Train and evaluate all three models using **RMSE** and **R-squared**.

3.  Perform **5-fold Cross-Validation** for robust performance estimation.

4.  Generate comparison plots for training performance.

5.  Generate plots comparing test set predictions.

6.  Save the best model (`random_forest_regressor_model.joblib`) and the test predictions (`house_price_predictions.csv`).


## 📊 Model Performance
--------------------

The script compares the models based on **Root Mean Squared Error (RMSE)** (lower is better) and **R-squared** (closer to 1.0 is better) on the training data.

### Training Performance Comparison

The Random Forest Regressor shows the best fit on the training data, as seen by the lowest RMSE and highest R-squared.

| **Model** | **Training RMSE** | **Training R-squared** | **Cross-Validation RMSE (Mean)** |
| --- | --- | --- | --- |
| **Linear Regression** | 0.1044 | 0.9388 | 0.1378 |
| **Ridge Regression** | 0.1074 | 0.9352 | 0.1328 |
| **RandomForest Regressor** | **0.0543** | **0.9845** | **0.1396** |
