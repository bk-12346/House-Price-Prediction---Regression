# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# NOTE: You must have the 'kagglehub' library installed to download the data.
# Run: pip install kagglehub
import kagglehub

# --- 1. Data Download and Loading ---

# Download the dataset from Kaggle
try:
    print("Downloading dataset from Kaggle...")
    # 'rishitaverma02/house-prices-advanced-regression-techniques' is the dataset
    path = kagglehub.dataset_download("rishitaverma02/house-prices-advanced-regression-techniques")
    print(f"Dataset files downloaded to: {path}")

    # Load data to dataframe
    # The file names need to be correctly referenced based on the downloaded path structure
    # The original script used 'train (1).csv' and 'test (1).csv', which might be specific to Colab.
    # We will try the standard file names first.
    train_file = os.path.join(path, "train.csv")
    test_file = os.path.join(path, "test.csv")

    # If standard names don't work, try the (1) names as in the notebook.
    if not os.path.exists(train_file):
         train_file = os.path.join(path, "train (1).csv")
         test_file = os.path.join(path, "test (1).csv")
         
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    print("Data loaded successfully.")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

except Exception as e:
    print(f"Error during data download or loading: {e}")
    print("Please ensure you have 'kagglehub' installed and your Kaggle API is configured.")
    exit()

# --- 2. Data Preprocessing (Identical to Notebook) ---

# Impute numerical columns with the mean
train_df.fillna(train_df.select_dtypes(include=np.number).mean(), inplace=True)
test_df.fillna(test_df.select_dtypes(include=np.number).mean(), inplace=True)

# Impute categorical columns with the mode
for col in train_df.select_dtypes(include='object').columns:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)
for col in test_df.select_dtypes(include='object').columns:
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)

# Select categorical columns
categorical_cols_train = train_df.select_dtypes(include='object').columns
categorical_cols_test = test_df.select_dtypes(include='object').columns

# Apply one-hot encoding
train_df_encoded = pd.get_dummies(train_df, columns=categorical_cols_train, dummy_na=False)
test_df_encoded = pd.get_dummies(test_df, columns=categorical_cols_test, dummy_na=False)

# Separate features (X) and target (y)
X = train_df_encoded.drop('SalePrice', axis=1)

# Get the target variable and apply log transformation (as in the original notebook)
y = np.log(train_df_encoded['SalePrice'])

# Align columns between the training and testing datasets
X_train, X_test = X.align(test_df_encoded, join='inner', axis=1, fill_value=0)

print("\nData Preprocessing complete.")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test (for prediction): {X_test.shape}")

# --- 3. Model Training and Evaluation ---

## Linear Regression

print("\n" + "="*50)
print("Training Linear Regression Model")
print("="*50)

model = LinearRegression()
model.fit(X_train, y)
y_train_pred_lr = model.predict(X_train)
rmse_train_lr = np.sqrt(mean_squared_error(y, y_train_pred_lr))
r2_train_lr = r2_score(y, y_train_pred_lr)

print(f"Linear Regression Training RMSE: {rmse_train_lr:.4f}")
print(f"Linear Regression Training R-squared: {r2_train_lr:.4f}")

# Cross-Validation
rmse_cv_lr = -cross_val_score(model, X_train, y, scoring='neg_mean_squared_error', cv=5)
rmse_cv_lr = np.sqrt(rmse_cv_lr)
print(f"Linear Regression Cross-Validation RMSE (5-fold Mean): {rmse_cv_lr.mean():.4f} (+/- {rmse_cv_lr.std():.4f})")

# Making predictions on X_test
y_test_pred_lr = model.predict(X_test)
print(f"Predictions made on X_test for Linear Regression.")


# ----------------------------------------------------------------------
## Ridge Regression

print("\n" + "="*50)
print("Training Ridge Regression Model")
print("="*50)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y)
y_train_pred_ridge = ridge_model.predict(X_train)
rmse_train_ridge = np.sqrt(mean_squared_error(y, y_train_pred_ridge))
r2_train_ridge = r2_score(y, y_train_pred_ridge)

print(f"Ridge Regression Training RMSE: {rmse_train_ridge:.4f}")
print(f"Ridge Regression Training R-squared: {r2_train_ridge:.4f}")

# Cross-Validation
rmse_cv_ridge = -cross_val_score(ridge_model, X_train, y, scoring='neg_mean_squared_error', cv=5)
rmse_cv_ridge = np.sqrt(rmse_cv_ridge)
print(f"Ridge Regression Cross-Validation RMSE (5-fold Mean): {rmse_cv_ridge.mean():.4f} (+/- {rmse_cv_ridge.std():.4f})")

# Making predictions on X_test
y_test_pred_ridge = ridge_model.predict(X_test)
print(f"Predictions made on X_test for Ridge Regression.")


# ----------------------------------------------------------------------
## RandomForest Regressor

print("\n" + "="*50)
print("Training RandomForest Regressor Model")
print("="*50)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y)
y_train_pred_rf = rf_model.predict(X_train)
rmse_train_rf = np.sqrt(mean_squared_error(y, y_train_pred_rf))
r2_train_rf = r2_score(y, y_train_pred_rf)

print(f"RandomForest Regressor Training RMSE: {rmse_train_rf:.4f}")
print(f"RandomForest Regressor Training R-squared: {r2_train_rf:.4f}")

# Cross-Validation
rmse_cv_rf = -cross_val_score(rf_model, X_train, y, scoring='neg_mean_squared_error', cv=5)
rmse_cv_rf = np.sqrt(rmse_cv_rf)
print(f"RandomForest Regressor Cross-Validation RMSE (5-fold Mean): {rmse_cv_rf.mean():.4f} (+/- {rmse_cv_rf.std():.4f})")

# Making predictions on X_test
y_test_pred_rf = rf_model.predict(X_test)
print(f"Predictions made on X_test for RandomForest Regressor.")


# ----------------------------------------------------------------------
# --- 4. Model Comparison and Visualization ---

print("\n" + "="*50)
print("Model Performance Comparison (Training & CV)")
print("="*50)

print("Model | Train RMSE | Train R2 | CV RMSE (Mean) | CV RMSE (Std)")
print("-" * 60)
print(f"LR    | {rmse_train_lr:.4f}     | {r2_train_lr:.4f} | {rmse_cv_lr.mean():.4f}      | {rmse_cv_lr.std():.4f}")
print(f"Ridge | {rmse_train_ridge:.4f}     | {r2_train_ridge:.4f} | {rmse_cv_ridge.mean():.4f}      | {rmse_cv_ridge.std():.4f}")
print(f"RF    | {rmse_train_rf:.4f}     | {r2_train_rf:.4f} | {rmse_cv_rf.mean():.4f}      | {rmse_cv_rf.std():.4f}")

# Create a DataFrame to hold the performance metrics for plotting
performance_data = {
    'Model': ['Linear Regression', 'Ridge Regression', 'RandomForest Regressor'],
    'RMSE': [rmse_train_lr, rmse_train_ridge, rmse_train_rf],
    'R-squared': [r2_train_lr, r2_train_ridge, r2_train_rf]
}
performance_df = pd.DataFrame(performance_data)

# Melt the DataFrame for easier plotting with seaborn
performance_df_melted = performance_df.melt('Model', var_name='Metric', value_name='Score')

# Create bar plots for RMSE and R-squared
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Score', data=performance_df_melted[performance_df_melted['Metric'] == 'RMSE'])
plt.title('Model Comparison - Training RMSE')
plt.ylabel('RMSE')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='Score', data=performance_df_melted[performance_df_melted['Metric'] == 'R-squared'])
plt.title('Model Comparison - Training R-squared')
plt.ylabel('R-squared')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# --- 5. Saving and Loading the Best Model (Random Forest in this case) ---

# Save the trained RandomForest model
model_filename = 'random_forest_regressor_model.joblib'
# joblib.dump(rf_model, model_filename)
# print(f"\nRandomForest Regressor model saved as: {model_filename}")

# Load the trained model from the file
loaded_rf_model = joblib.load(model_filename)
print(f"RandomForest Regressor model loaded from: {model_filename}")

# Use the loaded model to make predictions on X_test
y_test_pred_loaded_rf = loaded_rf_model.predict(X_test)
print("Predictions made on X_test using the loaded RandomForest Regressor model.")

# --- 6. Visualization of Test Set Predictions ---

print("\n" + "="*50)
print("Visualizing Test Set Predictions")
print("="*50)

# The predictions are on a log scale, so transform them back to original price (USD)
test_ids = test_df['Id'] # Get the Id column from the original (un-encoded) test set

# Exponentiate to get the predicted SalePrice in USD
predictions_df = pd.DataFrame({
    'Id': test_ids,
    'LR_Predicted_Price': np.exp(y_test_pred_lr),
    'Ridge_Predicted_Price': np.exp(y_test_pred_ridge),
    'RF_Predicted_Price': np.exp(y_test_pred_rf) # Use the best model (RandomForest)
})

# Display the first 5 predictions
print("First 5 Predicted Prices (in USD):")
print(predictions_df.head())

# --- Plot 1: Distribution of the Best Model's Predictions ---

plt.figure(figsize=(10, 6))
sns.histplot(predictions_df['RF_Predicted_Price'], kde=True, bins=50)
plt.title('Distribution of Predicted Sale Prices (RandomForest Regressor)')
plt.xlabel('Predicted Sale Price (USD)')
plt.ylabel('Frequency')
plt.ticklabel_format(style='plain', axis='x')
plt.show()

# --- Plot 2: Model Comparison for a Subset of Houses ---

# Compare the predictions for the first 50 houses to see model agreement/disagreement
subset_df = predictions_df.set_index('Id').head(50)
subset_df.plot(kind='line', figsize=(12, 7))

plt.title('Model Comparison for Predicted Sale Prices (First 50 Test Houses)')
plt.xlabel('Test House ID')
plt.ylabel('Predicted Sale Price (USD)')
plt.ticklabel_format(style='plain', axis='y')
plt.legend(title='Model')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Save predictions to a CSV file

predictions_df.to_csv('house_price_predictions.csv', index=False)
print("\nPredicted values saved to house_price_predictions.csv")