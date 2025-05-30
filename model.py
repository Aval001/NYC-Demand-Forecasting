import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Paths ---
data_folder = "data/processed"
model_folder = "data/models"
os.makedirs(model_folder, exist_ok=True)

file_names = [f"processed_{i:02d}.csv" for i in range(1, 8)]

# --- Step 1: Load 12 monthly files ---
all_months = []
for file in file_names:
    df = pd.read_csv(os.path.join(data_folder, file), index_col=0)
    df.index = pd.to_datetime(df.index)
    all_months.append(df)
full_df = pd.concat(all_months)
zones = full_df.columns

# --- Step 2: Time features incl. is_weekday ---
def create_time_features(df):
    df_feat = df.copy()

    # Extract and encode time-based features directly from the index
    hour = df_feat.index.hour
    minute_bin = (df_feat.index.minute // 5).astype(int)
    dow = df_feat.index.dayofweek
    month = df_feat.index.month

    df_feat["is_weekday"] = (dow < 5).astype(int)
    df_feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df_feat["bin5_sin"] = np.sin(2 * np.pi * minute_bin / 12)
    df_feat["bin5_cos"] = np.cos(2 * np.pi * minute_bin / 12)
    df_feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df_feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df_feat["month_sin"] = np.sin(2 * np.pi * month / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Drop original demand columns (zone columns)
    return df_feat.drop(columns=df.columns)

# --- Step 3: Train XGBoost per zone and save model ---
results = {}
val_error_distributions = {}

for zone in zones:
    print(f"📈 Training model for zone: {zone}")
    
    y = full_df[zone].copy()
    X = create_time_features(full_df)

    # Add lag features
    X["lag_1"] = y.shift(1)
    X["lag_24"] = y.shift(24)
    X["lag_168"] = y.shift(168)

    # Target
    y = y.shift(-1)

    # Drop final 3 rows to align target and features
    X = X.iloc[:-3]
    y = y.iloc[:-3]

    # Drop rows with NaNs (due to lagging)
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # Split
    train_idx = X.index < "2010-06-01"
    val_idx = (X.index >= "2010-06-01") & (X.index < "2010-07-01")
    test_idx = X.index >= "2010-07-01"

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    # Fit model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
    model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(model_folder, f"xgb_zone_{zone}.joblib")
    joblib.dump(model, model_path)

    # Validation predictions and error distribution
    y_val_pred = model.predict(X_val)
    val_errors = y_val_pred - y_val
    val_mean = np.mean(val_errors)
    val_std = np.std(val_errors)

    val_error_distributions[zone] = {"mean": val_mean, "std": val_std}

    # Test set evaluation
    y_test_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    results[zone] = {"MAE": mae, "MSE": mse}

# --- Step 4: Export results and error distributions ---
results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join(model_folder, "xgboost_forecast_results.csv"))

val_error_df = pd.DataFrame(val_error_distributions).T
val_error_df.to_csv(os.path.join(model_folder, "val_error_distributions.csv"))

print("✅ Models and results saved to data/models/")
