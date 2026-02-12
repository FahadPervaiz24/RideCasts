import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import xgboost as xgb

# Load features
df = pd.read_parquet("data/processed/features_hourly.parquet")

# Week-hour feature (0-167)
df["week_hour"] = df["day_of_week"] * 24 + df["hour_of_day"]
df["month"] = df["month"].astype(int)
df["day_of_year"] = df["day_of_year"].astype(int)
df["week_of_year"] = df["week_of_year"].astype(int)

# Train/val split: last 28 days as validation
cutoff = df["hour"].max() - pd.Timedelta(days=28)
train = df[df["hour"] < cutoff]
val = df[df["hour"] >= cutoff]

# Baseline-as-feature: mean trips per zone x week_hour (train only)
baseline = (
    train.groupby(["PULocationID", "week_hour"], as_index=False)["trip_count"]
    .mean()
    .rename(columns={"trip_count": "baseline_week_hour_mean"})
)
train = train.merge(baseline, on=["PULocationID", "week_hour"], how="left")
val = val.merge(baseline, on=["PULocationID", "week_hour"], how="left")

global_mean = train["trip_count"].mean()
train["baseline_week_hour_mean"] = train["baseline_week_hour_mean"].fillna(global_mean)
val["baseline_week_hour_mean"] = val["baseline_week_hour_mean"].fillna(global_mean)

feature_cols = [
    "PULocationID",
    "week_hour",
    "month",
    "day_of_year",
    "week_of_year",
    "baseline_week_hour_mean",
    "temperature",
    "wind_speed",
    "relative_humidity",
    "precipitation",
    "is_rain",
    "is_weekend",
    "is_holiday",
]

X_train = train[feature_cols].copy()
y_train = train["trip_count"]
X_val = val[feature_cols].copy()
y_val = val["trip_count"]

# Treat these as categorical for XGBoost (requires recent xgboost)
cat_cols = ["PULocationID", "week_hour", "month", "week_of_year"]
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_val[col] = X_val[col].astype("category")

# Log-transform target to stabilize variance
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=10,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="mae",
    tree_method="hist",
    enable_categorical=True,
    random_state=0,
)

model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    verbose=False,
)

# Predict in log space, then invert
y_pred_log = model.predict(X_val)
y_pred = np.expm1(y_pred_log)

mae = np.mean(np.abs(y_val - y_pred))
smape = np.mean(2 * np.abs(y_pred - y_val) / (np.abs(y_pred) + np.abs(y_val) + 1e-8))

print("MAE:", mae)
print("sMAPE:", smape)

# Save model + metrics
out_dir = Path("models") / "XGBoost"
out_dir.mkdir(parents=True, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = out_dir / f"xgboost_week_hour_{run_id}.json"
model.save_model(str(model_path))
print("saved model:", model_path)

metrics_path = out_dir / f"xgboost_week_hour_{run_id}_metrics.txt"
metrics_path.write_text(f"MAE: {mae}\nsMAPE: {smape}\n")
print("saved metrics:", metrics_path)

latest_model = out_dir / "xgboost_week_hour_latest.json"
latest_metrics = out_dir / "xgboost_week_hour_latest_metrics.txt"
latest_model.write_text(model_path.read_text())
latest_metrics.write_text(metrics_path.read_text())
print("saved latest model:", latest_model)
print("saved latest metrics:", latest_metrics)
