import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/features_hourly.parquet")
df = df.sort_values(["PULocationID", "hour"])

# Seasonal-naive baseline: same hour last week (t-168).
df["baseline_pred"] = df.groupby("PULocationID")["trip_count"].shift(168)
df = df.dropna(subset=["baseline_pred"])

y_actual = df["trip_count"].to_numpy()
y_pred = df["baseline_pred"].to_numpy()

mae = np.mean(np.abs(y_actual - y_pred))
smape = np.mean(2 * np.abs(y_pred - y_actual) / (np.abs(y_pred) + np.abs(y_actual) + 1e-8))

print("MAE:", mae)
print("sMAPE:", smape)


