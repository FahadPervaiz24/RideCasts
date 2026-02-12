import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/features_hourly.parquet")

df["hour_of_week"] = (df["day_of_week"] * 24 ) + df["hour_of_day"]

df = df.sort_values(["hour", "PULocationID"]).reset_index(drop=True)

#split data so were validating on last 28 days
cutoff = df["hour"].max() - pd.Timedelta(days=28)

train = df[df["hour"] < cutoff]
val = df[df["hour"] >= cutoff]


averages = train.groupby(["PULocationID","hour_of_week"])["trip_count"].mean()

averages = averages.to_frame(name="pred")

val_with_preds = val.merge(averages, on=["PULocationID","hour_of_week"], how="left")

print(val_with_preds[["trip_count", "pred"]].head())
print(val_with_preds["pred"].isna().mean())

# ---- Fallback for rare missing (zone, hour_of_week) pairs ----
zone_mean = train.groupby("PULocationID")["trip_count"].mean()

val_with_preds["pred"] = val_with_preds["pred"].fillna(
    val_with_preds["PULocationID"].map(zone_mean)
)

# Sanity check: should be 0 after fallback
missing_after = val_with_preds["pred"].isna().mean()
print("Missing pred rate after fallback:", missing_after)

# ---- Metrics: MAE + sMAPE ----
y_true = val_with_preds["trip_count"].to_numpy()
y_pred = val_with_preds["pred"].to_numpy()

mae = np.mean(np.abs(y_true - y_pred))
smape = np.mean(
    2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
)

print("Calendar baseline MAE:", mae)
print("Calendar baseline sMAPE:", smape)







