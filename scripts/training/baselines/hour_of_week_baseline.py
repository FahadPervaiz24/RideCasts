import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/features_hourly.parquet")

#168 hours in a week, grab the average for each per zone
df["hour_of_week"] = (df["day_of_week"]*24) + df["hour_of_day"]

cutoff = df["hour"].max() - pd.Timedelta(days=28)
train = df[df["hour"] < cutoff]
val = df[df["hour"] >= cutoff]

averages = (
    train.groupby(["PULocationID", "hour_of_week"])["trip_count"]
    .mean()
    .reset_index()
)
val = val.merge(
    averages,
    on=["PULocationID", "hour_of_week"],
    how="left"
)

y_true = val["trip_count_x"]
y_pred = val["trip_count_y"]

mae = np.mean(np.abs(y_true - y_pred))
smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))
print("MAE:", mae)
print("sMAPE:", smape)
