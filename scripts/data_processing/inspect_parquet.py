import pandas as pd

df = pd.read_parquet("data/processed/features_hourly.parquet")
group = df.groupby(["PULocationID", "hour"])
print(df["hour"].head(5))
