import argparse
import json
from pathlib import Path

import pandas as pd


FEATURES_DEFAULT = "data/processed/features_hourly.parquet"
BASELINE_OUT = "data/serving/baseline_week_hour_mean.csv"
META_OUT = "data/serving/baseline_meta.json"


def build_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, float, list[int]]:
    df = df.copy()
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"])
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["week_hour"] = df["day_of_week"] * 24 + df["hour_of_day"]

    cutoff = df["hour"].max() - pd.Timedelta(days=28)
    train = df[df["hour"] < cutoff].copy()

    baseline = (
        train.groupby(["PULocationID", "week_hour"], as_index=False)["trip_count"]
        .mean()
        .rename(columns={"trip_count": "baseline_week_hour_mean"})
    )
    global_mean = float(train["trip_count"].mean())
    zone_ids = sorted(train["PULocationID"].unique().tolist())
    return baseline, global_mean, zone_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build serving baseline artifacts.")
    parser.add_argument("--features-path", default=FEATURES_DEFAULT)
    parser.add_argument("--baseline-out", default=BASELINE_OUT)
    parser.add_argument("--meta-out", default=META_OUT)
    args = parser.parse_args()

    df = pd.read_parquet(args.features_path, columns=["hour", "PULocationID", "trip_count"])
    baseline, global_mean, zone_ids = build_baseline(df)

    baseline_path = Path(args.baseline_out)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(baseline_path, index=False)

    meta = {
        "baseline_global_mean": global_mean,
        "zone_ids": zone_ids,
    }
    meta_path = Path(args.meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))

    print("saved:", baseline_path)
    print("saved:", meta_path)
    print("zones:", len(zone_ids))


if __name__ == "__main__":
    main()
