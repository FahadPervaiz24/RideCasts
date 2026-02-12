import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate GHCNh weather to hourly bins.")
    parser.add_argument("--infile", required=True, help="Input hourly weather parquet.")
    parser.add_argument("--outfile", required=True, help="Output aggregated parquet.")
    args = parser.parse_args()

    in_path = Path(args.infile)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_parquet(in_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "station_id"])
    df["hour"] = df["datetime"].dt.floor("h")

    # Mean for most columns, sum precipitation per hour.
    agg = df.groupby(["station_id", "hour"], as_index=False).agg(
        {
            "temperature": "mean",
            "dew_point_temperature": "mean",
            "station_level_pressure": "mean",
            "sea_level_pressure": "mean",
            "wind_speed": "mean",
            "wind_gust": "mean",
            "relative_humidity": "mean",
            "precipitation": "sum",
        }
    )

    agg["is_rain"] = (agg["precipitation"] > 0).astype(int)
    agg = agg.sort_values(["station_id", "hour"]).reset_index(drop=True)

    # Also create a citywide hourly view by averaging across stations.
    citywide = (
        agg.groupby("hour", as_index=False)
        .agg(
            {
                "temperature": "mean",
                "dew_point_temperature": "mean",
                "station_level_pressure": "mean",
                "sea_level_pressure": "mean",
                "wind_speed": "mean",
                "wind_gust": "mean",
                "relative_humidity": "mean",
                "precipitation": "mean",
                "is_rain": "max",
            }
        )
        .sort_values("hour")
        .reset_index(drop=True)
    )

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_station_path = out_path.with_name(out_path.stem + "_by_station" + out_path.suffix)
    agg.to_parquet(by_station_path, index=False)
    citywide.to_parquet(out_path, index=False)
    print("saved:", by_station_path, "rows:", len(agg))
    print("saved:", out_path, "rows:", len(citywide))


if __name__ == "__main__":
    main()
