import argparse
from pathlib import Path 

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def load_concat(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["hour"]
    df["hour_of_day"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df["hour"].min(), end=df["hour"].max())
    holiday_dates = set(holidays.date)
    df["is_holiday"] = df["hour"].dt.date.isin(holiday_dates).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hourly features for TLC demand forecasting.")
    parser.add_argument("--tlc", nargs="+", required=True, help="TLC hourly parquet files.")
    parser.add_argument("--weather", nargs="+", required=True, help="Weather hourly parquet files.")
    parser.add_argument("--out", default="data/processed/features_hourly.parquet", help="Output file.")
    parser.add_argument(
        "--ffill-weather",
        action="store_true",
        help="Forward-fill missing weather hours after merging.",
    )
    args = parser.parse_args()

    tlc = load_concat(args.tlc)
    weather = load_concat(args.weather)

    tlc["hour"] = pd.to_datetime(tlc["hour"], errors="coerce")
    weather["hour"] = pd.to_datetime(weather["hour"], errors="coerce")

    if tlc["hour"].isna().any():
        tlc = tlc.dropna(subset=["hour"])
    if weather["hour"].isna().any():
        weather = weather.dropna(subset=["hour"])

    df = tlc.merge(weather, on="hour", how="left")
    if args.ffill_weather:
        weather_cols = [c for c in weather.columns if c != "hour"]
        df = df.sort_values(["PULocationID", "hour"])
        df[weather_cols] = df.groupby("PULocationID")[weather_cols].ffill()

    df = add_time_features(df)
    df = add_holiday_flag(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print("saved:", out_path, "rows:", len(df))


if __name__ == "__main__":
    main()
