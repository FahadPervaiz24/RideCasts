import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import lightgbm as lgb


MODEL_DEFAULT = "models/LGBM/lightgbm_week_hour_20260210_132138.txt"
FEATURES_DEFAULT = "data/processed/features_hourly.parquet"
TIMEZONE_DEFAULT = "America/New_York"
HORIZON_DEFAULT = 48

FEATURE_COLS = [
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

CAT_COLS = ["PULocationID", "week_hour", "month", "week_of_year"]


def next_top_of_hour(local_now: datetime) -> datetime:
    return local_now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


def fetch_open_meteo_hourly(
    latitude: float,
    longitude: float,
    timezone_name: str,
    start_hour: datetime,
    horizon_hours: int,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "forecast_days": 3,
        "timezone": timezone_name,
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    with urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    hourly = payload.get("hourly", {})
    if not hourly:
        raise ValueError("Open-Meteo response missing 'hourly'.")

    if "time" not in hourly:
        raise ValueError("Open-Meteo response missing hourly.time.")

    df = pd.DataFrame(
        {
            "hour": pd.to_datetime(hourly["time"]),
            "temperature": hourly["temperature_2m"],
            "relative_humidity": hourly["relative_humidity_2m"],
            "precipitation": hourly["precipitation"],
            "wind_speed": hourly["wind_speed_10m"],
        }
    )
    # Open-Meteo times are local to the requested timezone when timezone is provided.
    df["hour"] = pd.to_datetime(df["hour"]).dt.tz_localize(ZoneInfo(timezone_name))
    df = df.sort_values("hour")
    df = df[df["hour"] >= start_hour].head(horizon_hours).reset_index(drop=True)
    if len(df) < horizon_hours:
        raise ValueError(
            f"Open-Meteo returned only {len(df)} hourly rows from {start_hour}, expected {horizon_hours}."
        )
    return df


def make_dummy_weather(start_hour: datetime, horizon_hours: int) -> pd.DataFrame:
    hours = pd.date_range(start=start_hour, periods=horizon_hours, freq="h")
    t = np.arange(horizon_hours)
    temp = 8 + 6 * np.sin(2 * np.pi * (t - 5) / 24)
    rh = 65 + 20 * np.cos(2 * np.pi * (t - 2) / 24)
    wind = 14 + 3 * np.sin(2 * np.pi * (t + 3) / 24)
    prcp = np.where((t % 24 >= 14) & (t % 24 <= 17), 0.2, 0.0)
    return pd.DataFrame(
        {
            "hour": hours,
            "temperature": temp,
            "relative_humidity": np.clip(rh, 20, 100),
            "precipitation": prcp,
            "wind_speed": np.clip(wind, 0, None),
        }
    )


def build_baseline_lookup(features_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    df = features_df.copy()
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"])
    cutoff = df["hour"].max() - pd.Timedelta(days=28)
    train = df[df["hour"] < cutoff].copy()

    baseline = (
        train.groupby(["PULocationID", "week_hour"], as_index=False)["trip_count"]
        .mean()
        .rename(columns={"trip_count": "baseline_week_hour_mean"})
    )
    global_mean = float(train["trip_count"].mean())
    return baseline, global_mean


def build_inference_frame(
    zone_ids: np.ndarray,
    weather_df: pd.DataFrame,
    baseline_lookup: pd.DataFrame,
    baseline_global_mean: float,
) -> pd.DataFrame:
    hours = weather_df[["hour"]].copy()
    hours["hour"] = pd.to_datetime(hours["hour"])
    hours["hour_of_day"] = hours["hour"].dt.hour
    hours["day_of_week"] = hours["hour"].dt.dayofweek
    hours["month"] = hours["hour"].dt.month.astype(int)
    hours["day_of_year"] = hours["hour"].dt.dayofyear.astype(int)
    hours["week_of_year"] = hours["hour"].dt.isocalendar().week.astype(int)
    hours["week_hour"] = hours["day_of_week"] * 24 + hours["hour_of_day"]
    hours["is_weekend"] = (hours["day_of_week"] >= 5).astype(int)
    hours["is_holiday"] = 0

    zones = pd.DataFrame({"PULocationID": zone_ids})
    zones["_k"] = 1
    hours["_k"] = 1
    df = zones.merge(hours, on="_k", how="inner").drop(columns=["_k"])
    df = df.merge(weather_df, on="hour", how="left")
    df["is_rain"] = (df["precipitation"] > 0).astype(int)
    df = df.merge(baseline_lookup, on=["PULocationID", "week_hour"], how="left")
    df["baseline_week_hour_mean"] = df["baseline_week_hour_mean"].fillna(baseline_global_mean)

    for col in CAT_COLS:
        df[col] = df[col].astype("category")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 48-hour zone forecasts.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--model-path", default=MODEL_DEFAULT, help="LightGBM model path.")
    parser.add_argument("--features-path", default=FEATURES_DEFAULT, help="features_hourly parquet.")
    parser.add_argument("--horizon-hours", type=int, default=HORIZON_DEFAULT, help="Forecast horizon.")
    parser.add_argument("--timezone", default=TIMEZONE_DEFAULT, help="Timezone, e.g. America/New_York.")
    parser.add_argument("--latitude", type=float, default=40.7128, help="Open-Meteo latitude.")
    parser.add_argument("--longitude", type=float, default=-74.0060, help="Open-Meteo longitude.")
    parser.add_argument(
        "--dummy-weather",
        action="store_true",
        help="Use synthetic weather and skip Open-Meteo call.",
    )
    args = parser.parse_args()

    model = lgb.Booster(model_file=args.model_path)
    features_df = pd.read_parquet(args.features_path, columns=["hour", "PULocationID", "trip_count"])

    zone_ids = np.sort(features_df["PULocationID"].unique())
    if len(zone_ids) == 0:
        raise ValueError("No zones found in features file.")

    tz = ZoneInfo(args.timezone)
    start_hour = next_top_of_hour(datetime.now(tz))

    baseline_source = features_df.copy()
    baseline_source["hour"] = pd.to_datetime(baseline_source["hour"])
    baseline_source["hour_of_day"] = baseline_source["hour"].dt.hour
    baseline_source["day_of_week"] = baseline_source["hour"].dt.dayofweek
    baseline_source["week_hour"] = baseline_source["day_of_week"] * 24 + baseline_source["hour_of_day"]
    baseline_lookup, baseline_global_mean = build_baseline_lookup(baseline_source)

    if args.dummy_weather:
        weather_df = make_dummy_weather(start_hour, args.horizon_hours)
        weather_source = "dummy"
    else:
        try:
            weather_df = fetch_open_meteo_hourly(
                latitude=args.latitude,
                longitude=args.longitude,
                timezone_name=args.timezone,
                start_hour=start_hour,
                horizon_hours=args.horizon_hours,
            )
            weather_source = "open-meteo"
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"Failed to fetch Open-Meteo forecast: {exc}") from exc

    inf_df = build_inference_frame(zone_ids, weather_df, baseline_lookup, baseline_global_mean)
    X = inf_df[FEATURE_COLS]
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    inf_df["prediction"] = np.clip(np.rint(y_pred), 0, None).astype(int)

    predictions = [
        {
            "hour": ts.isoformat(),
            "PULocationID": int(zone),
            "prediction": int(pred),
        }
        for ts, zone, pred in zip(inf_df["hour"], inf_df["PULocationID"], inf_df["prediction"])
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timezone": args.timezone,
        "horizon_hours": args.horizon_hours,
        "zone_count": int(len(zone_ids)),
        "prediction_count": int(len(predictions)),
        "model_path": args.model_path,
        "weather_source": weather_source,
        "predictions": predictions,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print("saved:", out_path)
    print("zones:", len(zone_ids))
    print("rows:", len(predictions))
    print("hours:", weather_df["hour"].min(), "to", weather_df["hour"].max())


if __name__ == "__main__":
    main()
