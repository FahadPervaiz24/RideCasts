import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def expand_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for val in values:
        p = Path(val)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.parquet")))
        elif "*" in val:
            paths.extend(sorted(Path().glob(val)))
        else:
            paths.append(p)
    return paths


def aggregate_counts(
    paths: list[Path],
    pickup_col: str,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> dict[tuple[pd.Timestamp, int], int]:
    counts: dict[tuple[pd.Timestamp, int], int] = defaultdict(int)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        print("reading:", path)
        df = pd.read_parquet(path, columns=[pickup_col, "PULocationID"])
        df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
        df["PULocationID"] = pd.to_numeric(df["PULocationID"], errors="coerce")
        df = df.dropna(subset=[pickup_col, "PULocationID"])
        if start_ts is not None:
            df = df[df[pickup_col] >= start_ts]
        if end_ts is not None:
            df = df[df[pickup_col] < end_ts]
        df["hour"] = df[pickup_col].dt.floor("h")
        grouped = df.groupby(["hour", "PULocationID"]).size()
        for (hour, puloc), cnt in grouped.items():
            counts[(hour, int(puloc))] += int(cnt)
    return counts


def counts_to_frame(counts: dict[tuple[pd.Timestamp, int], int]) -> pd.DataFrame:
    rows = [(hour, puloc, cnt) for (hour, puloc), cnt in counts.items()]
    df = pd.DataFrame(rows, columns=["hour", "PULocationID", "trip_count"])
    df = df.sort_values(["hour", "PULocationID"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate TLC trips to hourly counts by pickup zone.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Parquet file(s), directories, or globs to aggregate.",
    )
    parser.add_argument(
        "--pickup-col",
        default="tpep_pickup_datetime",
        help="Pickup datetime column name.",
    )
    parser.add_argument(
        "--out",
        default="data/processed/tlc_hourly_zone.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--start",
        default="",
        help="Filter pickup datetimes >= this (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default="",
        help="Filter pickup datetimes < this (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="If output exists, append and re-aggregate to sum counts.",
    )
    args = parser.parse_args()

    paths = expand_paths(args.inputs)
    if not paths:
        raise ValueError("No parquet files found.")

    start_ts = pd.to_datetime(args.start) if args.start else None
    end_ts = pd.to_datetime(args.end) if args.end else None

    counts = aggregate_counts(paths, args.pickup_col, start_ts, end_ts)
    df = counts_to_frame(counts)
    if df.empty:
        raise ValueError("Aggregation result is empty.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.append and out_path.exists():
        existing = pd.read_parquet(out_path)
        df = pd.concat([existing, df], ignore_index=True)
        df = (
            df.groupby(["hour", "PULocationID"], as_index=False)["trip_count"]
            .sum()
            .sort_values(["hour", "PULocationID"])
            .reset_index(drop=True)
        )

    print("rows:", len(df))
    print("hour_range:", df["hour"].min(), "to", df["hour"].max())
    print("zones:", df["PULocationID"].nunique())

    df.to_parquet(out_path, index=False)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
