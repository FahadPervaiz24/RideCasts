import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize HVFHV pickup column to match yellow schema."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input HVFHV parquet file(s), directories, or globs.",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Output directory (default: same folder as input).",
    )
    args = parser.parse_args()

    paths = expand_paths(args.inputs)
    if not paths:
        raise ValueError("No parquet files found.")

    out_dir = Path(args.outdir) if args.outdir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for in_path in paths:
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        out_path = (
            out_dir / (in_path.stem + "_normalized.parquet")
            if out_dir
            else in_path.with_name(in_path.stem + "_normalized.parquet")
        )

        df = pd.read_parquet(in_path)
        if "pickup_datetime" not in df.columns:
            raise ValueError(f"Expected column 'pickup_datetime' not found in {in_path}")

        df = df.rename(columns={"pickup_datetime": "tpep_pickup_datetime"})
        df.to_parquet(out_path, index=False)
        print("saved:", out_path)


if __name__ == "__main__":
    main()
