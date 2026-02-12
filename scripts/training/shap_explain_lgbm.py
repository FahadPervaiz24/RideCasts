import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt


# ----------------------------
# CONFIG
# ----------------------------
OUTPUT_DIR = Path("assets/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEPENDENCE_FEATURE = "precipitation"
MAX_ROWS_FOR_SHAP = 5000
RANDOM_SEED = 42

FEATURES_PATH = "data/processed/features_hourly.parquet"
MODEL_PATH = "models/LGBM/lightgbm_week_hour_20260210_132138.txt"


def _sample_df(X: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(X) <= max_rows:
        return X
    return X.sample(n=max_rows, random_state=RANDOM_SEED)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week_hour"] = df["day_of_week"] * 24 + df["hour_of_day"]
    df["month"] = df["month"].astype(int)
    df["day_of_year"] = df["day_of_year"].astype(int)
    df["week_of_year"] = df["week_of_year"].astype(int)
    return df


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"])
    df = build_features(df)

    cutoff = df["hour"].max() - pd.Timedelta(days=28)
    train = df[df["hour"] < cutoff].copy()
    val = df[df["hour"] >= cutoff].copy()

    baseline = (
        train.groupby(["PULocationID", "week_hour"], as_index=False)["trip_count"]
        .mean()
        .rename(columns={"trip_count": "baseline_week_hour_mean"})
    )
    train = train.merge(baseline, on=["PULocationID", "week_hour"], how="left")
    val = val.merge(baseline, on=["PULocationID", "week_hour"], how="left")

    global_mean = train["trip_count"].mean()
    train["baseline_week_hour_mean"] = train["baseline_week_hour_mean"].fillna(global_mean)
    val["baseline_week_hour_mean"] = val["baseline_week_hour_mean"].fillna(global_mean)

    feature_cols = [
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

    X_train = train[feature_cols].copy()
    X_test = val[feature_cols].copy()

    # Keep categorical columns consistent with training
    cat_cols = ["PULocationID", "week_hour", "month", "week_of_year"]
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    X_bg = _sample_df(X_train, MAX_ROWS_FOR_SHAP)
    X_eval = _sample_df(X_test, MAX_ROWS_FOR_SHAP)

    model = lgb.Booster(model_file=MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)

    # 1) Global importance (beeswarm)
    fig = plt.figure(facecolor="#111111")
    shap.summary_plot(shap_values, X_eval, show=False, max_display=25)
    ax = plt.gca()
    ax.set_facecolor("#111111")
    ax.tick_params(colors="#e5e5e5")
    for spine in ax.spines.values():
        spine.set_color("#3a3a3a")
    plt.tight_layout()
    out1 = OUTPUT_DIR / "shap_summary_beeswarm.png"
    plt.savefig(out1, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 2) Global importance (bar)
    plt.figure()
    shap.summary_plot(shap_values, X_eval, plot_type="bar", show=False, max_display=25)
    plt.tight_layout()
    out2 = OUTPUT_DIR / "shap_summary_bar.png"
    plt.savefig(out2, dpi=200)
    plt.close()

    # 3) Dependence plot for one feature
    if DEPENDENCE_FEATURE in X_eval.columns:
        plt.figure()
        shap.dependence_plot(
            DEPENDENCE_FEATURE,
            shap_values,
            X_eval,
            show=False,
            interaction_index="auto",
        )
        plt.tight_layout()
        out3 = OUTPUT_DIR / f"shap_dependence_{DEPENDENCE_FEATURE}.png"
        plt.savefig(out3, dpi=200)
        plt.close()
    else:
        print(f"Dependence feature '{DEPENDENCE_FEATURE}' not in columns.")

    # 4) Single prediction explanation (waterfall)
    i = 0
    base_value = explainer.expected_value
    row_shap = shap_values[i, :]
    exp = shap.Explanation(
        values=row_shap,
        base_values=base_value,
        data=X_eval.iloc[i, :].values,
        feature_names=X_eval.columns.tolist(),
    )
    plt.figure()
    shap.plots.waterfall(exp, show=False, max_display=15)
    out4 = OUTPUT_DIR / "shap_waterfall_row0.png"
    plt.savefig(out4, dpi=200, bbox_inches="tight")
    plt.close()

    contrib = (
        pd.Series(row_shap, index=X_eval.columns)
        .sort_values(key=np.abs, ascending=False)
        .head(10)
    )

    print("Saved:", out1)
    print("Saved:", out2)
    print("Saved:", out3 if DEPENDENCE_FEATURE in X_eval.columns else "skipped")
    print("Saved:", out4)
    print("\nTop contributions (abs) for row 0:")
    print(contrib)


if __name__ == "__main__":
    main()
