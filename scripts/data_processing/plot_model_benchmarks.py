import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/reports/model_benchmarks.csv"
OUT_PATH = "data/reports/model_benchmarks.png"

BG = "#111111"
GRID = "#5a5a5a"
TEXT = "#e5e5e5"
DOT1 = "#d9d9d9"
DOT2 = "#9fb2ff"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Model", "MAE", "sMAPE"])

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor(BG)
    ax2.set_facecolor(BG)

    df_mae = df.sort_values("MAE", ascending=True)
    ax1.hlines(df_mae["Model"], 0, df_mae["MAE"], color=GRID, alpha=0.45, linewidth=2)
    ax1.scatter(df_mae["MAE"], df_mae["Model"], s=70, color=DOT1, zorder=3)
    ax1.set_title("Mean Absolute Error (MAE)")
    ax1.set_xlabel("MAE")
    ax1.grid(color=GRID, alpha=0.25, axis="x")

    df_smape = df.sort_values("sMAPE", ascending=True)
    ax2.hlines(df_smape["Model"], 0, df_smape["sMAPE"], color=GRID, alpha=0.45, linewidth=2)
    ax2.scatter(df_smape["sMAPE"], df_smape["Model"], s=70, color=DOT2, zorder=3)
    ax2.set_title("Mean Absolute Percentage Error (sMAPE)")
    ax2.set_xlabel("sMAPE")
    ax2.grid(color=GRID, alpha=0.25, axis="x")

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", colors=TEXT)
        ax.tick_params(axis="y", colors=TEXT)
        for spine in ax.spines.values():
            spine.set_color("#3a3a3a")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, facecolor=fig.get_facecolor())
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
