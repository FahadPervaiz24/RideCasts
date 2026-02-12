import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

df = pd.read_parquet("data/processed/features_hourly.parquet")

# Citywide hourly total
city = df.groupby("hour")["trip_count"].sum().reset_index()

# Daily total
daily = city.set_index("hour").resample("D").sum()
daily = daily[(daily.index >= "2023-01-01") & (daily.index < "2024-01-01")]

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#111111")
ax.set_facecolor("#111111")
ax.plot(daily.index, daily["trip_count"], color="white")
ax.grid(color="gray", alpha=0.3)
ax.set_title("Citywide Daily TLC Demand")
ax.set_xlabel("Day")
ax.set_ylabel("Trips")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M"))
plt.tight_layout()
plt.savefig("data/reports/citywide_timeserie1.png")

# Two-panel: trips (top) and weather (bottom)
weather = df.groupby("hour")[["temperature", "precipitation"]].mean().reset_index()
daily_weather = weather.set_index("hour").resample("D").mean()
daily_weather = daily_weather[(daily_weather.index >= "2023-01-01") & (daily_weather.index < "2024-01-01")]

plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)
fig.patch.set_facecolor("#111111")
ax1.set_facecolor("#111111")
ax2.set_facecolor("#111111")

# Top: trips (with rolling smoothers)
roll7 = daily["trip_count"].rolling(7).mean()
roll28 = daily["trip_count"].rolling(28).mean()
ax1.plot(daily.index, daily["trip_count"], color="white", alpha=0.25, label="Daily")
ax1.plot(daily.index, roll7, color="#00FF66", linewidth=1.5, label="7-day avg")
ax1.plot(daily.index, roll28, color="#7A00FF", linewidth=2.0, label="28-day avg")
# Fixed y-range for consistent scale
ax1.set_ylim(400_000, 1_300_000)
ax1.set_yticks(np.arange(400_000, 1_300_001, 150_000))
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1_000_000:.2f}M"))
ax1.grid(color="gray", alpha=0.3)
ax1.set_title("Daily Trips + Daily Weather (Citywide)")
ax1.set_ylabel("Trips")
ax1.legend()

# Bottom: weather (dual axis)
ax2.plot(daily_weather.index, daily_weather["temperature"], color="#00FF66", label="Temp")
ax2.set_ylabel("Temp (°C)")

ax2b = ax2.twinx()
ax2b.plot(daily_weather.index, daily_weather["precipitation"], color="#7A00FF", label="Precip")
ax2b.set_ylabel("Precip (mm)")

ax2.grid(color="gray", alpha=0.3)
ax2.set_xlabel("Day")

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2b.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.tight_layout()
plt.savefig("data/reports/trips_weather_two_panel.png")
