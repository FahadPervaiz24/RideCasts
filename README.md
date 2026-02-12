# RideCasts

Applied machine learning for forecasting NYC taxi demand.

## Overview
RideCasts is an end-to-end time-series ML project that forecasts short-horizon hourly
demand in New York City using historical trip data and real weather signals. The goal
is a reproducible pipeline with temporal validation and clear interpretability.

## Data Sources
- NYC TLC trip records (Yellow + High Volume FHV), aggregated to hourly demand by pickup zone.
- NOAA GHCNh hourly weather from NYC-area stations (Central Park, JFK, LaGuardia, Newark).

## Modeling
- Baselines: weekly lag (t-168) and hour-of-week.
- Primary model: LightGBM with calendar + weather + week-hour baseline feature.
- Validation: rolling holdout (last 28 days), MAE and sMAPE.

## Frontend
Interactive map with zone extrusions and 48-hour playback, plus a method page that
includes model benchmarks, SHAP, and trips vs weather plots.

## Repo Notes
- Only the production LightGBM model artifact is kept under `models/`.
- Raw/processed data and large artifacts are ignored by default.
