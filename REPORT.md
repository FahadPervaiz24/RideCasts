# RideCasts: Notes to Self

## Issues hit
- DST gap hour shows up (expected), left as‑is.
- Ridge underfits badly (MAE ~55, sMAPE ~0.64).
- Weekly naive baseline (t‑168) is very strong but not deployable due to limited recent/live TLC data.

## Progress
- Built hourly pipeline (2023–2024) for TLC + weather.
- Aggregated TLC to hour × zone and built feature table.
- Baselines are running (t‑168 + week_hour).
- LightGBM w/ baseline feature beats deployable + benchmark baselines.
- LGBM tuning: added train-only baseline feature (zone × week_hour mean), tried higher min_data_in_leaf + subsample/colsample; baseline feature helped, extra regularization didn’t.
- t‑168: MAE ~22.49, sMAPE ~0.262 (benchmark only).
- week_hour mean: MAE ~28.61, sMAPE ~0.247 (deployable baseline).
- LightGBM (+ baseline feature): MAE ~20.05, sMAPE ~0.201 (best so far).
- XGBoost: MAE ~20.71, sMAPE ~0.205 (not better than LGBM).
- Deployed map UI with 48-hour playback, zone extrusions, and legend tiers.
- Added method page with model benchmarks + SHAP + trips vs weather plots.

## What’s left
- Add forecast freshness indicator to frontend.
- Wire GitHub Action to refresh forecast hourly.
- Finalize clean repo + README for recruiters.
