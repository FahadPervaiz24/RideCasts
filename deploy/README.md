# Deployment (Option A: Netlify + GitHub Actions)

## Overview
Frontend is static (Netlify). Forecasts are generated hourly by a GitHub Action and pushed
to object storage (S3/R2/GCS). The frontend fetches the latest JSON.

## What runs where
- **Netlify**: static site hosting (Deck.gl frontend).
- **GitHub Actions**: hourly forecast job.
- **Object storage**: `forecast_latest.json` (public read).

## Setup steps
1) Create `frontend/` app and deploy it to Netlify.
2) Make the forecast JSON publicly readable.
3) Point the frontend to the JSON URL.

## GitHub Actions schedule
The workflow exists at `.github/workflows/forecast_job.yml`.
It runs hourly at :15 UTC **only** if repo variable `ENABLE_SCHEDULE` is set to `true`.

## Required secrets (GitHub)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `FORECAST_BUCKET`
- `FORECAST_PREFIX` (e.g., `taxiflow/`)

## Forecast generator
`scripts/serve/generate_forecast.py` now runs real inference:
- loads `models/LGBM/lightgbm_week_hour_20260210_132138.txt`
- fetches Open‑Meteo hourly forecast
- builds features for all zones (48 hours x 263 zones)
- writes predictions to `data/forecast/forecast_latest.json`

## Frontend data contract
`forecast_latest.json` should look like:
```
{
  "generated_at": "2026-02-12T18:00:00Z",
  "horizon_hours": 48,
  "predictions": [
    {"hour": "2026-02-12T19:00:00Z", "PULocationID": 161, "prediction": 142.3}
  ]
}
```
