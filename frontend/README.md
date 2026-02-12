# Frontend

Static Deck.gl frontend lives in `frontend/public`.

## Run locally
From repo root:

```bash
python3 -m http.server 8000
```

Open:

`http://localhost:8000/frontend/public/index.html`

## Data files
- Zones: `frontend/public/data/taxi_zones.geojson`
- Forecast: `frontend/public/data/forecast_latest.json`

If you regenerate forecast, copy it over:

```bash
cp data/forecast/forecast_latest.json frontend/public/data/forecast_latest.json
```
