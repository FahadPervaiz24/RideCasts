# Taxiflow
Taxiflow

Applied machine learning for forecasting NYC taxi demand

Overview

Taxiflow is an end-to-end time-series ML project that forecasts short-horizon taxi demand in New York City using historical trip data and real weather signals. The focus is on building a reproducible forecasting pipeline with proper temporal validation and clear model explanations, rather than chasing complex architectures.

This project is designed to mirror how forecasting systems are built and evaluated in production environments.

⸻

Problem

Taxi demand in NYC exhibits strong:
	•	daily and weekly seasonality
	•	long-term trends
	•	sensitivity to weather conditions (rain, temperature)

Accurately modeling these patterns is essential for understanding demand dynamics and evaluating forecasting performance under real-world conditions.

⸻

Data Sources
	•	NYC Taxi & Limousine Commission (TLC) trip records
Aggregated into daily demand time series.
	•	NOAA NCEI Daily Summaries
Observed weather data (precipitation, max/min temperature) queried using an NYC bounding box.

No simulated or user-controlled inputs are used — all features are learned from historical observations.

⸻

Approach
	1.	Data ingestion
	•	Programmatic data pulls
	•	Explicit row counts and date range checks
	•	Saved locally as Parquet for reproducibility
	2.	Feature engineering
	•	Lag features (e.g. t-1, t-7)
	•	Rolling statistics
	•	Calendar effects
	•	Weather indicators (e.g. rain flag)
	3.	Modeling
	•	Seasonal baseline models
	•	One primary ML forecasting model
	•	Walk-forward (rolling) temporal validation
	•	Error tracked over time, not just aggregate metrics
	4.	Interpretability
	•	Feature attribution to understand what drives forecasts
	•	Explicit discussion of limitations and failure modes

⸻

Evaluation

Models are evaluated using:
	•	MAE
	•	MAPE / sMAPE
	•	Rolling backtests to reflect real forecasting conditions

Random train/test splits are intentionally avoided.

⸻

What this project is (and isn’t)

This is:
	•	An applied ML forecasting system
	•	Focused on correctness, validation, and clarity
	•	Designed to be understandable and inspectable

This is not:
	•	A traffic simulation
	•	A deep learning benchmark
	•	A causal inference system
	•	A real-time dispatch tool

⸻

Motivation

The goal of Taxiflow is to demonstrate practical ML engineering skills:
	•	working with messy real-world data
	•	designing clean pipelines
	•	validating time-series models correctly
	•	communicating results clearly
