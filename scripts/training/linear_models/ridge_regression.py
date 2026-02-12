import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_parquet("data/processed/features_hourly.parquet")

cutoff = df["hour"].max() - pd.Timedelta(days=28)
train = df[df["hour"] < cutoff]
val = df[df["hour"] >= cutoff]

feature_vals = [
    "hour_of_day", "day_of_week", "month", "PULocationID", "temperature",
    "wind_speed","relative_humidity", "precipitation",
    "is_rain", "is_weekend", "is_holiday"]

x_train = train[feature_vals]
y_train = train["trip_count"]

x_val = val[feature_vals]
y_val = val["trip_count"]


#split features into appropriate groups before preprocessing
categorical_cols = ["hour_of_day", "day_of_week", "month", "PULocationID"]
numerical_cols = ["temperature","wind_speed", "relative_humidity", "precipitation"]
binary_cols = ["is_rain", "is_weekend", "is_holiday"]

"""
Step before pre-processing: Need to fill in NaN values with 
median/frequent values - 6093 rows out of 4.28M are NaN for weather features
"""
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

bin_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

#preprocess features with imputation
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, numerical_cols),
        ("cat", cat_pipe, categorical_cols),
        ("bin", bin_pipe, binary_cols)
    ]
)


X_train = preprocessor.fit_transform(x_train)
X_val = preprocessor.transform(x_val)


model = Ridge(alpha=0.0, solver="sag", random_state=0)
model.fit(X_train,y_train)
y_pred = model.predict(X_val)

mae = np.mean(np.abs(y_val-y_pred))
smape = np.mean(2 * np.abs(y_pred - y_val) / (np.abs(y_pred) + np.abs(y_val) + 1e-8))
print("MAE:", mae)
print("sMAPE:", smape)

print("mean:", y_val.mean())
print("MAE % of mean:", 100 * mae / y_val.mean())

