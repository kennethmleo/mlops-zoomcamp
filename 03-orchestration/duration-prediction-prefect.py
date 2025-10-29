from prefect import flow, task
from datetime import datetime, timedelta
import pandas as pd
import pickle
import xgboost as xgb
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path("models")
models_folder.mkdir(exist_ok=True, parents=True)


@task
def read_dataframe(year, month):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url)
    df["duration"] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    return df


@task
def create_X(df, dv=None):
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


@task
def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }
        mlflow.log_params(params)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=30,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=50,
        )

        y_pred = booster.predict(dvalid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open(models_folder / "preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(models_folder / "preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


@flow
def nyc_taxi_flow(year: int, month: int):
    df_train = read_dataframe(year, month)
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(next_year, next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    y_train, y_val = df_train["duration"].values, df_val["duration"].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"✅ MLflow run_id: {run_id}")
    return run_id

@flow
def scheduled_taxi_flow(date: str = None):
    """
    Wrapper flow that figures out which months to train/validate.
    - Training data = 2 months ago
    - Validation data = 1 month ago
    """
    if not date:
        date = datetime.today()
    train_date = date - timedelta(days=60)
    val_date = date - timedelta(days=30)
    nyc_taxi_flow(train_date.year, train_date.month)


if __name__ == "__main__":
    scheduled_taxi_flow.serve(
        name="monthly-taxi",
        cron="0 0 1 * *",  # run every 1st day of the month
    )
    print("✅ Deployment 'monthly-taxi' registered successfully!")
