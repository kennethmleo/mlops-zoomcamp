from prefect import flow, task
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pickle
import mlflow

@task
def read_data(filename):
    df = pd.read_parquet(filename)
    return df

@task
def prepare_features(df):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df['duration'].dt.total_seconds() / 60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna('-1').astype(str)
    dicts = df[categorical].to_dict(orient='records')
    y = df['duration'].values
    return dicts, y

@task
def train_and_log_model(dicts, y):
    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts)
    model = LinearRegression()
    model.fit(X_train, y)

    print("Intercept:", model.intercept_)

    with open("dict_vectorizer.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-linear-regression")

    with mlflow.start_run():
        mlflow.set_tag("developer", "kenneth")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("feature_type", "PU and DO separately")

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact("dict_vectorizer.pkl")

        mlflow.log_metric("intercept", model.intercept_)

    return dv, model


@flow
def train_flow(train_path: str = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"):
    df = read_data(train_path)
    dicts, y = prepare_features(df)
    dv, model = train_and_log_model(dicts, y)


if __name__ == "__main__":
    train_flow()
