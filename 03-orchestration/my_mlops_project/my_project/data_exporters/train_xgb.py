# train_xgb.py
from mage_ai.data_preparation.decorators import custom
from sklearn.metrics import root_mean_squared_error
from pathlib import Path
import xgboost as xgb
import pickle
import mlflow
import os

# ---------- IMPORTANT ----------
# If MLflow runs on your HOST and Mage is in Docker:
# mlflow_tracking_uri = "http://host.docker.internal:5000"
# If both are on same docker network, use the service name:
# mlflow_tracking_uri = "http://mlflow:5000"
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')

@data_loader
def train_xgb(data, **kwargs):
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    dv = data['dv']

    # Make sure models/ exists
    Path('models').mkdir(exist_ok=True)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("nyc-taxi-experiment")

    best_params = {
        'learning_rate': 0.09585355369315604,
        'max_depth': 30,
        'min_child_weight': 1.060597050922164,
        'objective': 'reg:linear',
        'reg_alpha': 0.018060244040060163,
        'reg_lambda': 0.011658731377413597,
        'seed': 42
    }

    with mlflow.start_run() as run:
        mlflow.log_params(best_params)
        # Also log which months were used
        mlflow.log_params({
            "train_year": data['train_year'],
            "train_month": data['train_month'],
            "val_year": data['val_year'],
            "val_month": data['val_month'],
        })

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val)

        booster = xgb.train(
            params=best_params,
            dtrain=dtrain,
            num_boost_round=30,
            evals=[(dvalid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(dvalid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # Save and log preprocessor
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        # Log model
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        print(f"MLflow run_id: {run.info.run_id}, RMSE={rmse:.4f}")
        return {"run_id": run.info.run_id, "rmse": float(rmse)}
