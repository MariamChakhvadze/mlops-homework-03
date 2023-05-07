from typing import Any

import hydra
from omegaconf import DictConfig
import mlflow
import pandas as pd
from sklearn import base, feature_extraction, metrics, pipeline

mlflow.set_experiment("my_experiment")
mlflow.set_tracking_uri("http://localhost:5000")

def read_data_into_feats_target(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration < 80)]

    pickup = df.lpep_pickup_datetime.dt
    df["month"] = pickup.month
    df["day_of_month"] = pickup.day
    df["day_of_week"] = pickup.dayofweek
    df["hour"] = pickup.hour

    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)

    features = ["PU_DO", "month", "day_of_month", "day_of_week", "hour"]

    return df[features].copy(), df["duration"].copy()


class TabularToDict(base.BaseEstimator, base.TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> "TabularToDict":
        return self

    def transform(self, X: pd.DataFrame) -> dict[str, Any] | list[dict[str, Any]]:
        return X.to_dict(orient="records")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3.2")
def train(config: DictConfig):
    model = hydra.utils.instantiate(config.model)

    X_train, y_train = read_data_into_feats_target(config.train_data)
    X_valid, y_valid = read_data_into_feats_target(config.valid_data)

    pipe = pipeline.make_pipeline(
        TabularToDict(),
        feature_extraction.DictVectorizer(),
        model,
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_valid)

    print(f"RMSE: {metrics.mean_squared_error(y_valid, y_pred, squared=False)}")
    print(f"MAPE: {metrics.mean_absolute_percentage_error(y_valid, y_pred)}")

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        mlflow.log_param("train_data", config.train_data)
        mlflow.log_param("valid_data", config.valid_data)

        # for name, value in model.get_params().items():
        #     mlflow.log_param(name, value)
        mlflow.log_params(config.model)

        mlflow.log_metric("RMSE", metrics.mean_squared_error(y_valid, y_pred, squared=False))
        mlflow.log_metric("MAPE", metrics.mean_absolute_percentage_error(y_valid, y_pred))

        mlflow.sklearn.log_model(pipe, "pipeline_model")

        with open("predictions.txt", "w") as pred_file:
            pred_file.write("\n".join([str(pred) for pred in y_pred]))
        mlflow.log_artifact("predictions.txt")


if __name__ == '__main__':
    train()