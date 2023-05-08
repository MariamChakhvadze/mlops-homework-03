"""Script to train the model."""
import os
import sys

sys.path.insert(1, f"{os.path.dirname(__file__)}/../")
sys.path.insert(1, f"{os.path.dirname(__file__)}/../../")

import hydra
import omegaconf
import mlflow
import pandas as pd
from sklearn import feature_extraction, metrics, pipeline

from features import utils


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3.2")
def train(config: omegaconf.DictConfig):
    """"""
    model = hydra.utils.instantiate(config.model)

    X_train = pd.read_csv(config.train_features)
    y_train = pd.read_csv(config.train_target)

    X_valid = pd.read_csv(config.valid_features)
    y_valid = pd.read_csv(config.valid_target)

    pipe = pipeline.make_pipeline(
        utils.TabularToDict(),
        feature_extraction.DictVectorizer(),
        model,
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_valid)

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        mlflow.log_param("train_features", config.train_features)
        mlflow.log_param("train_target", config.train_target)
        mlflow.log_param("valid_features", config.valid_features)
        mlflow.log_param("valid_target", config.valid_target)

        # for name, value in model.get_params().items():
        #     mlflow.log_param(name, value)
        mlflow.log_params(config.model)

        mlflow.log_metric(
            "RMSE", metrics.mean_squared_error(y_valid, y_pred, squared=False)
        )
        mlflow.log_metric(
            "MAPE", metrics.mean_absolute_percentage_error(y_valid, y_pred)
        )

        mlflow.sklearn.log_model(pipe, "pipeline_model")

        with open("models/predictions.txt", "w") as pred_file:
            pred_file.write("\n".join([str(pred) for pred in y_pred]))
        mlflow.log_artifact("models/predictions.txt")


if __name__ == "__main__":
    mlflow.set_experiment("trainings")
    mlflow.set_tracking_uri("http://localhost:5000")

    train()
