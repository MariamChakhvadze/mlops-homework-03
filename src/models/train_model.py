"""Script to train the model."""
from typing import Any

import hydra
import omegaconf
import mlflow
import pandas as pd
from sklearn import base, feature_extraction, metrics, pipeline


class TabularToDict(base.BaseEstimator, base.TransformerMixin):
    """Class to transform tabular data into dictionary or list of dictionaries."""

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None
    ) -> "TabularToDict":
        """Return self instance.

        Args:
            X: features.
            y: target.

        Returns:
            Self instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> dict[str, Any] | list[dict[str, Any]]:
        """Transform features into dictionary or list of dictionaries.

        Args:
            X: features.

        Returns:
            Transformed features.
        """
        return X.to_dict(orient="records")


def read_data(filepath: str | omegaconf.ListConfig[str]) -> pd.DataFrame:
    """Read data into pandas DataFrame.

    If filepath is a list of paths, data will be concatenated.

    Args:
        filepath: single file path or list of them.

    Returns:
        Data as pandas DataFrame.
    """
    if isinstance(filepath, omegaconf.ListConfig):
        dfs = []

        for data_file in filepath:
            dfs.append(pd.read_csv(data_file))

        df = pd.concat(dfs)
    else:
        df = pd.read_csv(filepath)

    return df


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3.2")
def train(config: omegaconf.DictConfig) -> None:
    """Train model and log all information in MLFlow.

    Args:
        config: configuration for training.
    """
    model = hydra.utils.instantiate(config.model)

    X_train = read_data(config.train_features)
    y_train = read_data(config.train_target).squeeze()

    X_valid = read_data(config.valid_features)
    y_valid = read_data(config.valid_target).squeeze()

    pipe = pipeline.make_pipeline(
        TabularToDict(),
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

        if config.log_all_model_params:
            for name, value in model.get_params().items():
                mlflow.log_param(name, value)
        else:
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
