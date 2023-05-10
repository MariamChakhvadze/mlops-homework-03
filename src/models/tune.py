"""Script to tune XGBRegressor."""
import os
import sys

sys.path.insert(1, f"{os.path.dirname(__file__)}/../")

import numpy as np
import omegaconf
import optuna
import pandas as pd
from sklearn import feature_extraction, metrics, model_selection, pipeline
import xgboost as xg

from features import utils


def tune(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 10,
    n_jobs: int = 1,
    tracking_uri: str | None = None,
) -> None:
    """Tune XGBRegressor.

    Args:
        X: features.
        y: target.
        n_trials: number of trials.
        n_jobs: number of jobs.
        tracking_uri: MLFlow tracking URI.
    """
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        """Tune model.

        Args:
            trial: optuna trial object.

        Returns:
            Mean RMSE.
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10),
            "random_state": 42,
        }

        metrics_list = []
        for train_idx, test_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[test_idx], y.iloc[test_idx]

            pipe = pipeline.make_pipeline(
                utils.TabularToDict(),
                feature_extraction.DictVectorizer(),
                xg.XGBRegressor(**params),
            )
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            metrics_list.append(
                metrics.mean_squared_error(y_val, y_pred, squared=False)
            )

        return np.round(np.mean(metrics_list), decimals=4)

    mlflow_callback = optuna.integration.MLflowCallback(
        tracking_uri=tracking_uri,
        metric_name="RMSE",
        mlflow_kwargs={"nested": True},
    )

    study = optuna.create_study(
        study_name="tuning_XGBRegressor",
    )
    study.optimize(
        func=objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[mlflow_callback],
    )


if __name__ == "__main__":
    config = omegaconf.OmegaConf.load("configs/tune.yaml")

    X_train = pd.read_csv(config.features)
    y_train = pd.read_csv(config.target)

    tune(X_train, y_train, tracking_uri="http://localhost:5000")
