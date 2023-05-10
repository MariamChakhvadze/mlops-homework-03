"""Script to compare old and newly trained models' performances."""
import mlflow
import pandas as pd
from sklearn import metrics

import batch_inference


def compare(
    features_filepath: str,
    target_filepath: str,
    old_model_uri: str,
    new_model_uri: str,
    predictions_filename: str,
) -> None:
    """Compare metrics of old and newly trained models.

    All predictions and metrics will be logged in MLFlow.

    Args:
        features_filepath: processed features file path.
        target_filepath: target file path.
        old_model_uri: old model URI.
        new_model_uri: newly trained model URI.
        predictions_filename: name for predictions CSV file.
    """
    result_old = batch_inference.batch_predict(features_filepath, old_model_uri)
    result_new = batch_inference.batch_predict(features_filepath, new_model_uri)

    df = pd.DataFrame(
        {
            "old": result_old,
            "new": result_new,
        }
    )

    df.to_csv(f"models/{predictions_filename}.csv", index=False)
    mlflow.log_artifact(f"models/{predictions_filename}.csv")

    target = pd.read_csv(target_filepath)

    mlflow.log_metric(
        f"{old_model_uri.replace(':', '')}_RMSE",
        metrics.mean_squared_error(target, result_old, squared=False),
    )
    mlflow.log_metric(
        f"{old_model_uri.replace(':', '')}_MAPE",
        metrics.mean_absolute_percentage_error(target, result_old),
    )

    mlflow.log_metric(
        f"{new_model_uri.replace(':', '')}_RMSE",
        metrics.mean_squared_error(target, result_new, squared=False),
    )
    mlflow.log_metric(
        f"{new_model_uri.replace(':', '')}_MAPE",
        metrics.mean_absolute_percentage_error(target, result_new),
    )


if __name__ == "__main__":
    mlflow.set_experiment("offline_comparisons")
    mlflow.set_tracking_uri("http://localhost:5000")

    compare(
        "data/processed/green_tripdata_2021-06_features.csv",
        "data/processed/green_tripdata_2021-06_target.csv",
        "models:/XGBRegressor/1",
        "models:/XGBRegressor/2",
        "predictions_xgboost_2021_06",
    )

    compare(
        "data/processed/green_tripdata_2021-06_features.csv",
        "data/processed/green_tripdata_2021-06_target.csv",
        "models:/LinearRegression/1",
        "models:/LinearRegression/2",
        "predictions_linear_regression_2021_06",
    )
