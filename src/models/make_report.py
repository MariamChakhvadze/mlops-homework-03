"""Script to generate report."""
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset
import pandas as pd

import batch_inference


def render_drift_report(
    save_location: str,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    sample_size: int = 5000,
    drift_threshold: float = 0.2,
) -> None:
    """Render Drift report.

    Args:
        save_location: file path where report should be saved.
        reference_data: previous month data.
        current_data: current month data.
        feature_columns: names of feature columns.
        sample_size: size of random sample.
        drift_threshold: threshold for DataDrift report.
    """
    reference_data = reference_data.copy()
    current_data = current_data.copy()

    reference_sample = reference_data.sample(n=sample_size, replace=False)
    current_sample = current_data.sample(n=sample_size, replace=False)

    report = Report(
        metrics=[
            DataDriftPreset(feature_columns, drift_share=drift_threshold),
            TargetDriftPreset(),
            RegressionPreset(feature_columns),
        ]
    )

    report.run(
        reference_data=reference_sample,
        current_data=current_sample,
    )
    report.save_html(save_location)


if __name__ == "__main__":
    reference_features = pd.read_csv(
        "data/processed/green_tripdata_2021-05_features.csv"
    )
    reference_features["target"] = pd.read_csv(
        "data/processed/green_tripdata_2021-05_target.csv"
    ).squeeze()

    current_features = pd.read_csv("data/processed/green_tripdata_2021-06_features.csv")
    current_features["target"] = pd.read_csv(
        "data/processed/green_tripdata_2021-06_target.csv"
    ).squeeze()

    features = reference_features.columns[:-1]

    # XGBRegressor version 2
    reference_pred = batch_inference.batch_predict(
        "data/processed/green_tripdata_2021-05_features.csv", "models:/XGBRegressor/2"
    )
    reference_features["prediction"] = reference_pred

    current_pred = batch_inference.batch_predict(
        "data/processed/green_tripdata_2021-06_features.csv", "models:/XGBRegressor/2"
    )
    current_features["prediction"] = current_pred

    render_drift_report(
        "models/xgboost_report.html",
        reference_features,
        current_features,
        features,
    )
