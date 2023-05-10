"""Script for batch inference."""
import mlflow
from numpy.typing import NDArray
import pandas as pd


mlflow.set_tracking_uri("http://localhost:5000")


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for prediction.

    Args:
        data: data (features).

    Returns:
        Prepared data.
    """
    df = data.copy()
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    pickup = df.lpep_pickup_datetime.dt
    df["month"] = pickup.month
    df["day_of_month"] = pickup.day
    df["day_of_week"] = pickup.dayofweek
    df["hour"] = pickup.hour

    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)

    return df[["PU_DO", "month", "day_of_month", "day_of_week", "hour"]]


def batch_predict(data_filepath: str, model_uri: str) -> NDArray[float]:
    """Make a batch prefiction.

    Args:
        data_filepath: file path of data.
        model_uri: model URI.

    Returns:
        Predictions as NumPy array.
    """
    data = pd.read_parquet(data_filepath)
    features = prepare_features(data)

    model = mlflow.pyfunc.load_model(model_uri)

    return model.predict(features)
