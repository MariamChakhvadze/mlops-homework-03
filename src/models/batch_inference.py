"""Script for batch inference."""
import mlflow
from numpy.typing import NDArray
import pandas as pd


def batch_predict(data_filepath: str, model_uri: str) -> NDArray[float]:
    """Make a batch prediction.

    Args:
        data_filepath: file path of processed data (CSV file).
        model_uri: model URI.

    Returns:
        Predictions as NumPy array.
    """
    data = pd.read_csv(data_filepath)

    model = mlflow.pyfunc.load_model(model_uri)

    return model.predict(data)
