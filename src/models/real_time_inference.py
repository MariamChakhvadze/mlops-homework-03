"""Script to serve the model in real-time."""
from typing import Any

import mlflow
import pandas as pd
from flask import Flask, request, jsonify, Response


mlflow.set_tracking_uri("http://localhost:5000")

model_uri_1 = "models:/XGBRegressor/1"
model_uri_2 = "models:/LinearRegression/1"
loaded_model_1 = mlflow.pyfunc.load_model(model_uri_1)
loaded_model_2 = mlflow.pyfunc.load_model(model_uri_2)


def prepare_features(ride: dict[str, Any]) -> pd.DataFrame:
    """Prepare features for prediction.

    Args:
        ride: features from request.

    Returns:
        Prepared features as pandas DataFrame.
    """
    df = pd.DataFrame([ride])
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    pickup = df.lpep_pickup_datetime.dt
    df["month"] = pickup.month
    df["day_of_month"] = pickup.day
    df["day_of_week"] = pickup.dayofweek
    df["hour"] = pickup.hour

    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)

    return df[["PU_DO", "month", "day_of_month", "day_of_week", "hour"]]


def predict(
    features: pd.DataFrame, model_id: int
) -> tuple[float, str] | tuple[Response, int]:
    """Make prediction.

    Args:
        features: features as pandas DataFrame.
        model_id: model ID which should be used for prediction.

    Returns:
        Prediction and model version (run ID) if `model_id` is valid. Otherwise,
        the response is returned with a 400 status code.
    """
    if model_id == 1:
        loaded_model = loaded_model_1
    elif model_id == 2:
        loaded_model = loaded_model_2
    else:
        return jsonify({"error": "Invalid model ID"}), 400

    return loaded_model.predict(features).item(), loaded_model.metadata.run_id


app = Flask("duration-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint() -> Response:
    """Endpoint for predicting ride duration.

    Returns:
        Predicted duration and model version as a JSON.
    """
    ride = request.get_json()
    features = prepare_features(ride)
    pred, version = predict(features, ride["model_id"])

    result = {"duration": pred, "model_version": version}  # actually, run ID

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
