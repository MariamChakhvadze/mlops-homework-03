"""Fake client script to query ML endpoint (first versions of models)."""

import requests


if __name__ == "__main__":
    data = {
        "VendorID": 2,
        "lpep_pickup_datetime": 1609460156000,
        "lpep_dropoff_datetime": 1609460392000,
        "store_and_fwd_flag": "N",
        "RatecodeID": 1.0,
        "PULocationID": 43,
        "DOLocationID": 151,
        "passenger_count": 1.0,
        "trip_distance": 1.01,
        "fare_amount": 5.5,
        "extra": 0.5,
        "mta_tax": 0.5,
        "tip_amount": 0.0,
        "tolls_amount": 0.0,
        "ehail_fee": None,
        "improvement_surcharge": 0.3,
        "total_amount": 6.8,
        "payment_type": 2.0,
        "trip_type": 1.0,
        "congestion_surcharge": 0.0,
    }

    # first model - XGBRegressor
    data["model_id"] = 1
    response = requests.post("http://localhost:9696/predict", json=data)
    print(response.json())

    # second model - LinearRegression
    data["model_id"] = 2
    response = requests.post("http://localhost:9696/predict", json=data)
    print(response.json())
