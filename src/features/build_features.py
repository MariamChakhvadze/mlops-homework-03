"""Script to transform data into features and target."""
import omegaconf
import pandas as pd


def transform_data_into_feats_target_csv(
    filepath: str | list[str], prefix: str, save_dir: str
) -> None:
    """Transform NYC Taxi dataset into features and target and save them as CSV files.

    Args:
        filepath: path of dataset or list of paths.
        prefix: prefix for created files.
        save_dir: directory where transsformed data should be saved.
    """
    if isinstance(filepath, list):
        dfs = []

        for data_file in filepath:
            df = pd.read_parquet(data_file)

        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_parquet(filepath)

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
    df[features].to_csv(f"{save_dir}/{prefix}_features.csv", index=False)

    df["duration"].to_csv(f"{save_dir}/{prefix}_target.csv", index=False)


if __name__ == "__main__":
    config = omegaconf.OmegaConf.load("configs/build_features.yaml")

    for data in config.data:
        transform_data_into_feats_target_csv(
            filepath=data["filepath"], prefix=data["prefix"], save_dir=config.save_dir
        )
