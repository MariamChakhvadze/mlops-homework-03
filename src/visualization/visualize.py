"""Script to visualize target and log plot in the MLFlow."""
import matplotlib.pyplot as plt
import mlflow
import omegaconf
import pandas as pd


def visualize_target(filepath: str, fig_name: str) -> None:
    """Plot target and log figure in the MLFlow.

    Args:
        filepath: path of target file.
        fig_name: logged figure name.
    """
    target = pd.read_csv(filepath).squeeze()

    fig, ax = plt.subplots()
    target.plot(ax=ax, subplots=True, kind="hist", bins=40)

    mlflow.log_figure(fig, f"{fig_name}.png")


if __name__ == "__main__":
    mlflow.set_experiment("visualizations")
    mlflow.set_tracking_uri("http://localhost:5000")

    config = omegaconf.OmegaConf.load("configs/visualize.yaml")

    for index, data in enumerate(config.data):
        mlflow.log_param(f"filepath_{index}", data["filepath"])
        mlflow.log_param(f"fig_name_{index}", data["fig_name"])

        visualize_target(filepath=data["filepath"], fig_name=data["fig_name"])
