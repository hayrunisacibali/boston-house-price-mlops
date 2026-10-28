import mlflow

from src.data.splitter import split_data
from src.preprocessing.features import select_features
from src.evaluation.metrics import calculate_metrics
from src.training.train import train_model
from src.tracking.mlflow_tracker import log_champion_selection


TRACKING_URI = (
    "sqlite:///mlflow.db"
)

MODEL_NAME = "BostonHousePriceModel"
MODEL_VERSION = "1"


def compare_models():
    """
    Compare Registry Version 1 with the initial production model
    using the same validation dataset.
    """

    mlflow.set_tracking_uri(TRACKING_URI)

    # Load validation data
    _, X_val, _, _, y_val, _ = split_data()
    X_val_selected = select_features(X_val)

    # Load Registry Version 1
    version_1 = mlflow.xgboost.load_model(
        f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )

    # Train production model
    production_model = train_model()

    # Evaluate Version 1
    version_1_metrics = calculate_metrics(
        y_val,
        version_1.predict(X_val_selected),
    )

    # Evaluate Production model
    production_metrics = calculate_metrics(
        y_val,
        production_model.predict(X_val_selected),
    )

    print("\nVersion 1:")
    print(version_1_metrics)

    print("\nProduction Model:")
    print(production_metrics)

    # Log comparison to MLflow
    log_champion_selection(
        version_1_metrics,
        production_metrics,
    )

    # Select winner using validation RMSE
    if version_1_metrics["RMSE"] < production_metrics["RMSE"]:
        winner = "Version 1"
    else:
        winner = "Production Model"

    print(f"\nWinner: {winner}")


if __name__ == "__main__":
    compare_models()