import mlflow
from scipy.stats import ks_2samp

from src.data.splitter import split_data
from src.prediction.storage import (
    get_current_cycle_id,
    get_cycle_predictions,
)
from src.preprocessing.features import SELECTED_FEATURES


TRACKING_URI = (
    "sqlite:///mlflow.db"
)

EXPERIMENT_NAME = "Boston House Price Prediction - Production"

MODEL_NAME = "BostonHousePriceModel"

CHAMPION_ALIAS = "champion"

DRIFT_THRESHOLD = 0.05


def detect_feature_drift(
    reference_data,
    production_data,
    threshold=DRIFT_THRESHOLD,
):
    """
    Compare reference and production distributions
    for each selected feature using the KS test.
    """

    results = {}

    for feature in SELECTED_FEATURES:

        statistic, p_value = ks_2samp(
            reference_data[feature],
            production_data[feature],
        )

        results[feature] = {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift": bool(p_value < threshold),
        }

    return results


def detect_drift():
    """
    Compare the training data distribution
    with the current production prediction cycle.
    """

    X_train, _, _, _, _, _ = split_data()

    production_data = get_cycle_predictions()

    if production_data.empty:
        raise ValueError(
            "No production prediction data available "
            "for the current cycle."
        )

    reference_data = X_train[
        SELECTED_FEATURES
    ].copy()

    production_data = production_data[
        SELECTED_FEATURES
    ].copy()

    feature_results = detect_feature_drift(
        reference_data,
        production_data,
    )

    overall_drift = any(
        result["drift"]
        for result in feature_results.values()
    )

    return {
        "overall_drift": overall_drift,
        "reference_rows": len(reference_data),
        "production_rows": len(production_data),
        "features": feature_results,
    }


def log_monitoring_run(result):
    """
    Log the monitoring result to MLflow.
    """

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    cycle_id = get_current_cycle_id()

    client = mlflow.MlflowClient()

    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        CHAMPION_ALIAS,
    )

    champion_version = str(
        champion.version
    )

    run_name = (
        f"monitoring-cycle-{cycle_id}"
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------

        mlflow.log_params({
            "monitoring_type": "data_drift",
            "cycle_id": cycle_id,
            "reference_rows": result[
                "reference_rows"
            ],
            "production_rows": result[
                "production_rows"
            ],
            "drift_threshold": DRIFT_THRESHOLD,
            "features_checked": len(
                SELECTED_FEATURES
            ),
            "reference_dataset": "training_data",
            "production_dataset": (
                f"production_cycle_{cycle_id}"
            ),
        })

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        for feature, values in (
            result["features"].items()
        ):

            mlflow.log_metrics({
                f"{feature}_ks_statistic":
                    values["statistic"],

                f"{feature}_p_value":
                    values["p_value"],
            })

        # --------------------------------------------------
        # Tags
        # --------------------------------------------------

        mlflow.set_tags({
            "stage": "monitoring",
            "monitoring_type": "data_drift",
            "cycle_id": str(cycle_id),
            "model_name": MODEL_NAME,
            "model_alias": CHAMPION_ALIAS,
            "champion_version": champion_version,
            "overall_drift": str(
                result["overall_drift"]
            ),
            "result": (
                "drift"
                if result["overall_drift"]
                else "no_drift"
            ),
        })

        # Feature-level drift tags

        for feature, values in (
            result["features"].items()
        ):

            mlflow.set_tag(
                f"{feature}_drift",
                str(values["drift"]),
            )

    print(
        "\nMonitoring result logged to MLflow."
    )

    return {
        "run_name": run_name,
        "cycle_id": cycle_id,
        "champion_version": champion_version,
    }


if __name__ == "__main__":

    result = detect_drift()

    print("\n==============================")
    print("Data Drift Detection")
    print("==============================")

    print(
        f"Reference rows  : "
        f"{result['reference_rows']}"
    )

    print(
        f"Production rows : "
        f"{result['production_rows']}"
    )

    print("\nFeature Results:")

    for feature, values in (
        result["features"].items()
    ):

        status = (
            "DRIFT"
            if values["drift"]
            else "NO DRIFT"
        )

        print(
            f"{feature:8} | "
            f"p-value: "
            f"{values['p_value']:.4f} | "
            f"{status}"
        )

    print("\n==============================")

    if result["overall_drift"]:
        print(
            "Overall Result: DRIFT DETECTED"
        )
    else:
        print(
            "Overall Result: NO DRIFT"
        )

    print("==============================")

    log_monitoring_run(result)