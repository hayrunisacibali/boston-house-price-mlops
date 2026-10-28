import mlflow
from mlflow import MlflowClient

from src.data.splitter import split_data
from src.preprocessing.features import select_features
from src.evaluation.metrics import calculate_metrics
from src.retraining.candidate import prepare_candidate_data
from src.prediction.storage import get_training_state, get_cycle_predictions


TRACKING_URI = (
    "sqlite:///mlflow.db"
)

EXPERIMENT_NAME = "Boston House Price Prediction - Production"

MODEL_NAME = "BostonHousePriceModel"

CHAMPION_ALIAS = "champion"


def evaluate_candidate(candidate_version):
    """
    Compare the given Candidate with the current Champion.
    """

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient()

    # --------------------------------------------------
    # 1. Get current Champion
    # --------------------------------------------------

    champion_version = client.get_model_version_by_alias(
        MODEL_NAME,
        CHAMPION_ALIAS,
    ).version

    candidate_version = int(candidate_version)

    if candidate_version == int(champion_version):
        raise ValueError(
            "Candidate version is already the Champion."
        )

    # --------------------------------------------------
    # 2. Training information
    # --------------------------------------------------

    X_candidate, _ = prepare_candidate_data()

    training_state = get_training_state()

    source_cycle = training_state["last_trained_cycle"]

    candidate_training_rows = len(X_candidate)

    X_train, _, _, _, _, _ = split_data()

    original_training_rows = len(X_train)

    current_cycle_predictions = get_cycle_predictions()

    new_production_rows = len(
        current_cycle_predictions
    )

    # --------------------------------------------------
    # 3. Validation and test data
    # --------------------------------------------------

    _, X_val, X_test, _, y_val, y_test = split_data()

    X_val_selected = select_features(X_val)
    X_test_selected = select_features(X_test)

    # --------------------------------------------------
    # 4. Run information
    # --------------------------------------------------

    run_name = (
        f"candidate-validation-v"
        f"{candidate_version}"
        f"-vs-champion-v"
        f"{champion_version}"
    )

    print("\n================================")
    print("Candidate Evaluation")
    print("================================")

    print(f"Champion : V{champion_version}")
    print(f"Candidate: V{candidate_version}")
    print(f"Source Cycle: {source_cycle}")
    print(f"Run Name : {run_name}")

    # --------------------------------------------------
    # 5. Load Champion
    # --------------------------------------------------

    champion_uri = (
        f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    )

    champion_model = mlflow.pyfunc.load_model(
        champion_uri
    )

    print("\nChampion model loaded.")

    # --------------------------------------------------
    # 6. Load Candidate
    # --------------------------------------------------

    candidate_uri = (
        f"models:/{MODEL_NAME}/{candidate_version}"
    )

    candidate_model = mlflow.xgboost.load_model(
        candidate_uri
    )

    print("Candidate model loaded.")

    # --------------------------------------------------
    # 7. Validation comparison
    # --------------------------------------------------

    champion_predictions = champion_model.predict(
        X_val_selected
    )

    candidate_predictions = candidate_model.predict(
        X_val_selected
    )

    champion_metrics = calculate_metrics(
        y_val,
        champion_predictions,
    )

    candidate_metrics = calculate_metrics(
        y_val,
        candidate_predictions,
    )

    # --------------------------------------------------
    # 8. Candidate test evaluation
    # --------------------------------------------------

    candidate_test_predictions = candidate_model.predict(
        X_test_selected
    )

    candidate_test_metrics = calculate_metrics(
        y_test,
        candidate_test_predictions,
    )

    # --------------------------------------------------
    # 9. Decision
    # --------------------------------------------------

    if candidate_metrics["RMSE"] < champion_metrics["RMSE"]:
        result = "candidate_better"
    else:
        result = "champion_better_or_equal"

    # --------------------------------------------------
    # 10. Print results
    # --------------------------------------------------

    print("\n================================")
    print("Champion Validation")
    print("================================")

    print(
        f"MAE  : {champion_metrics['MAE']:.4f}"
    )
    print(
        f"RMSE : {champion_metrics['RMSE']:.4f}"
    )
    print(
        f"R2   : {champion_metrics['R2']:.4f}"
    )

    print("\n================================")
    print("Candidate Validation")
    print("================================")

    print(
        f"MAE  : {candidate_metrics['MAE']:.4f}"
    )
    print(
        f"RMSE : {candidate_metrics['RMSE']:.4f}"
    )
    print(
        f"R2   : {candidate_metrics['R2']:.4f}"
    )

    print("\n================================")
    print("Candidate Test")
    print("================================")

    print(
        f"MAE  : {candidate_test_metrics['MAE']:.4f}"
    )
    print(
        f"RMSE : {candidate_test_metrics['RMSE']:.4f}"
    )
    print(
        f"R2   : {candidate_test_metrics['R2']:.4f}"
    )

    print("\n================================")
    print(f"Result: {result}")
    print("================================")

    # --------------------------------------------------
    # 11. Log comparison to MLflow
    # --------------------------------------------------

    with mlflow.start_run(
        run_name=run_name
    ):

        mlflow.set_tags({
            "stage": "candidate_validation",
            "candidate_version": str(
                candidate_version
            ),
            "champion_version": str(
                champion_version
            ),
            "source_cycle": str(source_cycle),
            "model_name": MODEL_NAME,
            "result": result,
        })

        mlflow.log_params({
            "comparison": "Candidate vs Champion",
            "decision_metric": "RMSE",

            "candidate_version":
                candidate_version,

            "champion_version":
                champion_version,

            "candidate_training_rows":
                candidate_training_rows,

            "original_training_rows":
                original_training_rows,

            "new_production_rows":
                new_production_rows,

            "source_cycle":
                source_cycle,

            "validation_rows":
                len(X_val_selected),

            "test_rows":
                len(X_test_selected),
        })

        mlflow.log_metrics({
            "champion_MAE":
                champion_metrics["MAE"],

            "champion_RMSE":
                champion_metrics["RMSE"],

            "champion_R2":
                champion_metrics["R2"],

            "candidate_MAE":
                candidate_metrics["MAE"],

            "candidate_RMSE":
                candidate_metrics["RMSE"],

            "candidate_R2":
                candidate_metrics["R2"],

            "candidate_test_MAE":
                candidate_test_metrics["MAE"],

            "candidate_test_RMSE":
                candidate_test_metrics["RMSE"],

            "candidate_test_R2":
                candidate_test_metrics["R2"],
        })

    print("\nCandidate evaluation logged to MLflow.")

    # Return information to the next stage
    return {
        "candidate_version": candidate_version,
        "champion_version": int(champion_version),
        "source_cycle": source_cycle,
        "result": result,
    }


if __name__ == "__main__":
    raise ValueError(
        "This module now requires a candidate_version."
    )