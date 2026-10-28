import mlflow
from mlflow import MlflowClient

from src.data.splitter import split_data
from src.preprocessing.features import select_features
from src.evaluation.metrics import calculate_metrics
from src.retraining.candidate import (
    train_candidate,
    prepare_candidate_data,
    add_current_cycle_to_training_history,
)
from src.prediction.storage import (
    get_current_cycle_id,
    get_cycle_predictions,
    get_training_state,
    update_training_state,
)


TRACKING_URI = (
    "sqlite:///mlflow.db"
)

EXPERIMENT_NAME = "Boston House Price Prediction - Production"

MODEL_NAME = "BostonHousePriceModel"

CHAMPION_ALIAS = "champion"


def get_next_model_version(client):
    """
    Find the next available model version number.
    """

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        return 1

    current_versions = [
        int(version.version)
        for version in versions
    ]

    return max(current_versions) + 1


def register_candidate():
    """
    Add the current production cycle to the training history,
    train the Candidate once, and register the same model in MLflow.
    """

    # --------------------------------------------------
    # 1. Connect to MLflow
    # --------------------------------------------------

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient()

    # --------------------------------------------------
    # 2. Get current Champion
    # --------------------------------------------------

    champion_version = client.get_model_version_by_alias(
        MODEL_NAME,
        CHAMPION_ALIAS,
    ).version

    # --------------------------------------------------
    # 3. Get current production cycle
    # --------------------------------------------------

    current_cycle = get_current_cycle_id()

    # --------------------------------------------------
    # 4. Check training state
    # --------------------------------------------------

    training_state = get_training_state()

    last_trained_cycle = training_state[
        "last_trained_cycle"
    ]

    if current_cycle <= last_trained_cycle:
        raise ValueError(
            f"Cycle {current_cycle} has already "
            f"been incorporated into training history."
        )

    # --------------------------------------------------
    # 5. Add current cycle to training history
    # --------------------------------------------------

    print("\nAdding current cycle to training history...")

    added_rows = add_current_cycle_to_training_history()

    if added_rows == 0:
     print(
        f"Cycle {current_cycle} is already present in training history. "
        "Continuing candidate training."
    )

    print(
        f"Added {added_rows} production rows "
        f"from Cycle {current_cycle}."
    )

    # --------------------------------------------------
    # 6. Determine Candidate version
    # --------------------------------------------------

    candidate_version = get_next_model_version(client)

    run_name = (
        f"candidate-training-v{candidate_version}"
    )

    # --------------------------------------------------
    # 7. Prepare evaluation data
    # --------------------------------------------------

    _, X_val, X_test, _, y_val, y_test = split_data()

    X_val_selected = select_features(X_val)
    X_test_selected = select_features(X_test)

    # --------------------------------------------------
    # 8. Prepare Candidate training data
    # --------------------------------------------------

    X_candidate, _ = prepare_candidate_data()

    X_train, _, _, _, _, _ = split_data()

    original_training_rows = len(X_train)

    candidate_training_rows = len(
        X_candidate
    )

    # Current cycle's production rows
    current_cycle_predictions = get_cycle_predictions() 

    new_production_rows = len(
        current_cycle_predictions
    )

    candidate_features = list(
        X_candidate.columns
    )

    # --------------------------------------------------
    # 9. Print Candidate information
    # --------------------------------------------------

    print("\n================================")
    print("Candidate Training")
    print("================================")

    print(
        f"Current Champion : V{champion_version}"
    )

    print(
        f"Production Cycle : {current_cycle}"
    )

    print(
        f"Candidate        : V{candidate_version}"
    )

    print(
        f"Training rows    : "
        f"{candidate_training_rows}"
    )

    print(
        f"New production   : "
        f"{new_production_rows}"
    )

    # --------------------------------------------------
    # 10. Train Candidate ONCE
    # --------------------------------------------------

    print("\nTraining Candidate model...")

    candidate_model = train_candidate()

    print("Candidate model trained.")

    # --------------------------------------------------
    # 11. Evaluate Candidate
    # --------------------------------------------------

    validation_predictions = candidate_model.predict(
        X_val_selected
    )

    test_predictions = candidate_model.predict(
        X_test_selected
    )

    validation_metrics = calculate_metrics(
        y_val,
        validation_predictions,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    # --------------------------------------------------
    # 12. Model parameters
    # --------------------------------------------------

    model_params = candidate_model.get_params()

    # --------------------------------------------------
    # 13. MLflow Run
    # --------------------------------------------------

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        # ----------------------------------------------
        # Tags
        # ----------------------------------------------

        mlflow.set_tags({
            "stage": "candidate_training",

            "candidate_version":
                str(candidate_version),

            "previous_champion_version":
                str(champion_version),

            "source_cycle":
                str(current_cycle),

            "model_name":
                MODEL_NAME,
        })

        # ----------------------------------------------
        # General parameters
        # ----------------------------------------------

        mlflow.log_params({
            "model_type":
                type(candidate_model).__name__,

            "model_registry_name":
                MODEL_NAME,

            "candidate_version":
                candidate_version,

            "champion_version":
                champion_version,

            "source_cycle":
                current_cycle,

            "candidate_training_rows":
                candidate_training_rows,

            "original_training_rows":
                original_training_rows,

            "new_production_rows":
                new_production_rows,

            "candidate_features":
                len(candidate_features),

            "validation_rows":
                len(X_val_selected),

            "test_rows":
                len(X_test_selected),

            "decision_metric":
                "RMSE",
        })

        # ----------------------------------------------
        # Feature names
        # ----------------------------------------------

        mlflow.log_param(
            "features",
            ",".join(candidate_features),
        )

        # ----------------------------------------------
        # Model parameters
        # ----------------------------------------------

        for parameter_name, parameter_value in (
            model_params.items()
        ):
            if parameter_value is not None:
                mlflow.log_param(
                    parameter_name,
                    str(parameter_value),
                )

        # ----------------------------------------------
        # Validation metrics
        # ----------------------------------------------

        mlflow.log_metrics({
            "validation_MAE":
                validation_metrics["MAE"],

            "validation_RMSE":
                validation_metrics["RMSE"],

            "validation_R2":
                validation_metrics["R2"],
        })

        # ----------------------------------------------
        # Test metrics
        # ----------------------------------------------

        mlflow.log_metrics({
            "test_MAE":
                test_metrics["MAE"],

            "test_RMSE":
                test_metrics["RMSE"],

            "test_R2":
                test_metrics["R2"],
        })

        # ----------------------------------------------
        # Log and register SAME trained model
        # ----------------------------------------------

        model_info = mlflow.xgboost.log_model(
            candidate_model,
            name="model",
            registered_model_name=MODEL_NAME,
        )

        print(
            "\nCandidate model logged to MLflow."
        )

        print(
            f"Run ID    : {run.info.run_id}"
        )

        print(
            f"Model URI : {model_info.model_uri}"
        )

    # --------------------------------------------------
    # 14. Update training state
    # --------------------------------------------------

    update_training_state(
        last_trained_cycle=current_cycle,
        training_rows=candidate_training_rows,
    )

    # --------------------------------------------------
    # 15. Final information
    # --------------------------------------------------

    print("\n================================")
    print("Candidate Training Completed")
    print("================================")

    print(
        f"Candidate Version : "
        f"V{candidate_version}"
    )

    print(
        f"Run Name          : "
        f"{run_name}"
    )

    print(
        f"Training rows     : "
        f"{candidate_training_rows}"
    )

    print(
        f"Cycle             : "
        f"{current_cycle}"
    )

    # IMPORTANT:
    # Return the actual Candidate version
    # to the next pipeline stage.
    return int(candidate_version)


if __name__ == "__main__":
    register_candidate()