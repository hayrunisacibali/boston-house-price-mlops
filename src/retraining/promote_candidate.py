import mlflow
from mlflow import MlflowClient

from src.prediction.storage import get_current_cycle_id


TRACKING_URI = (
    "sqlite:///mlflow.db"
)

EXPERIMENT_NAME = (
    "Boston House Price Prediction - Production"
)

MODEL_NAME = "BostonHousePriceModel"

CHAMPION_ALIAS = "champion"


def get_current_champion(client):
    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        CHAMPION_ALIAS
    )

    return int(champion.version)


def get_comparison_run(
    client,
    candidate_version,
    champion_version
):
    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    runs = client.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],
        filter_string=(
            "tags.stage = 'candidate_validation' "
            f"AND tags.candidate_version = "
            f"'{candidate_version}' "
            f"AND tags.champion_version = "
            f"'{champion_version}'"
        ),
        order_by=["start_time DESC"],
    )

    if not runs:
        raise ValueError(
            "No candidate comparison Run found."
        )

    return runs[0]


def promote_candidate(
    client,
    candidate_version,
    champion_version,
    cycle_id,
    comparison_run,
):
    result = comparison_run.data.tags.get(
        "result"
    )

    candidate_rmse = (
        comparison_run.data.metrics.get(
            "candidate_RMSE"
        )
    )

    champion_rmse = (
        comparison_run.data.metrics.get(
            "champion_RMSE"
        )
    )

    # --------------------------------------------------
    # Candidate is better
    # --------------------------------------------------

    if result == "candidate_better":

        client.set_registered_model_alias(
            MODEL_NAME,
            CHAMPION_ALIAS,
            candidate_version,
        )

        run_name = (
            f"candidate-promotion-v"
            f"{candidate_version}"
        )

        with mlflow.start_run(
            run_name=run_name
        ):

            mlflow.set_tags({
                "stage": "promotion",
                "result": "promoted",
                "model_name": MODEL_NAME,

                "candidate_version":
                    str(candidate_version),

                "previous_champion_version":
                    str(champion_version),

                "champion_version":
                    str(candidate_version),

                "source_cycle":
                    str(cycle_id),
            })

            mlflow.log_params({
                "decision_metric": "RMSE",

                "candidate_validation_RMSE":
                    candidate_rmse,

                "previous_champion_validation_RMSE":
                    champion_rmse,
            })

            mlflow.log_metrics({
                "candidate_RMSE":
                    candidate_rmse,

                "previous_champion_RMSE":
                    champion_rmse,
            })

        print(
            f"\nCandidate V{candidate_version} "
            "PROMOTED."
        )

        print(
            f"Champion changed: "
            f"V{champion_version} "
            f"-> V{candidate_version}"
        )

        return "promoted"

    # --------------------------------------------------
    # Candidate is not better
    # --------------------------------------------------

    if result == "champion_better_or_equal":

        run_name = (
            f"candidate-rejection-v"
            f"{candidate_version}"
        )

        with mlflow.start_run(
            run_name=run_name
        ):

            mlflow.set_tags({
                "stage": "rejection",
                "result": "rejected",
                "model_name": MODEL_NAME,

                "candidate_version":
                    str(candidate_version),

                "champion_version":
                    str(champion_version),

                "source_cycle":
                    str(cycle_id),
            })

            mlflow.log_params({
                "decision_metric": "RMSE",

                "candidate_validation_RMSE":
                    candidate_rmse,

                "champion_validation_RMSE":
                    champion_rmse,
            })

            mlflow.log_metrics({
                "candidate_RMSE":
                    candidate_rmse,

                "champion_RMSE":
                    champion_rmse,
            })

        print(
            f"\nCandidate V{candidate_version} "
            "REJECTED."
        )

        print(
            f"Champion remains "
            f"V{champion_version}."
        )

        return "rejected"

    raise ValueError(
        f"Unknown comparison result: {result}"
    )


def run_promotion_decision(candidate_version):
    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    client = MlflowClient()

    cycle_id = get_current_cycle_id()

    champion_version = get_current_champion(
        client
    )

    candidate_version = int(
        candidate_version
    )

    comparison_run = get_comparison_run(
        client,
        candidate_version,
        champion_version,
    )

    print("\n================================")
    print("Candidate Promotion Decision")
    print("================================")

    print(
        f"Champion : V{champion_version}"
    )

    print(
        f"Candidate: V{candidate_version}"
    )

    print(
        f"Cycle    : {cycle_id}"
    )

    print(
        f"Comparison Run: "
        f"{comparison_run.info.run_id}"
    )

    result = promote_candidate(
        client=client,
        candidate_version=candidate_version,
        champion_version=champion_version,
        cycle_id=cycle_id,
        comparison_run=comparison_run,
    )

    print(
        f"\nDecision completed: {result}"
    )

    return result


if __name__ == "__main__":
    raise ValueError(
        "This module now requires a candidate_version."
    )