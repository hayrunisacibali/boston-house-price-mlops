from src.monitoring.drift import (
    detect_drift,
    log_monitoring_run,
)

from src.prediction.storage import (
    get_current_cycle_id,
    get_cycle_prediction_count,
    start_new_cycle,
    get_training_state,
)

from src.retraining.register_candidate import (
    register_candidate,
)

from src.retraining.evaluate_candidate import (
    evaluate_candidate,
)

from src.retraining.promote_candidate import (
    run_promotion_decision,
)


MONITORING_THRESHOLD = 50


def run_automatic_monitoring():
    """
    Run the complete automatic MLOps lifecycle
    after every 50 predictions.
    """

    # --------------------------------------------------
    # 1. Check prediction count
    # --------------------------------------------------

    prediction_count = get_cycle_prediction_count()

    if prediction_count < MONITORING_THRESHOLD:
        return None

    # --------------------------------------------------
    # 2. Get current cycle
    # --------------------------------------------------

    cycle_id = get_current_cycle_id()

    print("\n================================")
    print(f"Monitoring started for cycle {cycle_id}")
    print("================================")

    # --------------------------------------------------
    # 3. Detect drift
    # --------------------------------------------------

    result = detect_drift()

    mlflow_info = log_monitoring_run(result)

    # --------------------------------------------------
    # 4. NO DRIFT
    # --------------------------------------------------

    if not result["overall_drift"]:

        print("\nMonitoring result: NO DRIFT")

        start_new_cycle()

        print("New prediction cycle started.")

        return {
            "status": "no_drift",
            "cycle_id": cycle_id,
            "mlflow_run": mlflow_info,
        }

    # --------------------------------------------------
    # 5. DRIFT DETECTED
    # --------------------------------------------------

    print("\nMonitoring result: DRIFT DETECTED")

    # --------------------------------------------------
    # 6. Check whether this cycle was already trained
    # --------------------------------------------------

    training_state = get_training_state()

    last_trained_cycle = training_state["last_trained_cycle"]

    if cycle_id <= last_trained_cycle:

        print(
            f"\nCycle {cycle_id} has already been processed."
        )

        print(
            f"last_trained_cycle = {last_trained_cycle}"
        )

        print(
            "Skipping duplicate candidate training."
        )

        start_new_cycle()

        print(
            "New prediction cycle started."
        )

        return {
            "status": "already_processed",
            "cycle_id": cycle_id,
            "mlflow_run": mlflow_info,
        }

    # --------------------------------------------------
    # 7. Train + Register Candidate
    # --------------------------------------------------

    print(
        "\nStarting candidate training..."
    )

    candidate_version = register_candidate()

    print(
        f"\nCandidate V{candidate_version} "
        "registered."
    )

    # --------------------------------------------------
    # 8. Compare Candidate vs Champion
    # --------------------------------------------------

    comparison_result = evaluate_candidate(
        candidate_version
    )

    print(
        "\nCandidate comparison completed."
    )

    # --------------------------------------------------
    # 9. Promote or Reject
    # --------------------------------------------------

    decision = run_promotion_decision(
        candidate_version
    )

    # --------------------------------------------------
    # 10. Start next prediction cycle
    # --------------------------------------------------

    start_new_cycle()

    print(
        "\nNew prediction cycle started."
    )

    # --------------------------------------------------
    # 11. Return lifecycle result
    # --------------------------------------------------

    return {
        "status": "drift",
        "cycle_id": cycle_id,
        "mlflow_run": mlflow_info,
        "candidate_version": candidate_version,
        "comparison": comparison_result,
        "decision": decision,
    }