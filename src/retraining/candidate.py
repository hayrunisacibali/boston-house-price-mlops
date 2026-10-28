import sqlite3

import pandas as pd

from src.prediction.storage import (
    DB_PATH,
    get_current_cycle_id,
    get_training_state,
    initialize_database,
    update_training_state,
)
from src.training.model import create_model


FEATURES = [
    "RM",
    "LSTAT",
    "DIS",
    "NOX",
    "PTRATIO",
    "CRIM",
]


def get_training_history():
    """
    Load the cumulative training history.
    """

    initialize_database()

    connection = sqlite3.connect(DB_PATH)

    rows = connection.execute(
        """
        SELECT
            id,
            source_type,
            source_prediction_id,
            cycle_id,
            RM,
            LSTAT,
            DIS,
            NOX,
            PTRATIO,
            CRIM,
            target
        FROM training_history
        ORDER BY id
        """
    ).fetchall()

    connection.close()

    columns = [
        "id",
        "source_type",
        "source_prediction_id",
        "cycle_id",
        "RM",
        "LSTAT",
        "DIS",
        "NOX",
        "PTRATIO",
        "CRIM",
        "target",
    ]

    return pd.DataFrame(rows, columns=columns)


def add_current_cycle_to_training_history():
    """
    Add the current production cycle to training history.

    This function must only be called after DRIFT has been detected.
    """

    initialize_database()

    current_cycle = get_current_cycle_id()
    training_state = get_training_state()

    last_trained_cycle = training_state["last_trained_cycle"]

    if current_cycle <= last_trained_cycle:
        print(
            f"Cycle {current_cycle} has already been "
            "incorporated into training history."
        )
        return 0

    connection = sqlite3.connect(DB_PATH)

    rows = connection.execute(
        """
        SELECT
            id,
            cycle_id,
            RM,
            LSTAT,
            DIS,
            NOX,
            PTRATIO,
            CRIM,
            prediction,
            timestamp
        FROM predictions
        WHERE cycle_id = ?
        ORDER BY id
        """,
        (current_cycle,),
    ).fetchall()

    if not rows:
        connection.close()

        raise ValueError(
            f"No production predictions found for "
            f"cycle {current_cycle}."
        )

    added_rows = 0

    for row in rows:
        (
            prediction_id,
            cycle_id,
            rm,
            lstat,
            dis,
            nox,
            ptratio,
            crim,
            prediction,
            timestamp,
        ) = row

        already_exists = connection.execute(
            """
            SELECT 1
            FROM training_history
            WHERE source_prediction_id = ?
            """,
            (prediction_id,),
        ).fetchone()

        if already_exists:
            continue

        connection.execute(
            """
            INSERT INTO training_history (
                source_type,
                source_prediction_id,
                cycle_id,
                RM,
                LSTAT,
                DIS,
                NOX,
                PTRATIO,
                CRIM,
                target,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "production",
                prediction_id,
                cycle_id,
                float(rm),
                float(lstat),
                float(dis),
                float(nox),
                float(ptratio),
                float(crim),
                float(prediction),
                timestamp,
            ),
        )

        added_rows += 1

    connection.commit()
    connection.close()

    print(
        f"Added {added_rows} production rows "
        f"from Cycle {current_cycle}."
    )

    return added_rows


def prepare_candidate_data():
    """
    Prepare cumulative candidate training data.
    """

    history = get_training_history()

    if history.empty:
        raise ValueError(
            "Training history is empty."
        )

    X_candidate = history[FEATURES].copy()
    y_candidate = history["target"].copy()

    X_candidate = X_candidate.apply(
        pd.to_numeric,
        errors="coerce",
    )

    y_candidate = pd.to_numeric(
        y_candidate,
        errors="coerce",
    )

    if X_candidate.isna().any().any():
        raise ValueError(
            "Candidate features contain invalid "
            "or missing values."
        )

    if y_candidate.isna().any():
        raise ValueError(
            "Candidate target contains invalid "
            "or missing values."
        )

    return X_candidate, y_candidate


def train_candidate():
    """
    Train a candidate model using cumulative training history.
    """

    X_candidate, y_candidate = prepare_candidate_data()

    print(
        f"Training candidate with "
        f"{len(X_candidate)} rows..."
    )

    model = create_model()

    model.fit(
        X_candidate,
        y_candidate,
    )

    print("Candidate training completed.")

    return model


def run_candidate_training():
    """
    Complete candidate training pipeline.

    This function should be called only after DRIFT detection.
    """

    current_cycle = get_current_cycle_id()

    print("\n================================")
    print(
        f"Candidate training for Cycle {current_cycle}"
    )
    print("================================")

    added_rows = add_current_cycle_to_training_history()

    if added_rows == 0:
        raise ValueError(
            "No new production data was added "
            "to training history."
        )

    X_candidate, y_candidate = prepare_candidate_data()

    print(
        f"\nCandidate dataset: "
        f"{len(X_candidate)} rows"
    )

    model = train_candidate()

    new_training_rows = len(X_candidate)

    update_training_state(
        last_trained_cycle=current_cycle,
        training_rows=new_training_rows,
    )

    print("\nTraining state updated:")
    print(
        f"last_trained_cycle = {current_cycle}"
    )
    print(
        f"training_rows = {new_training_rows}"
    )

    return model


if __name__ == "__main__":
    current_cycle = get_current_cycle_id()
    training_state = get_training_state()

    print("Current training state:")
    print(
        f"Last trained cycle: "
        f"{training_state['last_trained_cycle']}"
    )
    print(
        f"Current production cycle: "
        f"{current_cycle}"
    )
    print(
        f"Current training rows: "
        f"{training_state['training_rows']}"
    )

    print("\nCandidate module is ready.")