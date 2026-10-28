import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "predictions.db"


def initialize_database():
    connection = sqlite3.connect(DB_PATH)

    # -------------------------------------------------
    # Predictions table
    # -------------------------------------------------
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            RM REAL NOT NULL,
            LSTAT REAL NOT NULL,
            DIS REAL NOT NULL,
            NOX REAL NOT NULL,
            PTRATIO REAL NOT NULL,
            CRIM REAL NOT NULL,
            prediction REAL NOT NULL,
            model_alias TEXT NOT NULL,
            model_version TEXT,
            cycle_id INTEGER
        )
        """
    )

    columns = connection.execute(
        "PRAGMA table_info(predictions)"
    ).fetchall()

    column_names = [column[1] for column in columns]

    if "model_version" not in column_names:
        connection.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN model_version TEXT
            """
        )

    if "cycle_id" not in column_names:
        connection.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN cycle_id INTEGER
            """
        )

    # -------------------------------------------------
    # Pipeline state
    # -------------------------------------------------
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            current_cycle_id INTEGER NOT NULL
        )
        """
    )

    state = connection.execute(
        """
        SELECT current_cycle_id
        FROM pipeline_state
        WHERE id = 1
        """
    ).fetchone()

    if state is None:
        # Old prediction records become Cycle 1
        connection.execute(
            """
            UPDATE predictions
            SET cycle_id = 1
            WHERE cycle_id IS NULL
            """
        )

        # Predictions already associated with a model version
        # belong to the next production cycle
        connection.execute(
            """
            UPDATE predictions
            SET cycle_id = 2
            WHERE model_version IS NOT NULL
            """
        )

        connection.execute(
            """
            INSERT INTO pipeline_state (
                id,
                current_cycle_id
            )
            VALUES (1, 2)
            """
        )

    else:
        connection.execute(
            """
            UPDATE predictions
            SET cycle_id = ?
            WHERE cycle_id IS NULL
            """,
            (state[0],),
        )

    # -------------------------------------------------
    # Training state
    # -------------------------------------------------
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS training_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_trained_cycle INTEGER NOT NULL,
            training_rows INTEGER NOT NULL
        )
        """
    )

    training_state = connection.execute(
        """
        SELECT id
        FROM training_state
        WHERE id = 1
        """
    ).fetchone()

    if training_state is None:
        connection.execute(
            """
            INSERT INTO training_state (
                id,
                last_trained_cycle,
                training_rows
            )
            VALUES (1, 0, 303)
            """
        )

    # -------------------------------------------------
    # Cumulative training history
    # -------------------------------------------------
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_prediction_id INTEGER,
            cycle_id INTEGER,
            RM REAL NOT NULL,
            LSTAT REAL NOT NULL,
            DIS REAL NOT NULL,
            NOX REAL NOT NULL,
            PTRATIO REAL NOT NULL,
            CRIM REAL NOT NULL,
            target REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def get_current_cycle_id():
    initialize_database()

    connection = sqlite3.connect(DB_PATH)

    result = connection.execute(
        """
        SELECT current_cycle_id
        FROM pipeline_state
        WHERE id = 1
        """
    ).fetchone()

    connection.close()

    return result[0]


def start_new_cycle():
    initialize_database()

    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        """
        UPDATE pipeline_state
        SET current_cycle_id = current_cycle_id + 1
        WHERE id = 1
        """
    )

    connection.commit()
    connection.close()


def save_prediction(
    rm,
    lstat,
    dis,
    nox,
    ptratio,
    crim,
    prediction,
    model_alias="champion",
    model_version=None,
):
    initialize_database()

    cycle_id = get_current_cycle_id()

    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        """
        INSERT INTO predictions (
            timestamp,
            RM,
            LSTAT,
            DIS,
            NOX,
            PTRATIO,
            CRIM,
            prediction,
            model_alias,
            model_version,
            cycle_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            float(rm),
            float(lstat),
            float(dis),
            float(nox),
            float(ptratio),
            float(crim),
            float(prediction),
            model_alias,
            model_version,
            cycle_id,
        ),
    )

    connection.commit()
    connection.close()


def get_prediction_count():
    initialize_database()

    connection = sqlite3.connect(DB_PATH)

    result = connection.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        """
    ).fetchone()

    connection.close()

    return result[0]


def get_cycle_prediction_count():
    initialize_database()

    cycle_id = get_current_cycle_id()

    connection = sqlite3.connect(DB_PATH)

    result = connection.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE cycle_id = ?
        """,
        (cycle_id,),
    ).fetchone()

    connection.close()

    return result[0]


def get_cycle_predictions():
    initialize_database()

    cycle_id = get_current_cycle_id()

    connection = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        """
        SELECT
            RM,
            LSTAT,
            DIS,
            NOX,
            PTRATIO,
            CRIM,
            prediction,
            model_alias,
            model_version,
            cycle_id
        FROM predictions
        WHERE cycle_id = ?
        ORDER BY id
        """,
        connection,
        params=(cycle_id,),
    )

    connection.close()

    return df


def get_predictions():
    initialize_database()

    connection = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        """
        SELECT
            RM,
            LSTAT,
            DIS,
            NOX,
            PTRATIO,
            CRIM,
            prediction,
            model_alias,
            model_version,
            cycle_id
        FROM predictions
        ORDER BY id
        """,
        connection,
    )

    connection.close()

    return df


def get_training_state():
    initialize_database()

    connection = sqlite3.connect(DB_PATH)

    result = connection.execute(
        """
        SELECT
            last_trained_cycle,
            training_rows
        FROM training_state
        WHERE id = 1
        """
    ).fetchone()

    connection.close()

    return {
        "last_trained_cycle": result[0],
        "training_rows": result[1],
    }


def update_training_state(
    last_trained_cycle,
    training_rows,
):
    initialize_database()

    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        """
        UPDATE training_state
        SET
            last_trained_cycle = ?,
            training_rows = ?
        WHERE id = 1
        """,
        (
            last_trained_cycle,
            training_rows,
        ),
    )

    connection.commit()
    connection.close()


def initialize_training_history():
    """
    Initialize the cumulative training history.

    Initial history:
    - 303 original training samples
    - 50 production samples used for V3
    """

    initialize_database()

    connection = sqlite3.connect(DB_PATH)

    existing_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM training_history
        """
    ).fetchone()[0]

    # Prevent duplicate initialization
    if existing_count > 0:
        connection.close()

        print(
            f"Training history already initialized "
            f"with {existing_count} rows."
        )

        return

    # -------------------------------------------------
    # Original training data
    # -------------------------------------------------
    from src.data.splitter import split_data

    X_train, _, _, y_train, _, _ = split_data()

    for (_, row), target in zip(
        X_train.iterrows(),
        y_train,
    ):
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
                "original",
                None,
                None,
                float(row["RM"]),
                float(row["LSTAT"]),
                float(row["DIS"]),
                float(row["NOX"]),
                float(row["PTRATIO"]),
                float(row["CRIM"]),
                float(target),
                datetime.now().isoformat(),
            ),
        )

    # -------------------------------------------------
    # First 50 production records used for V3
    # -------------------------------------------------
    production_rows = connection.execute(
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
        WHERE cycle_id = 1
        ORDER BY id
        LIMIT 50
        """
    ).fetchall()

    if len(production_rows) != 50:
        connection.close()

        raise ValueError(
            "Expected 50 Cycle 1 production records, "
            f"but found {len(production_rows)}."
        )

    for row in production_rows:
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

    connection.commit()
    connection.close()

    print(
        "Training history initialized successfully: "
        "303 original + 50 production = 353 rows."
    )