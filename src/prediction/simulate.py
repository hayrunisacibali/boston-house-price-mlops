import sys

import mlflow
import numpy as np
import pandas as pd

from src.data.data_loader import load_data
from src.prediction.storage import save_prediction


TRACKING_URI = (
    "sqlite:///mlflow.db"
)

MODEL_URI = "models:/BostonHousePriceModel@champion"

FEATURES = [
    "RM",
    "LSTAT",
    "DIS",
    "NOX",
    "PTRATIO",
    "CRIM",
]


def generate_random_inputs(number_of_samples):
    """
    Generate realistic random production inputs
    based on the real Boston Housing feature distributions.
    """

    df = load_data()

    rng = np.random.default_rng()

    generated_data = {}

    for feature in FEATURES:
        values = df[feature].values

        generated_data[feature] = rng.choice(
            values,
            size=number_of_samples,
            replace=True,
        )

    return pd.DataFrame(generated_data)


def generate_drifted_inputs(number_of_samples):
    """
    Generate production-like inputs with
    controlled feature distribution drift.
    """

    df = load_data()

    if number_of_samples > len(df):
        raise ValueError(
            "Number of samples cannot exceed dataset size."
        )

    # Take real feature values as the starting point.
    production_data = (
        df[FEATURES]
        .sample(
            n=number_of_samples,
            random_state=42,
        )
        .reset_index(drop=True)
        .copy()
    )

    # -------------------------------------------------
    # Controlled distribution drift
    # -------------------------------------------------

    production_data["RM"] *= 1.20
    production_data["LSTAT"] *= 1.25
    production_data["DIS"] *= 0.80

    return production_data


def predict_and_save(input_data):
    """
    Predict using the current Champion model
    and save every prediction to the production database.
    """

    mlflow.set_tracking_uri(TRACKING_URI)

    model = mlflow.pyfunc.load_model(
        MODEL_URI
    )

    predictions = model.predict(input_data)

    for (_, row), prediction in zip(
        input_data.iterrows(),
        predictions,
    ):
        save_prediction(
            rm=float(row["RM"]),
            lstat=float(row["LSTAT"]),
            dis=float(row["DIS"]),
            nox=float(row["NOX"]),
            ptratio=float(row["PTRATIO"]),
            crim=float(row["CRIM"]),
            prediction=float(prediction),
            model_alias="champion",
        )


def simulate_predictions(number_of_samples):
    """
    Generate normal production inputs,
    predict with Champion, and save predictions.
    """

    if number_of_samples <= 0:
        raise ValueError(
            "Number of samples must be greater than 0."
        )

    print(
        f"Generating {number_of_samples} "
        "normal production predictions..."
    )

    input_data = generate_random_inputs(
        number_of_samples
    )

    predict_and_save(input_data)

    print(
        f"Successfully added {number_of_samples} "
        "normal production predictions."
    )


def simulate_drift_predictions(number_of_samples):
    """
    Generate drifted production inputs,
    predict with Champion, and save predictions.
    """

    if number_of_samples <= 0:
        raise ValueError(
            "Number of samples must be greater than 0."
        )

    print(
        f"Generating {number_of_samples} "
        "drifted production predictions..."
    )

    input_data = generate_drifted_inputs(
        number_of_samples
    )

    predict_and_save(input_data)

    print(
        f"Successfully added {number_of_samples} "
        "drifted production predictions."
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage:"
        )
        print(
            "  python -m src.prediction.simulate normal <number>"
        )
        print(
            "  python -m src.prediction.simulate drift <number>"
        )
        sys.exit(1)

    simulation_type = sys.argv[1]
    number_of_samples = int(sys.argv[2])

    if simulation_type == "normal":

        simulate_predictions(
            number_of_samples
        )

    elif simulation_type == "drift":

        simulate_drift_predictions(
            number_of_samples
        )

    else:

        print(
            "Simulation type must be "
            "'normal' or 'drift'."
        )

        sys.exit(1)