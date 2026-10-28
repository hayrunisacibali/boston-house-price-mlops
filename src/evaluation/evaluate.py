from src.data.splitter import split_data
from src.preprocessing.features import select_features
from src.evaluation.metrics import calculate_metrics


def evaluate_model(model):
    """
    Evaluate a trained model on the test dataset.
    """

    _, _, X_test, _, _, y_test = split_data()

    X_test_selected = select_features(X_test)

    y_pred = model.predict(X_test_selected)

    metrics = calculate_metrics(
        y_test,
        y_pred,
    )

    return metrics


def evaluate_validation(model):
    """
    Evaluate a trained model on the validation dataset.
    """

    _, X_val, _, _, y_val, _ = split_data()

    X_val_selected = select_features(X_val)

    y_pred = model.predict(X_val_selected)

    metrics = calculate_metrics(
        y_val,
        y_pred,
    )

    return metrics