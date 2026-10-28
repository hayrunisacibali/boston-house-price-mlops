SELECTED_FEATURES = [
    "RM",
    "LSTAT",
    "DIS",
    "NOX",
    "PTRATIO",
    "CRIM",
]


def select_features(X):
    """
    Select the features used by the final model.
    """
    return X[SELECTED_FEATURES]