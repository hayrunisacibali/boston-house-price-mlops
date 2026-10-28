from sklearn.model_selection import train_test_split

from src.data.data_loader import load_data


TARGET_COLUMN = "MEDV"


def split_data(
    test_size=0.2,
    validation_size=0.2,
    random_state=42,
):
    """
    Load the dataset and split it into
    training, validation, and test sets.
    """

    df = load_data()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Separate the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Separate validation from training
    adjusted_validation_size = validation_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=adjusted_validation_size,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test