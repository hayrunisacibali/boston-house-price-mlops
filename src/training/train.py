from src.data.splitter import split_data
from src.preprocessing.features import select_features
from src.training.model import create_model


def train_model():
    """
    Train the final XGBoost model using the training dataset.
    """

    X_train, _, _, y_train, _, _ = split_data()

    X_train_selected = select_features(X_train)

    model = create_model()

    model.fit(
        X_train_selected,
        y_train,
    )

    return model