from xgboost import XGBRegressor


def create_model():
    """
    Create the final XGBoost regression model.
    """

    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    return model