import mlflow


EXPERIMENT_NAME = "Boston House Price Prediction - Production"

TRACKING_URI = (
    "sqlite:///mlflow.db"
)


def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_production_baseline(
    model,
    validation_metrics,
    test_metrics,
):
    """
    Log the clean production baseline model to MLflow.
    """

    setup_mlflow()

    with mlflow.start_run(
        run_name="production-baseline-v2"
    ):

        mlflow.log_params({
            "model": "XGBoost",
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_features": 6,
            "training_rows": 303,
            "validation_rows": 101,
            "test_rows": 102,
            "feature_set": (
                "RM,LSTAT,DIS,NOX,PTRATIO,CRIM"
            ),
        })

        mlflow.log_metrics({
            "validation_MAE": validation_metrics["MAE"],
            "validation_RMSE": validation_metrics["RMSE"],
            "validation_R2": validation_metrics["R2"],
        })

        mlflow.log_metrics({
            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_R2": test_metrics["R2"],
        })

        mlflow.set_tags({
            "model_role": "production_baseline",
            "training_type": "clean_baseline",
            "dataset_split": "303_train_101_val_102_test",
        })

        model_info = mlflow.xgboost.log_model(
            model,
            name="model",
        )

        print("Production baseline logged to MLflow.")

        return model_info