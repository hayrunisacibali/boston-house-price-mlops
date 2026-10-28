import mlflow


TRACKING_URI = (
    "sqlite:///mlflow.db"
)

MODEL_NAME = "BostonHousePriceModel"

RUN_ID = "a08e24cc6c514988b642a2b6afe9b49c"

MODEL_URI = f"runs:/{RUN_ID}/model"


mlflow.set_tracking_uri(TRACKING_URI)

print("Registering production baseline as Version 2...")

model_version = mlflow.register_model(
    model_uri=MODEL_URI,
    name=MODEL_NAME,
)

print("\nModel registered successfully.")

print(f"Model name : {model_version.name}")
print(f"Version    : {model_version.version}")