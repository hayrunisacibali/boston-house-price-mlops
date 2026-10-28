import mlflow
from mlflow import MlflowClient


TRACKING_URI = (
TRACKING_URI = "sqlite:///mlflow.db"
)

MODEL_NAME = "BostonHousePriceModel"
CHAMPION_ALIAS = "champion"


mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()


print("Checking model registry...\n")

model = client.get_registered_model(MODEL_NAME)

print(f"Model name: {model.name}")

print("\nVersions:")

versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

for version in versions:
    print(
        f"Version {version.version} | "
        f"Run ID: {version.run_id} | "
        f"Aliases: {version.aliases}"
    )


print("\nCurrent champion:")

try:
    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        CHAMPION_ALIAS,
    )

    print(
        f"Champion -> Version {champion.version}"
    )

except Exception:
    print("Champion alias is not currently assigned.")


# Find the newest registered model version
newest_version = max(
    int(version.version)
    for version in versions
)

print(
    f"\nMoving champion alias to Version {newest_version}..."
)

client.set_registered_model_alias(
    MODEL_NAME,
    CHAMPION_ALIAS,
    str(newest_version),
)

print(
    f"Champion alias moved to Version {newest_version}."
)


print("\nFinal champion:")

champion = client.get_model_version_by_alias(
    MODEL_NAME,
    CHAMPION_ALIAS,
)

print(
    f"Champion -> Version {champion.version}"
)