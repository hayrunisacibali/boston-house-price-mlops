from src.training.train import train_model
from src.evaluation.evaluate import evaluate_validation, evaluate_model
from src.tracking.mlflow_tracker import log_production_baseline


print("Training clean production baseline...")

model = train_model()

print("Evaluating validation set...")
validation_metrics = evaluate_validation(model)

print("Evaluating test set...")
test_metrics = evaluate_model(model)

print("\nValidation Metrics:")
print(validation_metrics)

print("\nTest Metrics:")
print(test_metrics)

print("\nLogging production baseline to MLflow...")

log_production_baseline(
    model=model,
    validation_metrics=validation_metrics,
    test_metrics=test_metrics,
)

print("\nProduction baseline completed.")