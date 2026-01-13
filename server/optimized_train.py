import model_predict
import optuna
import pipeline

def objective(trial):
    dataset_train, dataset_test = pipeline.load_datasets(version="latest")
    epochs = trial.suggest_int("epochs", 1, 10)
    per_device_batch_size = trial.suggest_int("per_device_batch_size", 1, 8)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4)
    model_cache = pipeline.train_model(
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        output_dir="./cache/output",
        base_model="camembert-base",
        epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=8,
        learning_rate=learning_rate,
    )
    evals = pipeline._eval_model(pipeline._get_model(model_cache))
    return evals["loss"]

@pipeline
def train_optimization():
    study = optuna.create_study()
    study.optimize()