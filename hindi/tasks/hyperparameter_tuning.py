import mlflow
import optuna

import argparse
import os
import sys
from pathlib import Path
import json

import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

sys.path.append(os.getcwd())

def train(trial):
    # Setting nested=True will create a child run under the parent run.
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32)
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 300, step=10)
        rf_max_features = trial.suggest_float("rf_max_features", 0.2, 1.0)
        params = {
            "max_depth": rf_max_depth,
            "n_estimators": rf_n_estimators,
            "max_features": rf_max_features,
        }
        # Log current trial's parameters
        mlflow.log_params(params)

        error = rng.random()  # Placeholder for actual model training and evaluation logic
        mlflow.log_metrics({"error": error})

        # Log the model file
        # mlflow.sklearn.log_model(regressor_obj, name="model")
 
        # Make it easy to retrieve the best-performing child run later
        trial.set_user_attr("run_id", child_run.info.run_id)
        return error


def write_best_params(out_dir, best_params):
    path = Path(out_dir) / "best_params.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(best_params, ensure_ascii=False))
    
    print(f"Wrote best params to {path}")

def run_hyperparameter_tuning(df, out_dir, mlflow_uri):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("F5TTS Hindi Hyperparameter Tuning")
    print(f"Using MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Create a parent run that contains all child runs for different trials
    with mlflow.start_run(run_name="study"):
        # Log the experiment settings
        n_trials = 30
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("manifest_rows", int(len(df)))

        study = optuna.create_study(direction="minimize")
        study.optimize(train, n_trials=n_trials)

        # Log the best trial and its run ID
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics({"best_error": study.best_value})
        if best_run_id := study.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id)
        
        write_best_params(out_dir, study.best_trial.params)


def get_args():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for training F5TTS-hindi model.")
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest with text and audio file path fields.")
    parser.add_argument("--out-dir", required=True, help="Folder for saving the best hyperparameters (best_params.json)")
    parser.add_argument(
        "--mlflow-uri",
        default="http://mlflow:5000",
        help=("MLflow tracking URI. Defaults to http://mlflow:5000 in Docker."),
    )
    return parser.parse_args()
 
def cli():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(args.manifest, lines=True)

    run_hyperparameter_tuning(df,out_dir, args.mlflow_uri)

    print(f"\nHyperparameters saved to: {args.out_dir}/best_params.json")

if __name__ == "__main__":
    cli()
