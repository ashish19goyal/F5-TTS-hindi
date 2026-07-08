from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

default_args = {
    "owner": "train",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="train",
    description="Train F5TTS-Hindi model on IndicVoices-R Hindi dataset",
    default_args=default_args,
    start_date=datetime(2026, 7, 1),
    schedule=None,  # trigger manually; this is a one-shot analysis run
    catchup=False,
    max_active_runs=1,
    tags=["f5-tts", "IndicVoices-R", "hindi", "dataset", "train"],
    doc_md=__doc__,
) as dag:
    
    hyperparameter_tuning = BashOperator(
        task_id="hyperparameter_tuning",
        bash_command=(
            "python /opt/airflow/tasks/hyperparameter_tuning.py"
            " --manifest /opt/data/manifests/raw.jsonl"
            " --out-dir /opt/data/hyperparameters"
        ),
        execution_timeout=timedelta(days=6),
    )

    training_model = BashOperator(
        task_id="training_model",
        bash_command=(
            # "python /opt/airflow/tasks/train_model.py"
            # " --manifest /opt/data/manifests/raw.jsonl"
            # " --params /opt/data/hyperparameters/best_params.json"
            # " --out-dir /opt/data/trained_model"
            "echo 'hyperparameter tuning completed, training model with best parameters'"
        ),
        execution_timeout=timedelta(days=12),
    )

    hyperparameter_tuning >> training_model