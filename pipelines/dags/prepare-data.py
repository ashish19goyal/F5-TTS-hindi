from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

default_args = {
    "owner": "prepare",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="prepare",
    description="Preprocess IndicVoices-R Hindi dataset into Arrow files for F5-TTS training",
    default_args=default_args,
    start_date=datetime(2026, 7, 1),
    schedule=None,  # trigger manually; this is a one-shot data build
    catchup=False,
    max_active_runs=1,
    tags=["f5-tts", "hindi", "dataset", "IndicVoices-R", "prepare"],
    doc_md=__doc__,
) as dag:

    build_vocab = BashOperator(
        task_id="build_vocab",
        application="/opt/airflow/tasks/build-vocab.py",
        conn_id="spark_default",
        application_args=[
            "--manifest", "/opt/app/data/manifests/raw.jsonl",
            "--out-dir", "/opt/app/data/prepare/text",
        ],
        execution_timeout=timedelta(minutes=10),
    )

    wav_to_arrow = SparkSubmitOperator(
        task_id="wav_to_arrow",
        application="/opt/airflow/tasks/wav-to-arrow.py",
        conn_id="spark_default",
        application_args=[
            "--manifest", "/opt/app/data/manifests/raw.jsonl",
            "--out-dir", "/opt/app/data/prepare/arrow",
        ],
        execution_timeout=timedelta(hours=6),
    )

    build_vocab >> wav_to_arrow
