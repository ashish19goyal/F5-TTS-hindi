from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

default_args = {
    "owner": "analyse",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="analyse",
    description="Analyse IndicVoices-R Hindi dataset",
    default_args=default_args,
    start_date=datetime(2026, 7, 1),
    schedule=None,  # trigger manually; this is a one-shot analysis run
    catchup=False,
    max_active_runs=1,
    tags=["f5-tts", "IndicVoices-R", "hindi", "dataset", "analyse"],
    doc_md=__doc__,
) as dag:
    download = BashOperator(
        task_id="download_dataset",
        bash_command=(
            "python /opt/airflow/tasks/download.py"
            " --work-dir /opt/airflow/data"
            " --hf-token {{ var.value.hf_token }}"
        ),
        execution_timeout=timedelta(hours=6),
    )

    analyse_text = SparkSubmitOperator(
        task_id="analyse_text",
        application="/opt/airflow/tasks/analyse-text.py",
        conn_id="spark_default",
        application_args=[
            "--manifest", "/opt/app/data/manifests/raw.jsonl",
            "--out-dir", "/opt/app/data/analysis/text",
        ],
        execution_timeout=timedelta(hours=4),
    )

    analyse_audio_pauses = BashOperator(
        task_id="analyse_audio_pauses",
        bash_command=(
            "python /opt/airflow/tasks/analyse-audio.py"
            " --manifest /opt/airflow/data/manifests/raw.jsonl"
            " --out-dir /opt/airflow/data/analysis/audio"
        ),
        execution_timeout=timedelta(hours=6),
    )

    report = BashOperator(
        task_id="report",
        bash_command=(
            "echo '=== Text Analysis ===' &&"
            " cat /opt/airflow/data/analysis/text/duration_stats.json &&"
            " cat /opt/airflow/data/analysis/text/correlation.json &&"
            " echo '=== Audio Pause Analysis ===' &&"
            " cat /opt/airflow/data/analysis/audio/pause_stats.json"
            " > /opt/airflow/data/analysis/report.txt"
        ),
        execution_timeout=timedelta(minutes=5),
    )

    download >> [analyse_text, analyse_audio_pauses] >> report
