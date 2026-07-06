from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "analyse_indicvoices_r_hindi",
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="analyse_indicvoices_r_hindi",
    description="Analyse IndicVoices-R Hindi dataset",
    default_args=default_args,
    start_date=datetime(2026, 7, 1),
    schedule=None,  # trigger manually; this is a one-shot data build
    catchup=False,
    max_active_runs=1,
    tags=["f5-tts", "IndicVoices-R", "hindi", "dataset"],
    doc_md=__doc__,
) as dag:
    download = BashOperator(
        task_id="download_dataset",
        bash_command=f"echo Downloading dataset",
        execution_timeout=timedelta(hours=12),
    )

    build_vocab = BashOperator(
        task_id="build_vocab",
        bash_command=f"Building vocabulary",
        execution_timeout=timedelta(hours=1),
    )

    data_distribution = BashOperator(
        task_id="data_distribution",
        bash_command=f"Finding data distribution",
        execution_timeout=timedelta(hours=1),
    )

    report = BashOperator(
        task_id="report",
        bash_command=f"Generating report as markdown file with relevant plots",
        execution_timeout=timedelta(hours=12),
    )

    download >> build_vocab >> data_distribution >> report
