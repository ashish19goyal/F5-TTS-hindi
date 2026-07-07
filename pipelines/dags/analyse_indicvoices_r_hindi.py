from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "analyse_indicvoices_r_hindi",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="analyse_indicvoices_r_hindi",
    description="Analyse IndicVoices-R Hindi dataset",
    default_args=default_args,
    start_date=datetime(2026, 7, 1),
    schedule=None,  # trigger manually; this is a one-shot analysis run
    catchup=False,
    max_active_runs=1,
    tags=["f5-tts", "IndicVoices-R", "hindi", "dataset"],
    doc_md=__doc__,
) as dag:
    install_dependencies = BashOperator(
        task_id="install_dependencies",
        bash_command="pip install -q -r /opt/airflow/tasks/requirements.txt",
        execution_timeout=timedelta(hours=1),
    )

    download = BashOperator(
        task_id="download_dataset",
        bash_command=(
            "python /opt/airflow/tasks/download_indicvoices_r_hindi.py"
            " --work-dir {{ var.value.work_dir }}"
            " --hf-token {{ var.value.hf_token }}"
        ),
        execution_timeout=timedelta(hours=6),
    )

    analyse_text = BashOperator(
        task_id="analyse_text",
        bash_command=(
            "python /opt/airflow/tasks/analyse_text_indicvoices_r_hindi.py"
            " --manifest {{ var.value.work_dir }}/manifests/raw.jsonl"
            " --out-dir {{ var.value.work_dir }}/analysis/text"
        ),
        execution_timeout=timedelta(hours=4),
    )

    analyse_audio_pauses = BashOperator(
        task_id="analyse_audio_pauses",
        bash_command=(
            "python /opt/airflow/tasks/analyse_audio_pauses_indicvoices_r_hindi.py"
            " --manifest {{ var.value.work_dir }}/manifests/raw.jsonl"
            " --out-dir {{ var.value.work_dir }}/analysis/audio_pauses"
        ),
        execution_timeout=timedelta(hours=6),
    )

    report = BashOperator(
        task_id="report",
        bash_command=(
            "echo '=== Text Analysis ===' &&"
            " cat {{ var.value.work_dir }}/analysis/text/duration_stats.json &&"
            " cat {{ var.value.work_dir }}/analysis/text/correlation.json &&"
            " echo '=== Audio Pause Analysis ===' &&"
            " cat {{ var.value.work_dir }}/analysis/audio_pauses/pause_stats.json"
            " > {{ var.value.work_dir }}/analysis/report.txt"
        ),
        execution_timeout=timedelta(minutes=5),
    )

    install_dependencies >> download >> [analyse_text, analyse_audio_pauses] >> report
