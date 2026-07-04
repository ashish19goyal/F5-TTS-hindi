"""
Airflow DAG orchestrating IndicVoices-R (Hindi) preparation for F5-TTS.

Each pipeline stage of src/f5_tts/train/datasets/prepare_indicvoices_r.py runs as
its own Airflow task, so a failed stage can be retried without redoing the earlier
ones (all state lives in work_dir as WAVs + JSONL manifests).

Airflow Variables (Admin -> Variables):
    hf_token          Hugging Face token with access to ai4bharat/indicvoices_r (gated).
    work_dir          Shared filesystem dir for intermediate audio/manifests
                      (must be visible to all Spark executors). Default: /data/work
    out_dir           Final dataset dir (mel arrow shards, duration.json, vocab.txt).
                      Default: /data/indicvoices_r_hindi
    spark_master      Spark master URL. Default: local[*]
    repo_dir          Checkout of this repository on the Airflow workers.
                      Default: /opt/F5-TTS-hindi
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

REPO_DIR = "{{ var.value.get('ivr_repo_dir', '/opt/F5-TTS-hindi') }}"
SCRIPT = f"{REPO_DIR}/src/f5_tts/train/datasets/prepare_indicvoices_r.py"
WORK_DIR = "{{ var.value.get('ivr_work_dir', '/data/ivr_work') }}"
OUT_DIR = "{{ var.value.get('ivr_out_dir', '/data/indicvoices_r_hindi') }}"
SPARK_MASTER = "{{ var.value.get('ivr_spark_master', 'local[*]') }}"

COMMON_ARGS = f"--work-dir {WORK_DIR}"

# Spark stages are submitted to the cluster; executors resolve f5_tts and the
# processing deps from their own environment (bake them into the worker image).
SPARK_SUBMIT = (
    f"cd {REPO_DIR} && "
    f"spark-submit --master {SPARK_MASTER} "
    f"--conf spark.executorEnv.PYTHONPATH={REPO_DIR}/src "
    f"{SCRIPT}"
)
PYTHON_RUN = f"cd {REPO_DIR} && PYTHONPATH={REPO_DIR}/src python {SCRIPT}"

default_args = {
    "owner": "prepare_indicvoices_r",
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="prepare_indicvoices_r_hindi",
    description="Download + preprocess IndicVoices-R Hindi into F5-TTS training format",
    default_args=default_args,
    start_date=datetime(2026, 7, 1),
    schedule=None,  # trigger manually; this is a one-shot data build
    catchup=False,
    max_active_runs=1,
    tags=["f5-tts", "hindi", "dataset"],
    doc_md=__doc__,
) as dag:
    download = BashOperator(
        task_id="download_dataset",
        bash_command=f"{PYTHON_RUN} download {COMMON_ARGS} --hf-token {{{{ var.value.hf_token }}}}",
        execution_timeout=timedelta(hours=12),
    )

    demix = BashOperator(
        task_id="audio_stage1_demix_htdemucs",
        bash_command=f"{SPARK_SUBMIT} demix {COMMON_ARGS}",
        execution_timeout=timedelta(hours=24),
    )

    dereverb = BashOperator(
        task_id="audio_stage2_dereverb_voicefixer",
        bash_command=f"{SPARK_SUBMIT} dereverb {COMMON_ARGS}",
        execution_timeout=timedelta(hours=24),
    )

    enhance = BashOperator(
        task_id="audio_stage3_enhance_deepfilternet3",
        bash_command=f"{SPARK_SUBMIT} enhance {COMMON_ARGS}",
        execution_timeout=timedelta(hours=24),
    )

    quality_filter = BashOperator(
        task_id="audio_stage4_quality_filter",
        bash_command=f"{SPARK_SUBMIT} quality_filter {COMMON_ARGS}",
        execution_timeout=timedelta(hours=12),
    )

    standardize = BashOperator(
        task_id="audio_stage5_standardize",
        bash_command=f"{SPARK_SUBMIT} standardize {COMMON_ARGS}",
        execution_timeout=timedelta(hours=6),
    )

    text_process = BashOperator(
        task_id="text_preprocess",
        bash_command=f"{SPARK_SUBMIT} text_process {COMMON_ARGS}",
        execution_timeout=timedelta(hours=2),
    )

    build_vocab = BashOperator(
        task_id="build_vocab",
        bash_command=f"{PYTHON_RUN} build_vocab {COMMON_ARGS}",
        execution_timeout=timedelta(hours=1),
    )

    package = BashOperator(
        task_id="package_arrow_dataset",
        bash_command=f"{SPARK_SUBMIT} package {COMMON_ARGS} --out-dir {OUT_DIR}",
        execution_timeout=timedelta(hours=12),
    )

    download >> demix >> dereverb >> enhance >> quality_filter >> standardize >> text_process >> build_vocab >> package
