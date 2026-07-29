#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <command> [options]

Commands:
  analyse         Run data analytics DAG on Airflow
  prepare         Run data preparation DAG on Airflow
  train           Run training DAG on Airflow
  eval            Run evaluation DAG on Airflow
  inference       Run inference (TTS generation)

Inference options (set via environment variables):
  MODEL           Model name (default: F5TTS_Hindi)
  REF_AUDIO       Reference audio file path
  REF_TEXT        Reference audio transcript
  GEN_TEXT        Text to synthesize (inline)
  GEN_FILE        File containing text to synthesize (overrides GEN_TEXT)
  OUTPUT_DIR      Output directory (default: tests)
  OUTPUT_FILE     Output filename (default: infer_out.wav)
  DEVICE          Device override (cuda/cpu)
  HF_CACHE_DIR    HuggingFace cache directory
  VOCAB_FILE      Path to custom vocab file
  CKPT_FILE       Path to model checkpoint
  SEED            Random seed for reproducibility

Examples:
  $0 inference
  REF_AUDIO=demo.wav REF_TEXT="..." GEN_TEXT="Hello world" $0 inference
EOF
    exit 1
}

log() {
    local msg="$1"
    printf "# %s\n" "${msg}"
}

# ---------------------------------------------------------------------------
# Pick docker compose CLI
# ---------------------------------------------------------------------------
if docker compose version >/dev/null 2>&1; then
    DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DC="docker-compose"
else
    echo "ERROR: docker compose is not installed." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Bring up the cluster
# ---------------------------------------------------------------------------
deploy_cluster() {
    ${DC} -f docker-compose.yml up -d
    log "Spark and Airflow clusters are deployed using docker containers."
}

# ---------------------------------------------------------------------------
# Run an airflow DAG on the Airflow cluster running in docker.
# Current directory is mounted to airflow container to detect dags automatically.
# ---------------------------------------------------------------------------
trigger_dag() {
    local dag_id="$1"
    local run_id="${dag_id}_$(date +%Y%m%d_%H%M%S)"

    log "Triggering Airflow DAG ${dag_id} on the cluster"

    # Wait for scheduler to parse and register DAG from filesystem before triggering it.
    for _ in $(seq 1 30); do
        if docker exec airflow-scheduler airflow dags list 2>/dev/null | grep -qw "${dag_id}"; then
            break
        fi
        sleep 5
    done

    docker exec airflow-scheduler airflow dags unpause "${dag_id}" >> /dev/null 2>&1
    docker exec airflow-scheduler airflow dags trigger "${dag_id}" --run-id "${run_id}" >> /dev/null 2>&1
    log "Triggered ${dag_id} on airflow. Check the run details on airflow UI at http://localhost:8081"
}

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
run_inference() {
    local py="python3"
    if ! command -v python3 >/dev/null 2>&1; then
        py="python"
    fi

    log "Running inference pipeline"
    log "  MODEL=${MODEL:-F5TTS_Hindi}"
    log "  REF_AUDIO=${REF_AUDIO:-}"
    log "  OUTPUT_DIR=${OUTPUT_DIR:-tests}"

    # Build environment variables to pass through
    export MODEL REF_AUDIO REF_TEXT GEN_TEXT GEN_FILE
    export OUTPUT_DIR OUTPUT_FILE DEVICE HF_CACHE_DIR
    export VOCAB_FILE CKPT_FILE SEED

    ${py} inferencing.py 2>&1
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
[ $# -ge 1 ] || usage

case "$1" in
    analyse|prepare|train|eval)
        deploy_cluster
        trigger_dag "${1}"
        ;;
    inference)
        shift
        run_inference "$@"
        ;;
    *)
        usage
        ;;
esac