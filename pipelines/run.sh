#!/usr/bin/env bash

set -euo pipefail

LOG_FILE="results.log"
DATA_PREPARATION_DAG="prepare_indicvoices_r_hindi"
ANALYTICS_DAG="analyse_indicvoices_r_hindi"
TRAINING_DAG="train_f5tts_hindi"
EVAL_DAG="eval_f5TTS_hindi"
INFERENCING_SCRIPT="inferencing_f5tts_hindi"

log() {
    local msg="$1"
    local log_file="${2:-${LOG_FILE}}"
    printf "\n\n# ${msg}\n" | tee -a "${log_file}"
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
${DC} -f docker-compose.yml up -d
log "Spark and Airflow clusters are deployed using docker containers."

# ---------------------------------------------------------------------------
# Run airflow dag on the Airflow cluster running in docker
# ---------------------------------------------------------------------------

log "Triggering Airflow DAG to prepare indicVoices-R dataset on the cluster"

# Wait for scheduler to parse and register DAG from filesystem before triggering it.
for _ in $(seq 1 30); do
    if docker exec airflow-scheduler airflow dags list 2>/dev/null | grep -qw "${DAG_ID}"; then
        break
    fi
    sleep 5
done


# ---------------------------------------------------------------------------
# Current directory is mounted to airflow container to detect dags automatically
# ---------------------------------------------------------------------------
docker exec airflow-scheduler airflow dags unpause "${DAG_ID}" >> "${LOG_FILE}" 2>&1
docker exec airflow-scheduler airflow dags trigger "${DAG_ID}" --run-id "${RUN_ID}" >> "${LOG_FILE}" 2>&1

log "Done"