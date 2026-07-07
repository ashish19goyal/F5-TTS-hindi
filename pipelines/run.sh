#!/usr/bin/env bash

set -euo pipefail

DATA_PREPARATION_DAG="prepare-data"
ANALYTICS_DAG="analyse"
TRAINING_DAG="train"
EVAL_DAG="eval"
INFERENCING_SCRIPT="inference.py"

usage() {
    cat <<EOF
Usage: $0 <command>

Commands:
  analytics         Run data analytics DAG on Airflow
  data-preparation  Run data preparation DAG on Airflow
  training          Run training DAG on Airflow
  evaluation        Run evaluation DAG on Airflow
  inferencing       Run the inferencing python script
EOF
    exit 1
}

log() {
    local msg="$1"
    printf "\n\n# ${msg}\n"
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

    docker exec airflow-scheduler airflow variables import /opt/airflow/variables.json >> /dev/null 2>&1
    docker exec airflow-scheduler airflow dags unpause "${dag_id}" >> /dev/null 2>&1
    docker exec airflow-scheduler airflow dags trigger "${dag_id}" --run-id "${run_id}" >> /dev/null 2>&1
    log "Triggered ${dag_id} on airflow. Check the run details on airflow UI at http://localhost:8081"
}

run_analytics() {
    deploy_cluster
    trigger_dag "${ANALYTICS_DAG}"
}

run_data_preparation() {
    deploy_cluster
    trigger_dag "${DATA_PREPARATION_DAG}"
}

run_training() {
    deploy_cluster
    trigger_dag "${TRAINING_DAG}"
}

run_evaluation() {
    deploy_cluster
    trigger_dag "${EVAL_DAG}"
}

run_inferencing() {
    log "Running inferencing script ${INFERENCING_SCRIPT}"
    python "${INFERENCING_SCRIPT}" 2>&1
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
[ $# -eq 1 ] || usage

case "$1" in
    analytics)        run_analytics ;;
    data-preparation) run_data_preparation ;;
    training)         run_training ;;
    evaluation)       run_evaluation ;;
    inferencing)      run_inferencing ;;
    *)                usage ;;
esac