#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <command>

Commands:
  analyse         Run data analytics DAG on Airflow
  prepare-data  Run data preparation DAG on Airflow
  train          Run training DAG on Airflow
  eval        Run evaluation DAG on Airflow
  inference       Run the inferencing python script
EOF
    exit 1
}

log() {
    local msg="$1"
    printf "# ${msg}\n"
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
# Dispatch
# ---------------------------------------------------------------------------
[ $# -eq 1 ] || usage

if $1 == "inference"; then
    log "Running inferencing script ${INFERENCING_SCRIPT}"
    python3 inferencing.py 2>&1
else
    deploy_cluster
    trigger_dag "${1}"
fi