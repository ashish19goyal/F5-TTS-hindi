# Automated Data preparation and training 

## Configuration
This setup uses mutliple tools for different purposes as listed below
- Apache airflow: To run pipelines for each of the objectives listed above.
- Apache spark: To run each of the tasks in the pipelines in a distributed manner. This helps in partitioning the data and avoiding out-of-memory errors
- MLFlow: To track multiple training runs with different configurations
- DVC: To store processed data and model checkpoints.
- docker: To deploy Apache airflow and spark clusters.

## Usage
This folder contains run.sh script. This script can be used to trigger
- Data analytics: `./run.sh analyitcs`
- Data preaparation: `./run.sh data-preparation`
- Training: `./run.sh training`
- Evaluation: `./run.sh evaluation`
- Inferencing: `./run.sh inferencing`