#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e
# start model
docker build -t mlflow/train -f DockerFile.train .
docker run -p 5000:5000 -d -v ./mlruns:/proj/mlruns  --name training_container mlflow/train

## start server
docker build -t mlflow/serve -f DockerFile.serve .
docker run -p 8000:8000 -d -v ./mlruns:/proj/mlruns  --name server_container mlflow/serve