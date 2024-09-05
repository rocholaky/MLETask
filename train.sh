#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Run the train.py script
echo "Starting training..."
python train.py
echo "Training complete."

# Run the evaluate.py script
echo "Starting evaluation..."
python evaluate.py
echo "Evaluation complete."

# Start the MLflow UI
echo "Starting MLflow UI..."
mlflow ui --host 0.0.0.0 --port 5000
