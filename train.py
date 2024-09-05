# importing mlflow
import mlflow
from mlflow.tracking import MlflowClient
## virtual environment 
import argparse
import pandas as pd
from app.models.trainingModel import set_training_experiment
from app.data_pipelines import createTitanicPipeline

## dotenv
from dotenv import load_dotenv
import os
from datetime import datetime

def train(model_name, train_data_path, test_data_path):
    pipeline = createTitanicPipeline()
    set_training_experiment(model_name, pipeline,train_data_path, test_data_path)




if __name__ == "__main__":
    # Set MLflow tracking URI
    print(os.getcwd())
    parser = argparse.ArgumentParser()    
    parser.add_argument("--train_data_path", type=str, default="Data/train.csv",required=False, help="Path to the training data")
    parser.add_argument("--test_data_path", type=str, default="Data/test.csv", help="Testing Data")
    args = parser.parse_args()
    load_dotenv("config.env")
    experiment_name = os.getenv("train_experiment")
    mlflow_tracking_path = os.getenv("mlflow_train_tracking")
    mlflow.set_tracking_uri(mlflow_tracking_path)
    print(mlflow.get_tracking_uri())
    print("experiment_name", experiment_name, mlflow_tracking_path)
    model_name = os.getenv("model_name")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run: 
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("date",datetime.now().strftime("%Y-%m-%d %H:%M"))
        train(model_name, args.train_data_path, args.test_data_path)
        run_id = mlflow.active_run().info.run_id
        client = MlflowClient()
        mlflow.register_model(f"runs:/{run_id}/surivival_classifier", 
                              f"{experiment_name}_{model_name}_survival_classifier")
        print("loggingModel:", f"{experiment_name}_{model_name}_survival_classifier")
        model_metadata = client.get_latest_versions(f"{experiment_name}_{model_name}_survival_classifier")






