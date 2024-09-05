# importing mlflow
import mlflow
from mlflow import mlflowClient
## virtual environment 
import argparse
import pandas as pd
from models.trainingModel import set_training_experiment
from data_pipelines import createTitanicPipeline

## dotenv
from dotenv import load_dotenv
import os

def train(model_name, train_data_path, test_data_path):
    pipeline = createTitanicPipeline()
    set_training_experiment(model_name, pipeline,train_data_path, test_data_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--train_data_path", type=str, default="Data/train.csv",required=False, help="Path to the training data")
    parser.add_argument("--test_data_path", type=str, default="Data/test.csv", help="Testing Data")
    args = parser.parse_args()
    load_dotenv("config.env")
    experiment_name = os.getenv("train_experiment")
    model_name = os.getenv("model_name")
    client= mlflowClient()
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run: 
        train(model_name, args.train_data_path, args.test_data_path)
    





