from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import argparse
from app.models.evaluation import evaluate
import os
import pandas as pd

if __name__ == "__main__":

    '''
    Script to start the evaluation process
    '''
    parser = argparse.ArgumentParser()    
    parser.add_argument("--data_path", type=str, default="Data/test.csv",required=False, help="Path to the training data")
    args = parser.parse_args()
    load_dotenv("config.env")
    experiment_name = os.getenv("train_experiment")
    model_name= os.getenv("model_name")

    try: 
        mlflow.set_experiment(experiment_name)
        client = MlflowClient()
        model_metadata = client.get_latest_versions(f"{experiment_name}_{model_name}_survival_classifier")
        run_id = model_metadata[0].run_id
        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/survival_classifier")
    except: 
        raise ValueError(f"There is no model in staging called {model_name}_survival_classifier please run the train rutine")
    # load the data
    data = pd.read_csv(args.data_path)
    X = data.drop(columns=["Survived"])
    y= data["Survived"]
    
    # get evaluation
    with mlflow.start_run(): 
        mlflow.set_tag("model", mlflow.get_run(run_id).data.tags["model"])
        mlflow.set_tag("date",datetime.now().strftime("%Y-%m-%d %H:%M"))
        evaluate(model, X, y)