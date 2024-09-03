import mlflow
import os
from dotenv import load_dotenv
from datetime import datetime
import argparse



def trainWorkFlow(train_data_path, test_data_path):
    load_dotenv("../config.env")

    experiment_name = os.getenv("train_experiment")
    model_name = os.getenv("model_name")
    mlflow.set_experiment(experiment_name)

    ## preprocessing of data: 
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        mlflow.log_param("run_date", datetime.now().strftime("%Y-%m-%d"))
        pipeline = mlflow.projects.run(uri="./", 
                                    entry_point="preprocessing", 
                                    parameters={
                                        "train_data_path": train_data_path
                                    })
        model = mlflow.projects.run(uri="./", 
                                    entry_point="training",
                                    parameters={
                                        "pipeline_path": pipeline, 
                                        "train_data_path": train_data_path, 
                                        "test_data_path": test_data_path
                                    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the testing data")
    parser.add_argument("--evaluation_data_path", type=str, required=False,default="Data/test.csv",  help="Path to the evaluation data")
    args = parser.parse_args()

    trainWorkFlow(args.train_data_path, args.test_data_path, args.evaluation_data_path)