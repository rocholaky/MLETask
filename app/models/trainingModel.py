import mlflow

from sklearn.pipeline import Pipeline
import optuna
from models.piped_models import RandomForestPiped, XGBPiped
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
import numpy as np
import psutil
import time
import pandas as pd
def get_system_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_info = psutil.virtual_memory()
    ram_percent = ram_info.percent
    start_time = time.time()
    return cpu_percent, ram_percent, start_time


class pipedWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)

def train_model(model, X_train, y_train, 
                        X_test, y_test, n_trials=100): 
    
    np.random.seed(55)
    cpu_initial, ram_initial, time_initial = get_system_usage()
    # model Optimization
    model = model.optimize_hyper_parameters(X_train, y_train, X_test, y_test, 
                                      f1_score)
    ## metrics: 
    cpu_end, ram_end, time_end = get_system_usage()
    y_pred = model.predict(X_test)
    recall_test = recall_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)
    accuracy_test = accuracy_score(y_test, y_pred)
    precision_test = precision_score(y_test, y_pred)
    cpu_usage = cpu_end-cpu_initial
    ram_usage = ram_end - ram_initial
    time_usage = time_end - time_initial


    # logging results
    mlflow.log_param(model.get_params(), "param.json")
    mlflow.log_metric("f1", f1_test)
    mlflow.log_metric("recall", recall_test)
    mlflow.log_metric("precision", precision_test)
    mlflow.log_metric("CPU", cpu_usage)
    mlflow.log_metric("RAM", ram_usage)
    mlflow.log_metric("time", time_usage)
    wrapped_model = pipedWrapper(model=model.pipeline)
    run_id = mlflow.active_run().info.run_id
    mlflow.pyfunc.save_model(path=f"runs:/{run_id}/survival_classifier", python_model=wrapped_model)

def set_training_experiment(model_name, pipeline_obj,train_path, test_path):
    if model_name == "random_forest":
        model= RandomForestPiped(pipeline_obj)
    elif model_name == "xgboost": 
        model = XGBPiped(pipeline_obj)
    else: 
        ValueError(f"{model_name} is not one of the possible names")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X_train = train_data.drop(columns=["Survived"])
    y_train = train_data["Survived"]
    X_test = test_data.drop(columns=["Survived"])
    y_test = test_data["Survived"]


    train_model(model,
                X_train, y_train, X_test, y_test)