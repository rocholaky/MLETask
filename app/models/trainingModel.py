import mlflow

from sklearn.pipeline import Pipeline
from app.models.piped_models import RandomForestPiped, XGBPiped
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from app.models.util_func import get_system_usage
from evaluate import evaluate
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
    cpu_usage = cpu_end-cpu_initial
    ram_usage = ram_end - ram_initial
    time_usage = time_end - time_initial
    mlflow.log_dict(model.get_params(), "param.json")
    mlflow.log_metric("CPU", cpu_usage)
    mlflow.log_metric("RAM", ram_usage)
    mlflow.log_metric("time", time_usage)
    evaluate(model, X_test, y_test)
    mlflow.pyfunc.log_model(artifact_path="survival_classifier", python_model=pipedWrapper(model)
                            )


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

    mlflow.log_param("columns", train_data.columns)
    train_model(model,
                X_train, y_train, X_test, y_test)
    