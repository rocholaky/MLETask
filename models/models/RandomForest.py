import mlflow
from sklearn.ensemble import RandomForestClassifier
import optuna
import xgboost as xgb
from sklearn.metrics import recall_score, f1_score, precision_score
import numpy as np
import psutil
import time
import pandas as pd
from parameter_optimizer import rf_objective, xgb_objective
def get_system_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_info = psutil.virtual_memory()
    ram_percent = ram_info.percent
    start_time = time.time()
    return cpu_percent, ram_percent, start_time




def train_model(model_class, optimize_function, X_train, y_train, 
                        X_test, y_test, n_trials): 
    
    np.random.seed(55)
    cpu_initial, ram_initial, time_initial = get_system_usage()
    # model Optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_function, n_trials=n_trials)
    optimal_parameters = study.best_params
    optimal_parameters["random_state"] = 33
    # model retraining
    model = model_class(**optimal_parameters)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    
    ## metrics: 
    cpu_end, ram_end, time_end = get_system_usage()
    recall_test = recall_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)
    precision_test = precision_score(y_test, y_pred)
    cpu_usage = cpu_end-cpu_initial
    ram_usage = ram_end - ram_initial
    time_usage = time_end - time_initial


    # logging results
    mlflow.log_dict(study.best_params, "param.json")
    mlflow.log_metric("f1", f1_test)
    mlflow.log_metric("recall", recall_test)
    mlflow.log_metric("precision", precision_test)
    mlflow.log_metric("CPU", cpu_usage)
    mlflow.log_metric("RAM", ram_usage)
    mlflow.log_metric("time", time_usage)
    mlflow.sklearn.log_model("survival_classifier", model)

def set_training_experiment(model_name, train_path, test_path):
    if model_name == "random_forest":
        model_class = RandomForestClassifier
        optimize_function = rf_objective
    elif model_name == "xgboost": 
        model_class = xgb.XGBClassifier
        optimize_function = xgb_objective
    else: 
        ValueError(f"{model_name} is not one of the possible names")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X_train = train_data.drop(columns=["survived"])
    y_train = train_data["survived"]