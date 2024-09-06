import pandas as pd
import os
import mlflow
from app.models.util_func import get_system_usage

### SKLEARN ###
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
### METRICS  ######
from sklearn.metrics import accuracy_score
### MODELS: ######
import argparse
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

''''
This method is generated to evaluate the trained model 
on a different set of data of the titatnic
'''
def evaluate(model, X_test, y_test):
    # measure initial cpu, ram usage and time 
    cpu_start, ram_start, time_start = get_system_usage()
    # generate prediction
    y_pred = model.predict(X_test)
    # measure the final cpu, ram usage and time
    cpu_end, ram_end, time_end = get_system_usage()
    ## LOG METRICS: 
    recall_test = recall_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)
    accuracy_test = accuracy_score(y_test, y_pred)
    precision_test = precision_score(y_test, y_pred)

    # GET HARDWARE METRICS AND TIME
    cpu_usage_inference = cpu_end-cpu_start
    ram_usage_inference = ram_end - ram_start
    time_usage_inference = time_end - time_start
   

    # logging results
    mlflow.log_metric("f1", f1_test)
    mlflow.log_metric("recall", recall_test)
    mlflow.log_metric("precision", precision_test)
    mlflow.log_metric("accuracy", accuracy_test)
    mlflow.log_metric("CPU", cpu_usage_inference)
    mlflow.log_metric("RAM", ram_usage_inference)
    mlflow.log_metric("time", time_usage_inference)

def get_model(experiment_name, model_name):
    '''
    Function to get the model that is in the model registry
    '''
    client = MlflowClient()
    model_metadata = client.get_latest_versions(f"{experiment_name}_{model_name}_survival_classifier")
    run_id = model_metadata[0].run_id
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/survival_classifier")
    return model