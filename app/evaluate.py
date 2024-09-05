import pandas as pd
import os
import mlflow
from models.util_func import get_system_usage

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
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def evaluate(model, X_test, y_test):
    cpu_start, ram_start, time_start = get_system_usage()
    y_pred = model.predict(X_test)
    cpu_end, ram_end, time_end = get_system_usage()
    recall_test = recall_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)
    accuracy_test = accuracy_score(y_test, y_pred)
    precision_test = precision_score(y_test, y_pred)
    cpu_usage_inference = cpu_end-cpu_start
    ram_usage_inference = ram_end - ram_start
    time_usage_inference = time_end - time_start
   

    # logging results
    mlflow.log_param(model.get_params(), "param.json")
    mlflow.log_metric("f1", f1_test)
    mlflow.log_metric("recall", recall_test)
    mlflow.log_metric("precision", precision_test)
    mlflow.log_metric("accuracy", accuracy_test)
    mlflow.log_metric("CPU", cpu_usage_inference)
    mlflow.log_metric("RAM", ram_usage_inference)
    mlflow.log_metric("time", time_usage_inference)
