# importing mlflow
import mlflow

### importing sklearn libraries
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

## virtual environment 
from dotenv import load_doatenv
import os
load_doatenv("../config.env")

experiment_name = os.getenv("train_experiment")
model_name = os.getenv("model_name")
mlflow.set_experiment(experiment_name)






