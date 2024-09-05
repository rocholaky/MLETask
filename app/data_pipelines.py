import pandas as pd
import os
import argparse
import mlflow

### SKLEARN ###
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class PreFixedPipeline(TransformerMixin, BaseEstimator): 
    def __init__(self, pipeline): 
        super()
        self.pipeline = pipeline
    
    def fit(self, X, y=None): 
        # in this case do nothing as the pipe 
        # is already fitted
        return self

    def transform(self, X): 
        return self.pipeline.transform(X)

    def predict(self, X): 
        return self.pipeline.transform(X)
    

def createTitanicPipeline(): 

    # Defining the transformations: 
    age_imputer = SimpleImputer(strategy="median")
    embarked_imputer = SimpleImputer(strategy="most_frequent")
    one_hot_encoder = OneHotEncoder(drop="first") # we do this to avoid redundant features
    pass_through = FunctionTransformer()
    inputer= ColumnTransformer(
        transformers=[
            ("age_imputer", age_imputer, ["Age"]), # input age
            ("embarked_imputer", embarked_imputer, ["Embarked"])# input embarked
        ],
        remainder="passthrough", # let the other columns passthrough
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    preprocessor_transformer = ColumnTransformer(
        transformers=[
            ("one hot", one_hot_encoder, ["Embarked", "Sex", "Pclass"]),
            ("selector", pass_through, ["Age", "SibSp", "Parch", "Fare"])
        ],
        remainder="drop"
    )


    preprocessor_pipeline = Pipeline(steps =[
        ("input_variables", inputer), 
        ("preprocessing", preprocessor_transformer), 

    ])
    return preprocessor_pipeline
    #mlflow.sklearn.log_model(preprocessor_pipeline, "pipeline")
