import pandas as pd
import os

### SKLEARN ###
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split

### METRICS  ######
from sklearn.metrics import accuracy_score
### MODELS: ######
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


