from sklearn.ensemble import RandomForestClassifier
import optuna
import xgboost as xgb
import pandas as pd
from sklearn.metrics import f1_score

def train_model_for_opt(model, X_train, y_train, X_test, y_test):
     
    # fit the model
    model.fit(X_train, y_train)
    # generate prediction
    y_pred = model.predict(X_test, y_test)

    test_f1 = f1_score(y_test, y_pred)
    return test_f1

def rf_objective(trial): 
    param_grid = {
        "n_estimators": trial.suggest_int('n_estimators', 10, 100),
        "max_depth": trial.suggest_int('max_depth', 2, 20),
        "min_samples_split": trial.suggest_int('min_samples_split', 2, 20),
        "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 20),
        "max_features": trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
    }
    model = RandomForestClassifier(**param_grid)
    # initialize the model
    test_f1 = train_model_for_opt(model, X_train, y_train, X_test, y_test)
    return test_f1


def xgb_objective(trial):
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-4, 1e2, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1e2, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1e2, log=True),
    }
    model = xgb.XGBClassifier(**param_grid)
    # initialize the model
    test_f1 = train_model_for_opt(model, X_train, y_train, X_test, y_test)
    return test_f1