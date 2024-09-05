from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import optuna
import mlflow 

class PipedModel(ABC, BaseEstimator, TransformerMixin):
    def __init__(self, pipeline: Pipeline, **model_kwargs):
        self.pipeline =self.__prepare_pipeline(pipeline)
        self.set_model(self.pipeline, **model_kwargs)


    @abstractmethod
    def set_model(self, pipeline, **model_kwargs):
        '''
        Abstract Class, method should implement the model
        '''
        pass

    def set_parameters(self, **model_kwargs):
        self.set_model(self.pipeline, **model_kwargs)
    
    def __prepare_pipeline(self, pipeline):
        '''
        private method just for the inner use of the class
        '''
        return Pipeline(steps=[("preprocessing", pipeline),
                               ("model", None)])

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self
    
    @property
    def preprocessing_pipeline(self): 
        return self.pipeline.named_steps['preprocessing']
    
    @property
    @abstractmethod
    def model_name(self):
        """This method should be implemented"""
        pass

    def predict(self, X):
        return self.pipeline.predict(X)

    def transform(self, X):
        return self.pipeline.transform(X)
    
    def get_params(self, deep=True):
        return self.pipeline.get_params()
    
    def set_params(self, **params):
        self.pipeline = params.get("pipeline", self.pipeline)
        self.model_name = params.get("model_name", self.model_name)
        return self
    
    @abstractmethod
    def optimize_hyper_parameters(self, X_train, y_train, X_test, y_test, 
                                  optimization_score, n_trials=100): 
        '''
        This method should be implemented
        '''
        pass
    

class RandomForestPiped(PipedModel): 
    def set_model(self, pipeline, **model_kwargs):

        steps = list(pipeline.steps)
        steps[-1] = ("model", RandomForestClassifier(**model_kwargs))
        self.pipeline = Pipeline(steps)

    @property
    def model_name(self): 
        return "random_forest"

    def optimize_hyper_parameters(self, X_train, y_train, X_test, y_test, 
                                  optimization_score, n_trials=100): 
        def optimize_function(trial): 
            param_grid = {
                "n_estimators": trial.suggest_int('n_estimators', 10, 100),
                "max_depth": trial.suggest_int('max_depth', 2, 20),
                "min_samples_split": trial.suggest_int('min_samples_split', 2, 20),
                "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 20),
                "max_features": trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
            self.set_parameters(**param_grid)
            self.pipeline.fit(X_train, y_train)
            # generate prediction
            y_pred = self.predict(X_test)

            test_f1 = optimization_score(y_test, y_pred)
            return test_f1

        study = optuna.create_study(direction="maximize")
        study.optimize(optimize_function, n_trials=n_trials)
        optimal_parameters = study.best_params
        optimal_parameters["random_state"] = 33
        # model retraining
        self.set_parameters(**optimal_parameters)
        self.fit(X_train, y_train)
        return self


class XGBPiped(PipedModel): 
    def set_model(self, pipeline, **model_kwargs):
            
        steps = list(pipeline.steps)
        steps[-1] = ("model", xgb.XGBClassifier(**model_kwargs))
        self.pipeline = Pipeline(steps)
    @property
    def model_name(self): 
        return "xgboost"
    
    def optimize_hyper_parameters(self, X_train, y_train, X_test, y_test, 
                                  optimization_score, n_trials=100): 
        def optimize_function(trial): 
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
            self.set_parameters(**param_grid)
            self.pipeline.fit(X_train, y_train)
            # generate prediction
            y_pred = self.predict(X_test)

            test_f1 = optimization_score(y_test, y_pred)
            return test_f1

        study = optuna.create_study(direction="maximize")
        study.optimize(optimize_function, n_trials=n_trials)
        optimal_parameters = study.best_params
        optimal_parameters["random_state"] = 33
        # model retraining
        self.set_parameters(**optimal_parameters)
        self.fit(X_train, y_train)
        return self