from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat, PositiveFloat
from typing import List, Optional, Literal
import joblib
import pandas as pd
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
import os
import uvicorn
from app.models.evaluation import get_model


## load environmental variables
load_dotenv("config.env")
experiment_name = os.getenv("train_experiment")
model_name= os.getenv("model_name")
mlflow_tracking_path = os.getenv("mlflow_train_tracking")
mlflow.set_tracking_uri(mlflow_tracking_path)

# start server
app = FastAPI()


# Define the passanger body
# this represents a row in the Passenger 
class Passenger(BaseModel):
    PassengerId: str
    Pclass: conint(ge=1, le=3)
    Name: str
    Sex: Literal["male", "female"]
    Age: Optional[confloat(gt=0, lt=120)] = None
    SibSp: conint(ge=0)
    Parch: conint(ge=0)
    Fare: PositiveFloat
    Cabin: Optional[str] = None
    Embarked: Optional[Literal["S", "C", "Q"]]
    Ticket: Optional[str] =None

    
# this represents a list of passengers
class PassengerList(BaseModel):
    passengers: List[Passenger]


## post method to predict
@app.post("/predict/")
async def predict_survival(passenger_list: PassengerList):
    print("Experiment", mlflow_tracking_path, experiment_name)
    # Convert the list of passengers to a DataFrame
    df = pd.DataFrame([passenger.model_dump() for passenger in passenger_list.passengers])
    
    try: 
        model = get_model(experiment_name, model_name)
    except: 
        raise HTTPException(code=400, detail=f"There is no model in staging called {model_name}_survival_classifier please run the train rutine")
    
    # Make predictions
    predictions = model.predict(df)
    
    # Create a response with PassengerId and prediction
    results = [{"PassengerId": passenger.PassengerId, "Survived": bool(pred)} 
               for passenger, pred in zip(passenger_list.passengers, predictions)]
    
    return results


    
if __name__=="__main__":
     uvicorn.run(app, host='127.0.0.1', port=8000, workers=5 )