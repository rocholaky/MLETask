from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
from dotenv import load_dotenv
import mlflow
import uvicorn


load_dotenv("config.env")

app = FastAPI()


# Define the request model
class Passenger(BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: Optional[float] = None
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: Optional[str] = None
    Embarked: str

class PassengerList(BaseModel):
    passengers: List[Passenger]

@app.post("/predict/")
async def predict_survival(passenger_list: PassengerList):
    # Convert the list of passengers to a DataFrame
    df = pd.DataFrame([passenger.model_dump() for passenger in passenger_list.passengers])
    
    # Prepare the features
    df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    
    
    # Make predictions
    predictions = model.predict(df)
    
    # Create a response with PassengerId and prediction
    results = [{"PassengerId": passenger.PassengerId, "Survived": bool(pred)} 
               for passenger, pred in zip(passenger_list.passengers, predictions)]
    
    return results


    
if __name__=="__main__":
     uvicorn.run(app, host='127.0.0.1', port=8000, workers=5 )