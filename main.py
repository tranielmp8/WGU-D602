#!/usr/bin/env python
# coding: utf-8

# import statements
from fastapi import FastAPI, HTTPException
import json
from pydantic import BaseModel
import numpy as np
import pickle
from datetime import datetime

# Load airport encodings and the model
with open('airport_encodings.json', 'r') as f:
    airports = json.load(f)

with open('finalized_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
print(model)

def create_airport_encoding(airport: str, airports: dict) -> np.array:
    """
    create_airport_encoding is a function that creates an array the length of all arrival airports from the chosen
    departure aiport.  The array consists of all zeros except for the specified arrival airport, which is a 1.  

    Parameters
    ----------
    airport : str
        The specified arrival airport code as a string
    airports: dict
        A dictionary containing all of the arrival airport codes served from the chosen departure airport
        
    Returns
    -------
    np.array
        A NumPy array the length of the number of arrival airports.  All zeros except for a single 1 
        denoting the arrival airport.  Returns None if arrival airport is not found in the input list.
        This is a one-hot encoded airport array.

    """
    temp = np.zeros(len(airports))
    if airport in airports:
        temp[airports.get(airport)] = 1
        temp = temp.T
        return temp
    else:
        return None

# TODO:  write the back-end logic to provide a prediction given the inputs
# requires finalized_model.pkl to be loaded 
# the model must be passed a NumPy array consisting of the following:
# (polynomial order, encoded airport array, departure time as seconds since midnight, arrival time as seconds since midnight)
# the polynomial order is 1 unless you changed it during model training in Task 2
# YOUR CODE GOES HERE
def time_to_seconds(time_str: str) -> int:
    """
    Convert time in HH:MM format to seconds since midnight.

    Parameters:
    - time_str (str): Time in HH:MM format.

    Returns:
    - int: Seconds since midnight.

    Raises:
    - ValueError: If the input string is not in HH:MM format.
    """
    from datetime import datetime
    try:
        time_obj = datetime.strptime(time_str, "%H:%M")
        return time_obj.hour * 3600 + time_obj.minute * 60
    except ValueError:
        raise ValueError("Time must be in HH:MM format")

# TODO:  write the API endpoints.  
# YOUR CODE GOES HERE

# Initialize FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "API is functional"} 
  
# Define the request body schema
class DelayPredictionRequest(BaseModel):
    arrival_airport: str
    departure_airport: str
    departure_time: str
    arrival_time: str
    
    
@app.get("/predict/delays")
async def predict_delays(request: DelayPredictionRequest):
    try:
        # Extract data from request
        arrival_airport = request.arrival_airport.upper()
        departure_airport = request.departure_airport.upper()
        departure_time = request.departure_time.upper()
        arrival_time = request.arrival_time.upper()
        
        # Explicitly validate the length of the airport codes
        if len(departure_airport) != 3 or len(arrival_airport) != 3:
          raise HTTPException(status_code=400, detail="Airport Code should = 3 characters ")
        
        if request.arrival_airport.upper() not in airports:
            raise HTTPException(status_code=400, detail="Invalid arrival airport code")
        
        encoded_airport = np.zeros(len(airports))
        encoded_airport[airports[arrival_airport]] = 1
        

        dep_time_seconds = time_to_seconds(departure_time)
        arr_time_seconds = time_to_seconds(arrival_time)

        # Create feature array
        polynomial_order = 1
        features = np.concatenate((
            [polynomial_order], encoded_airport, [dep_time_seconds, arr_time_seconds]
        ))

        # Make prediction
        prediction = model.predict([features])

        # Convert prediction to JSON-serializable type
        # Construct the response with all details
        return {
            "arrival_airport": arrival_airport,
            "departure_airport": departure_airport,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "average_departure_delay_minutes": float(prediction[0])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))