# randomforest-api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
# Import the specific preprocessing function for this model
from data.preprocessing_speed import preprocess_data_speed, ALL_INTERSECTION_NAMES
import logging
import numpy as np # For potential NaN checks later

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Initialize FastAPI app ---
app = FastAPI(title="Random Forest Speed Prediction API")

# --- 2. Load the trained model ---
MODEL_PATH = "data/rf_speed_model.pkl"
model = None
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    # Consider exiting if the model fails to load in production

# --- 3. Define the input data structure ---
# Matches what the Node.js server sends
class Coordinates(BaseModel):
    lat: float
    lng: float

class PredictionInput(BaseModel):
    model: str # "randomforest"
    coordinates: Coordinates
    predictionTime: str # e.g., "Next Hour"
    # Add other optional fields if they become relevant later
    event: str | None = None

# --- 4. Define the prediction endpoint ---
@app.post("/predict/")
async def make_prediction(input_data: PredictionInput):
    logger.info(f"Received prediction request: {input_data.dict()}")

    if model is None:
        logger.error("Model is not loaded. Cannot make predictions.")
        raise HTTPException(status_code=500, detail="Model is not available")

    try:
        # --- a. Convert input to DataFrame ---
        # We need DateTime and a JunctionName for the current preprocessing
        data = {
            # Placeholder DateTime - Needs proper handling
            'DateTime': [pd.Timestamp.now()],
            # !! IMPORTANT TEMPORARY STEP !!
            # Map coordinates to the CLOSEST known JunctionName.
            # This requires a mapping function (e.g., using distances).
            # For now, HARDCODING a default for testing:
            'JunctionName': ['Intersection_Trinity Circle'] # HARDCODED - NEEDS PROPER MAPPING
        }
        input_df = pd.DataFrame(data)

        # --- b. Preprocess the input data ---
        processed_df = preprocess_data_speed(input_df)
        logger.info(f"Preprocessed data for prediction: \n{processed_df}")
        logger.info(f"Columns sent to model: {processed_df.columns.tolist()}")

        # --- c. Make prediction ---
        prediction_raw = model.predict(processed_df)
        predicted_speed = prediction_raw[0] # Get the single prediction value
        logger.info(f"Raw model prediction (speed): {predicted_speed}")

        # --- d. Translate prediction to frontend format ---
        # Ensure speed is non-negative
        predicted_speed = max(0, predicted_speed)

        # --- e. Format the response ---
        # Focus on sending the avgSpeed as expected by the RF display function
        response_data = {
            "predictions": {
                # Add dummy congestion for compatibility if needed, but speed is key
                "congestion": {"level": 0.0, "label": "Unknown"},
                "avgSpeed": round(predicted_speed, 1) # Speed prediction
            },
            # We can add alternative route logic here later
            "alternativeRoute": None
        }
        logger.info(f"Sending response: {response_data}")
        return response_data

    except ValueError as ve: # Catch errors from preprocessing
        logger.error(f"Data preprocessing error: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {ve}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        # Log feature mismatch errors specifically
        if "feature_names mismatch" in str(e):
             logger.error(f"Feature mismatch detail: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --- 5. Add a simple root endpoint ---
@app.get("/")
def read_root():
    return {"message": "Random Forest Speed Prediction API is running!"}