import numpy as np
from fastapi import APIRouter
from tensorflow.keras.models import load_model
from src.utils.img_preprocessing import img_preprocessing

router = APIRouter()

# Load the trained model
Model = load_model("src/models/finger_count_model1.h5")


@router.post("/predicts")
def predict_img_label(payload: dict):
    # Get the base64-encoded image from the request and process it
    input_tensor = img_preprocessing(payload["imgBase64"])

    # Make a class prediction using the model
    prediction = Model.predict(input_tensor)
    predicted_class = np.argmax(prediction)
    return {"number_of_fingers": int(predicted_class)}
