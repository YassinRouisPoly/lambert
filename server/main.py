from transformers import CamembertTokenizer, CamembertForMaskedLM
import torch
from pydantic import BaseModel
from fastapi import FastAPI

from model_predict import predict, load_model, is_loaded

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

load_model("v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # List of allowed origins
    allow_credentials=True,        # Allow cookies/authorization headers
    allow_methods=["*"],           # Allow all methods (GET, POST, PUT, etc.)
    allow_headers=["*"],           # Allow all headers
)

class TextInput(BaseModel):
    text: str

prediction_types = ["this", "next", "pendingNext"]

@app.get("/")
def root():
    return {"status": "API opérationnelle"}

@app.post("/get_prediction")
def get_prediction(input: TextInput):
    text = input.text

    prediction = predict(text)

    return {
        "preferredPrediction": prediction_types[prediction[0]],
        "predictions": {
            "this": prediction[1],
            "next": prediction[2],
            "pendingNext": prediction[2],
        },
        "updates": {
            "this": prediction[3],
            "next": prediction[4],
            "pendingNext": prediction[4],
        }
    }