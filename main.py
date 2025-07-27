from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Sarcasm Detection API",
    description="Detect whether a sentence is sarcastic or not using a BERT model.",
    version="1.0"
)

# Load sarcasm model once
sarcasm_model = pipeline("text-classification", model="manishiitg/sarcasm-detector-bert")

# Define request model
class InputText(BaseModel):
    sentence: str

# API endpoint
@app.post("/predict")
async def predict(input_text: InputText):
    sentence = input_text.sentence
    prediction = sarcasm_model(sentence)[0]
    label = "Sarcastic" if prediction['label'] == 'LABEL_1' else "Not Sarcastic"
    confidence = round(prediction['score'], 4)
    
    return {
        "sentence": sentence,
        "prediction": label,
        "confidence": confidence
    }

# Run using: uvicorn app:app --reload
