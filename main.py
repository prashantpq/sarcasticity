from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline

# Load pretrained sarcasm detection model (RoBERTa base fine-tuned)
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sarcasm")

# FastAPI app
app = FastAPI(title="Sarcasm Detection API")

# Input model
class TextInput(BaseModel):
    sentence: str

# Routes
@app.get("/")
async def root():
    return {"message": "Sarcasm Detection API is running!"}

@app.post("/predict")
async def predict(input: TextInput):
    result = classifier(input.sentence)[0]
    label = result["label"]  # 'LABEL_1' for sarcastic, 'LABEL_0' for not
    score = round(result["score"], 4)
    is_sarcastic = "Yes" if label == "LABEL_1" else "No"
    
    return {
        "sentence": input.sentence,
        "sarcastic": is_sarcastic,
        "confidence": score
    }
