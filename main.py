from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

class TextInput(BaseModel):
    text: str

app = FastAPI()

sarcasm_classifier = pipeline("text-classification", model="helinivan/english-sarcasm-detector")

@app.get("/")
async def root():
    return {"message": "Sarcasm detection API is running"}

@app.post("/predict")
async def predict(input: TextInput):
    text = input.text
    if not text:
        raise HTTPException(status_code=400, detail='Please provide input text.')

    result = sarcasm_classifier(text)[0]  
    label = result["label"]
    confidence = round(result["score"], 4)

    label_map = {
    "LABEL_0": "sarcastic 😏",
    "LABEL_1": "not sarcastic 😊",
    "sarcastic": "sarcastic 😏",
    "not_sarcastic": "not sarcastic 😊",
    "POSITIVE": "sarcastic 😏",
    "NEGATIVE": "not sarcastic 😊"
    }


    readable_label = label_map.get(label, label.lower())

    return {
    "prediction": readable_label,
    "confidence": confidence,
    "raw_label": label  
     }

