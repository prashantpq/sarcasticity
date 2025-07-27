from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

sarcasm_classifier = pipeline("text-classification", model="mrm8488/t5-base-finetuned-sarcasm-twitter")
class TextINput(BaseModel):
    sentence : str

app = FastAPI()

@app.get('/')
async def root():
    return {'message' : 'Sarcasm detection API is running'}

@app.post('/predict')
def predict_sarcasm(input_text: TextINput):
    result = sarcasm_classifier(input_text.sentence)[0]
    label = result['label']
    score = result['score']
    
    return {
        "sentence": input_text.sentence,
        "prediction": "sarcastic" if "sarcasm" in label.lower() else "not sarcastic",
        "confidence": round(score, 3)
    }


