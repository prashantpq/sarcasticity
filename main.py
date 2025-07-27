from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")

sarcasm_classifier = pipeline("text-classification", model="nikesh66/Sarcasm-Detection-using-BERT")

class TextInput(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(input: TextInput):
    text = input.text
    if not text:
        raise HTTPException(status_code=400, detail='Please provide input text.')

    result = sarcasm_classifier(text)[0]
    label = result["label"]
    confidence = round(result["score"], 4)

    label_map = {
        "0": "Not Sarcasm 😊",
        "1": "Sarcasm 🙄",
        "Not Sarcasm": "Not Sarcasm 😊",
        "Sarcasm": "Sarcasm 🙄",
        "LABEL_0": "Not Sarcasm 😊",
        "LABEL_1": "Sarcasm 🙄"
    }

    readable_label = label_map.get(label, label)

    return {
        "prediction": readable_label,
        "confidence": confidence,
        "raw_label": label
    }
