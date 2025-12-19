from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

MODEL_NAME = "ric4j4/bert-fake-review"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

class TextInput(BaseModel):
    text: str

label_map = {0: "CG", 1: "OR"}

@app.post("/predict")
def predict(data: TextInput):
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return {
        "label": label_map[pred],
        "confidence": float(probs[0][pred])
    }
