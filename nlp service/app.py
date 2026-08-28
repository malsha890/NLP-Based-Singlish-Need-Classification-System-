from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoModel
from peft import PeftModel

from hybrid_tokenizer import hybrid_encode
from model import get_aux_features

app = FastAPI()

# Allow the React dev server (localhost:5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CATEGORIES = ["medical aid", "shelter", "food/water", "rescue/missing", "other"]

CATEGORY_TO_AGENCY = {
    "medical aid": {"responder": "Ministry of Health / Red Cross", "priority": True},
    "shelter": {"responder": "DMC Local Office", "priority": False},
    "food/water": {"responder": "WFP / Partner NGOs", "priority": False},
    "rescue/missing": {"responder": "Tri Forces / Police", "priority": True},
    "other": {"responder": "No action required", "priority": False},
}

# ---- Load the trained model (base + LoRA adapter + classifier head) ----
print("Loading base XLM-R...")
base = AutoModel.from_pretrained("xlm-roberta-base")

print("Loading LoRA adapter...")
encoder = PeftModel.from_pretrained(base, "lora_adapter")
encoder.eval()

print("Loading classifier head...")
HIDDEN_SIZE = base.config.hidden_size  # 768
AUX_FEATURE_DIM = 2
classifier = nn.Sequential(
    nn.Linear(HIDDEN_SIZE + AUX_FEATURE_DIM, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, len(CATEGORIES)),
)
classifier.load_state_dict(torch.load("classifier_head.pt", map_location="cpu"))
classifier.eval()

print("Model ready.")

# ---- Request / response schemas ----
class NeedReport(BaseModel):
    text: str

class ClassificationResult(BaseModel):
    category: str
    confidence: float
    responder: str
    priority: bool
    all_scores: dict

# ---- Endpoint ----
@app.post("/classify", response_model=ClassificationResult)
def classify_report(report: NeedReport):
    input_ids, attention_mask = hybrid_encode(report.text)
    aux = get_aux_features(report.text)

    with torch.no_grad():
        out = encoder(
            input_ids=torch.tensor([input_ids]),
            attention_mask=torch.tensor([attention_mask]),
        )
        pooled = out.last_hidden_state[:, 0, :]
        combined = torch.cat([pooled, torch.tensor([aux], dtype=torch.float)], dim=1)
        logits = classifier(combined)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    category = CATEGORIES[pred_idx]
    mapping = CATEGORY_TO_AGENCY[category]
    all_scores = {CATEGORIES[i]: round(probs[i].item(), 3) for i in range(len(CATEGORIES))}

    return ClassificationResult(
        category=category,
        confidence=round(probs[pred_idx].item(), 3),
        responder=mapping["responder"],
        priority=mapping["priority"],
        all_scores=all_scores,
    )

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Singlish Need Classifier API is running"}