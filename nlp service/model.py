from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn

NEED_KEYWORDS = ["one", "epa", "please", "ikmanin", "ikmanata", "help",
                  "urgent", "puluwanda", "asaneepa",
                  "kadila", "thuwala", "amarui", "ambulance", "doctor", "beheth", "ospital"]

def get_aux_features(text):
    words = text.lower().split()
    length_feature = min(len(words) / 20, 1.0)
    keyword_hits = sum(1 for w in words if w in NEED_KEYWORDS)
    keyword_feature = min(keyword_hits / 3, 1.0)
    return [length_feature, keyword_feature]

class SinglishNeedClassifier(nn.Module):
    def __init__(self, base_model_name="xlm-roberta-base", num_categories=5, aux_feature_dim=2):
        super().__init__()
        base = AutoModel.from_pretrained(base_model_name)
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
        )
        self.encoder = get_peft_model(base, lora_config)
        hidden = base.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden + aux_feature_dim, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_categories),
        )

    def forward(self, input_ids, attention_mask, aux_features):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        combined = torch.cat([pooled, aux_features], dim=1)
        return self.classifier(combined)
