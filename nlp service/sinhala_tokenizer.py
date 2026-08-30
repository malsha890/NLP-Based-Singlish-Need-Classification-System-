from transformers import AutoTokenizer

si_tokenizer = AutoTokenizer.from_pretrained("keshan/sinhala-roberta-oscar")
MAX_LEN = 64

def sinhala_encode(text_si):
    out = si_tokenizer(text_si, padding="max_length", truncation=True, max_length=MAX_LEN)
    return out["input_ids"], out["attention_mask"]
