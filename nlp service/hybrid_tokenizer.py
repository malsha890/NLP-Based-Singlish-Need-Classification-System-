from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
MAX_LEN = 64

def hybrid_encode(text, oov_threshold=0.4):
    ids = [tokenizer.bos_token_id]
    for word in text.strip().split():
        pieces = tokenizer.tokenize(word)
        fragmentation = len(pieces) / max(len(word), 1)
        if fragmentation > oov_threshold or tokenizer.unk_token in pieces:
            for ch in word:
                ch_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ch))
                ids.extend(ch_ids if ch_ids else [tokenizer.unk_token_id])
        else:
            ids.extend(tokenizer.convert_tokens_to_ids(pieces))
    ids.append(tokenizer.eos_token_id)
    ids = ids[:MAX_LEN]
    attention_mask = [1] * len(ids)
    pad_len = MAX_LEN - len(ids)
    ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len
    return ids, attention_mask

def baseline_encode(text):
    out = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LEN)
    return out["input_ids"], out["attention_mask"]
