import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("baseline_tokenizer.model")

def build_char_vocab(train_txt_path):
    chars = set()
    with open(train_txt_path, encoding="utf-8") as f:
        for line in f:
            chars.update(line.strip())
    vocab = {ch: i for i, ch in enumerate(sorted(chars))}
    vocab["<unk>"] = len(vocab)
    return vocab

CHAR_VOCAB = build_char_vocab("../data/train.txt")

def hybrid_tokenize(text, oov_threshold=0.4):
    token_ids = []
    for word in text.strip().split():
        pieces = sp.encode(word, out_type=str)
        fragmentation = len(pieces) / max(len(word), 1)
        if fragmentation > oov_threshold or "<unk>" in pieces:
            token_ids.extend(CHAR_VOCAB.get(c, CHAR_VOCAB["<unk>"]) for c in word)
        else:
            token_ids.extend(sp.encode(word, out_type=int))
    return token_ids

if __name__ == "__main__":
    variants = ["kiyanna", "kianna", "kynna", "kynn", "kiynna"]
    for word in variants:
        print(f"{word:10} -> {hybrid_tokenize(word)}")