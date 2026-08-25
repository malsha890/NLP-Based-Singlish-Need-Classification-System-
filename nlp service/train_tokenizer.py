import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="../data/train.txt",
    model_prefix="baseline_tokenizer",
    vocab_size=800,
    model_type="unigram",
    character_coverage=1.0,
)
print("Done — baseline_tokenizer.model and baseline_tokenizer.vocab created")