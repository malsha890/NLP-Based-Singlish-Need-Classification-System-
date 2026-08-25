import pandas as pd
import random

VOWELS = ['a', 'e', 'i', 'o', 'u']

SIMILAR_CONSONANTS = {
    'th': 't', 't': 'th',
    'd': 'dh', 'dh': 'd',
    'k': 'c', 'c': 'k',
    'v': 'w', 'w': 'v',
}

def double_vowel(word):
    positions = [i for i, ch in enumerate(word) if ch in VOWELS]
    if not positions:
        return word
    i = random.choice(positions)
    return word[:i] + word[i] + word[i:]

def drop_vowel(word):
    positions = [i for i, ch in enumerate(word) if ch in VOWELS and i != 0]
    if not positions:
        return word
    i = random.choice(positions)
    return word[:i] + word[i+1:]

def swap_consonant(word):
    for original, replacement in SIMILAR_CONSONANTS.items():
        if original in word:
            return word.replace(original, replacement, 1)
    return word

def apply_random_transformation(word):
    transformation = random.choice([double_vowel, drop_vowel, swap_consonant])
    return transformation(word)

def generate_variant(sentence):
    words = sentence.split()
    if len(words) == 0:
        return sentence
    num_words_to_change = random.randint(1, min(2, len(words)))
    indices_to_change = random.sample(range(len(words)), num_words_to_change)
    for i in indices_to_change:
        words[i] = apply_random_transformation(words[i])
    return " ".join(words)

def messages_augmented(input_csv, output_csv, variants_per_message=4):
    df = pd.read_csv(input_csv)
    augmented_rows = []

    for _, row in df.iterrows():
        augmented_rows.append({"text": row["text"], "category": row["category"], "source": "original"})

    for _, row in df.iterrows():
        for _ in range(variants_per_message):
            variant_text = generate_variant(row["text"])
            augmented_rows.append({"text": variant_text, "category": row["category"], "source": "synthetic"})

    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df.to_csv(output_csv, index=False)
    print(f"Original messages: {len(df)}")
    print(f"Total after augmentation: {len(augmented_df)}")

if __name__ == "__main__":
    messages_augmented("../data/train_original.csv", "../data/train_augmented.csv", variants_per_message=4)