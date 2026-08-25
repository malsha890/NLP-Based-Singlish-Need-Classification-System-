import pandas as pd
import re

df = pd.read_csv("../data/disaster_need_messages.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()      # collapse multiple spaces
    text = re.sub(r'!{2,}', '!', text)             # !!! -> !
    text = re.sub(r'\?{2,}', '?', text)            # ??? -> ?
    text = re.sub(r'\.{2,}', '.', text)            # ... -> .
    return text

df["text"] = df["text"].apply(clean_text)
df.to_csv("../data/disaster_need_messages_cleaned.csv", index=False)
print(f"Cleaned {len(df)} messages")
print(df.head())