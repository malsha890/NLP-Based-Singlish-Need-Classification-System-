import pandas as pd
df = pd.read_csv("../data/train_augmented.csv")
with open("../data/train.txt", "w", encoding="utf-8") as f:
    for text in df["text"]:
        f.write(str(text) + "\n")
print(f"Wrote {len(df)} lines")