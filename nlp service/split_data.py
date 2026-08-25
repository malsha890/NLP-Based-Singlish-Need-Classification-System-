import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/disaster_need_messages_cleaned.csv")

train_df, test_df = train_test_split(
    df, test_size=0.30, stratify=df["category"], random_state=42
)

train_df.to_csv("../data/train_original.csv", index=False)
test_df.to_csv("../data/test_original.csv", index=False)

print(f"Train: {len(train_df)}")
print(train_df["category"].value_counts())
print(f"Test: {len(test_df)}")
print(test_df["category"].value_counts())