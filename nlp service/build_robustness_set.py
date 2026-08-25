import pandas as pd
from augment_data import generate_variant

test_df = pd.read_csv("../data/test_original.csv")

clean_test = test_df.sample(frac=0.5, random_state=42)
robustness_source = test_df.drop(clean_test.index)

robustness_test = robustness_source.copy()
robustness_test["text"] = robustness_test["text"].apply(generate_variant)

clean_test.to_csv("../data/test_clean.csv", index=False)
robustness_test.to_csv("../data/test_robustness.csv", index=False)
print(f"Clean test: {len(clean_test)}, Robustness test: {len(robustness_test)}")