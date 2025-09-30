import pandas as pd

df = pd.read_csv("../datasets/scenarios_cleaned.csv")
df[["category", "full_prompt"]].to_csv("../datasets/full_dataset.csv", index=False)
train = df.groupby("category").sample(16)
train = train[["category", "full_prompt"]]

test = df.drop(train.index)

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)


