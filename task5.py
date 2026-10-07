import pandas as pd

df = pd.read_csv("signs.csv")
print(df)
fire_signs = df[df["element"] == "Fire"]
print(fire_signs)
counts = df.groupby("element").size()
print(counts)