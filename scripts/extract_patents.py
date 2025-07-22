import pandas as pd
import os

DATASET_PATH = "C:\\Users\\Jahnavi Undela\\Downloads\\archive (2)\\train_mini.csv"

df = pd.read_csv(DATASET_PATH, usecols=["publication_number", "abstract"])
print(f"🔹 Loaded dataset with {len(df)} rows")

df.dropna(subset=["abstract"], inplace=True)
df.drop_duplicates(subset=["abstract"], inplace=True)
print(f"✅ Cleaned dataset → {len(df)} rows remaining")

sample_size = min(1000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

OUTPUT_PATH = os.path.join("..", "backend", "data", "patents_sample.csv")
df_sample.to_csv(OUTPUT_PATH, index=False)
print(f"📁 Saved sample to {OUTPUT_PATH}")
