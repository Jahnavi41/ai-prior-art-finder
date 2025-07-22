import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Load CSV
df = pd.read_csv("../backend/data/patents_sample.csv")

# Extract abstracts
abstracts = df["abstract"].fillna("").tolist()

# Load SBERT model
print("🔍 Loading Sentence-BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
print("⚙️ Generating embeddings...")
embeddings = model.encode(abstracts, show_progress_bar=True)

# Save to .npy
os.makedirs("../backend/embeddings", exist_ok=True)
np.save("../backend/embeddings/abstract_embeddings.npy", embeddings)
print("✅ Embeddings saved to backend/embeddings/abstract_embeddings.npy")
