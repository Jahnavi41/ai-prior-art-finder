import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# Load sample dataset
df = pd.read_csv("../backend/data/patents_sample.csv")

# Initialize MiniLM-L6-v2
print("🔍 Loading Sentence-BERT model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Generate embeddings
print("⚙️ Generating embeddings...")
abstracts = df["abstract"].tolist()
embeddings = model.encode(abstracts, batch_size=32, show_progress_bar=True)

# Save the embeddings
output_path = "../backend/embeddings/minilm_embeddings.npy"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.save(output_path, embeddings)
np.save("../backend/embeddings/minilm_texts.npy", abstracts)
print(f"✅ Embeddings saved to {output_path}")
