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

# Prepare structured text records
print("📦 Structuring data...")
text_records = []

for _, row in df.iterrows():
    record = {
        "publication": row.get("publication_number", "N/A"),
        "abstract": row.get("abstract", "")
    }
    text_records.append(record)

# Generate embeddings for abstracts only
print("⚙️ Generating embeddings...")
abstracts = [record["abstract"] for record in text_records]
embeddings = model.encode(abstracts, batch_size=32, show_progress_bar=True)

# Save embeddings and text records
output_dir = "../backend/embeddings"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "minilm_embeddings.npy"), embeddings)
np.save(os.path.join(output_dir, "minilm_texts.npy"), text_records, allow_pickle=True)

print("✅ Saved structured texts and embeddings.")
