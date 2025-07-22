import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

CSV_PATH = "../data/patents_sample.csv"
EMBEDDINGS_PATH = "../embeddings/abstract_embeddings.npy"
TOP_K = 5

def load_data():
    print("📥 Loading dataset and embeddings...")
    df = pd.read_csv(CSV_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    return df, embeddings

def embed_input(text, model):
    print("🧠 Embedding input abstract...")
    return model.encode(text, convert_to_tensor=True)

def find_similar_abstracts(input_text, df, embeddings, model, top_k=TOP_K):
    input_embedding = embed_input(input_text, model)

    print("🔎 Calculating cosine similarities...")
    cosine_scores = util.cos_sim(input_embedding, embeddings)[0]

    print(f"✅ Top {top_k} similar patents:")
    top_results = torch.topk(cosine_scores, k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        idx = int(idx)
        results.append({
            "publication_number": df.iloc[idx]["publication_number"],
            "abstract": df.iloc[idx]["abstract"],
            "similarity_score": score.item()
        })
    return results

if __name__ == "__main__":
    import torch

    input_abstract = """A device for closing skin wounds using biodegradable clips with adjustable tension,
    designed to minimize scarring and infection."""

    model = SentenceTransformer("all-MiniLM-L6-v2")

    df, embeddings = load_data()
    results = find_similar_abstracts(input_abstract, df, embeddings, model)

    for i, res in enumerate(results, 1):
        print(f"\n🔹 Match #{i}")
        print(f"📘 Publication #: {res['publication_number']}")
        print(f"🧾 Abstract: {res['abstract']}")
        print(f"📊 Similarity Score: {res['similarity_score']:.4f}")
