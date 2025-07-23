from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import fitz
import os
import re

# Load model & embeddings on server start
EMBEDDING_PATH = "backend/embeddings/minilm_embeddings.npy"
TEXT_PATH = "backend/embeddings/minilm_texts.npy"

embedding_matrix = np.load(EMBEDDING_PATH)
texts = np.load(TEXT_PATH, allow_pickle=True)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
app = FastAPI()

def extract_abstract_only(text):
    text = text.lower()
    text = ' '.join(text.split())

    # Find where abstract starts
    abstract_start_keywords = ['abstract:', 'abstract -', 'abstract']
    start_idx = -1
    for keyword in abstract_start_keywords:
        if keyword in text:
            start_idx = text.find(keyword) + len(keyword)
            break
    if start_idx == -1:
        print("⚠️ 'Abstract' not found in the text.")
        return ""

    after_abstract = text[start_idx:]

    # Heuristic patterns to end extraction
    end_patterns = [
        r'sheet \d+ of \d+',
        r'figure \d+',
        r'claims',
        r'field of the invention',
        r'background',
        r'description',
        r'summary',
        r'brief description',
        r'us \d{4}/\d+',
        r'\d{4}/\d{7}', 
    ]

    end_idx = len(after_abstract)
    for pattern in end_patterns:
        match = re.search(pattern, after_abstract)
        if match and match.start() < end_idx:
            end_idx = match.start()

    abstract_cleaned = after_abstract[:end_idx].strip()
    abstract_words = abstract_cleaned.split()
    if len(abstract_words) > 250:
        abstract_cleaned = ' '.join(abstract_words[:250])

    return abstract_cleaned

def find_similar_abstracts(query, k=5):
    query_embedding = model.encode([query], convert_to_tensor=True)
    dataset_embeddings = torch.tensor(embedding_matrix)

    similarities = torch.nn.functional.cosine_similarity(query_embedding, dataset_embeddings)
    top_k_indices = torch.topk(similarities, k=k).indices.tolist()

    return [
        {
            "publication": texts[i].get("publication", "N/A"),
            "abstract": texts[i].get("abstract", ""),
            "score": float(similarities[i])
        }
        for i in top_k_indices
    ]

@app.post("/search")
async def search_similar_patents(file: UploadFile = File(...), top_k: int = Query(5, description="Number of top similar results to return")):
    try:
        # Save file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Extract full PDF text
        with fitz.open(file_path) as pdf:
            full_text = "".join([page.get_text() for page in pdf])

        os.remove(file_path)

        # Extract abstract from PDF
        abstract = extract_abstract_only(full_text)
        if not abstract:
            return JSONResponse(content={"abstract": "", "results": []})

        # Find similar patents
        results = find_similar_abstracts(abstract, k=top_k)

        return JSONResponse(content={"abstract": abstract, "results": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
