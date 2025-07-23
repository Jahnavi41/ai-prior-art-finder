# backend/api/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import fitz  
import os

# Load embeddings and texts once when server starts
EMBEDDING_PATH = "../embeddings/minilm__embeddings.npy"
TEXT_PATH = "../embeddings/minilm__text.npy"

# Load dataset embeddings
embedding_matrix = np.load(EMBEDDING_PATH)
texts = np.load(TEXT_PATH, allow_pickle=True)

# Load model once
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

app = FastAPI()

def extract_abstract_from_pdf(file_path):
    """Extracts the first 1000 characters of the first page as a mock abstract."""
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
            break  
    return text.strip()[:1000]  

def find_similar_abstracts(query, k=5):
    query_embedding = model.encode([query], convert_to_tensor=True)
    dataset_embeddings = torch.tensor(embedding_matrix)

    # Cosine similarity
    similarities = torch.nn.functional.cosine_similarity(query_embedding, dataset_embeddings)
    top_k_indices = torch.topk(similarities, k=k).indices.tolist()

    # Return top-k results
    return [{"text": texts[i], "score": float(similarities[i])} for i in top_k_indices]

@app.post("/search")
async def search_similar_patents(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Extract abstract
        abstract = extract_abstract_from_pdf(file_path)

        # Clean up temp file
        os.remove(file_path)

        # Run similarity search
        results = find_similar_abstracts(abstract)

        return JSONResponse(content={"abstract": abstract, "results": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
