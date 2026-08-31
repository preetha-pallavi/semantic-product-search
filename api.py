import pickle
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Semantic product search based on Amazon Review")

# Load everything once at startup
index = faiss.read_index('product_index.faiss')
with open('product_ids.pkl', 'rb') as f:
    product_ids = pickle.load(f)
with open('product_sample_text.pkl', 'rb') as f:
    product_sample_text = pickle.load(f)
with open('product_keywords.pkl', 'rb') as f:
    product_titles = pickle.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

pid_to_pos = {pid: i for i, pid in enumerate(product_ids)}


def get_title(pid: str) -> str:
    return product_titles.get(pid, "Unknown product")


def score_label(score: float) -> str:
    if score >= 0.9:
        return "Strong match"
    elif score >= 0.75:
        return "Good match"
    else:
        return "Similar"


@app.get("/recommend/{product_id}")
def recommend(product_id: str, k: int = 5):
    if product_id not in pid_to_pos:
        raise HTTPException(status_code=404, detail="Product not found")

    idx_pos = pid_to_pos[product_id]
    query_vec = index.reconstruct(idx_pos).reshape(1, -1)
    D, I = index.search(query_vec, k + 1)

    results = []
    for score, i in zip(D[0], I[0]):
        pid = product_ids[i]
        if pid == product_id:
            continue
        results.append({
            "product_id": pid,
            "title": get_title(pid),
            "match_quality": score_label(float(score)),
            "sample_review": product_sample_text.get(pid, "")[:200]
        })

    return {
        "query_product": product_id,
        "query_title": get_title(product_id),
        "recommendations": results[:k]
    }


@app.get("/search")
def search(query: str, k: int = 5):
    query_vec = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec, k)

    results = []
    for score, i in zip(D[0], I[0]):
        pid = product_ids[i]
        results.append({
            "product_id": pid,
            "title": get_title(pid),
            "match_quality": score_label(float(score)),
            "sample_review": product_sample_text.get(pid, "")[:200]
        })

    return {"query": query, "results": results}