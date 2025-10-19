from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import json
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Load TypeScript Book chunks with precomputed embeddings
with open("chunks.json", "r") as f:
    chunks = [json.loads(line) for line in f]

# Ensure each chunk has "embedding"
if "embedding" not in chunks[0]:
    raise Exception("chunks.json must have embeddings precomputed for each chunk.")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.get("/search")
def search(q: str = Query(..., description="Question to retrieve answer from TypeScript Book")):
    # Embed query
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=q
    ).data[0].embedding

    # Compute similarity for all chunks
    best_chunk = max(chunks, key=lambda c: cosine_similarity(c["embedding"], query_emb))

    # Return answer and source
    return {
        "answer": best_chunk["content"],
        "sources": best_chunk["id"]
    }
