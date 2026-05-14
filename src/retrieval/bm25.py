import os
import pickle
import re
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
DB_DIR = "data/processed/chroma"
COLLECTION_NAME = "rag_docs"
INDEX_PATH = Path("data/processed/bm25.pkl")
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)
def tokenize(text: str):
    return re.findall(r"[A-Za-z0-9_]+", text.lower())
def build_bm25():
    collection = get_collection()
    all_items = collection.get(include=["documents", "metadatas"])
    documents = all_items.get("documents", [])
    metadatas = all_items.get("metadatas", [])
    chunks = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(documents, metadatas)
    ]
    if not chunks:
        raise ValueError("No chunks found in ChromaDB. Run ingestion first: python -m src.ingest.run_ingest")
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"Built BM25 index with {len(chunks)} chunks.")
    return bm25, chunks
def load_bm25():
    if not os.path.exists(INDEX_PATH):
        return build_bm25()
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]
def retrieve_bm25(query: str, top_k: int = 10):
    bm25, chunks = load_bm25()
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])[:top_k]
    return [
        {
            "text": c["text"],
            "metadata": c["metadata"],
            "score": float(s)
        }
        for c, s in ranked
    ]
