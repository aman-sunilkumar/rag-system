import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_DIR = "data/processed/chroma"
COLLECTION_NAME = "rag_docs"

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

def retrieve(query: str, top_k: int = 10):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    out = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        out.append({
            "text": doc,
            "metadata": meta,
            "score": 1 - dist
        })

    return out
