import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_DIR = "data/processed/chroma"
COLLECTION_NAME = "rag_docs"

def embed_and_store(chunks: list[dict]):
    if not chunks:
        print("No chunks to store.")
        return

    client = chromadb.PersistentClient(path=DB_DIR)

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")
