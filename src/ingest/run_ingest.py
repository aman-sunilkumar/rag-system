from pathlib import Path
from src.ingest.loaders import load_any, chunk_id
from src.ingest.chunkers import splitter
from src.ingest.embed import embed_and_store
RAW = Path("data/raw")
def main():
    all_chunks = []
    for path in RAW.glob("*"):
        if path.is_dir():
            continue
        print(f"Loading {path.name}...")
        try:
            pages = load_any(path)
        except ValueError as e:
            print(f"Skipping: {e}")
            continue
        for page in pages:
            for offset, chunk_text in enumerate(splitter.split_text(page["text"])):
                cid = chunk_id(path.name, len(all_chunks))
                all_chunks.append({
                    "id": cid,
                    "text": chunk_text,
                    "metadata": {
                        "file": path.name,
                        "page": page["page"],
                        "chunk_id": cid,
                    },
                })
    print(f"Total chunks: {len(all_chunks)}")
    embed_and_store(all_chunks)
    print("Done.")
if __name__ == "__main__":
    main()
