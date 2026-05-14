from src.retrieval.vector_store import retrieve
from src.generation.generator import generate

q = "What is the AI Risk Management Framework?"

chunks = retrieve(q, top_k=5)

print("=== TOP CHUNKS ===")

for i, c in enumerate(chunks):
    print(f"[source_{i+1}] {c['metadata']['file']} p{c['metadata'].get('page')}")
    print(c["text"][:200], "...\n")

print("=== ANSWER ===")
print(generate(q, chunks))
