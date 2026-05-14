from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.rerank import rerank
from src.generation.generator import generate
from src.generation.citation_check import validate_citations
def ask(question: str, retrieve_k: int = 20, top_n: int = 5):
    candidates = hybrid_retrieve(question, top_k=retrieve_k)
    sources = rerank(question, candidates, top_n=top_n)
    answer = generate(question, sources)
    check = validate_citations(answer, len(sources))
    if not check["ok"]:
        retry_q = (
            f"{question}\n\nIMPORTANT: cite every factual claim with [source_N]. "
            f"If the sources do not support the answer, refuse."
        )
        answer = generate(retry_q, sources)
        check = validate_citations(answer, len(sources))
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "validation": check
    }
