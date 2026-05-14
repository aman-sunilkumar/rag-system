from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")
def rerank(query: str, candidates: list, top_n: int = 5):
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [
        {
            **c,
            "rerank_score": float(s)
        }
        for c, s in scored[:top_n]
    ]
