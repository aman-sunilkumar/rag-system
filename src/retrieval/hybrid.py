from src.retrieval.vector_store import retrieve as vector_retrieve
from src.retrieval.bm25 import retrieve_bm25
def get_doc_id(doc):
    meta = doc.get("metadata", {})
    if meta.get("chunk_id"):
        return meta["chunk_id"]
    return f"{meta.get('file', 'unknown')}:{meta.get('page', 'unknown')}:{doc.get('text', '')[:80]}"
def reciprocal_rank_fusion(result_lists, k: int = 60, top_k: int = 20):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = get_doc_id(doc)
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            docs[doc_id] = doc
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    fused = []
    for doc_id, score in ranked[:top_k]:
        doc = docs[doc_id]
        fused.append({
            **doc,
            "rrf_score": float(score)
        })
    return fused
def hybrid_retrieve(query: str, top_k: int = 20):
    vec = vector_retrieve(query, top_k=top_k)
    bm = retrieve_bm25(query, top_k=top_k)
    return reciprocal_rank_fusion([vec, bm], top_k=top_k)
