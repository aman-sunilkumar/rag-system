import re
CITE_RE = re.compile(r"\[source_(\d+)\]")
def validate_citations(answer: str, num_sources: int):
    cited = set(int(m) for m in CITE_RE.findall(answer))
    invalid = [n for n in cited if n < 1 or n > num_sources]
    has_citations = len(cited) > 0
    refused = "cannot answer" in answer.lower() or "not enough information" in answer.lower()
    return {
        "has_citations": has_citations,
        "invalid_citations": invalid,
        "refused": refused,
        "ok": (has_citations and not invalid) or refused
    }
