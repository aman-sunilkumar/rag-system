from pathlib import Path
import hashlib
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader
def chunk_id(file_name: str, index: int) -> str:
    return hashlib.md5(f"{file_name}-{index}".encode()).hexdigest()
def load_pdf(path: Path):
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({
            "page": i + 1,
            "text": text
        })
    return pages
def load_text(path: Path):
    return [{
        "page": 1,
        "text": path.read_text(encoding="utf-8")
    }]
def load_any(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in [".txt", ".md"]:
        return load_text(path)
    raise ValueError(f"Unsupported file type: {suffix}")
