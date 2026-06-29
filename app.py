import os, json, uuid
from pathlib import Path
from typing import Generator

import numpy as np
import fitz                          # PyMuPDF
import requests as http
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# ── Init ───────────────────────────────────────────────────────────────────────
print("Loading embedding model (first run downloads ~90 MB)…")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")

app = FastAPI(title="PDF RAG")

# ── In-memory store ────────────────────────────────────────────────────────────
docs:   dict[str, dict] = {}
chunks: list[dict]      = []

CHUNK_SIZE = 900
OVERLAP    = 150

# ── PDF helpers ────────────────────────────────────────────────────────────────
def pdf_to_pages(data: bytes) -> list[dict]:
    doc = fitz.open(stream=data, filetype="pdf")
    return [
        {"page": i + 1, "text": p.get_text("text").strip()}
        for i, p in enumerate(doc)
        if p.get_text("text").strip()
    ]

def split(text: str) -> list[str]:
    result, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        piece = text[start:end].strip()
        if piece:
            result.append(piece)
        if end >= len(text):
            break
        start += CHUNK_SIZE - OVERLAP
    return result

# ── Ollama helpers ─────────────────────────────────────────────────────────────
def ollama_models() -> list[str]:
    try:
        r = http.get(f"{OLLAMA}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []

def ollama_stream(model: str, system: str, user: str) -> Generator[str, None, None]:
    """Yield text tokens from Ollama streaming chat."""
    try:
        r = http.post(
            f"{OLLAMA}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                yield token
            if data.get("done"):
                break
    except http.exceptions.ConnectionError:
        yield "\n\n⚠ Could not connect to Ollama. Is it running? (ollama serve)"
    except Exception as e:
        yield f"\n\n⚠ Error: {e}"

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    models = ollama_models()
    return {"ollama": bool(models), "models": models}

@app.get("/models")
def list_models():
    return ollama_models()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    data   = await file.read()
    doc_id = uuid.uuid4().hex[:8]
    pages  = pdf_to_pages(data)

    if not pages:
        raise HTTPException(400, "Could not extract text from this PDF.")

    raw = [
        {"page": p["page"], "text": t}
        for p in pages
        for t in split(p["text"])
    ]

    texts      = [r["text"] for r in raw]
    embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    new_chunks = [
        {
            "id":        f"{doc_id}_{i}",
            "doc_id":    doc_id,
            "doc_name":  file.filename,
            "page":      r["page"],
            "text":      r["text"],
            "embedding": emb,
        }
        for i, (r, emb) in enumerate(zip(raw, embeddings))
    ]

    docs[doc_id] = {
        "id": doc_id, "name": file.filename,
        "pages": len(pages), "chunk_count": len(new_chunks),
    }
    chunks.extend(new_chunks)
    return docs[doc_id]

@app.get("/documents")
def list_docs():
    return list(docs.values())

@app.delete("/documents/{doc_id}")
def delete_doc(doc_id: str):
    if doc_id not in docs:
        raise HTTPException(404, "Document not found.")
    docs.pop(doc_id)
    chunks[:] = [c for c in chunks if c["doc_id"] != doc_id]
    return {"ok": True}

class ChatRequest(BaseModel):
    question: str
    model:    str            = "llama3.2"
    doc_ids:  list[str] | None = None

@app.post("/chat")
def chat(req: ChatRequest):
    if not chunks:
        raise HTTPException(400, "Upload at least one PDF first.")

    pool = chunks if not req.doc_ids else [c for c in chunks if c["doc_id"] in req.doc_ids]
    if not pool:
        raise HTTPException(400, "No chunks found for the selected documents.")

    # Embed + cosine search (embeddings are already L2-normalised)
    q_emb   = embedder.encode([req.question], normalize_embeddings=True)[0]
    matrix  = np.stack([c["embedding"] for c in pool])
    scores  = matrix @ q_emb
    top_idx = np.argsort(scores)[::-1][:6]
    top     = [pool[i] for i in top_idx]

    context = "\n\n---\n\n".join(
        f"[{c['doc_name']}, page {c['page']}]\n{c['text']}"
        for c in top
    )
    sources = [
        {"doc_name": c["doc_name"], "page": c["page"], "score": round(float(scores[top_idx[i]]), 3)}
        for i, c in enumerate(top)
    ]

    system = (
        "You are a precise research assistant. Answer using only the provided document excerpts. "
        "Cite sources naturally, e.g. '(filename, p. N)'. "
        "If the context is insufficient, say so instead of guessing."
    )
    user_msg = f"Excerpts:\n{context}\n\nQuestion: {req.question}"

    def stream() -> Generator[str, None, None]:
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        for token in ollama_stream(req.model, system, user_msg):
            yield f"data: {json.dumps({'type': 'text', 'text': token})}\n\n"
        yield 'data: {"type":"done"}\n\n'

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no"})

# ── Serve frontend ─────────────────────────────────────────────────────────────
static = Path(__file__).parent / "static"
static.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=static, html=True), name="static")
