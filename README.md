# PDF RAG

Ask questions about your PDFs using a local AI. **No API key. No cloud. Free forever.**

## What it does

- Upload PDFs via drag & drop
- Ask questions in a chat interface
- Get answers with source citations (document name + page number)
- Everything runs on your machine — nothing is sent anywhere

## Setup

### 1. Install Ollama
Download from **https://ollama.com** and install it.

Then pull a model (pick one):
```bash
ollama pull llama3.2        # recommended — fast, good quality (~2 GB)
ollama pull mistral         # alternative (~4 GB)
ollama pull phi3            # smaller, faster (~2 GB)
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```
> First run downloads the embedding model (~90 MB, one time only).

### 3. Run

**Windows:** double-click `start.bat`

**Mac / Linux:**
```bash
python -m uvicorn app:app --port 8000
```

Then open **http://localhost:8000**

## How it works

```
PDF → extract text → chunk (900 chars) → embed (all-MiniLM-L6-v2)
                                                        ↓
Question → embed → cosine similarity search → top 6 chunks
                                                        ↓
                                          Ollama LLM → streamed answer
```

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local)
- **LLM:** any Ollama model you have installed
- **Vector search:** in-memory numpy (fast for <20 docs)
- **Backend:** FastAPI + Python
- **Frontend:** vanilla HTML/CSS/JS

## Tech stack

| Component | Library |
|-----------|---------|
| PDF parsing | PyMuPDF |
| Embeddings | sentence-transformers |
| Vector search | numpy |
| LLM | Ollama (local) |
| API | FastAPI |
| Frontend | Vanilla JS |
