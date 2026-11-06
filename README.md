# PDF Chat — RAG over your documents

Upload PDFs and ask questions. Answers are grounded in your documents and cite the source file, page and passage.

## Run it

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key                     # Windows: set ANTHROPIC_API_KEY=your-key
uvicorn main:app --reload
```
Open http://127.0.0.1:8000. The first upload downloads the embedding model (~90 MB) once.

## How it works

| Step | Implementation (`main.py`) |
|---|---|
| Extract | `pypdf` reads text page by page |
| Chunk | 150-word sliding window, 30-word overlap, page number kept per chunk |
| Embed | `all-MiniLM-L6-v2` (sentence-transformers, runs locally) |
| Store | SQLite: vectors as float32 blobs next to doc id + page |
| Retrieve | Cosine similarity with NumPy, top 5, across one or many PDFs |
| Generate | Claude answers from numbered passages only and cites `[n]` |

Other features: conversation memory (last 6 turns), follow-up questions rewritten into standalone search queries, a similarity threshold plus a strict prompt so unanswerable questions get "I couldn't find that", short/detailed summaries with key points, terms and chapter summaries, rename/delete, and Pending → Processing → Completed/Failed status.

## Known limits (good to mention in interviews)
- Scanned PDFs have no text layer, so they fail (fix: add OCR with `pytesseract`).
- Passage highlighting inside the PDF viewer is best effort and depends on the browser.
- Brute-force NumPy search is fine for dozens of PDFs; for more, swap in FAISS or ChromaDB.
- No user accounts; chats are keyed to a browser session id.

## Ideas to extend
Flashcards/MCQs from chunks, document comparison, streaming responses, a reranker, auth, Docker.

## Resume bullets
- Built a full-stack RAG application (FastAPI, SQLite, sentence-transformers, Claude API) that answers questions over multiple PDFs with page-level citations.
- Implemented the retrieval pipeline from scratch: page-aware chunking with overlap, local embeddings, cosine-similarity search, and context construction with hallucination control.
- Added conversational memory with follow-up query rewriting, plus short and detailed document summarization.
- Built a responsive UI with background PDF processing, live status, and click-to-open citations that jump to the cited PDF page.
