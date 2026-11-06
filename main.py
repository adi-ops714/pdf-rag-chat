"""PDF Chat — a small RAG app.
Pipeline: upload -> extract text -> chunk -> embed -> store -> retrieve -> generate.
"""
import json, os, re, sqlite3, time
from pathlib import Path

import anthropic
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)
DB_PATH = BASE / "app.db"

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
EMBED_MODEL = "all-MiniLM-L6-v2"   # runs locally, no API key needed
CHUNK_WORDS, OVERLAP_WORDS = 150, 30
TOP_K = 5
MIN_SCORE = 0.25                    # below this we treat the answer as "not in documents"

app = FastAPI(title="PDF Chat")
llm_client = None
_embedder = None


# ---------- helpers ----------
def db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS documents(
            id INTEGER PRIMARY KEY, name TEXT, path TEXT, size INTEGER, pages INTEGER,
            uploaded_at TEXT, status TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY, doc_id INTEGER, page INTEGER, text TEXT, emb BLOB);
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
            sources TEXT, created_at TEXT);
        """)


init_db()


def embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def embed(texts):
    return embedder().encode(texts, normalize_embeddings=True).astype(np.float32)


def llm(system, messages, max_tokens=1000):
    global llm_client
    if llm_client is None:
        llm_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    r = llm_client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system, messages=messages)
    return r.content[0].text


def chunk_page(text):
    """Sliding window over words so chunks overlap and keep context."""
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    out = []
    for i in range(0, max(len(words), 1), step):
        piece = " ".join(words[i:i + CHUNK_WORDS])
        if len(piece) > 40:
            out.append(piece)
        if i + CHUNK_WORDS >= len(words):
            break
    return out


# ---------- ingestion ----------
def process_document(doc_id: int):
    with db() as c:
        c.execute("UPDATE documents SET status='Processing' WHERE id=?", (doc_id,))
        path = c.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()["path"]
    try:
        reader = PdfReader(path)
        rows = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            rows += [(page_no, ch) for ch in chunk_page(text)]
        if not rows:
            raise ValueError("No extractable text (scanned PDF? OCR is not supported).")
        vecs = embed([t for _, t in rows])
        with db() as c:
            c.executemany(
                "INSERT INTO chunks(doc_id,page,text,emb) VALUES(?,?,?,?)",
                [(doc_id, p, t, v.tobytes()) for (p, t), v in zip(rows, vecs)])
            c.execute("UPDATE documents SET status='Completed' WHERE id=?", (doc_id,))
    except Exception as e:
        with db() as c:
            c.execute("UPDATE documents SET status='Failed', error=? WHERE id=?", (str(e), doc_id))


# ---------- document endpoints ----------
@app.post("/api/documents")
async def upload(bg: BackgroundTasks, files: list[UploadFile] = File(...)):
    created = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"{f.filename} is not a PDF")
        data = await f.read()
        safe = f"{int(time.time()*1000)}_{re.sub(r'[^A-Za-z0-9._-]', '_', f.filename)}"
        path = UPLOADS / safe
        path.write_bytes(data)
        try:
            pages = len(PdfReader(path).pages)
        except Exception:
            path.unlink(missing_ok=True)
            raise HTTPException(400, f"{f.filename} is not a valid PDF")
        with db() as c:
            cur = c.execute(
                "INSERT INTO documents(name,path,size,pages,uploaded_at,status) VALUES(?,?,?,?,?,'Pending')",
                (f.filename, str(path), len(data), pages, time.strftime("%Y-%m-%d %H:%M")))
        bg.add_task(process_document, cur.lastrowid)
        created.append(cur.lastrowid)
    return {"ids": created}


@app.get("/api/documents")
def list_docs():
    with db() as c:
        return [dict(r) for r in c.execute(
            "SELECT id,name,size,pages,uploaded_at,status,error FROM documents ORDER BY id DESC")]


class Rename(BaseModel):
    name: str


@app.patch("/api/documents/{doc_id}")
def rename(doc_id: int, body: Rename):
    with db() as c:
        c.execute("UPDATE documents SET name=? WHERE id=?", (body.name.strip(), doc_id))
    return {"ok": True}


@app.delete("/api/documents/{doc_id}")
def delete(doc_id: int):
    with db() as c:
        row = c.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Not found")
        Path(row["path"]).unlink(missing_ok=True)
        c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    return {"ok": True}


@app.get("/api/documents/{doc_id}/file")
def get_file(doc_id: int):
    with db() as c:
        row = c.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Not found")
    return FileResponse(row["path"], media_type="application/pdf")


# ---------- retrieval ----------
def retrieve(query, doc_ids=None, k=TOP_K):
    q = embed([query])[0]
    sql = ("SELECT c.id,c.doc_id,c.page,c.text,c.emb,d.name FROM chunks c "
           "JOIN documents d ON d.id=c.doc_id WHERE d.status='Completed'")
    params = []
    if doc_ids:
        sql += f" AND c.doc_id IN ({','.join('?'*len(doc_ids))})"
        params = list(doc_ids)
    with db() as c:
        rows = c.execute(sql, params).fetchall()
    if not rows:
        return []
    mat = np.vstack([np.frombuffer(r["emb"], dtype=np.float32) for r in rows])
    scores = mat @ q                      # cosine similarity (vectors are normalised)
    top = np.argsort(-scores)[:k]
    return [{"doc_id": rows[i]["doc_id"], "doc_name": rows[i]["name"],
             "page": rows[i]["page"], "text": rows[i]["text"],
             "score": float(scores[i])} for i in top if scores[i] >= MIN_SCORE]


# ---------- chat ----------
class ChatReq(BaseModel):
    session_id: str
    question: str
    doc_ids: list[int] | None = None


SYSTEM = ("You answer questions using ONLY the numbered context passages provided. "
          "Cite the passages you use with their numbers like [1] or [2][3]. "
          "If the passages do not contain the answer, say: \"I couldn't find that in the "
          "uploaded documents.\" Do not use outside knowledge. Be concise.")


def history_for(session_id, limit=6):
    with db() as c:
        rows = c.execute("SELECT role,content FROM messages WHERE session_id=? "
                         "ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]


@app.post("/api/chat")
def chat(req: ChatReq):
    hist = history_for(req.session_id)
    search_q = req.question
    if hist:  # turn a follow-up like "what about its cost?" into a standalone query
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in hist)
        search_q = llm("Rewrite the user's last question as a standalone search query, "
                       "using the conversation for context. Output only the query.",
                       [{"role": "user", "content": f"{convo}\nuser: {req.question}"}], 100).strip()

    hits = retrieve(search_q, req.doc_ids)
    if not hits:
        answer, sources = "I couldn't find that in the uploaded documents.", []
    else:
        ctx = "\n\n".join(f"[{i}] ({h['doc_name']}, page {h['page']})\n{h['text']}"
                          for i, h in enumerate(hits, 1))
        msgs = [{"role": m["role"], "content": m["content"]} for m in hist]
        msgs.append({"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {req.question}"})
        answer = llm(SYSTEM, msgs, 800)
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        sources = [{"n": i, **h} for i, h in enumerate(hits, 1) if i in cited] or \
                  [{"n": i, **h} for i, h in enumerate(hits, 1)]

    now = time.strftime("%Y-%m-%d %H:%M")
    with db() as c:
        c.execute("INSERT INTO messages(session_id,role,content,sources,created_at) VALUES(?,?,?,?,?)",
                  (req.session_id, "user", req.question, None, now))
        c.execute("INSERT INTO messages(session_id,role,content,sources,created_at) VALUES(?,?,?,?,?)",
                  (req.session_id, "assistant", answer, json.dumps(sources), now))
    return {"answer": answer, "sources": sources}


@app.get("/api/history")
def history(session_id: str):
    with db() as c:
        rows = c.execute("SELECT role,content,sources FROM messages WHERE session_id=? ORDER BY id",
                         (session_id,)).fetchall()
    return [{"role": r["role"], "content": r["content"],
             "sources": json.loads(r["sources"]) if r["sources"] else []} for r in rows]


@app.delete("/api/history")
def clear_history(session_id: str):
    with db() as c:
        c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    return {"ok": True}


# ---------- summarisation ----------
@app.post("/api/documents/{doc_id}/summary")
def summarize(doc_id: int, mode: str = "short"):
    with db() as c:
        rows = c.execute("SELECT page,text FROM chunks WHERE doc_id=? ORDER BY id", (doc_id,)).fetchall()
    if not rows:
        raise HTTPException(400, "Document is not processed yet")
    budget, step = 30000, max(1, len(rows) // 40)       # sample evenly for long docs
    text, total = "", 0
    for r in rows[::step]:
        piece = f"[p.{r['page']}] {r['text']}\n"
        if total + len(piece) > budget:
            break
        text, total = text + piece, total + len(piece)
    detail = ("Write a detailed summary (3-5 paragraphs) and chapter_summaries: a list of "
              "{title, summary} for each major section." if mode == "detailed"
              else "Write a short summary (3-4 sentences) and set chapter_summaries to [].")
    out = llm("You summarise documents. Respond with ONLY valid JSON, no markdown fences.",
              [{"role": "user", "content": f"{detail}\nAlso include key_points (5-8 strings) and "
                f"key_terms (list of {{term, definition}}).\nKeys: summary, key_points, "
                f"key_terms, chapter_summaries.\n\nDocument:\n{text}"}], 2500)
    try:
        return json.loads(re.sub(r"^```(?:json)?|```$", "", out.strip(), flags=re.M).strip())
    except json.JSONDecodeError:
        raise HTTPException(502, "Model returned invalid JSON, try again")


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
