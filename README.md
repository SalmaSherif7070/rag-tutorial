# A Beginner's RAG Tutorial 🧠🔎

A small, readable project that teaches **Retrieval-Augmented Generation (RAG)** by
building a working question-answering system over real documents.

If you've never heard of RAG before, start here. 👇

---

## 1. What is RAG, in one minute?

A large language model (LLM) like Gemini is smart, but it only knows what it saw
during training. It can't see your private documents, and it sometimes makes
things up ("hallucinates").

**RAG fixes this by giving the model notes to read before it answers.**

> Think of an open-book exam. Instead of answering from memory, you first *find*
> the relevant pages, then write your answer *based on those pages*.

The two halves of the name:

| Part | What happens | In this project |
|------|--------------|-----------------|
| **Retrieval** | Find the most relevant text for the question | Search a vector database (Qdrant) |
| **Generation** | Write an answer using that text | Ask Gemini, grounded in the found passages |

### The flow

```
            your question
                 |
                 v
   [ 1. embed the question into a vector ]      <- Jina
                 |
                 v
   [ 2. find the closest passages ]             <- Qdrant
                 |
                 v
   [ 3. give passages + question to the LLM ]   <- Gemini
                 |
                 v
          grounded answer (+ sources)
```

### What is an "embedding"?

An embedding is a list of numbers (a *vector*) that captures the *meaning* of a
piece of text. Texts with similar meaning get similar vectors. So to find
relevant passages we just look for the vectors closest to the question's vector.
That is what a **vector database** like Qdrant is built to do, fast.

---

## 2. What this specific app does

Most tutorials build a single-folder search. We go one step further and keep an
**orchestrator** (great for learning how real systems scale):

```
                 START
                   |
            ┌──────────────┐
            │ orchestrator │   the "router": which domains can answer this?
            └──────────────┘
                   |  (fan out, one worker per chosen domain)
        ┌──────────┼──────────┬──────────┐
     [health]  [finance]   [legal]    [tech]      each worker:
        │          │          │          │          - searches ITS collection
        └──────────┴────┬─────┴──────────┘          - drafts an answer from passages
                        |  (answers merged)
                 ┌──────────────┐
                 │ synthesizer  │   combine into one answer + list the sources
                 └──────────────┘
                        |
                       END
```

The knowledge base comes from the **[RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench)**
dataset on Hugging Face. We use 4 of its subsets, one per domain:

| Domain  | RAGBench subset | Qdrant collection | Example question |
|---------|-----------------|-------------------|------------------|
| health  | `covidqa`       | `rag_health`      | "What are common symptoms of COVID-19?" |
| finance | `finqa`         | `rag_finance`     | "How is operating margin calculated?" |
| legal   | `cuad`          | `rag_legal`       | "What does an indemnification clause cover?" |
| tech    | `techqa`        | `rag_tech`        | "How do I fix an SSL handshake error?" |

---

## 3. The project layout (clean architecture)

Each file has **one job**. Read them in this order:

```
rag/
├── config.py         # settings loaded from .env (no hard-coded keys!)
├── domains.py        # the 4 domains -> 4 Qdrant collections
├── schemas.py        # data shapes (Pydantic) + the graph State
├── embeddings.py     # STEP: text -> vector, using the Jina Cloud API
├── vector_store.py   # STEP: store & search vectors in Qdrant
├── data_loader.py    # download RAGBench from Hugging Face
├── ingest.py         # INDEXING pipeline: load -> embed -> store
├── retrieval.py      # SEARCH step: embed question -> find passages
├── llm.py            # the Gemini model (generation)
├── dependencies.py   # builds & wires all the pieces together
├── graph/
│   ├── nodes.py      # orchestrator / worker / synthesizer
│   └── builder.py    # connects the nodes into a runnable graph
├── api.py            # FastAPI web service (gives us Swagger UI)
└── cli.py            # run it from the terminal
```

And at the project root, for running in containers:

```
Dockerfile           # image for the API service
docker-compose.yml   # runs qdrant + api together
```

---

## 4. Setup (common to both run modes)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Two free API keys:
  - **Jina** (embeddings): https://jina.ai/embeddings/
  - **Google AI Studio** (Gemini): https://aistudio.google.com/app/apikey
- Python 3.10+ (only needed for the local CLI mode, Option B below)

### Add your keys

```powershell
Copy-Item .env.example .env
notepad .env       # paste your JINA_API_KEY and GOOGLE_API_KEY
```

---

## 5. Run it

You have two ways to run the project. **Option A runs everything in containers and
gives you Swagger UI** — recommended.

### Option A — Everything in Docker, with Swagger UI 🐳 (recommended)

This starts **two containers**: `qdrant` (the database) and `api` (our app).

```powershell
docker compose up -d --build
```

Now open the interactive API docs — this *is* Swagger UI:

### 👉 http://localhost:8000/docs

From that page you can click any endpoint, hit **"Try it out"**, and run it in the
browser — no curl needed. Do it in this order:

1. **`POST /ingest`** → click *Try it out* → `Execute`. Body (edit as you like):
   ```json
   { "max_rows": 50 }
   ```
   This downloads the data, embeds it with Jina, and stores it in Qdrant.
2. **`POST /ask`** → *Try it out* → set the body and `Execute`:
   ```json
   { "question": "What are the symptoms of COVID-19?" }
   ```

You'll get back the answer plus the source passages. Prefer the terminal? Same
endpoints work with curl:

```powershell
curl.exe -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d "{\"max_rows\":50}"
curl.exe -X POST http://localhost:8000/ask    -H "Content-Type: application/json" -d "{\"question\":\"What is an indemnification clause?\"}"
```

Useful URLs while it's running:
- Swagger UI: http://localhost:8000/docs
- ReDoc (read-only docs): http://localhost:8000/redoc
- Qdrant dashboard: http://localhost:6333/dashboard

Logs and shutdown:
```powershell
docker compose logs -f api   # watch the app's output
docker compose down          # stop everything
```

### Option B — Local CLI (no API container)

Run only the database in Docker, and the Python app on your machine:

```powershell
docker compose up -d qdrant          # just the database
pip install -r requirements.txt

python -m rag.cli ingest --max-rows 50
python -m rag.cli ask "What are the symptoms of COVID-19?"
python -m rag.cli demo --domain tech
```

Example output:

```
=== ANSWER ===
Common symptoms include fever, dry cough, and fatigue...

=== SOURCES ===
- [health] (score 0.812) Patients with COVID-19 commonly present with fever...
- [health] (score 0.779) A dry cough and shortness of breath were reported in...
```

When you're done: `docker compose down`.

---

## 6. How a single question flows through the code

1. **`cli.py`** builds the graph and calls `graph.invoke({"query": ...})`.
2. **`graph/nodes.py → orchestrator`** asks Gemini which domains are relevant.
3. **`assign_workers`** fans out: one `worker` per chosen domain, running together.
4. Each **`worker`** calls **`retrieval.retrieve`** → embeds the question
   (**`embeddings.py`**), searches Qdrant (**`vector_store.py`**), then asks
   Gemini to answer *using only those passages*.
5. **`synthesizer`** merges the workers' answers into one and attaches the sources.

---

## 7. Ideas to extend it (great for practice)

- Add a 5th domain in `domains.py` (pick another RAGBench subset) and re-ingest.
- Show the retrieved passages *before* the answer, to see retrieval in action.
- Add a `--top-k` flag and watch how more/fewer passages change the answer.
- Evaluate quality: compare answers to RAGBench's ground-truth `response`.
- Swap Gemini for a local model (e.g. via Ollama) in `llm.py`.

---

## 8. Common issues

| Symptom | Fix |
|---------|-----|
| `Connection refused` to localhost:6333 | Qdrant isn't running — `docker compose up -d`. |
| `/docs` opens but `/ask` returns 500 | Usually missing/invalid keys in `.env`, or you haven't run `/ingest` yet. Check `docker compose logs -f api`. |
| `api` container keeps restarting | Open its logs: `docker compose logs api`. Most often a bad key or a typo in `.env`. |
| Changed the code but Docker shows the old version | Rebuild the image: `docker compose up -d --build`. |
| `pydantic ... validation error` on startup | Missing keys in `.env` (JINA_API_KEY / GOOGLE_API_KEY). |
| Ingestion is slow | Use a small `max_rows` (e.g. 50) while learning. |
| Empty / weird answers | Make sure you ran ingest first, and the question fits one of the 4 domains. |
