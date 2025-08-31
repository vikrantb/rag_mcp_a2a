a
# Project Summary

> **Goal:** Build, run, and extend a **weekend‑scale Retrieval‑Augmented Generation (RAG) demo** with clean interfaces for **MCP tools**, an **OpenAI Agents SDK** agent, and a thin **A2A** façade. The code must be portable, file‑based, and easy to hand to prospects.  
> **Note:** This document intentionally contains **no company names** and targets **public help/docs sites** only.

---

## 0) Shared Principles (LLM × Human Contract)

1. **Deterministic & Idempotent:** Re‑running the same config must reproduce the same outputs (unless crawling new content).  
2. **Cache First:** Never discard previously downloaded HTML unless explicitly allowed. Prefer **reparse** over re‑crawl.  
3. **Small, Simple, Swappable:** Components are modular; backends (embeddings/DB/reranker) can be swapped with minimal edits.  
4. **Polite & Legal:** Respect `robots.txt` by default. A special **crawl_as_human** mode exists only for permitted scenarios.  
5. **Transparent:** Stream one‑line progress for every major step. Persist intermediate artifacts (HTML → parsed → chunks → index).  
6. **No PHI / Sensitive Data:** Index public help/docs only. Defensively scrub nav/ads/footers; keep headings/lists/tables.

---

## 1) What We’re Building (Scope)

A compact kit that can:

- **Discover & fetch** a public documentation site
- **Parse & clean** pages into structured JSONL records
- **Chunk & enrich** text into passages with metadata
- **Index** with a **hybrid** pipeline (BM25 + vectors via FAISS)
- **Retrieve** (keyword + vector → optional rerank)
- **Answer with citations** (demo script)
- **Evaluate quickly** on 10–20 golden questions
- **Expose tools** via MCP, wrap with an Agents SDK agent, and offer a simple A2A HTTP façade

### Explicit Non‑Goals (for now)
- Full production multi‑tenant service; advanced auth/quotas; streaming LLM generation; complex UI.

---

## 2) Repository Layout (expected)

```
assets_<site>/            # cached HTML & plain-text snapshots
archive/<epoch>/          # prior outputs moved here
chunks_<site>.jsonl       # chunked passages
parsed_<site>.jsonl       # parsed article records
index/                    # FAISS/BM25 artifacts + meta
web_crawler.py            # crawl → parse → chunk, plus offline reparse
index_build.py            # build hybrid index
query_local.py            # ad-hoc retrieve/rerank
ask_demo.py               # grounded answer + citations (no LLM by default)
eval_runner.py            # tiny evaluation harness
<site>_config.json        # JSON config for a run
README.md                 # user-facing walkthrough
project_summary.md        # THIS file — contract/spec for LLMs & agents
```

---

## 3) Configuration Schema (single JSON)

The crawler/chunker consumes one config file. Example (generic):

```json
{
  "crawl": {
    "seed_urls": ["https://<docs-host>/help"],
    "allow_patterns": ["/help/", "/docs/"],
    "deny_patterns": ["/search", "/login", "[?#]preview="],
    "allowed_domains": ["<docs-host>"],
    "same_host": true,
    "max_pages": 300,
    "max_depth": 5,
    "respect_robots": true,
    "delay_seconds": 0.6,
    "timeout_seconds": 10.0,
    "user_agent": "RAG-DiscoveryBot/1.0 (+https://example.org)",

    "crawl_as_human": false,               // if true: ignore robots, ~2s delay, progress streaming
    "force_redownload_from_web": false,    // if true: re-crawl even if cache exists
    "force_reparse_from_assets": false     // if true: rebuild parsed JSONL from cached HTML
  },
  "outputs": {
    "assets_dir": "assets_<site>",
    "parsed_jsonl_path": "parsed_<site>.jsonl",
    "chunks_jsonl_path": "chunks_<site>.jsonl",
    "fresh_run": true                      // archive only what will be regenerated
  },
  "chunk": {
    "target_tokens": 600,
    "overlap_tokens": 100,
    "site": "<docs-host>"
  }
}
```

### Run‑Mode Resolution (automated)
- If `force_reparse_from_assets=true` → **reparse** (offline)  
- Else if `force_redownload_from_web=true` → **online** crawl  
- Else if parsed JSONL exists → **offline** (skip crawl)  
- Else if `assets_dir/raw_html` exists → **reparse**  
- Else → **online** crawl  
- Robots preflight is **only** run in **online** mode. If seeds are blocked but cache exists, auto‑switch to **reparse**.

### Archiving
- When `fresh_run=true`, move only the files that will be regenerated into `archive/<epoch>/...`.  
- Cached HTML is preserved unless performing a brand‑new crawl.

---

## 4) Data Models (JSONL)

### 4.1 Parsed Page (`parsed_*.jsonl`)
Each line is one page:
```json
{
  "url": "https://<docs-host>/help/example",
  "title": "Example Page",
  "breadcrumbs": ["Section", "Subsection"],
  "updated_at": "2024-06-12",
  "body_text": "...clean plain text...",
  "html": "... (optional, sanitized HTML) ..."
}
```

### 4.2 Chunk (`chunks_*.jsonl`)
```json
{
  "chunk_id": "<stable-hash>",
  "url": "https://<docs-host>/help/example",
  "title": "Example Page",
  "section": "H2/H3 heading if known",
  "site": "<docs-host>",
  "updated_at": "2024-06-12",
  "text": "chunk body (≈400–700 tokens)",
  "idx": 0
}
```

Chunking rules: 600 target tokens, 100 overlap; prefer heading‑aware boundaries, then paragraph fallback. Hash is computed from `{url, idx, text[:N]}` to be stable.

---

## 5) Crawling & Parsing (LLM build spec)

**Discovery**
- BFS starting from `seed_urls`, honoring `allow_patterns` + `deny_patterns`, `same_host`, and `allowed_domains`.
- De‑duplicate by normalized URL; enqueue only HTML links.
- Stream: `[DISCOVER] depth=k visiting: <url>` plus reasons for skips.

**Fetching**
- Online: `respect_robots=true` by default; if `crawl_as_human=true`, skip robots and sleep ~2s/page.
- Save **raw HTML** to `assets_<site>/raw_html/<hash>.html` with a manifest line: `{url, path, ts}`.

**Parsing**
- Strip headers/footers/nav; keep headings, lists, tables.  
- Extract title, breadcrumbs, last‑updated if present.  
- Produce `parsed_*.jsonl`. Short/empty pages can be logged and skipped.

---

## 6) Indexing (Hybrid: BM25 + Vectors)

**Why**
- BM25 excels at exact/rare terms (IDs, URLs, product nouns).  
- Vectors capture semantics when phrasing differs.  
- Fusing both yields strong, inexpensive recall for small/medium corpora.

**Default**
- **Vectors:** FAISS index; default embedding model is a small English model (configurable).  
- **Keyword:** Lightweight BM25 (stored as a pickle).  
- `index/` stores: `faiss.index`, `bm25.pkl`, `meta.json`, `model.txt`.

**Swap Options**
- Replace FAISS with Qdrant/pgvector/Milvus by updating `index_build.py` adapter.  
- Replace embeddings with API (e.g., OpenAI) by swapping the encoder call.

Build command:
```bash
python index_build.py --chunks chunks_<site>.jsonl --out index
```

---

## 7) Retrieval & Rerank

Baseline hybrid flow:
1. Keyword search (BM25) → top **k1**  
2. Vector search (FAISS) → top **k2**  
3. Merge (reciprocal rank fusion or score‑normalize + union) → **k≈20**  
4. (Optional) **Rerank** using a cross‑encoder or API → top 3–5 to the prompt

CLI:
```bash
python query_local.py --query "how do I contact support?" --index index --chunks chunks_<site>.jsonl
python query_local.py --query "what can clinicians do?" --rerank --index index --chunks chunks_<site>.jsonl
```

---

## 8) Grounded Answer Demo

`ask_demo.py` takes the top passages and prints:
- A short **draft answer** composed only from retrieved text  
- A **citation list** (`title — url`)

System prompt (skeleton used inside the script):
- *“Answer only from provided passages. If unsure, say you don’t have enough info. Keep it short. Cite sources.”*

---

## 9) Tiny Evaluation Harness

`eval_runner.py` consumes `eval_golden.jsonl`:
```jsonl
{"q": "Where do I find Hours of Operation and how do I contact Support?", "a_contains": ["Hours", "Support"]}
{"q": "Show me the Getting Started steps with a link to training videos.", "a_contains": ["Getting Started", "training"]}
{"q": "What can clinicians do in the Console?", "a_contains": ["Console"]}
```
Metrics reported:
- **Top‑1‑ish accuracy**: does top document contain any of the expected strings?  
- **p95 latency**: retrieval timing only  
- **$/100 calls**: zero (no LLM)

Use this to compare: different chunk sizes, models, reranker on/off.

---

## 10) Observability

Log per query: input text, retrieved chunk IDs + URLs, scores, elapsed ms, errors.  
Optionally emit CSV for quick p50/p95 charts.

---

## 11) MCP Tools (to be implemented)

Expose a minimal **Knowledge Base** device:

- `kb.search(query, filters?) → [{url, title, snippet, score}]`  
  - **Args:** `query` (string), optional `filters` (`{domain?, section?, updated_after?}`)  
  - **Errors:** `INVALID_QUERY`, `BACKEND_UNAVAILABLE`

- `kb.get(url) → {title, body, updated_at}`  
  - **Args:** `url` (string)  
  - **Errors:** `NOT_FOUND`

- `kb.ask(question) → {answer, citations:[{title,url}]}`  
  - **Args:** `question` (string)  
  - **Notes:** uses retrieval + (optional) rerank + grounded draft

Implementation hint: put adapters in `mcp/` later; wire to local index + chunks JSONL.

---

## 12) Agents SDK Plan

Create a **DocsAgent** that owns the KB tools and a **WriterAgent** that formats replies for email/chat.  
Use **handoffs**: DocsAgent → WriterAgent when an answer is found.  
Expose both via a small orchestrator that routes intents.

---

## 13) A2A Façade (HTTP JSON‑RPC)

Single endpoint `/a2a/invoke`:
```json
{
  "agent": "docs",
  "task": {"question": "..."},
  "params": {"rerank": true}
}
```
Response:
```json
{
  "answer": "...",
  "citations": [{"title": "...", "url": "..."}],
  "latency_ms": 1234,
  "trace_id": "<uuid>"
}
```
Auth can be a bearer token or nothing for demo; include rate‑limit headers in the response if desired.

---

## 14) Safety, Legal, and Etiquette

- Default to `respect_robots=true`. Use `crawl_as_human=true` **only with permission**.  
- Crawl **public help/docs**. Avoid login‑gated pages.  
- Never process PHI or other sensitive personal data.  
- Provide clear user‑agent and a gentle delay when crawling.

---

## 15) Performance & Sizing Guidance

- Corpora: 50–5,000 pages; target chunk count < 20k for the demo.  
- Embedding model: small English model by default; swap for multilingual if needed.  
- Retrieval defaults: k1=k2=20 (pre‑fusion), top_k=3–5 after rerank.  
- Typical build laptop: ~2–6 minutes for 200–500 chunks.

---

## 16) Runbook (happy path)

```bash
# 0) venv
python3 -m venv .venv && source .venv/bin/activate
bash dependencies.sh

# 1) Configure
cp example_config.jsonl  demo_config.json  # or author new JSON as per §3

# 2) Crawl/Parse/Chunk (or reuse cache)
python web_crawler.py --config demo_config.json

# 3) Index
python index_build.py --chunks chunks_<site>.jsonl --out index

# 4) Query
python query_local.py --query "Where is X" --index index --chunks chunks_<site>.jsonl

# 5) Eval (optional)
python eval_runner.py --golden eval_golden.jsonl
```

**Offline rebuild:** If you have `assets_<site>/raw_html` but no parsed file, the crawler auto‑switches to **reparse** and reconstructs `parsed_*.jsonl` and `chunks_*.jsonl`.

---

## 17) Extensibility & Multi‑Customer Onboarding

- Keep per‑customer configs (`<customer>_config.json`).  
- Separate constants and utility functions; keep script entrypoints thin.  
- Add a **profile** section (future) for default chunk size/model per customer.

---

## 18) Quality Bar (Acceptance Criteria)

- **Discovery:** BFS finds ≥90% of article URLs reachable from seeds under `allow_patterns` within `max_depth` and `max_pages`.  
- **Parsing:** Titles present for ≥98% of parsed records; empty/short pages are logged.  
- **Chunking:** Mean chunk length 400–700 tokens; overlap applied; metadata present.  
- **Index:** `index/` contains FAISS + BM25 + meta; query returns results in <300ms for ≤20k chunks on a laptop.  
- **Eval:** Script prints Top‑1‑ish, p95, and $/100.  
- **Archiving:** Prior outputs moved to `archive/<epoch>/...`; cached HTML preserved unless re‑crawl is forced.  
- **Progress:** One‑line logs for every major step.

---

## 19) Prompts & Coding Guidelines (for LLM agents)

- Prefer explicit, small functions; avoid hidden state.  
- Print human‑readable progress: `[DISCOVER]`, `[PARSE]`, `[CHUNK]`, `[INDEX]`, `[QUERY]`.  
- Never hardcode site‑specific strings in logic (only in config).  
- Treat file paths as parameters; raise helpful errors with next‑step hints.  
- When refactoring, keep CLI flags stable or add shims.

---

## 20) Roadmap (light)

- MCP server exposing `kb.search|get|ask`  
- Agents SDK orchestration + handoffs  
- A2A façade with simple auth & tracing  
- Better eval (faithfulness & citation checks)  
- Observability CSV → plot  
- Swap‑in vector DB backends and embedding models via adapters
