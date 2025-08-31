a
# RAG · MCP · A2A — Weekend-Scale Demo Kit

A clean, small toolkit to:

- **Crawl** a public help center or docs site (politely, or in a human-like mode)
- **Parse → chunk → index** content for hybrid retrieval (BM25 + vectors)
- **Query locally** (keyword, vector, or hybrid with rerank)
- **Run a tiny eval** (3–20 golden Qs)
- (Next steps) expose as **MCP tools**, wrap with an **OpenAI Agents SDK** agent, and add a thin **A2A** façade

> ⚠️ This repo avoids any customer names or proprietary content. Use only public pages you’re allowed to crawl.

---

## Quickstart

```bash
# 0) Create and activate a venv (optional)
python3 -m venv .venv && source .venv/bin/activate

# 1) Install deps
bash dependencies.sh

# 2) Configure a crawl (see config section)
cp 98point6_config.json example_config.json   # or create your own

# 3) Crawl / parse / chunk (or re-use cached HTML)
python web_crawler.py --config example_config.json

# 4) Build an index (BM25 + FAISS)
python index_build.py --chunks chunks_<site>.jsonl --out index

# 5) Try retrieval
python query_local.py --query "Where is X?" --index index --chunks chunks_<site>.jsonl

# 6) (Optional) Rerank
python query_local.py --query "What can I do in the Console?" --rerank --index index --chunks chunks_<site>.jsonl

# 7) (Optional) Tiny eval
python eval_runner.py --golden eval_golden.jsonl
```

> Tip: Replace `<site>` with your short site label (e.g., `chunks_docs.jsonl`).

---

## Repo layout

- `web_crawler.py` — discovery, fetching, parsing, chunking
- `index_build.py` — hybrid index (BM25 + vectors)
- `query_local.py` — ad‑hoc retrieve/rerank
- `ask_demo.py` — simple grounded answer demo (no LLM by default)
- `eval_runner.py` — tiny evaluation harness
- `archive/` — older runs moved here with **epoch** folder names
- `assets_<site>/` — cached HTML and plain‑text snapshots
- `parsed_<site>.jsonl` — parsed article records
- `chunks_<site>.jsonl` — chunked passages with metadata
- `index/` — built retrieval indexes (BM25 pickle, FAISS, meta)

---

## Configuration (JSON)

You pass a single JSON config to `web_crawler.py`. Keep it small and explicit.

```json
{
  "crawl": {
    "seed_urls": ["https://<docs-host>/start"],
    "allow_patterns": ["/docs/", "/help/"],
    "deny_patterns": ["/search", "/login", "[?#]preview="],
    "allowed_domains": ["<docs-host>"],
    "same_host": true,
    "max_pages": 300,
    "max_depth": 5,
    "respect_robots": true,
    "delay_seconds": 0.6,
    "timeout_seconds": 10.0,
    "user_agent": "RAG-DiscoveryBot/1.0 (+https://example.org)",

    "crawl_as_human": false,
    "force_redownload_from_web": false,
    "force_reparse_from_assets": false
  },
  "outputs": {
    "assets_dir": "assets_<site>",
    "parsed_jsonl_path": "parsed_<site>.jsonl",
    "chunks_jsonl_path": "chunks_<site>.jsonl",
    "fresh_run": true
  },
  "chunk": {
    "target_tokens": 600,
    "overlap_tokens": 100,
    "site": "<docs-host>"
  }
}
```

### Key flags & modes

- `crawl_as_human`  
  If **true**, the crawler **ignores robots.txt** and moves slowly (≈2s think time per page). Use only on sites where you have permission. Progress is streamed in one‑line messages; content is cached incrementally so you don’t lose work.

- `respect_robots`  
  If **true** and robots disallows your seeds, the script will **not crawl**. If you already have cached HTML, it will automatically fall back to **reparse** (offline) mode.

- `force_redownload_from_web`  
  If **true**, do a brand‑new crawl from the network (archives previous outputs). If **false** and cache exists, reuse the cache.

- `force_reparse_from_assets`  
  If **true**, skip crawling; rebuild `parsed_*.jsonl` from cached HTML (`assets_*/raw_html`). If a manifest is missing, it is bootstrapped automatically.

- `fresh_run`  
  Moves only the **outputs that would be regenerated** into `archive/<epoch>/...`. Cached HTML is preserved unless you explicitly choose an online crawl that replaces it.

### Output files

- **assets_dir**  
  `raw_html/` and `plain_text/` snapshots plus a minimal `cache_manifest.jsonl` (URL, path, timestamp).

- **parsed_jsonl_path**  
  One JSON object per page: `{url, title, updated_at?, breadcrumbs?, body_text, html?}`.

- **chunks_jsonl_path**  
  Passages after splitting: `{url, title, section?, updated_at?, site, chunk_id, text}`.

---

## Why this indexing recipe?

**Goal:** fast, cheap, high‑recall retrieval on small/medium doc sets.

- **BM25 (keyword)** surfaces exact term matches and handles rare tokens, URLs, and code‑like snippets well.
- **Vectors (FAISS + small embedding model)** capture semantic similarity when wording differs. Defaults target an English corpus and keep footprint light.
- **Hybrid fusion** (query both → merge → optional rerank) gives a strong baseline without heavy infra. You can:
  - swap FAISS for **Qdrant**, **pgvector**, or **Milvus**
  - swap embeddings for an API (e.g., OpenAI text‑embedding) or another local model
  - tune chunk sizes/overlap and metadata to fit your corpus

**Reranking** (cross‑encoder or API like Cohere Rerank) frequently delivers the largest accuracy bump for Q&A. It’s optional here but supported in `query_local.py` via `--rerank`.

---

## Building the index

```bash
# Build/refresh index from chunks
python index_build.py --chunks chunks_<site>.jsonl --out index

# What gets written
#  - index/faiss.index          (vector index)
#  - index/bm25.pkl            (BM25 keyword index)
#  - index/meta.json           (model + config)
#  - index/model.txt           (embedding model id)
```

To change embedding model or storage backend, edit the top of `index_build.py` (model name, dimension) and rerun.

---

## Querying locally

```bash
# Hybrid search
python query_local.py --query "your question" --index index --chunks chunks_<site>.jsonl

# With reranker (slow but more accurate)
python query_local.py --query "your question" --rerank --index index --chunks chunks_<site>.jsonl
```

`query_local.py` prints a small ranked list: title, URL, and snippet. For a grounded “answer + citations”, try `ask_demo.py`.

---

## Tiny evaluation

Create `eval_golden.jsonl` with a handful of questions:

```jsonl
{"q": "Where do I find Hours of Operation and how do I contact Support?", "a_contains": ["Hours", "Support"]}
{"q": "Show me the Getting Started steps with a link to training videos.", "a_contains": ["Getting Started", "training"]}
{"q": "What can clinicians do in the Console?", "a_contains": ["Console"]}
```

Run:

```bash
python eval_runner.py --golden eval_golden.jsonl
```

You’ll see:

- **Top‑1-ish accuracy** — did the top result actually contain the answer?
- **p95 latency** — simple timer around retrieval
- **$/100 calls** — zero here (no LLM in the loop)

Use this to compare: baseline vs. rerank, different chunk sizes, different models.

---

## Operational notes

- **Polite crawling**: keep `respect_robots=true` for public sites you don’t own. Use `crawl_as_human=true` **only** where you have explicit permission.
- **Idempotent runs**: outputs are archived under `archive/<epoch>/...`; caches are reused unless you force a new crawl.
- **Deterministic chunking**: chunk boundaries are text‑based so repeated runs are stable unless you change the tokenizer/params.
- **Portability**: everything is file‑based (JSONL + index dir). Easy to zip, move, and rebuild elsewhere.

---

## Roadmap (lightweight)

- MCP server exposing:
  - `kb.search(query, filters?)`
  - `kb.get(url)`
  - `kb.ask(question)`
- OpenAI Agents SDK integration and agent handoffs
- Simple A2A façade (`/a2a/invoke` → `{agent, task}`)
- Better eval (faithfulness, citation checks)
- Observability: CSV logs → quick latency charts

---

## FAQ

**Can I use another vector DB?**  
Yes. Replace FAISS calls with Qdrant/pgvector/Milvus; keep the same chunk schema.

**How do I avoid recrawling?**  
Set `force_redownload_from_web=false`. If parsed JSON is missing but you have `assets_*/raw_html`, the script will auto‑**reparse**.

**What about non‑English?**  
Swap to a multilingual embedding model and revisit token chunk sizes.

**What’s the license?**  
Choose a permissive license suitable for your use case (MIT/Apache‑2 recommended). Add it to the repo root.

---

## One‑liner demo

```bash
python web_crawler.py --config example_config.json && \
python index_build.py --chunks chunks_<site>.jsonl --out index && \
python query_local.py --query "Quick sanity check" --index index --chunks chunks_<site>.jsonl
```

---

### Credits

This kit is intentionally small and pragmatic. It uses standard Python tooling plus FAISS and a compact embedding model to keep setup fast.