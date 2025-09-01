
# Project Summary

> **Goal:** Build, run, and extend a **modular Retrieval‑Augmented Generation (RAG) toolkit** for help‑desk and documentation sites.  The kit is **file‑based** and heavily **configurable** via a single JSON schema, with clean interfaces for **MCP tools**, an **OpenAI Agents SDK** agent and a thin **A2A** façade.  Real company names are intentionally omitted; this framework operates exclusively on public help or documentation sites.

This document serves as the contract/spec for LLMs and agents when working with the codebase.  It defines how crawling, parsing, chunking, indexing, retrieval, evaluation and serving are orchestrated, and it highlights all of the knobs you can tune.  When copied into a prompt, it gives an LLM enough context to generate or modify the associated scripts.

---

## 0) Shared Principles (LLM × Human Contract)

1. **Deterministic & Idempotent:** Re‑running the same configuration must reproduce the same outputs (unless crawling new content).  Hashes and chunk identifiers are stable.
2. **Cache First:** Never discard previously downloaded HTML unless explicitly permitted.  Reuse cached assets and prefer **reparse** over re‑crawl whenever possible.
3. **Small, Simple, Swappable:** Components are modular.  Backends (embeddings, vector databases, rerankers) can be swapped with minimal edits.  Each script is thin and has a clear API.
4. **Polite & Legal:** Respect `robots.txt` by default.  A special `crawl_as_human` mode exists only for scenarios where you have explicit permission.  The user agent and delays are configurable.
5. **Transparent:** Stream one‑line progress messages for every major step (discovery, fetch, parse, chunk, index, query).  Persist intermediate artifacts (HTML → parsed → chunks → index) so that runs are auditable.
6. **No PHI / Sensitive Data:** Index only public help/docs content.  Defensively scrub navigation/ads/footers and retain only headings, lists and tables.

---

## 1) What We’re Building (Scope)

This kit can:

- **Discover & fetch** a public documentation site according to seed URLs and allow/deny patterns.
- **Parse & clean** pages into structured JSONL records (title, breadcrumbs, updated date, body text, optional HTML).
- **Chunk & enrich** text into passages with metadata, with adjustable size and overlap.
- **Index** the chunks using a **hybrid pipeline** combining BM25 for keywords and FAISS for dense vector search.
- **Retrieve** passages by keyword, vector or hybrid fusion, with optional reranking.
- **Answer with citations** using a simple grounded Q&A demo script.
- **Evaluate quickly** on a handful of golden questions to measure recall and latency.
- **Expose tools** via MCP, wrap them in an OpenAI Agents SDK agent, and offer a simple A2A HTTP façade.

### Explicit Non‑Goals (for now)

This kit is not intended to be a full production multi‑tenant service.  It does not provide advanced authentication/quotas, streaming LLM generation, or a complex user interface.  Those can be layered on later.

---

## 1a) Configurability & Extensibility

A key design goal of this project is **per‑site and per‑customer configurability**.  Every run is controlled by a single JSON configuration file (see §3) which allows you to tailor the behaviour of the crawler, chunker, indexer and retriever.  The main areas you can adjust are:

- **Discovery:** Define `seed_urls`, `allow_patterns`, `deny_patterns`, `allowed_domains`, `same_host`, `max_pages` and `max_depth` to constrain which pages are fetched.  Control politeness with `respect_robots`, `crawl_as_human`, `delay_seconds` and `timeout_seconds`.
- **Outputs:** Choose where assets, parsed JSONL and chunk JSONL files are written (`assets_dir`, `parsed_jsonl_path`, `chunks_jsonl_path`) and whether to archive previous outputs (`fresh_run`).  Each site/customer can have its own directory.
- **Chunking:** Tune `target_tokens` and `overlap_tokens` to adapt to the structure of your source documents.  The `site` field labels the data for downstream indexing and retrieval.
- **Indexing:** Edit the top of `index_build.py` to change the embedding model and vector dimension.  The script reads your chunks and writes a FAISS index, a BM25 pickle and metadata.  You can swap FAISS for another vector backend via an adapter.
- **Retrieval:** The hybrid search fuses results from BM25 (`k1`) and FAISS (`k2`).  Both values, as well as the final `top_k`, can be passed via command‑line flags to `query_local.py` or modified in the script.  A reranker can be enabled with `--rerank`, and its model is swappable.
- **Evaluation:** The evaluation harness reads a JSONL file of questions with expected substrings and optional required citations.  You curate this file per customer/site.  The harness reports top‑1 recall, p95 latency and cost estimates.

Because everything is file‑based, you can maintain multiple customer profiles (e.g. `example_config.json`, `customerB_config.json`) without changing code.  When invoking the scripts, simply point them at the appropriate config.  New features—such as a different vector database, a multilingual embedding model or a custom reranker—can be introduced by editing adapters or flags without touching the core data structures.

---

## 2) Repository Layout (expected)

```
assets_<site>/            # cached HTML & plain-text snapshots
archive/<epoch>/          # prior outputs moved here when fresh_run=true
chunks_<site>.jsonl       # chunked passages (configurable size/overlap)
parsed_<site>.jsonl       # parsed article records (title, breadcrumbs, body)
index/                    # FAISS/BM25 artifacts + meta
web_crawler.py            # crawl → parse → chunk, plus offline reparse
index_build.py            # build hybrid index (BM25 + vectors)
query_local.py            # ad‑hoc retrieve/rerank
ask_demo.py               # grounded answer + citations (no LLM by default)
eval_runner.py            # tiny evaluation harness
<site>_config.json        # JSON config for a run
README.md                 # user‑facing walkthrough for running the kit
project_summary.md        # THIS file — contract/spec for LLMs & agents
```

Keep per‑customer configs in this repository under descriptive names.  The scripts never hardcode a site name; they accept config paths via command‑line flags.

---

## 3) Configuration Schema (single JSON)

The crawler and chunker are driven by a single JSON file.  A minimal example:

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

All fields are explicit so both humans and models can parse them unambiguously.  To run against a different site, copy this template, adjust the URLs and patterns, tweak the crawl limits and chunk sizes, and save it under a new name.  Scripts receive the config via `--config path/to/config.json` and honour its settings throughout the pipeline.

### Run‑Mode Resolution (automated)

The crawler supports three modes and automatically picks the right one:

1. If `force_reparse_from_assets=true` → **reparse**: skip crawling entirely; rebuild parsed and chunked JSONL from cached HTML.
2. Else if `force_redownload_from_web=true` → **online**: start a brand‑new crawl, archiving previous outputs.
3. Else if a parsed JSONL file already exists → **offline**: skip crawling; use cached parsed and chunk data.
4. Else if `assets_dir/raw_html` exists → **reparse**: rebuild parsed and chunked files from cached raw HTML.
5. Otherwise → **online**: perform a fresh crawl.

When in **online** mode, a robots preflight is run on the seed URLs.  If the seeds are disallowed but you have cached HTML, the script automatically falls back to **reparse**.

### Archiving

When `fresh_run` is `true` the crawler moves only the outputs that will be regenerated (parsed JSONL, chunks JSONL, indexes) into `archive/<epoch>/...`.  Cached HTML under `assets_dir` is preserved unless you explicitly force a re‑crawl.  This keeps the run deterministic while letting you retain expensive downloads.

---

## 4) Data Models (JSONL)

### 4.1 Parsed Page (`parsed_*.jsonl`)

Each line is a JSON object representing a single page:

```json
{
  "url": "https://<docs-host>/help/example",
  "title": "Example Page",
  "breadcrumbs": ["Section", "Subsection"],
  "updated_at": "2024-06-12",
  "body_text": "...clean plain text...",
  "html": "...optional, sanitised HTML..."
}
```

Pages with very little content (e.g. just a title) may be logged and skipped.  Breadcrumbs and `updated_at` are extracted where available.

### 4.2 Chunk (`chunks_*.jsonl`)

Each chunk object contains the passage text and its provenance:

```json
{
  "chunk_id": "<stable-hash>",
  "url": "https://<docs-host>/help/example",
  "title": "Example Page",
  "section": "Heading if known",
  "site": "<docs-host>",
  "updated_at": "2024-06-12",
  "text": "chunk body (≈400–700 tokens)",
  "idx": 0
}
```

The hash is computed from `(url, idx, text[:N])` so that repeated runs produce identical identifiers.  `idx` is the zero‑based position of the chunk within its page.  Chunk boundaries are determined by aiming for `target_tokens` (600 by default) with an overlap of `overlap_tokens` (100 by default).  Headings are preferred as break points; otherwise paragraphs are used.  Adjust these values in the config to suit your content.

---

## 5) Crawling & Parsing

**Discovery:** The crawler uses breadth‑first search from `seed_urls`, following links that match `allow_patterns` and do not match `deny_patterns`.  It respects the `same_host` and `allowed_domains` constraints, stops at `max_pages` or `max_depth`, and de‑duplicates on normalised URLs.  Progress lines like `[DISCOVER] depth=k visiting: <url>` and skip reasons keep you informed.

**Fetching:** In online mode, the crawler fetches pages using the specified `user_agent`, `delay_seconds`, `timeout_seconds` and `respect_robots` settings.  If `crawl_as_human=true` it ignores robots and waits ~2s per page.  Raw HTML is saved to `assets_<site>/raw_html/<hash>.html` and a manifest records `{url, path, timestamp}`.

**Parsing:** Raw HTML is cleaned by removing headers, footers and navigation.  Headings, lists and tables are retained.  Titles, breadcrumbs and updated dates are extracted when possible.  The result is written to `parsed_*.jsonl`.  Short or empty pages are logged and skipped.

---

## 6) Indexing (Hybrid: BM25 + Vectors)

**Why hybrid?**  BM25 excels at exact/rare term matches (identifiers, URLs, product names), while dense vectors capture semantic similarity when wording differs.  Fusing both yields high recall on small/medium corpora without heavy infrastructure.

The default index builder writes:

- **Vectors:** a FAISS index built from sentence embeddings.  The embedding model and dimension are defined at the top of `index_build.py`.  You can replace FAISS with another vector database (e.g. Qdrant, pgvector, Milvus) by swapping the adapter.
- **Keyword:** a lightweight BM25 index stored as a pickle.
- **Meta:** a JSON file describing the model and config used for the index.

Run it with:

```bash
python index_build.py --chunks chunks_<site>.jsonl --out index
```

Adjust the embedding model, dimension or vector backend by editing `index_build.py`.  The script processes your chunked JSONL and writes all artifacts into `index/`.

---

## 7) Retrieval & Rerank

The baseline hybrid retrieval works as follows:

1. **Keyword search:** Query the BM25 index for the top `k1` chunks.
2. **Vector search:** Query the FAISS index for the top `k2` chunks.
3. **Merge:** Combine the two sets using reciprocal rank fusion or score normalisation, returning roughly 20 candidates.
4. **(Optional) Rerank:** Pass the candidates through a cross‑encoder reranker (or API) to select the top 3–5 most relevant chunks.

Run it with:

```bash
python query_local.py --query "your question" --index index --chunks chunks_<site>.jsonl

# With reranker (slow but often more accurate)
python query_local.py --query "your question" --rerank --index index --chunks chunks_<site>.jsonl
```

Use the `--topk`, `--k1` and `--k2` flags to adjust how many keyword and vector results are fused and how many top passages are returned.  Reranking is toggled with `--rerank`; its model can be swapped inside the script.

The CLI prints a ranked list of passages (title, URL and snippet).  For a grounded answer with citations, use `ask_demo.py`, which stitches together a short draft answer from the top passages and lists their sources.

---

## 8) Grounded Answer Demo

`ask_demo.py` takes a question, retrieves passages using the hybrid pipeline and produces:

- A concise **draft answer** composed solely from the retrieved text.  If there is insufficient information, it states so.
- A **citation list** of `title — url` for the passages used.

The system prompt embedded in the script instructs the answer generator to rely only on provided passages and to keep the response short.  This script does not call an LLM by default; it simply concatenates snippets.  Swap in a generative model if desired.

---

## 9) Tiny Evaluation Harness

To quickly gauge recall and latency, create an `eval_golden.jsonl` with questions and expected substrings:

```jsonl
{"q": "Where do I find Hours of Operation and how do I contact Support?", "a_contains": ["Hours", "Support"], "must_cite": ["support", "hours"]}
{"q": "Show me the Getting Started steps with a link to training videos.", "a_contains": ["Getting Started", "training"]}
{"q": "What can clinicians do in the Console?", "a_contains": ["Console"]}
```

Then run:

```bash
python eval_runner.py --golden eval_golden.jsonl
```

The harness reports:

- **Top‑1‑ish accuracy:** whether the top retrieved passage contains all of the expected substrings and, if provided, cites at least one of the `must_cite` keywords.
- **p95 latency:** a simple timer around retrieval.
- **$/100 calls:** an estimated cost assuming no LLM is in the loop (zero by default).

Use this script to compare different configurations (chunk sizes, models, reranker on/off).  The evaluation file can contain 3–20 questions; more will slow down the loop.

---

## 10) Observability

Each query and crawl step emits structured logs: input text, retrieved chunk IDs and scores, elapsed milliseconds, errors or skip reasons.  You can write these logs to CSV to quickly plot p50/p95 latency or recall.  Observability is key to understanding performance across customers.

---

## 11) MCP Tools (to be implemented)

Expose a minimal **Knowledge Base** device with three functions:

- `kb.search(query, filters?) → [{url, title, snippet, score}]`  – accept a text query and optional filters (domain, section, updated_after) and return ranked passages.
- `kb.get(url) → {title, body, updated_at}` – fetch a single document by URL.
- `kb.ask(question) → {answer, citations:[{title,url}]}` – run a retrieval + optional rerank + grounded answer, returning the answer and citations.

Errors such as `INVALID_QUERY` or `BACKEND_UNAVAILABLE` should be surfaced to callers.  Because the underlying retrieval functions are file‑based, wiring them into MCP adapters is straightforward: load the index and chunks per the current config and call the hybrid search.  Filters map naturally to metadata fields such as `site`, `section` and `updated_at`.

---

## 12) Agents SDK Plan

Create a **DocsAgent** that owns the KB tools and a **WriterAgent** that formats replies for email/chat.  The DocsAgent handles retrieval and grounded answering; the WriterAgent focuses on tone and delivery.  The orchestrator routes intents between them.  Because the underlying functions are deterministic and parameterised via config, the agents can be given strong guarantees about reproducibility.

---

## 13) A2A Façade (HTTP JSON‑RPC)

Offer a single endpoint `/a2a/invoke` which accepts a JSON request:

```json
{
  "agent": "docs",
  "task": {"question": "..."},
  "params": {"rerank": true}
}
```

The response includes the answer, citations, latency and a trace ID:

```json
{
  "answer": "...",
  "citations": [{"title": "...", "url": "..."}],
  "latency_ms": 1234,
  "trace_id": "<uuid>"
}
```

Authentication can be a simple bearer token.  Rate limiting and logging headers can be added as needed.  Because the KB functions are pure and file‑based, the HTTP layer remains thin.

---

## 14) Safety, Legal, and Etiquette

- Default to `respect_robots=true` and abide by each site’s crawl policies.  Use `crawl_as_human=true` only when you have explicit permission and be prepared to justify it.
- Crawl only public help or documentation pages.  Avoid login‑gated content and do not process personal data.
- Provide a clear `User-Agent` and a gentle delay between requests.  Do not hammer servers.

---

## 15) Performance & Sizing Guidance

- Designed for corpora of roughly 50–5,000 pages.  Aim for fewer than 20k chunks for the demo.
- The default small English embedding model is fast; swap in a multilingual model for non‑English content.
- Retrieval defaults are `k1=k2=20` with an overlap of 100 tokens.  Tune these to trade off recall and latency.  On a typical laptop, building an index for ~200–500 chunks takes 2–6 minutes.

---

## 16) Runbook (happy path)

```bash
# 0) Create a virtual environment and install dependencies
python3 -m venv .venv && source .venv/bin/activate
bash dependencies.sh

# 1) Prepare a config
cp example_config.json mydocs_config.json  # or author a new JSON per §3

# 2) Crawl, parse & chunk (or reuse cache)
python web_crawler.py --config mydocs_config.json

# 3) Build the hybrid index
python index_build.py --chunks chunks_<site>.jsonl --out index

# 4) Query the knowledge base
python query_local.py --query "Where is X?" --index index --chunks chunks_<site>.jsonl

# 5) Evaluate (optional)
python eval_runner.py --golden eval_golden.jsonl
```

If you already have raw HTML under `assets_<site>/raw_html` but no parsed file, the crawler automatically switches to **reparse** and reconstructs `parsed_*.jsonl` and `chunks_*.jsonl`.

---

## 17) Extensibility & Multi‑Customer Onboarding

- Keep per‑customer configs (`<customer>_config.json`) in the repository.  Each customer can have different crawl patterns, chunk sizes and models.
- Separate constants and utility functions; keep script entrypoints thin so they remain easy to wrap as tools.
- A future **profile** section could specify default chunk size and embedding model per customer; the scripts would read it to avoid repeating values in the config.

---

## 18) Quality Bar (Acceptance Criteria)

- **Discovery:** BFS should find ≥90% of article URLs reachable from seeds under the allow patterns within the defined depth and page limits.
- **Parsing:** Titles should be present for ≥98% of parsed records; empty or overly short pages should be logged.
- **Chunking:** Mean chunk length should be between 400 and 700 tokens with the configured overlap; metadata should be present on every chunk.
- **Index:** The `index/` directory should contain FAISS, BM25 and meta artifacts.  Queries should return results in under 300 ms for corpora up to ~20k chunks on a laptop.
- **Eval:** The evaluation harness should print top‑1 recall, p95 latency and estimated cost per 100 calls.
- **Archiving:** Prior outputs should be moved to `archive/<epoch>/...` when regenerating them; cached HTML should be preserved unless a fresh crawl is forced.
- **Progress:** Scripts should emit one‑line logs for every major step.

---

## 19) Prompts & Coding Guidelines (for LLM agents)

- Prefer explicit, small functions and avoid hidden state.
- Print human‑readable progress lines prefixed with tags such as `[DISCOVER]`, `[PARSE]`, `[CHUNK]`, `[INDEX]`, `[QUERY]`.
- Never hardcode site‑specific strings into logic; read them from the config file.
- Treat file paths and other parameters as arguments.  Raise helpful errors with next‑step hints when something goes wrong.
- When refactoring, keep CLI flags stable or add shims to avoid breaking consumers.

---

## 20) Roadmap (light)

- Implement the MCP knowledge base server exposing `kb.search`, `kb.get` and `kb.ask`.
- Integrate with the OpenAI Agents SDK for orchestrating DocsAgent and WriterAgent.
- Build a simple A2A façade with authentication, tracing and rate limiting.
- Expand the evaluation harness to measure faithfulness and citation quality, not just recall.
- Add observability: emit CSV logs that can be turned into latency and recall charts.
- Introduce adapters for other vector databases and embedding models.

---

This summary, together with the configuration schema and scripts, forms the complete “bible” for this project.  Copy it into your LLM prompts when generating code or instructions so that the model has a clear, unambiguous specification to follow.