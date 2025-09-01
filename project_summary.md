Project Summary

Goal: Build, run, and extend a modular Retrieval‑Augmented Generation (RAG) toolkit for help‑desk and documentation sites.  The kit is file‑based and heavily configurable via a single JSON schema, with clean interfaces for MCP tools, an OpenAI Agents SDK agent and a thin A2A façade.  Real company names are intentionally omitted; this framework operates exclusively on public help or documentation sites.

This document serves as the contract/spec for LLMs and agents when working with the codebase.  It defines how crawling, parsing, chunking, indexing, retrieval, evaluation and serving are orchestrated, and it highlights all of the knobs you can tune.  When copied into a prompt, it gives an LLM enough context to generate or modify the associated scripts.

⸻

0) Shared Principles (LLM × Human Contract)
	1.	Deterministic & Idempotent: Re‑running the same configuration must reproduce the same outputs (unless crawling new content).  Hashes and chunk identifiers are stable.
	2.	Cache First: Never discard previously downloaded HTML unless explicitly permitted.  Reuse cached assets and prefer reparse over re‑crawl whenever possible.
	3.	Small, Simple, Swappable: Components are modular.  Backends (embeddings, vector databases, rerankers) can be swapped with minimal edits.  Each script is thin and has a clear API.
	4.	Polite & Legal: Respect robots.txt by default.  A special crawl_as_human mode exists only for scenarios where you have explicit permission.  The user agent and delays are configurable.
	5.	Transparent: Stream one‑line progress messages for every major step (discovery, fetch, parse, chunk, index, query).  Persist intermediate artifacts (HTML → parsed → chunks → index) so that runs are auditable.
	6.	No PHI / Sensitive Data: Index only public help/docs content.  Defensively scrub navigation/ads/footers and retain only headings, lists and tables.

⸻

1) What We’re Building (Scope)

This kit can:
	•	Discover & fetch a public documentation site according to seed URLs and allow/deny patterns.
	•	Parse & clean pages into structured JSONL records (title, breadcrumbs, updated date, body text, optional HTML).
	•	Chunk & enrich text into passages with metadata, with adjustable size and overlap.
	•	Index the chunks using a hybrid pipeline combining BM25 for keywords and FAISS for dense vector search.
	•	Retrieve passages by keyword, vector or hybrid fusion, with optional reranking.
	•	Answer with citations using a simple grounded Q&A demo script.
	•	Evaluate quickly on a handful of golden questions to measure recall and latency.
	•	Expose tools via MCP, wrap them in an OpenAI Agents SDK agent, and offer a simple A2A HTTP façade.

Explicit Non‑Goals (for now)

This kit is not intended to be a full production multi‑tenant service.  It does not provide advanced authentication/quotas, streaming LLM generation, or a complex user interface.  Those can be layered on later.

⸻

1a) Configurability & Extensibility

A key design goal of this project is per‑site and per‑customer configurability.  Every run is controlled by a single JSON configuration file (see §3) which allows you to tailor the behaviour of the crawler, chunker, indexer and retriever.  The main areas you can adjust are:
	•	Discovery: Define seed_urls, allow_patterns, deny_patterns, allowed_domains, same_host, max_pages and max_depth to constrain which pages are fetched.  Control politeness with respect_robots, crawl_as_human, delay_seconds and timeout_seconds.
	•	Outputs: Choose where assets, parsed JSONL and chunk JSONL files are written (assets_dir, parsed_jsonl_path, chunks_jsonl_path) and whether to archive previous outputs (fresh_run).  Each site/customer can have its own directory.
	•	Chunking: Tune target_tokens and overlap_tokens to adapt to the structure of your source documents.  The site field labels the data for downstream indexing and retrieval.
	•	Indexing: Edit the top of index_build.py to change the embedding model and vector dimension.  The script reads your chunks and writes a FAISS index, a BM25 pickle and metadata.  You can swap FAISS for another vector backend via an adapter.
	•	Retrieval: The hybrid search fuses results from BM25 (k1) and FAISS (k2).  Both values, as well as the final top_k, can be passed via command‑line flags to query_local.py or modified in the script.  A reranker can be enabled with --rerank, and its model is swappable.
	•	Evaluation: The evaluation harness reads a JSONL file of questions with expected substrings and optional required citations.  You curate this file per customer/site.  The harness reports top‑1 recall, p95 latency and cost estimates.

Because everything is file‑based, you can maintain multiple customer profiles (e.g. example_config.json, customerB_config.json) without changing code.  When invoking the scripts, simply point them at the appropriate config.  New features—such as a different vector database, a multilingual embedding model or a custom reranker—can be introduced by editing adapters or flags without touching the core data structures.

⸻

2) Repository Layout (expected)