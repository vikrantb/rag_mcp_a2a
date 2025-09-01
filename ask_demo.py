import argparse, json, os
from query_local import (load_index, load_chunk_texts, bm25_topk, dense_topk, gather_candidates, pretty, load_query_config, resolve_from_config)
from sentence_transformers import SentenceTransformer
import numpy as np

def summarize_from_context(question, contexts, max_len=1200):
    # very basic extractive-style stitching for demo
    body = []
    used = []
    for c in contexts:
        snippet = c["text"][:400].replace("\n"," ")
        body.append(f"- {snippet}")
        used.append({"title": c.get("title",""), "url": c.get("url","")})
    answer = "Answer (draft):\n" + "\n".join(body)
    return answer, used

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Grounded answer demo. Supports per-customer --config so you don't repeat paths.")
    ap.add_argument("--config", help="Path to per-customer query config JSON (e.g., 98point6/query_config.json)")
    ap.add_argument("--index", default=None, help="Index directory; if omitted, read from --config")
    ap.add_argument("--chunks", default=None, help="Chunks JSONL; if omitted, read from --config")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--topn", type=int, default=4)
    args = ap.parse_args()

    # Resolve paths/params from optional config, mirroring query_local.py behavior
    dense_model_override = None
    cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # unused here, but kept for parity
    if args.config:
        qcfg = load_query_config(args.config)
        # reuse resolver to compute paths and defaults
        args.index, args.chunks, args.k, args.topn, _rerank, dense_model_override, cross_encoder_name = (
            resolve_from_config(qcfg, args.index, args.chunks, args.k, args.topn, None)
        )
    # Fallbacks if still missing
    if args.index is None:
        args.index = "index"
    if args.chunks is None:
        args.chunks = "chunks.jsonl"

    bm25, tokenized, faiss_idx, model_name, meta = load_index(args.index)
    texts = load_chunk_texts(args.chunks)
    enc = SentenceTransformer(model_name)

    bm_idx, bm_sc = bm25_topk(bm25, tokenized, args.query, k=args.k)
    d_idx, d_sc = dense_topk(faiss_idx, enc, args.query, k=args.k)
    cand = gather_candidates(bm_idx, bm_sc, d_idx, d_sc, alpha=0.5, topk=args.k)

    for i in range(len(meta)):
        if "text" not in meta[i] and i < len(texts):
            meta[i]["text"] = texts[i]
    contexts = [meta[i] for i in cand[:args.topn]]

    answer, cites = summarize_from_context(args.query, contexts)
    print("\n=== GROUNDED ANSWER (DRAFT) ===\n")
    print(answer)
    print("\nCitations:")
    for c in cites:
        print(f"- {c['title']} — {c['url']}")