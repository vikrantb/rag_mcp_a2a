import argparse, json
from query_local import load_index, load_chunk_texts, bm25_topk, dense_topk, gather_candidates, pretty
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index")
    ap.add_argument("--chunks", default="chunks_98point6.jsonl")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--topn", type=int, default=4)
    args = ap.parse_args()

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