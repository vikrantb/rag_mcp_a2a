import json, os, pickle, argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

def load_index(index_dir: str):
    index_dir = Path(index_dir)
    with open(index_dir/"bm25.pkl","rb") as f:
        bm25_obj = pickle.load(f)
    bm25 = bm25_obj["bm25"]
    tokenized = bm25_obj["tokenized"]

    faiss_idx = faiss.read_index(str(index_dir/"faiss.index"))
    with open(index_dir/"model.txt","r") as f:
        model_name = f.read().strip()
    with open(index_dir/"meta.jsonl","r",encoding="utf-8") as f:
        meta = [json.loads(line) for line in f]
    # texts live only in chunks file; for demo we keep them in meta? if not, pass a --chunks
    return bm25, tokenized, faiss_idx, model_name, meta

def normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    return m / n

def bm25_topk(bm25, tokenized, query: str, k: int) -> List[int]:
    scores = bm25.get_scores(query.split())
    return np.argsort(scores)[::-1][:k], scores

def dense_topk(faiss_idx, encoder: SentenceTransformer, query: str, k: int) -> Tuple[List[int], List[float]]:
    q = encoder.encode([query], convert_to_numpy=True)
    q = normalize(q.astype("float32"))
    D, I = faiss_idx.search(q, k)
    return I[0], D[0]

def gather_candidates(bm25_idx, bm25_scores, dense_idx, dense_scores, alpha=0.5, topk=20):
    # simple score fusion by z-normalized rank
    all_ids = list(set(bm25_idx.tolist() + dense_idx.tolist()))
    # build score maps
    bmap = {i: bm25_scores[i] for i in all_ids}
    dmap = {i: dense_scores[dense_idx.tolist().index(i)] if i in dense_idx else 0.0 for i in all_ids}
    # rank-normalize
    def rnorm(sorted_ids):
        ranks = {i: r for r, i in enumerate(sorted_ids, start=1)}
        return {i: 1.0/ranks.get(i, 1e9) for i in all_ids}
    b_r = rnorm(list(np.argsort(bm25_scores)[::-1]))
    d_r = rnorm(list(dense_idx))
    fused = [(i, alpha*b_r.get(i,0.0) + (1-alpha)*d_r.get(i,0.0)) for i in all_ids]
    fused.sort(key=lambda x: x[1], reverse=True)
    return [i for i,_ in fused[:topk]]

def pretty(passage: dict, max_chars=280) -> str:
    t = passage.get("title","")
    url = passage.get("url","")
    sec = passage.get("section","") or ""
    return f"{t} — {sec}\n{url}\n{passage.get('text','')[:max_chars].strip().replace('\\n',' ')}..."

def load_chunk_texts(chunks_path: str):
    texts = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            texts.append(j["text"])
    return texts

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hybrid query → (optional) rerank → print top docs")
    ap.add_argument("--index", default="index", help="Index folder with bm25.pkl, faiss.index, meta.jsonl, model.txt")
    ap.add_argument("--chunks", default="chunks_98point6.jsonl", help="Chunks JSONL containing text bodies")
    ap.add_argument("--query", required=True, help="Your question")
    ap.add_argument("--k", type=int, default=20, help="Candidate pool size")
    ap.add_argument("--topn", type=int, default=4, help="Final top N to display")
    ap.add_argument("--rerank", action="store_true", help="Use CrossEncoder reranker (slower, better)")
    args = ap.parse_args()

    bm25, tokenized, faiss_idx, model_name, meta = load_index(args.index)
    texts = load_chunk_texts(args.chunks)

    dense_encoder = SentenceTransformer(model_name)
    bm25_idx, bm25_scores = bm25_topk(bm25, tokenized, args.query, k=args.k)
    dense_idx, dense_scores = dense_topk(faiss_idx, dense_encoder, args.query, k=args.k)
    cand_ids = gather_candidates(bm25_idx, bm25_scores, dense_idx, dense_scores, alpha=0.5, topk=args.k)

    # attach text to meta on the fly for display
    for i in range(len(meta)):
        if "text" not in meta[i] and i < len(texts):
            meta[i]["text"] = texts[i]

    if args.rerank:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[args.query, meta[i]["text"]] for i in cand_ids]
        scores = reranker.predict(pairs)
        reranked = [i for i,_ in sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)]
        top_ids = reranked[:args.topn]
    else:
        top_ids = cand_ids[:args.topn]

    print("\n=== RESULTS ===")
    for rank, i in enumerate(top_ids, start=1):
        print(f"\n[{rank}] {pretty(meta[i])}")