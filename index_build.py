import json, os, pickle, argparse, sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# --- Config helpers ---
DEFAULT_CONFIG_PATH = "98point6_config.json"

def read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_chunks_path(args_chunks: str, cfg_path: str) -> str:
    # Priority 1: explicit CLI path
    if args_chunks:
        return args_chunks
    # Priority 2: outputs.chunks_jsonl_path in 98point6_config.json
    try:
        cfg = read_config(cfg_path)
        out = cfg.get("outputs", {})
        cp = out.get("chunks_jsonl_path")
        if cp:
            return cp
    except FileNotFoundError:
        pass
    # Fallback
    return "chunks_98point6.jsonl"

def load_chunks(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def main(chunks_path: str, out_dir: str = "index"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[index] output dir: {out_dir}")
    if not os.path.isfile(chunks_path):
        print(f"[index] ERROR: chunks file not found: {chunks_path}", file=sys.stderr)
        sys.exit(2)
    print(f"[index] reading chunks from: {chunks_path}")
    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]
    meta  = [{"url":c["url"],"title":c["title"],"section":c["section"],
              "updated_at":c["updated_at"],"site":c["site"],"chunk_id":c["chunk_id"]} for c in chunks]

    # ----- BM25 (keyword) -----
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(Path(out_dir)/"bm25.pkl","wb") as f:
        pickle.dump({"bm25":bm25, "tokenized":tokenized}, f)

    # ----- Vectors (dense) -----
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast
    encoder = SentenceTransformer(model_name)
    embs = encoder.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=True)
    embs = normalize(embs).astype("float32")

    index = faiss.IndexFlatIP(embs.shape[1])  # cosine via IP on normalized vectors
    index.add(embs)

    faiss.write_index(index, str(Path(out_dir)/"faiss.index"))
    with open(Path(out_dir)/"meta.jsonl","w",encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False)+"\n")
    with open(Path(out_dir)/"model.txt","w") as f:
        f.write(model_name+"\n")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"✅ Built index: {len(texts)} chunks → {out_dir}  ({ts})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build BM25 + FAISS index from chunked JSONL")
    ap.add_argument("--chunks", dest="chunks_jsonl", default=None,
                    help="Path to chunks JSONL. If omitted, will read outputs.chunks_jsonl_path from --config.")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                    help="Path to 98point6_config.json (used only to resolve default chunks path).")
    ap.add_argument("--out", default="index",
                    help="Output directory for the built index.")
    args = ap.parse_args()

    chunks_path = resolve_chunks_path(args.chunks_jsonl, args.config)
    main(chunks_path, args.out)