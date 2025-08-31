# index_build.py
# Build hybrid (BM25 + FAISS) indexes from chunked JSONL — config-driven.

import json, os, pickle, argparse, sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ------------------------
# Defaults
# ------------------------
DEFAULT_CONFIG_PATH = "98point6_config.json"  # any per-customer config is fine

# ------------------------
# IO helpers
# ------------------------
def read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------------------------
# Text utils
# ------------------------
def simple_tokenize(text: str) -> List[str]:
    return text.lower().split()

def normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

# ------------------------
# Builders
# ------------------------
def build_bm25(texts: List[str], tokenizer: str = "simple") -> Dict[str, Any]:
    if tokenizer == "simple":
        tokenized = [simple_tokenize(t) for t in texts]
    else:
        # Hook for future tokenizers
        tokenized = [simple_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return {"bm25": bm25, "tokenized": tokenized}

def build_embeddings(texts: List[str],
                     model_name: str,
                     batch_size: int,
                     do_normalize: bool) -> np.ndarray:
    encoder = SentenceTransformer(model_name)
    embs = encoder.encode(
        texts,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True
    )
    if do_normalize:
        embs = normalize(embs)
    return embs.astype("float32")

def build_faiss(embs: np.ndarray, index_type: str = "IndexFlatIP") -> faiss.Index:
    dim = embs.shape[1]
    if index_type == "IndexFlatL2":
        idx = faiss.IndexFlatL2(dim)
    else:
        # default: cosine via IP on normalized vectors
        idx = faiss.IndexFlatIP(dim)
    idx.add(embs)
    return idx

# ------------------------
# Path resolution from config + CLI
# ------------------------
def resolve_paths(cfg: dict,
                  cli_chunks: str | None,
                  cli_out: str | None) -> tuple[str, str]:
    outputs = cfg.get("outputs", {})
    index_cfg = cfg.get("index", {})

    chunks_path = (
        cli_chunks
        or outputs.get("chunks_jsonl_path")
        or "chunks.jsonl"
    )
    out_dir = (
        cli_out
        or index_cfg.get("out_dir")
        or "index"
    )
    return chunks_path, out_dir

# ------------------------
# Main
# ------------------------
ESSENTIAL_META = ["url", "title", "section", "updated_at", "site", "chunk_id"]

def main(config_path: str,
         chunks_override: str | None,
         out_dir_override: str | None) -> None:
    cfg = read_config(config_path)
    chunks_path, out_dir = resolve_paths(cfg, chunks_override, out_dir_override)

    # Load chunks
    if not os.path.isfile(chunks_path):
        print(f"[index] ERROR: chunks file not found: {chunks_path}", file=sys.stderr)
        sys.exit(2)
    chunks = read_jsonl(chunks_path)
    texts: List[str] = [c.get("text", "") for c in chunks]
    meta: List[Dict[str, Any]] = [{k: c.get(k) for k in ESSENTIAL_META} for c in chunks]

    # Index config
    index_cfg = cfg.get("index", {})
    bm25_cfg = index_cfg.get("bm25", {"enabled": True})
    emb_cfg = index_cfg.get("embedding", {})
    faiss_cfg = index_cfg.get("faiss", {})

    model_name = emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = int(emb_cfg.get("batch_size", 64))
    do_normalize = bool(emb_cfg.get("normalize", True))
    faiss_type = faiss_cfg.get("index_type", "IndexFlatIP")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[index] output dir: {out_dir}")
    print(f"[index] chunks: {chunks_path} (records={len(texts)})")

    # BM25 (optional)
    if bm25_cfg.get("enabled", True):
        tok_name = bm25_cfg.get("tokenizer", "simple")
        print(f"[bm25] building (tokenizer='{tok_name}') …")
        bm25_pack = build_bm25(texts, tokenizer=tok_name)
        with open(Path(out_dir) / "bm25.pkl", "wb") as f:
            pickle.dump(bm25_pack, f)
    else:
        print("[bm25] skipped (disabled)")

    # Embeddings + FAISS
    print(f"[emb] encoder='{model_name}' batch={batch_size} normalize={do_normalize}")
    embs = build_embeddings(texts, model_name=model_name,
                            batch_size=batch_size, do_normalize=do_normalize)
    print(f"[faiss] building index_type='{faiss_type}' …")
    faiss_index = build_faiss(embs, index_type=faiss_type)

    # Persist
    faiss.write_index(faiss_index, str(Path(out_dir) / "faiss.index"))
    write_jsonl(Path(out_dir) / "meta.jsonl", meta)
    with open(Path(out_dir) / "model.txt", "w", encoding="utf-8") as f:
        f.write(model_name + "\n")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"✅ Built index: {len(texts)} chunks → {out_dir}  ({ts})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build hybrid index (BM25 + FAISS) from chunked JSONL using a config file."
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                    help="Path to per-customer config JSON.")
    ap.add_argument("--chunks", dest="chunks_jsonl", default=None,
                    help="Optional override: path to chunks JSONL.")
    ap.add_argument("--out", dest="out_dir", default=None,
                    help="Optional override: output directory for the index.")
    args = ap.parse_args()
    main(args.config, args.chunks_jsonl, args.out_dir)