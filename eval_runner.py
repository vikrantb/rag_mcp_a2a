import json, argparse, time, os, sys
from ask_demo import summarize_from_context
from query_local import load_index, load_chunk_texts, bm25_topk, dense_topk, gather_candidates
from sentence_transformers import SentenceTransformer

def run_eval(index_dir, chunks_path, golden_path):
    if not os.path.isfile(golden_path):
        print(f"[eval] ERROR: golden file not found: {golden_path}", file=sys.stderr)
        print("Hint: create eval_golden.jsonl with lines like:")
        print('{"q":"Where do I find Hours of Operation and how do I contact Support?","a_contains":["Hours of Operation","Support"],"must_cite":["support","hours"]}')
        return 0, 0.0
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = [json.loads(l) for l in f]
    bm25, tokenized, faiss_idx, model_name, meta = load_index(index_dir)
    texts = load_chunk_texts(chunks_path)
    enc = SentenceTransformer(model_name)
    for i in range(len(meta)):
        if "text" not in meta[i] and i < len(texts):
            meta[i]["text"] = texts[i]

    ok = 0
    latencies = []
    for row in golden:
        t0 = time.time()
        bm_idx, bm_sc = bm25_topk(bm25, tokenized, row["q"], k=20)
        d_idx, d_sc = dense_topk(faiss_idx, enc, row["q"], k=20)
        cand = gather_candidates(bm_idx, bm_sc, d_idx, d_sc, alpha=0.5, topk=20)
        ctx = [meta[i] for i in cand[:4]]
        ans, cites = summarize_from_context(row["q"], ctx)
        latencies.append(time.time()-t0)
        # Accept either a string or a list for convenience
        contains_list = row.get("a_contains", [])
        if isinstance(contains_list, str):
            contains_list = [contains_list]
        pass_contains = all((s.lower() in ans.lower()) for s in contains_list)

        must_list = row.get("must_cite", [])
        if isinstance(must_list, str):
            must_list = [must_list]
        if not must_list:
            pass_cite = True
        else:
            pass_cite = any(
                any((k.lower() in (c.get("url", "").lower() + " " + c.get("title", "").lower())) for k in must_list)
                for c in cites
            )
        ok += 1 if (pass_contains and pass_cite) else 0

    p95 = sorted(latencies)[int(len(latencies)*0.95)-1] if latencies else 0.0
    print(f"Top-1-ish accuracy: {ok}/{len(golden)} = {ok/len(golden):.2f}")
    print(f"p95 latency: {p95:.2f}s over {len(golden)} queries")
    print(f"$/100 calls: ~0 (no LLM in loop)")
    return ok, p95

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index")
    ap.add_argument("--chunks", default="chunks_98point6.jsonl")
    ap.add_argument("--golden", default="eval_golden.jsonl")
    args = ap.parse_args()
    run_eval(args.index, args.chunks, args.golden)