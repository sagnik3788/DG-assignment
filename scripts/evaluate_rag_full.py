"""Compare current BM25 RAG, multilingual dense RAG, and hybrid RRF on the 70-question set.

Run: python scripts/evaluate_rag_full.py [--model intfloat/multilingual-e5-small]
Requires optional sentence-transformers. Generation calls are deliberately not made.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.listings import filter_listings, is_listing_query, load_listing_records
from app.main import ChatRequest, chat
from app.retrieval import chunks, retrieve, retrieve_bm25, sufficient_evidence, unsupported_intent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ranks_for(query: str, docs: list[dict], passages: list[dict], vectors, embed_query):
    bm_passages = retrieve_bm25(query, docs, limit=len(passages))
    bm_urls = list(dict.fromkeys(p["url"] for p in bm_passages))
    qv = embed_query(query)
    dense_idx = sorted(range(len(passages)), key=lambda i: float(vectors[i] @ qv), reverse=True)
    dense_urls = list(dict.fromkeys(passages[i]["url"] for i in dense_idx))
    bm_rank = {url: rank for rank, url in enumerate(bm_urls, 1)}
    dense_rank = {url: rank for rank, url in enumerate(dense_urls, 1)}
    hybrid_urls = sorted(set(bm_rank) | set(dense_rank), key=lambda url: (
        -(1 / (60 + bm_rank.get(url, 10**9)) + 1 / (60 + dense_rank.get(url, 10**9))),
        bm_rank.get(url, 10**9), dense_rank.get(url, 10**9), url))
    return {"BM25": bm_urls, "Dense": dense_urls, "Hybrid-RRF": hybrid_urls}, bm_passages, float(vectors[dense_idx[0]] @ qv)


def summarize(rank_values: list[int | None]) -> str:
    n = len(rank_values) or 1
    r1 = sum(r is not None and r <= 1 for r in rank_values) / n
    r5 = sum(r is not None and r <= 5 for r in rank_values) / n
    r10 = sum(r is not None and r <= 10 for r in rank_values) / n
    mrr = sum(1 / r for r in rank_values if r is not None) / n
    return f"n={len(rank_values)} R@1={r1:.1%} R@5={r5:.1%} R@10={r10:.1%} MRR={mrr:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("sentence-transformers", "fastembed"), default="sentence-transformers")
    parser.add_argument("--model")
    args = parser.parse_args()

    docs = read_jsonl(ROOT / "data/documents.jsonl")
    listings = load_listing_records(ROOT / "data/listings.jsonl")
    questions = json.loads((ROOT / "data/retrieval_questions.json").read_text(encoding="utf-8"))
    passages = [p for doc in docs for p in chunks(doc)]
    if args.backend == "fastembed":
        from fastembed import TextEmbedding
        model_name = args.model or "intfloat/multilingual-e5-small"
        if model_name == "intfloat/multilingual-e5-small" and not any(
            item["model"] == model_name for item in TextEmbedding.list_supported_models()
        ):
            from fastembed.common.model_description import ModelSource, PoolingType
            TextEmbedding.add_custom_model(
                model=model_name, dim=384,
                description="Multilingual E5 small, query/passage prefix required",
                model_file="onnx/model.onnx", sources=ModelSource(hf=model_name),
                pooling=PoolingType.MEAN, normalization=True,
            )
        model = TextEmbedding(model_name=model_name)
        vectors = list(model.passage_embed(["passage: " + p.get("title", "") + ". " + p["text"] for p in passages]))
        embed_query = lambda query: next(model.query_embed("query: " + query))
    else:
        from sentence_transformers import SentenceTransformer
        model_name = args.model or "intfloat/multilingual-e5-small"
        model = SentenceTransformer(model_name)
        vectors = model.encode(
            ["passage: " + p.get("title", "") + ". " + p["text"] for p in passages],
            normalize_embeddings=True, show_progress_bar=True,
        )
        embed_query = lambda query: model.encode(["query: " + query], normalize_embeddings=True)[0]

    rag = [q for q in questions if q["type"] == "rag" and not q["should_abstain"]]
    listing_qs = [q for q in questions if q["type"] == "listing" and not q["should_abstain"]]
    unsupported = [q for q in questions if q["should_abstain"]]
    unsupported += [
        {"query": "What time does the Louvre in Paris open?", "type": "rag"},
        {"query": "Who won the latest World Cup match?", "type": "rag"},
        {"query": "Explain how to bake sourdough bread", "type": "rag"},
        {"query": "What is the weather in Tokyo tomorrow?", "type": "rag"},
    ]

    ranked_cases = {"RAG questions": rag, "Listing prompts": listing_qs}
    rank_outputs: dict[str, dict[str, list[int | None]]] = {
        group: {method: [] for method in ("BM25", "Dense", "Hybrid-RRF")} for group in ranked_cases
    }
    missing_labels = []
    for group, items in ranked_cases.items():
        for item in items:
            ranked, _, _ = ranks_for(item["query"], docs, passages, vectors, embed_query)
            targets = set(item.get("expected_urls", []))
            if not targets.intersection(d["url"] for d in docs):
                missing_labels.append(item["query"])
                continue
            for method, urls in ranked.items():
                rank_outputs[group][method].append(next((i for i, url in enumerate(urls, 1) if url in targets), None))

    print(f"Snapshot: {len(docs)} pages / {len(passages)} chunks; {len(listings)} structured listings; backend={args.backend}; model={model_name}")
    print(f"Labeled groups: {len(rag)} prose RAG, {len(listing_qs)} listing, {len(unsupported)} abstention (including 4 generic OOD probes)")
    for group, method_results in rank_outputs.items():
        print(group + " source retrieval:")
        for method, values in method_results.items():
            print(f"  {method:10} {summarize(values)}")

    # Exercise the actual production chat route for property prompts; it is deterministic and needs no API key.
    list_hits = exact_sets = 0
    list_recall = 0.0
    for item in listing_qs:
        expected = set(item.get("expected_listing_ids", []))
        try:
            result = chat(ChatRequest(message=item["query"]))
        except Exception as exc:
            print(f"Listing route error for {item['query']}: {type(exc).__name__}: {exc}")
            continue
        actual = {row["listing_id"] for row in listings if any(row["listing_id"] in src["url"] for src in result.get("sources", []))}
        list_hits += bool(actual & expected) if expected else not actual
        exact_sets += actual == expected
        list_recall += len(actual & expected) / len(expected) if expected else float(not actual)
    n_listing = max(1, len(listing_qs))
    print(f"Production structured route: hit={list_hits}/{len(listing_qs)}; exact-set={exact_sets}/{len(listing_qs)}; mean ID recall={list_recall/n_listing:.1%}")

    # Follow production routing/guards to see which negative prompts would be sent to generation.
    abstained, false_support = 0, []
    for item in unsupported:
        q = item["query"]
        if unsupported_intent(q):
            stopped = True
        elif is_listing_query(q, listings):
            matches, _ = filter_listings(q, listings)
            stopped = not matches
        else:
            hits = retrieve(q, docs)
            stopped = not hits or hits[0].get("score", 0) < 0.18 or not sufficient_evidence(q, hits)
        abstained += stopped
        if not stopped:
            false_support.append(q)
    print(f"Production abstention: {abstained}/{len(unsupported)} = {abstained/max(1,len(unsupported)):.1%}")
    if false_support:
        print("Unsupported questions that pass retrieval guards:")
        for q in false_support:
            print("  -", q)
    if missing_labels:
        print("Expected source URLs missing from corpus:")
        for q in missing_labels:
            print("  -", q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
