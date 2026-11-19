"""Compare current BM25 RAG with dense and hybrid retrieval on labeled questions.

Run: python scripts/evaluate_rag.py [--model intfloat/multilingual-e5-small]
Requires the optional sentence-transformers package; it is not an app dependency.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.listings import filter_listings, load_listing_records
from app.main import ChatRequest, chat
from app.retrieval import chunks, retrieve, sufficient_evidence, unsupported_intent


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_urls(passages: list[dict]) -> list[str]:
    return list(dict.fromkeys(p["url"] for p in passages))


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "count": n,
        "recall@1": sum(r is not None and r <= 1 for r in ranks) / n if n else 0,
        "recall@5": sum(r is not None and r <= 5 for r in ranks) / n if n else 0,
        "recall@10": sum(r is not None and r <= 10 for r in ranks) / n if n else 0,
        "mrr": sum(1 / r for r in ranks if r is not None) / n if n else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    args = parser.parse_args()

    docs = load_jsonl(ROOT / "data/documents.jsonl")
    questions = json.loads((ROOT / "data/rag_eval_questions.json").read_text(encoding="utf-8"))
    passages = [p for doc in docs for p in chunks(doc)]
    model = SentenceTransformer(args.model)
    vectors = model.encode(
        ["passage: " + p.get("title", "") + ". " + p["text"] for p in passages],
        normalize_embeddings=True, show_progress_bar=True,
    )

    rank_lists = {name: [] for name in ("BM25", "Dense", "Hybrid-RRF")}
    listing_rag_ranks = {name: [] for name in ("BM25", "Dense", "Hybrid-RRF")}
    available, gaps, unanswerable = 0, [], []
    dense_positive, dense_negative = [], []
    current_positive_supported, current_positive_total = 0, 0
    current_positive_rejected = []
    current_negative_abstained, current_negative_total = 0, 0
    for item in questions:
        targets = item["relevant_any"]
        if not item["answerable"]:
            unanswerable.append(item)
            continue
        matching = [d["url"] for d in docs if any(target.lower() in d["url"].lower() for target in targets)]
        if not matching:
            gaps.append(item["query"])
            continue
        if item.get("type") == "rag":
            available += 1
        target_urls = set(matching)
        bm_passages = retrieve(item["query"], docs, limit=len(passages))
        bm_urls = unique_urls(bm_passages)
        qv = model.encode(["query: " + item["query"]], normalize_embeddings=True)[0]
        dense_idx = sorted(range(len(passages)), key=lambda i: float(vectors[i] @ qv), reverse=True)
        dense_urls = unique_urls([passages[i] for i in dense_idx])
        bm_rank = {url: i + 1 for i, url in enumerate(bm_urls)}
        dense_rank = {url: i + 1 for i, url in enumerate(dense_urls)}
        fused = sorted(set(bm_rank) | set(dense_rank), key=lambda url: (
            -(1 / (60 + bm_rank.get(url, 10**9)) + 1 / (60 + dense_rank.get(url, 10**9))), url))
        bucket = rank_lists if item.get("type") == "rag" else listing_rag_ranks
        for name, ranked in (("BM25", bm_urls), ("Dense", dense_urls), ("Hybrid-RRF", fused)):
            bucket[name].append(next((i for i, url in enumerate(ranked, 1) if url in target_urls), None))
        if item.get("type") == "rag":
            dense_positive.append(float(vectors[dense_idx[0]] @ qv))
            bm_ok = bool(bm_passages and bm_passages[0].get("score", 0) >= 0.18 and sufficient_evidence(item["query"], bm_passages))
            current_positive_supported += bm_ok
            current_positive_total += 1
            if not bm_ok:
                current_positive_rejected.append(item["query"])
        print(f"{item.get('type','rag').upper()} | {item['query']} | BM25={bucket['BM25'][-1]} Dense={bucket['Dense'][-1]} Hybrid={bucket['Hybrid-RRF'][-1]}")

    rag_gold = sum(q["answerable"] and q.get("type") == "rag" for q in questions)
    listing_gold = sum(q["answerable"] and q.get("type") == "listing" for q in questions)
    print(f"\nCorpus: {len(docs)} pages, {len(passages)} chunks; RAG gold sources in corpus: {available}/{rag_gold}; listing prompts: {listing_gold}; unsupported prompts: {len(unanswerable)}")
    print("Document retrieval (target source URL rank; does not score generated answer correctness):")
    for name, ranks in rank_lists.items():
        m = metrics(ranks)
        print(f"{name:10} n={m['count']} R@1={m['recall@1']:.1%} R@5={m['recall@5']:.1%} R@10={m['recall@10']:.1%} MRR={m['mrr']:.3f}")
    print("Embedding-only RAG on listing-ID prompts (contrast with structured route):")
    for name, ranks in listing_rag_ranks.items():
        m = metrics(ranks)
        print(f"{name:10} n={m['count']} R@1={m['recall@1']:.1%} R@5={m['recall@5']:.1%} R@10={m['recall@10']:.1%} MRR={m['mrr']:.3f}")
    if gaps:
        print("Labeled answerable questions whose target URL is absent from corpus:")
        for query in gaps:
            print("  -", query)

    # Measure the production structured-listing route on exact-ID questions.
    records = load_listing_records(ROOT / "data/listings.jsonl")
    id_cases = [
        ("listing id 5879365", "5879365"), ("property number 5914584", "5914584"),
        ("listing 5577", "5577"), ("listing id 5879277", "5879277"),
        ("property id 5173", "5173"), ("رقم العقار 5421", "5421"),
        ("رقم العقار 5879276", "5879276"), ("listing #5582", "5582"),
    ]
    id_ok = 0
    for query, expected in id_cases:
        result = chat(ChatRequest(message=query))
        actual = [expected for source in result["sources"] if expected in source["url"]]
        good = result.get("model") == "structured-filter" and actual == [expected]
        id_ok += good
        print(f"LISTING-ID | {query} | {'PASS' if good else 'FAIL'} {actual}")
    print(f"Structured exact-ID accuracy: {id_ok}/{len(id_cases)} = {id_ok/len(id_cases):.1%}")

    # Check representative attribute filters against independently computed gold sets.
    filter_cases = [
        ("daily rental apartments in Riyadh", {"transaction": "daily_rental", "property_type": "apartment", "city": "Riyadh"}),
        ("daily rental chalets in Jeddah", {"transaction": "daily_rental", "property_type": "chalet", "city": "Jeddah"}),
        ("apartments for sale in Jeddah", {"transaction": "sale", "property_type": "apartment", "city": "Jeddah"}),
        ("شقق للبيع في جدة", {"transaction": "sale", "property_type": "apartment", "city": "Jeddah"}),
        ("daily rental apartment in Riyadh with 1 bedroom", {"transaction": "daily_rental", "property_type": "apartment", "city": "Riyadh", "bedrooms": 1}),
    ]
    p_sum = r_sum = f_sum = 0.0
    for query, spec in filter_cases:
        gold = {row["listing_id"] for row in records if all(row.get(key) == value for key, value in spec.items())}
        actual_response = chat(ChatRequest(message=query))
        actual = {row["listing_id"] for row in records if any(row["listing_id"] in src["url"] for src in actual_response["sources"])}
        p = len(actual & gold) / len(actual) if actual else float(not gold)
        r = len(actual & gold) / len(gold) if gold else float(not actual)
        f = 2 * p * r / (p + r) if p + r else 0.0
        p_sum += p; r_sum += r; f_sum += f
        print(f"LISTING-FILTER | {query} | P={p:.2f} R={r:.2f} F1={f:.2f} gold={len(gold)} returned={len(actual)} route={actual_response.get('model')}")
    n = len(filter_cases)
    print(f"Structured filter macro: P={p_sum/n:.1%} R={r_sum/n:.1%} F1={f_sum/n:.1%}")

    for item in unanswerable:
        if not unsupported_intent(item["query"]):
            hits = retrieve(item["query"], docs)
            has_evidence = bool(hits and hits[0].get("score", 0) >= 0.18 and sufficient_evidence(item["query"], hits))
            dense_idx = sorted(range(len(passages)), key=lambda i: float(vectors[i] @ model.encode(["query: " + item["query"]], normalize_embeddings=True)[0]), reverse=True)
            dense_negative.append(float(vectors[dense_idx[0]] @ model.encode(["query: " + item["query"]], normalize_embeddings=True)[0]))
        else:
            has_evidence = False
            dense_negative.append(float("nan"))
        current_negative_abstained += not has_evidence
        current_negative_total += 1
        print(f"ABSTAIN | {item['query']} | current={'ABSTAIN' if not has_evidence else 'WOULD ANSWER'} | intent_guard={unsupported_intent(item['query'])}")
    print(f"Current pipeline accepts answerable general questions: {current_positive_supported}/{current_positive_total} = {current_positive_supported/current_positive_total:.1%}")
    if current_positive_rejected:
        print("Answerable general questions rejected by current evidence gate:")
        for query in current_positive_rejected:
            print("  -", query)
    print(f"Current pipeline abstains on unrelated questions: {current_negative_abstained}/{current_negative_total} = {current_negative_abstained/current_negative_total:.1%}")
    valid_neg = [score for score in dense_negative if score == score]
    if dense_positive and valid_neg:
        print(f"Dense max-cosine ranges (diagnostic only): positives {min(dense_positive):.3f}..{max(dense_positive):.3f}; unrelated {min(valid_neg):.3f}..{max(valid_neg):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
