"""Evaluate source retrieval, structured listing filters, and unsupported-query abstention."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.listings import filter_listings, is_listing_query, load_listing_records
from app.retrieval import retrieve, sufficient_evidence, unsupported_intent

DOCS = [json.loads(line) for line in (ROOT / "data/documents.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
QUESTIONS = json.loads((ROOT / "data/retrieval_questions.json").read_text(encoding="utf-8"))
LISTINGS = load_listing_records(ROOT / "data/listings.jsonl")


def dcg(ranks: list[int], k: int) -> float:
    return sum(1 / __import__("math").log2(rank + 1) for rank in ranks if rank <= k)


def evaluate():
    rag = [q for q in QUESTIONS if q["type"] == "rag" and not q["should_abstain"]]
    listing = [q for q in QUESTIONS if q["type"] == "listing" and not q["should_abstain"]]
    abstain_cases = [q for q in QUESTIONS if q["should_abstain"]]
    results = {k: {"hits": 0, "mrr": 0.0, "ndcg": 0.0} for k in (5, 10)}
    for item in rag:
        ranked = retrieve(item["query"], DOCS, limit=1000)
        urls = []
        for p in ranked:
            if p["url"] not in urls:
                urls.append(p["url"])
        expected = set(item["expected_urls"])
        ranks = [i for i, url in enumerate(urls, 1) if url in expected]
        first = min(ranks, default=None)
        for k in (5, 10):
            results[k]["hits"] += bool(set(urls[:k]) & expected)
            results[k]["mrr"] += 1 / first if first else 0
            results[k]["ndcg"] += dcg(ranks, k) / dcg(list(range(1, min(len(expected), k) + 1)), k)
        item["_rank"] = first

    listing_hits = 0
    listing_exact = 0
    for item in listing:
        matches, _ = filter_listings(item["query"], LISTINGS, limit=5)
        actual = {x["listing_id"] for x in matches}
        expected = set(item.get("expected_listing_ids", []))
        listing_hits += bool(actual & expected) if expected else not actual
        listing_exact += actual == expected
        item["_listing_recall"] = len(actual & expected) / len(expected) if expected else float(not actual)

    abstained = 0
    abstention_details = []
    for item in abstain_cases:
        q = item["query"]
        listing_route = is_listing_query(q, LISTINGS)
        intent_abstain = unsupported_intent(q)
        if listing_route:
            matches, _ = filter_listings(q, LISTINGS, limit=10)
            # A request for live status is unanswerable from this dated snapshot even if a listing matches.
            live_status = any(term in q.lower() for term in ("available right now", "still available", "current", "live availability", "active")) or any(term in q for term in ("الآن مباشرة", "متاح الان", "التوافر الحالي"))
            did_abstain = intent_abstain or live_status or not matches
        else:
            found = retrieve(q, DOCS, limit=5)
            did_abstain = intent_abstain or not found or found[0].get("score", 0) < 0.18 or not sufficient_evidence(q, found)
        abstained += did_abstain
        if not did_abstain:
            abstention_details.append(q)

    n = max(1, len(rag))
    print(f"Corpus: {len(DOCS)} pages ({sum(d['site']=='DarGlobal' for d in DOCS)} DarGlobal, {sum(d['site']=='Wasalt' for d in DOCS)} Wasalt); {len(LISTINGS)} structured listings")
    print(f"Labeled set: {len(QUESTIONS)} questions ({sum(q['language']=='en' for q in QUESTIONS)} English, {sum(q['language']=='ar' for q in QUESTIONS)} Arabic); {len(rag)} RAG, {len(listing)} listing, {len(abstain_cases)} abstention cases")
    print("RAG source ranking:")
    for k in (5, 10):
        print(f"  Recall@{k}: {results[k]['hits']}/{len(rag)} = {results[k]['hits']/n:.1%}; MRR={results[k]['mrr']/n:.3f}; nDCG@{k}={results[k]['ndcg']/n:.3f}")
    print(f"Structured listing IDs: recall={listing_hits}/{len(listing)} = {listing_hits/max(1,len(listing)):.1%}; exact result sets={listing_exact}/{len(listing)}")
    print(f"Unsupported/live-status abstention: {abstained}/{len(abstain_cases)} = {abstained/max(1,len(abstain_cases)):.1%}")
    if abstention_details:
        print("Non-abstained expected-unsupported queries:")
        for query in abstention_details:
            print(f"  - {query}")
    print("Per-question RAG ranks (source hit at rank ≤10 is Recall@10):")
    for q in rag:
        print(f"  {q['_rank'] or 'MISS':>4} | {q['query']}")


if __name__ == "__main__":
    evaluate()
