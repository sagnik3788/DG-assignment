"""Hybrid lexical + multilingual vector retrieval for mixed English/Arabic pages."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
import unicodedata
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
QUERY_FILLER = set("what is the a an and or about me tell can you find does provide offered current today now right tonight on in of to for its who where how when which are was by from with explain describe this that page information please project development property properties listing details exact can i tell it services homes type sale rent bedrooms under less than open opening hours time latest".split()) | set("ما هو هي في من على عن ماذا أين اين مشروع نوع الخدمات التي تقدّمها تقدمها منصة العقارات العقار للبيع للايجار الإيجار عدد اليوم الآن الحالي الحالية مباشرة تفاصيل أخبرني صف اشرح هل يمكن يكون يوجد شركة الشركه".split())
QUERY_ALIASES = {
    "ايدا": "aida", "آيدا": "aida", "تييرا": "tierra", "فيفا": "viva", "مريا": "marea",
    "ميسوني": "missoni", "ترامب": "trump", "قطر": "qatar", "عمان": "oman", "عمانيه": "oman",
    "العمانية": "oman", "عمانية": "oman", "اسبانيا": "spain", "إسبانيا": "spain",
        "شريكة": "partnership", "الشريكة": "partnership", "شريك": "partnership", "الشركة": "company",
    "فيندي": "fendi", "كازا": "casa", "فلل": "villas", "المحيط": "oceanfront", "المطلة": "oceanfront",
    "رئيسا": "chairman", "رئيس": "chairman", "مجلس": "board", "ادارة": "board",
    "دار": "dar", "جلوبال": "global", "دافنشي": "davinci", "باغاني": "pagani", "برج": "tower",
}
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
RRF_K = 60
_MODEL = None
_MODEL_FAILED = False
_MODEL_LOCK = threading.Lock()
_INDEX_LOCK = threading.Lock()
_INDEX_SIGNATURE = None
_INDEX = None
_BACKEND_STATUS = "hybrid (lazy)"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)
    return text.translate(str.maketrans({"أ":"ا", "إ":"ا", "آ":"ا", "ى":"ي", "ة":"ه"}))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


def query_tokens(text: str) -> list[str]:
    normalized = normalize(text)
    tokens = tokenize(normalized)
    expanded = tokens + [normalize(QUERY_ALIASES[t]) for t in tokens if t in QUERY_ALIASES]
    if "دار جلوبال" in normalized:
        expanded.append("darglobal")
    if "دي جي ون" in normalized:
        expanded.append("dg1")
    return list(dict.fromkeys(expanded))


def chunks(doc: dict, size: int = 850, overlap: int = 140) -> list[dict]:
    text = re.sub(r"\s+", " ", doc.get("text", "")).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?؟])\s+", text)
    result, buf = [], ""
    for sentence in sentences:
        if buf and len(buf) + len(sentence) > size:
            result.append({**doc, "text": buf.strip()})
            buf = (buf[-overlap:] + " " + sentence).strip()
        else:
            buf = (buf + " " + sentence).strip()
    if buf:
        result.append({**doc, "text": buf})
    return result


def _get_model():
    global _MODEL, _MODEL_FAILED, _BACKEND_STATUS
    if _MODEL is not None or _MODEL_FAILED:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None or _MODEL_FAILED:
            return _MODEL
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType

            if not any(item["model"] == EMBEDDING_MODEL_NAME for item in TextEmbedding.list_supported_models()):
                TextEmbedding.add_custom_model(
                    model=EMBEDDING_MODEL_NAME, dim=384,
                    description="Multilingual E5 small; query and passage prefixes required",
                    model_file="onnx/model.onnx", sources=ModelSource(hf=EMBEDDING_MODEL_NAME),
                    pooling=PoolingType.MEAN, normalization=True,
                )
            _MODEL = TextEmbedding(model_name=EMBEDDING_MODEL_NAME, threads=2)
        except Exception as exc:
            _MODEL_FAILED = True
            _BACKEND_STATUS = "BM25 fallback (embedding model unavailable)"
            print(f"Embedding model unavailable; using BM25: {exc}", file=sys.stderr)
    return _MODEL


def warm_embedding_model() -> bool:
    """Load/download the local ONNX model; used during image build and smoke tests."""
    return _get_model() is not None


def warm_retrieval_index(documents: list[dict]) -> bool:
    """Build the corpus vectors during image build so the first Space query is fast."""
    return _get_index(documents)["vectors"] is not None


def retrieval_backend() -> str:
    return _BACKEND_STATUS


def _signature(documents: list[dict]) -> str:
    payload = [(doc.get("url", ""), doc.get("content_hash") or hashlib.sha256(doc.get("text", "").encode()).hexdigest())
               for doc in documents]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _get_index(documents: list[dict]) -> dict:
    global _INDEX, _INDEX_SIGNATURE, _BACKEND_STATUS
    signature = _signature(documents)
    if signature == _INDEX_SIGNATURE and _INDEX is not None:
        return _INDEX
    with _INDEX_LOCK:
        if signature == _INDEX_SIGNATURE and _INDEX is not None:
            return _INDEX
        passages = [chunk for doc in documents for chunk in chunks(doc)]
        terms = [tokenize(p["text"] + " " + p.get("title", "") + " " + p.get("url", "")) for p in passages]
        df = Counter(term for row in terms for term in set(row))
        avg_len = sum(map(len, terms)) / len(terms) if terms else 1
        vectors = None
        cache_path = Path(os.getenv("FASTEMBED_INDEX_PATH", "data/fastembed_index.npz"))
        try:
            import numpy as np
            with np.load(cache_path, allow_pickle=False) as archive:
                if str(archive["signature"].item()) == signature and len(archive["vectors"]) == len(passages):
                    vectors = archive["vectors"]
        except (OSError, KeyError, ValueError):
            pass
        model = _get_model() if vectors is None else None
        if model and passages:
            try:
                import numpy as np
                vectors = np.asarray(list(model.passage_embed(["passage: " + p.get("title", "") + ". " + p["text"] for p in passages])), dtype="float32")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                with tmp_path.open("wb") as stream:
                    np.savez_compressed(stream, signature=signature, vectors=vectors)
                tmp_path.replace(cache_path)
                _BACKEND_STATUS = "hybrid (BM25 + multilingual E5 ONNX)"
            except Exception as exc:
                print(f"Embedding index failed; using BM25: {exc}", file=sys.stderr)
                _BACKEND_STATUS = "BM25 fallback (embedding index failed)"
        elif vectors is not None:
            _BACKEND_STATUS = "hybrid (cached BM25 + multilingual E5 ONNX)"
        _INDEX = {"passages": passages, "terms": terms, "df": df, "avg_len": avg_len, "vectors": vectors}
        _INDEX_SIGNATURE = signature
        return _INDEX


def _bm25_scores(query: str, index: dict) -> list[float]:
    q = set(query_tokens(query)) - QUERY_FILLER
    if not q:
        return [0.0] * len(index["passages"])
    total = len(index["passages"])
    scores = []
    for passage, row in zip(index["passages"], index["terms"]):
        counts, length = Counter(row), len(row)
        score = 0.0
        for term in q:
            freq = counts[term]
            if freq:
                idf = math.log(1 + (total - index["df"][term] + 0.5) / (index["df"][term] + 0.5))
                score += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * length / index["avg_len"]))
        url_terms = set(tokenize(passage.get("url", "")))
        score += 8 * sum(term in url_terms for term in q)
        scores.append(score)
    return scores


def _url_order(scores: list[float], passages: list[dict]) -> tuple[list[str], dict[str, int], dict[str, dict]]:
    best = {}
    for i in sorted(range(len(scores)), key=lambda j: (-float(scores[j]), passages[j].get("url", ""), j)):
        if float(scores[i]) <= 0:
            continue
        url = passages[i].get("url", "")
        best.setdefault(url, {"rank": len(best) + 1, "passage": passages[i], "score": float(scores[i])})
    order = sorted(best, key=lambda url: (best[url]["rank"], url))
    return order, {url: rank for rank, url in enumerate(order, 1)}, best


def _ranked(query: str, documents: list[dict]) -> tuple[list[str], dict[str, dict], list[float], dict]:
    global _BACKEND_STATUS
    index = _get_index(documents)
    passages = index["passages"]
    bm_scores = _bm25_scores(query, index)
    bm_urls, bm_rank, bm_best = _url_order(bm_scores, passages)
    vectors = index["vectors"]
    if vectors is None or not passages:
        return bm_urls, bm_best, bm_scores, index
    try:
        model = _get_model()
        query_vector = next(model.query_embed("query: " + query))
        dense_scores = [float(vector @ query_vector) for vector in vectors]
    except Exception as exc:
        print(f"Query embedding failed; using BM25: {exc}", file=sys.stderr)
        _BACKEND_STATUS = "BM25 fallback (query embedding failed)"
        return bm_urls, bm_best, bm_scores, index
    dense_urls, dense_rank, dense_best = _url_order(dense_scores, passages)
    fused = sorted(set(bm_rank) | set(dense_rank), key=lambda url: (
        -(1 / (RRF_K + bm_rank.get(url, 10**9)) + 1 / (RRF_K + dense_rank.get(url, 10**9))),
        bm_rank.get(url, 10**9), dense_rank.get(url, 10**9), url))
    best = {}
    for url in fused:
        b, d = bm_best.get(url), dense_best.get(url)
        selected = d["passage"] if d and (not b or dense_rank[url] < bm_rank[url]) else b["passage"]
        score = b["score"] if b else 0.0
        best[url] = {"passage": selected, "score": score}
    return fused, best, bm_scores, index


def retrieve_bm25(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    index = _get_index(documents)
    passages = index["passages"]
    scores = _bm25_scores(query, index)
    urls, _, best = _url_order(scores, passages)
    result = []
    for url in urls[:limit]:
        passage = dict(best[url]["passage"])
        passage["score"] = round(best[url]["score"], 3)
        result.append(passage)
    return result


def retrieve(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    urls, best, _, _ = _ranked(query, documents)
    result = []
    for url in urls[:limit]:
        passage = dict(best[url]["passage"])
        passage["score"] = round(best[url]["score"], 3)
        result.append(passage)
    return result


def sufficient_evidence(query: str, passages: list[dict]) -> bool:
    terms = {term for term in query_tokens(query) if len(term) > 2 and term not in QUERY_FILLER}
    if not terms:
        return bool(passages)
    evidence = set()
    for passage in passages[:10]:
        evidence.update(tokenize(" ".join((passage.get("title", ""), passage.get("text", ""), passage.get("url", "")))))
    matched = terms & evidence
    return bool(matched) if len(terms) == 1 else len(matched) / len(terms) >= 0.35


def unsupported_intent(query: str) -> bool:
    q = normalize(query)
    live = any(term in q for term in ("current", "currently", "live", "right now", "today", "tonight", "this week", "الان", "اليوم", "مباشر", "الحالي", "الحالية"))
    live_subject = any(term in q for term in ("available", "availability", "price", "open", "weather", "stock", "rate", "visitor", "visitors", "الاسعار", "السعر", "التوافر", "متاح", "زوار", "طقس", "سعر"))
    out_of_scope = any(term in q for term in ("mortgage", "weather", "stock price", "stock", "guarantee", "guaranteed", "appreciate", "appreciation", "predict", "forecast", "investment return", "roi", "owns", "owner", "private residence", "contact details", "world cup", "football match", "match result", "sports score", "الرهن", "الطقس", "عائد استثماري", "مضمون", "مالك", "المالك", "هاتف", "عدد الزوار"))
    forward_price_claim = any(term in q for term in ("next year", "in the future", "rise next", "increase next", "will rise", "will increase", "توقع", "سيرتفع", "سترتفع")) and any(term in q for term in ("price", "prices", "property", "العقار", "الاسعار", "السعر"))
    return (live and live_subject) or out_of_scope or forward_price_claim
