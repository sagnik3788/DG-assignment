"""Structured Wasalt listing extraction and deterministic query filters."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from .retrieval import normalize

DETAIL_PATH = re.compile(r"/(?:en/)?(?:property/(?:sale|rent)|dailyrental)/", re.I)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
CITY_PAIRS = {
    "riyadh": "الرياض", "jeddah": "جدة", "khobar": "الخبر", "dammam": "الدمام",
    "taif": "الطائف", "makkah": "مكة", "madinah": "المدينة", "abha": "أبها",
    "makkah al mukarramah": "مكة المكرمة", "najran": "نجران", "hail": "حائل",
    "hayil": "حائل", "jubail": "الجبيل", "yanbu": "ينبع", "tabuk": "تبوك",
}
TYPE_WORDS = {
    "apartment": ("apartment", "apartments", "شقة", "شقق"),
    "villa": ("villa", "villas", "فيلا", "فلل"),
    "chalet": ("chalet", "chalets", "شاليه", "شاليهات"),
    "hotel room": ("hotel room", "hotel rooms", "غرفة فندقية", "غرف فندقية"),
}
BED_COUNTS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
              "واحد": 1, "واحدة": 1, "غرفة": 1, "غرفتين": 2, "اثنين": 2, "اثنتين": 2,
              "ثلاث": 3, "ثلاثة": 3, "اربع": 4, "اربعة": 4, "أربع": 4, "أربعة": 4,
              "خمس": 5, "خمسة": 5, "ست": 6, "ستة": 6, "سبع": 7, "سبعة": 7,
              "ثمان": 8, "ثمانية": 8, "تسع": 9, "تسعة": 9, "عشر": 10, "عشرة": 10}


def _number(value: str) -> float | None:
    cleaned = value.translate(ARABIC_DIGITS).replace(",", "").replace("٬", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _first_number(pattern: str, text: str, flags: int = re.I) -> float | None:
    match = re.search(pattern, text, flags)
    return _number(match.group(1)) if match else None


def _canonical_city(value: str) -> str:
    q = normalize(value)
    if q.startswith(("north ", "south ", "east ", "west ")):
        q = q.split(" ", 1)[1]
    if q in ("hayil", "hail") or q == normalize("حائل"):
        return "Hail"
    if "makkah" in q or q in (normalize("مكة"), normalize("مكة المكرمة")):
        return "Makkah"
    return next((en.title() for en, ar in CITY_PAIRS.items() if normalize(en) == q or normalize(ar) == q), value.title())


def _extract_city_district(text: str) -> tuple[str | None, str | None]:
    cities = "|".join(re.escape(name) for name in sorted(CITY_PAIRS, key=len, reverse=True))
    match = re.search(rf"(?:for sale|for rent) in\s+(.{{2,65}}?),\s*({cities})\b", text, re.I)
    if match:
        return _canonical_city(match.group(2)), match.group(1).strip(" ,.")
    aliases = sorted((*CITY_PAIRS.keys(), "North Riyadh", "South Riyadh", "East Riyadh", "West Riyadh", "North Jeddah", "South Jeddah"), key=len, reverse=True)
    for alias in aliases:
        for match in re.finditer(rf"\b{re.escape(alias)}\b", text, re.I):
            prefix = text[:match.start()].rstrip()
            after = text[match.end():match.end() + 30]
            if re.match(r"\s+Map view\b", after, re.I):
                tokens = prefix.rstrip(" ,.").split()[-2:]
                while tokens and (tokens[0].isdigit() or normalize(tokens[0]) in ("road", "street", "avenue")):
                    tokens.pop(0)
                district = " ".join(tokens).strip(" ,.")
                return _canonical_city(alias), district
    for alias in sorted(CITY_PAIRS, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(alias)}\b", text, re.I):
            district = text[:match.start()].rstrip().rsplit(",", 1)[-1].strip(" ,.")
            if "," in text[:match.start()] and district and len(district) <= 45:
                return _canonical_city(alias), district
    arabic_cities = "|".join(re.escape(value) for value in CITY_PAIRS.values())
    match = re.search(rf"(?:في حي|في)\s+(.{{1,60}}?)[٬،,]\s*({arabic_cities})", text)
    if match:
        arabic = normalize(match.group(2))
        city = _canonical_city(match.group(2))
        return city, match.group(1).strip(" ،٬,.")
    return None, None


def _listing_row(doc: dict) -> dict | None:
    url = doc.get("url", "")
    if doc.get("site") != "Wasalt" or not DETAIL_PATH.search(url):
        return None
    ident = re.search(r"(\d{4,})/?(?:\?.*)?$", url)
    if not ident:
        return None
    title = doc.get("title", "")
    text = f"{title} {doc.get('text', '')}"
    path = url.lower()
    transaction = "daily_rental" if "dailyrental" in path else "rent" if "/rent/" in path else "sale"
    property_type = next((kind for kind, words in TYPE_WORDS.items()
                          if any(normalize(word) in normalize(title) for word in words)), None)
    if property_type is None:
        if "chalet-for-booking" in path: property_type = "chalet"
        elif "apartment-for-booking" in path: property_type = "apartment"
    area = _first_number(r"([\d,.]+)\s*(?:sqm|square meters?|متر\s*مربع)", text)
    bedrooms = _first_number(r"(\d{1,2})\s*(?:bedrooms?|bedroom|غرف(?:ة)?(?:\s*نوم)?)", text)
    price = _first_number(r"(?:price|سعر(?:ها|ه)?)\s*:?\s*([\d٠-٩٬,.]+)\s*(?:sar|ر\.س|ريال)?", text)
    if price is None:
        price = _first_number(r"([\d,]+(?:\.\d+)?)\s*/\s*(?:year|سنوي)", text)
    city, district = _extract_city_district(text)
    return {
        "listing_id": ident.group(1), "url": url, "title": title,
        "property_type": property_type, "transaction": transaction,
        "city": city, "district": district,
        "area_sqm": area, "bedrooms": int(bedrooms) if bedrooms is not None else None,
        "price_sar": price, "fetched_at": doc.get("fetched_at"),
        "language": doc.get("language", "en"),
    }


def build_listing_records(documents: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for doc in documents:
        row = _listing_row(doc)
        if row:
            groups[row["listing_id"]].append(row)
    records = []
    for ident, rows in groups.items():
        rows.sort(key=lambda row: (row["language"] != "en", row["fetched_at"] or ""), reverse=False)
        primary = next((r for r in rows if r["language"] == "en"), rows[0])
        arabic = next((r for r in rows if r["language"] == "ar"), None)
        record = dict(primary)
        # Fill missing structured values from the Arabic counterpart without losing its source.
        for key in ("property_type", "transaction", "city", "district", "area_sqm", "bedrooms", "price_sar"):
            if record.get(key) is None:
                record[key] = next((r[key] for r in rows if r.get(key) is not None), None)
        record["source_urls"] = list(dict.fromkeys(r["url"] for r in rows))
        record["source_pages"] = [{"url": r["url"], "title": r["title"], "language": r["language"], "fetched_at": r.get("fetched_at")} for r in rows]
        record["title_ar"] = arabic.get("title") if arabic else None
        record["district_ar"] = arabic.get("district") if arabic else None
        record["languages"] = sorted({r["language"] for r in rows})
        record["fetched_at"] = max((r["fetched_at"] for r in rows if r.get("fetched_at")), default=None)
        records.append(record)
    return sorted(records, key=lambda row: row["listing_id"])


def save_listing_records(documents: list[dict], path: Path) -> int:
    records = build_listing_records(documents)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    tmp.replace(path)
    return len(records)


def load_listing_records(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def _query_number(question: str, cues: tuple[str, ...]) -> float | None:
    q = unicodedata.normalize("NFKC", question).translate(ARABIC_DIGITS)
    cue = "|".join(re.escape(value) for value in cues)
    match = re.search(rf"(?:{cue})\s*(?:sar|riyal|ريال)?\s*([\d,]+(?:\.\d+)?)\s*(k|m|ألف|الف|مليون)?", q, re.I)
    if not match:
        return None
    value = _number(match.group(1))
    if value is not None and match.group(2):
        unit = normalize(match.group(2))
        value *= 1_000_000 if unit in ("m", "مليون") else 1000
    return value


def is_listing_query(question: str, records: list[dict] | None = None) -> bool:
    q = normalize(question)
    ids = {row.get("listing_id") for row in records or []}
    number_matches = list(re.finditer(r"(?<!\d)(\d{4,10})(?!\d)", q))
    if any(m.group(1) in ids or len(m.group(1)) >= 6 for m in number_matches):
        return True
    explicit = (
        "listing", "listing id", "property id", "رقم", "معرف", "apartment", "apartments", "villa", "villas", "bedroom", "bedrooms", "sqm",
        "price", "sar", "for sale", "for rent", "daily rental", "شقة", "شقق", "فيلا", "فلل",
        "غرف", "سعر", "ريال", "للبيع", "للايجار", "للايجاره", "مساحة",
    )
    if any(normalize(word) in q for word in explicit):
        return True
    has_property = any(normalize(word) in q for word in ("property", "properties", "home", "homes", "عقار", "عقارات"))
    locations = [*CITY_PAIRS.keys(), *CITY_PAIRS.values()]
    return has_property and any(normalize(place) in q for place in locations)


def filter_listings(question: str, records: list[dict], limit: int = 5) -> tuple[list[dict], dict]:
    q = normalize(question)
    q_digits = question.translate(ARABIC_DIGITS)
    ids = {row["listing_id"] for row in records}
    number_matches = list(re.finditer(r"(?<!\d)(\d{4,10})(?!\d)", q_digits))
    exact_id = next((m.group(1) for m in number_matches if m.group(1) in ids), None)
    explicit_id = re.search(r"(?:listing(?:\s+id)?|property\s+id|#|رقم(?:\s+العقار)?|معرف)\D{0,20}(\d{4,10})", q_digits, re.I)
    has_unknown_id = exact_id is None and bool(explicit_id or any(len(m.group(1)) >= 7 for m in number_matches))
    transaction = "daily_rental" if any(w in q for w in ("daily rental", "daily rent", "ايجار يومي", "للايجار اليومي")) else "rent" if any(w in q for w in ("for rent", "rent", "للايجار", "للايجاره")) else "sale" if any(w in q for w in ("for sale", "sale", "للبيع")) else None
    property_type = next((kind for kind, words in TYPE_WORDS.items() if any(normalize(w) in q for w in words)), None)
    beds_match = re.search(r"(?<!\d)(\d{1,2})\s*[- ]?\s*(?:bedrooms?|bedroom|غرف(?:ة)?(?:\s*نوم)?)", q_digits)
    beds = int(beds_match.group(1)) if beds_match else None
    if beds is None:
        for word, count in BED_COUNTS.items():
            if re.search(rf"\b{re.escape(word)}\b\s*(?:bedrooms?|bedroom|غرف(?:ة)?(?:\s*نوم)?)|(?:{re.escape(word)})\s*غرف|غرف(?:ة)?\s*(?:نوم\s*)?{re.escape(word)}\b", question, re.I):
                beds = count
                break
    max_price = _query_number(question, ("under", "below", "less than", "up to", "max", "أقل من", "دون", "بحد أقصى"))
    min_price = _query_number(question, ("over", "above", "more than", "at least", "أكثر من", "فوق"))
    locations = []
    for row in records:
        for field in ("city", "district"):
            place = row.get(field)
            place_tokens = [t for t in normalize(place or "").split() if len(t) >= 4]
            if place and (normalize(place) in q or any(token in q for token in place_tokens)) and normalize(place) not in locations:
                locations.append(normalize(place))
    # A query may use the Arabic spelling while the preferred listing record is English.
    for en, ar in CITY_PAIRS.items():
        if normalize(ar) in q and en not in locations:
            locations.append(en)

    filters = {"listing_id": exact_id, "transaction": transaction, "property_type": property_type,
               "bedrooms": beds, "max_price_sar": max_price, "min_price_sar": min_price,
               "locations": locations}
    if has_unknown_id:
        return [], filters
    results = []
    for row in records:
        if exact_id and row["listing_id"] != exact_id:
            continue
        if transaction and row.get("transaction") != transaction:
            continue
        if property_type and row.get("property_type") != property_type:
            continue
        if beds is not None and row.get("bedrooms") != beds:
            continue
        if max_price is not None and (row.get("price_sar") is None or row["price_sar"] > max_price):
            continue
        if min_price is not None and (row.get("price_sar") is None or row["price_sar"] < min_price):
            continue
        record_places = {normalize(row.get(k) or "") for k in ("city", "district")}
        if locations and not any(query_place == record_place or query_place in record_place or record_place in query_place for query_place in locations for record_place in record_places):
            continue
        results.append(row)
    return results[:limit], filters


def format_listing_answer(rows: list[dict], filters: dict, snapshot_date: str | None) -> str:
    date_label = snapshot_date or "unknown date"
    if not rows:
        return (f"No matching listing was found in the indexed public snapshot (latest capture: {date_label}). "
                "This does not confirm that no such property exists; availability and prices may have changed.")
    intro = f"Showing {len(rows)} matching indexed listing{'s' if len(rows) != 1 else ''} (latest Wasalt capture: {date_label}; not live availability)."
    lines = []
    for i, row in enumerate(rows, 1):
        details = [f"ID {row['listing_id']}"]
        if row.get("property_type"): details.append(row["property_type"].title())
        if row.get("transaction"): details.append({"sale":"for sale","rent":"for rent","daily_rental":"daily rental"}[row["transaction"]])
        if row.get("bedrooms") is not None: details.append(f"{row['bedrooms']} bedrooms")
        if row.get("area_sqm") is not None: details.append(f"{row['area_sqm']:g} sqm")
        if row.get("price_sar") is not None: details.append(f"SAR {row['price_sar']:,.0f}" + (" / year" if row.get("transaction") == "rent" else ""))
        place = ", ".join(p for p in (row.get("district"), row.get("city")) if p)
        if place: details.append(place)
        if row.get("fetched_at"): details.append(f"captured {row['fetched_at'][:10]}")
        lines.append(f"[{i}] " + " · ".join(details))
    return intro + "\n" + "\n".join(lines)


def format_listing_answer_ar(rows: list[dict], snapshot_date: str | None) -> str:
    date_label = snapshot_date or "غير معروف"
    if not rows:
        return (f"لم أجد عقاراً مطابقاً في اللقطة العامة المفهرسة (أحدث جمع للبيانات: {date_label}). "
                "هذا لا يعني عدم وجود عقار مطابق؛ قد تكون الأسعار والتوافر قد تغيرت.")
    intro = f"وجدت {len(rows)} من العقارات المطابقة في اللقطة العامة (أحدث جمع: {date_label}؛ البيانات ليست مباشرة)."
    type_names = {"apartment": "شقة", "villa": "فيلا", "chalet": "شاليه", "hotel room": "غرفة فندقية"}
    transaction_names = {"sale": "للبيع", "rent": "للإيجار", "daily_rental": "إيجار يومي"}
    lines = []
    for i, row in enumerate(rows, 1):
        parts = [f"رقم {row['listing_id']}"]
        if row.get("property_type"): parts.append(type_names.get(row["property_type"], row["property_type"]))
        if row.get("transaction"): parts.append(transaction_names[row["transaction"]])
        if row.get("bedrooms") is not None: parts.append(f"{row['bedrooms']} غرف نوم")
        if row.get("area_sqm") is not None: parts.append(f"{row['area_sqm']:g} م²")
        if row.get("price_sar") is not None: parts.append(f"{row['price_sar']:,.0f} ريال سعودي" + (" سنوياً" if row.get("transaction") == "rent" else ""))
        place = "، ".join(p for p in (row.get("district_ar") or row.get("district"), _arabic_city(row.get("city"))) if p)
        if place: parts.append(place)
        if row.get("fetched_at"): parts.append(f"جُمعت {row['fetched_at'][:10]}")
        lines.append(f"[{i}] " + " · ".join(parts))
    return intro + "\n" + "\n".join(lines)


def _arabic_city(city: str | None) -> str | None:
    if not city:
        return None
    return next((ar for en, ar in CITY_PAIRS.items() if en == city.lower()), city)
