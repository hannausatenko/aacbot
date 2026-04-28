"""RAG retrieval using sqlite-vec. Module-level model loads once at import time."""

import sqlite3
import struct
from collections import defaultdict
from pathlib import Path

import sqlite_vec
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "pictograms.db"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384

STRONG_THRESHOLD = 1.12   # category included only if its best result is within this

# symbols only shown when their trigger word appears in the query
CONDITIONAL_SYMBOLS = {
    "hard to breathe": ["breath", "breathe", "breathing", "asthma", "lungs"],
    "blood pressure":  ["pressure", "blood pressure", "bp"],
    "we":   ["we ", "we're", "our", "together", "family", "group"],
    "our":  ["our ", "we ", "together", "family"],
    "their": ["their", "they", "them"],
    "your": ["your", "you "],
    "father cristmas": ["christmas", "santa", "xmas", "new year", "holiday", "celebrate", "party"],
    "gift":     ["gift", "present", "give", "birthday", "christmas", "holiday", "celebrate", "party"],
    "greet":    ["hello", "greet", "meet", "welcome", "introduce", "say hello", "hi"],
    "surprise": ["surprise", "party", "birthday", "celebrate", "unexpected", "special"],
    "music":    ["music", "song", "dance", "party", "sing", "melody", "instrument", "listen"],
    # number symbols: their images depict everyday objects to convey quantity,
    # so they embed close to food/family/money — only show when counting is intended
    "zero":    ["zero", "0", "none", "nothing", "empty", "count", "number", "how many"],
    "one":     ["one", "1", "single", "alone", "only", "count", "number", "how many"],
    "two":     ["two", "2", "pair", "both", "count", "number", "how many"],
    "three":   ["three", "3", "count", "number", "how many"],
    "four":    ["four", "4", "count", "number", "how many"],
    "five":    ["five", "5", "count", "number", "how many"],
    "six":     ["six", "6", "count", "number", "how many"],
    "seven":   ["seven", "7", "count", "number", "how many"],
    "eight":   ["eight", "8", "count", "number", "how many"],
    "nine":    ["nine", "9", "count", "number", "how many"],
    "ten":     ["ten", "10", "count", "number", "how many"],
    "few":     ["few", "some", "couple", "not many", "a few", "count", "number"],
    "several": ["several", "some", "multiple", "count", "number"],
    "many":    ["many", "lot", "lots", "much", "plenty", "more", "count", "number"],
    "a little":["little", "bit", "small amount", "tiny", "count", "number"],
}

# communication symbols always shown when their exact concept is in the query keywords
COMMUNICATION_TRIGGERS = {
    "yes":              ["_____yes___"],
    "no":               ["_____no___"],
    "want":             ["_____want___"],
    "stop":             ["_____stop___"],
    "wait":             ["_____wait___"],
    "hello":            ["_____hello___"],
    "goodbye":          ["_____bye___"],
    "thank you":        ["_____thank_you___"],
    "don't understand": ["_____do_not_understand___"],
    "don't want":       ["_____do_not_want___", "_____dont_want___"],
}
SYMBOL_THRESHOLD = 1.20   # individual symbols shown up to this distance
RELATIVE_SPREAD = 0.06    # within category, only show symbols within this of the best match
MAX_PER_CATEGORY = 5      # max symbols shown per category

_model = SentenceTransformer(EMBED_MODEL)


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _query_once(query: str, n_results: int, audience: str | None, strong_threshold: float) -> list[dict]:
    embedding = _model.encode(query, normalize_embeddings=True)
    query_blob = struct.pack(f"{EMBED_DIM}f", *embedding)

    conn = _open_db()
    rows = conn.execute(
        """
        SELECT s.display_name, s.category, s.audience, s.png_path, v.distance,
               s.display_name_uk, s.category_uk
        FROM symbol_vss v
        JOIN symbols s ON s.symbol_id = v.symbol_id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (query_blob, n_results),
    ).fetchall()
    conn.close()

    all_results = [
        {"display_name": row[0], "category": row[1], "audience": row[2],
         "png_path": row[3], "distance": round(row[4], 4),
         "display_name_uk": row[5] or row[0], "category_uk": row[6] or row[1]}
        for row in rows
        if row[4] <= SYMBOL_THRESHOLD
    ]

    if audience is None:
        results = all_results
    else:
        # prefer requested audience; fall back to other variant when no preferred version exists
        preferred = [r for r in all_results if r["audience"] == audience]
        preferred_names = {(r["display_name"], r["category"]) for r in preferred}
        fallback = [r for r in all_results if (r["display_name"], r["category"]) not in preferred_names]
        results = preferred + fallback

    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    query_lower = query.lower()
    # sort categories by their best-match distance so most relevant appear first
    sorted_cats = sorted(by_category.values(), key=lambda syms: syms[0]["distance"])
    filtered = []
    for symbols in sorted_cats:
        best = symbols[0]["distance"]
        if best > strong_threshold:
            continue
        relevant = [s for s in symbols if s["distance"] <= best + RELATIVE_SPREAD]
        for s in relevant[:MAX_PER_CATEGORY]:
            triggers = CONDITIONAL_SYMBOLS.get(s["display_name"])
            if triggers and not any(t in query_lower for t in triggers):
                continue
            filtered.append(s)
    return filtered


def retrieve(query: str, n_results: int = 40, audience: str | None = None) -> list[dict]:
    """Return pictogram matches grouped by category, filtered by relevance and audience."""
    keywords = [k.strip() for k in query.split(",") if k.strip()]

    # short queries (≤4 keywords): use looser threshold to catch single-concept matches
    strong = STRONG_THRESHOLD if len(keywords) > 4 else 1.20

    results = _query_once(query, n_results, audience, strong)

    # multi-aspect queries: also search each half separately and merge
    if len(keywords) >= 6:
        mid = len(keywords) // 2
        for sub in [", ".join(keywords[:mid]), ", ".join(keywords[mid:])]:
            for r in _query_once(sub, n_results, audience, STRONG_THRESHOLD):
                if not any(e["display_name"] == r["display_name"] and e["audience"] == r["audience"]
                           for e in results):
                    results.append(r)

    # always inject communication symbols whose exact concept appears in the query
    query_lower = query.lower()
    existing_names = {r["display_name"] for r in results}
    conn = _open_db()
    for trigger_word, patterns in COMMUNICATION_TRIGGERS.items():
        if trigger_word not in query_lower:
            continue
        for pattern in patterns:
            rows = conn.execute(
                "SELECT display_name, category, audience, png_path, display_name_uk, category_uk "
                "FROM symbols WHERE INSTR(symbol_name, ?) > 0",
                (pattern,),
            ).fetchall()
            preferred = [r for r in rows if audience is None or r[2] == audience]
            if not preferred and rows:
                preferred = rows
            for row in preferred:
                if row[0] not in existing_names:
                    results.append({"display_name": row[0], "category": row[1],
                                    "audience": row[2], "png_path": row[3], "distance": 0.0,
                                    "display_name_uk": row[4] or row[0], "category_uk": row[5] or row[1]})
                    existing_names.add(row[0])
    conn.close()

    return results


PHRASE_MAX = 5           # max symbols in a phrase strip
PHRASE_FALLBACK_THRESHOLD = 1.20  # vector search fallback threshold


def retrieve_phrase(
    parts: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    audience: str | None = None,
) -> list[dict]:
    """Return one pictogram per (role, symbol) pair in phrase order.
    LLM provides exact symbol names; vector search is a fallback only."""
    if not parts:
        return []

    seen: set[str] = set()
    phrase: list[dict] = []
    conn = _open_db()

    for role, concept in list(parts)[:PHRASE_MAX]:
        concept = concept.strip()
        if not concept:
            continue

        # 1. Direct display_name lookup — LLM picked an exact symbol name
        rows = conn.execute(
            "SELECT display_name, category, audience, png_path, display_name_uk, category_uk "
            "FROM symbols WHERE display_name = ?",
            (concept,),
        ).fetchall()
        preferred = [r for r in rows if audience is None or r[2] == audience] or rows
        matched = False
        for name, cat, aud, path, name_uk, cat_uk in preferred:
            if name not in seen:
                seen.add(name)
                phrase.append({"role": role, "display_name": name, "category": cat,
                               "audience": aud, "png_path": path, "distance": 0.0,
                               "display_name_uk": name_uk or name, "category_uk": cat_uk or cat})
                matched = True
                break

        if matched:
            continue

        # 2. Vector search fallback — LLM gave a synonym or near-miss
        emb = _model.encode(concept, normalize_embeddings=True)
        blob = struct.pack(f"{EMBED_DIM}f", *emb)
        vrows = conn.execute(
            """
            SELECT s.display_name, s.category, s.audience, s.png_path, v.distance,
                   s.display_name_uk, s.category_uk
            FROM symbol_vss v JOIN symbols s ON s.symbol_id = v.symbol_id
            WHERE v.embedding MATCH ? AND k = 10
            ORDER BY v.distance
            """,
            (blob,),
        ).fetchall()
        for pass_num in range(2):
            for name, cat, aud, path, dist, name_uk, cat_uk in vrows:
                if dist > PHRASE_FALLBACK_THRESHOLD:
                    break
                if name in seen:
                    continue
                if pass_num == 0 and audience and aud != audience:
                    continue
                seen.add(name)
                phrase.append({"role": role, "display_name": name, "category": cat,
                               "audience": aud, "png_path": path, "distance": round(dist, 4),
                               "display_name_uk": name_uk or name, "category_uk": cat_uk or cat})
                matched = True
                break
            if matched:
                break

    conn.close()
    return phrase
