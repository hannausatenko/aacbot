"""LLM query understanding: converts a situation description into search keywords and phrase."""

import json
import os
import sqlite3
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import anthropic

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "pictograms.db"


def _read_api_key() -> str:
    env_file = BASE_DIR / ".env"
    try:
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "")


def _get_client() -> anthropic.Anthropic:
    key = _read_api_key()
    if not key:
        raise RuntimeError(
            f"ANTHROPIC_API_KEY not found. Create {BASE_DIR / '.env'} with ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=key)


def _build_catalog() -> str:
    """Build compact symbol catalog: display_name plus distinctive ai_tags grouped by category."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT display_name, category, ai_tags
            FROM symbols
            WHERE audience = 'adults' OR symbol_id NOT IN (
                SELECT symbol_id FROM symbols s2 WHERE s2.display_name = symbols.display_name
                AND s2.category = symbols.category AND s2.audience = 'adults'
            )
            ORDER BY category, display_name
        """).fetchall()
        conn.close()
        seen: set[tuple[str, str]] = set()
        by_cat: dict[str, list[str]] = defaultdict(list)
        for name, cat, ai_tags in rows:
            if (name, cat) in seen:
                continue
            seen.add((name, cat))
            tags = [t.strip() for t in (ai_tags or "").split(",") if t.strip()]
            name_words = set(name.lower().replace("-", " ").split())
            distinctive = [t for t in tags if t.lower() not in name_words][:3]
            entry = f"{name} ({', '.join(distinctive)})" if distinctive else name
            by_cat[cat].append(entry)
        return "\n".join(f"{cat}: {', '.join(entries)}" for cat, entries in sorted(by_cat.items()))
    except Exception:
        return ""


_CATALOG = _build_catalog()

EXPAND_SYSTEM = (
    "You help build AAC (pictogram) communication sequences from natural language.\n\n"
    "AVAILABLE SYMBOLS (use EXACT names from this list):\n"
    f"{_CATALOG}\n\n"
    "Tasks:\n"
    "1. keywords — 5-8 search terms covering ALL aspects of the situation.\n"
    "   Rules:\n"
    "   - Cover every aspect: sick+refusing food → include both health and food/emotion terms\n"
    "   - Concrete words only, no abstract: not 'help', 'comfort', 'activity', 'emotions'\n"
    "   - Use communication symbols (yes, no, want, thank you, hello, stop, wait, bye) ONLY for explicit "
    "greeting/agreeing/refusing queries — not for physical needs or emotional states\n\n"
    "2. phrase_parts — 2-5 items in AAC sentence order, each with a role label.\n"
    "   symbol MUST be an exact name from the symbol list above.\n"
    "   role must be one of: subject, feeling, symptom, want, action, object, person, place, response, condition\n"
    "   Order: subject → feeling/symptom → want → action/object → person/place\n"
    "   'want' always before what is wanted.\n"
    "   Examples:\n"
    "   'I am hungry' → [{\"role\":\"subject\",\"symbol\":\"I\"},{\"role\":\"want\",\"symbol\":\"want\"},{\"role\":\"object\",\"symbol\":\"eat\"}]\n"
    "   'I cannot breathe well' → [{\"role\":\"symptom\",\"symbol\":\"hard to breathe\"},{\"role\":\"condition\",\"symbol\":\"hurts\"},{\"role\":\"person\",\"symbol\":\"doctor man\"}]\n"
    "   'child is sick and refuses to eat' → [{\"role\":\"condition\",\"symbol\":\"ill\"},{\"role\":\"feeling\",\"symbol\":\"do not want\"},{\"role\":\"action\",\"symbol\":\"eat\"}]\n"
    "   'I feel sad and lonely' → [{\"role\":\"subject\",\"symbol\":\"I\"},{\"role\":\"feeling\",\"symbol\":\"sad\"},{\"role\":\"condition\",\"symbol\":\"tired\"}]\n"
    "   'I am scared and want my mother' → [{\"role\":\"subject\",\"symbol\":\"I\"},{\"role\":\"feeling\",\"symbol\":\"frightened\"},{\"role\":\"want\",\"symbol\":\"want\"},{\"role\":\"person\",\"symbol\":\"mother\"}]\n"
    "   'I need to go to the toilet' → [{\"role\":\"subject\",\"symbol\":\"I\"},{\"role\":\"want\",\"symbol\":\"want\"},{\"role\":\"object\",\"symbol\":\"toilet\"}]\n"
    "   'yes, thank you' → [{\"role\":\"response\",\"symbol\":\"yes\"},{\"role\":\"response\",\"symbol\":\"thank you\"}]\n\n"
    "3. audience — 'children', 'adults', or 'any'\n\n"
    "Respond with valid JSON only:\n"
    "{\"keywords\": \"word1, word2, ...\", "
    "\"phrase_parts\": [{\"role\": \"...\", \"symbol\": \"...\"}, ...], "
    "\"audience\": \"children|adults|any\"}"
)


@lru_cache(maxsize=512)
def expand_query(user_input: str) -> tuple[str, tuple[tuple[str, str], ...], str | None]:
    """Return (keywords, phrase_parts, audience). phrase_parts is ((role, symbol), ...)."""
    try:
        response = _get_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            temperature=0,
            system=EXPAND_SYSTEM,
            messages=[{"role": "user", "content": user_input}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        data = json.loads(text)
        keywords = data.get("keywords", "").strip()
        if not keywords or keywords.lower() == user_input.lower() or len(keywords.split()) > 12:
            keywords = user_input
        raw_parts = data.get("phrase_parts", [])
        phrase_parts = tuple(
            (str(p.get("role", "")).strip(), str(p.get("symbol", "")).strip())
            for p in raw_parts
            if str(p.get("symbol", "")).strip()
        )
        raw_aud = data.get("audience", "any").lower()
        audience = raw_aud if raw_aud in ("children", "adults") else None
        return keywords, phrase_parts, audience
    except Exception:
        return user_input, (), None
