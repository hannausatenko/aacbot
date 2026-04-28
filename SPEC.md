# Dimobi Search — Technical Specification

## Purpose

Pictogram search assistant for AAC (Augmentative and Alternative Communication) users and their communication partners. Converts a natural-language description of a situation (any language) into a curated set of Dyvogra pictograms and a suggested pictogram phrase.

---

## Architecture

```
user query
    │
    ▼
llm.py  expand_query()          ← Claude claude-sonnet-4-6
    │   returns: keywords, phrase_parts, audience
    │
    ├──► rag.py  retrieve()      ← sqlite-vec KNN + CONDITIONAL_SYMBOLS filter
    │            returns: list[dict] (category-grouped results, ≤40)
    │
    └──► rag.py  retrieve_phrase()
                 returns: list[dict] (≤5 phrase-strip symbols)
    │
    ▼
app.py  _render_constructor()   ← phrase strip HTML
app.py  _render_results()       ← category grid HTML
    │
    ▼
Gradio gr.HTML components
```

---

## Stack

| Component | Choice | Notes |
|---|---|---|
| UI | Gradio 5+ (`gr.Blocks`) | Dark theme, no footer |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | 384-dim, 50+ languages |
| Vector store | SQLite + `sqlite-vec` | KNN over `symbol_vss` virtual table |
| LLM | Claude `claude-sonnet-4-6` | Query expansion + phrase construction |
| Images | Pre-converted PNGs | `data/dyvogra-png/`, served as base64 inline |

---

## Data

### Database: `pictograms.db`

**`symbols` table**
| Column | Type | Description |
|---|---|---|
| `symbol_id` | INTEGER PK | |
| `symbol_name` | TEXT | Raw filename key (used for communication trigger matching) |
| `display_name` | TEXT | Human-readable English name |
| `display_name_uk` | TEXT | Ukrainian translation |
| `category` | TEXT | Category slug |
| `category_uk` | TEXT | Ukrainian category label |
| `audience` | TEXT | `'children'` \| `'adults'` |
| `png_path` | TEXT | Absolute path to PNG |
| `ai_tags` | TEXT | Comma-separated tags (used in LLM catalog) |

**`symbol_vss` virtual table** — sqlite-vec KNN index over 384-dim embeddings.

### Images: `data/dyvogra-png/`
Pre-converted PNG files (~6.7 MB total). Served inline as base64 data URIs (no separate file server needed).

---

## Modules

### `llm.py` — Query expansion

`expand_query(user_input: str) → (keywords, phrase_parts, audience)`

- Calls Claude with a system prompt containing the full symbol catalog (name + distinctive tags per category).
- Returns:
  - `keywords` — comma-separated search terms (5–8 words covering all aspects)
  - `phrase_parts` — `((role, symbol_name), ...)` for phrase strip construction
  - `audience` — `'children'` | `'adults'` | `None`
- Result is cached with `@lru_cache(maxsize=512)`.

### `rag.py` — Retrieval

`retrieve(query, n_results=40, audience=None) → list[dict]`
- Encodes query with the sentence-transformer model.
- KNN search in `symbol_vss`; filters by `SYMBOL_THRESHOLD = 1.20`.
- Applies `CONDITIONAL_SYMBOLS` guard (number/pronoun symbols only shown when their trigger words appear in the query).
- For audience filtering: preferred audience first, fallback to other variant when no preferred match exists.
- For multi-aspect queries (≥6 keywords): also searches each half separately and merges.
- Injects `COMMUNICATION_TRIGGERS` symbols (yes/no/want/stop/…) when their exact concept appears in the query.
- Category-level filter: only includes categories whose best result is within `STRONG_THRESHOLD = 1.12`.

`retrieve_phrase(parts, audience=None) → list[dict]`
- Per `(role, symbol_name)` pair: direct `display_name` lookup first, then vector search fallback.
- Returns ≤5 symbols in phrase order.

### `app.py` — UI

- `_search(query, language, audience_label)` — orchestrates `expand_query` → `retrieve_phrase` → `retrieve` → HTML render.
- `_render_constructor(parts, language)` — phrase strip: icons with role labels and arrows.
- `_render_results(results, language)` — category nav bar + symbol grid.
- Language state: `gr.State("English")` switched via 🇬🇧/🇺🇦 flag buttons.
- Audience filter: dropdown (Any / Children / Adults), overrides LLM-detected audience.
- 10 example query buttons shown at the top, labels swap on language change.

---

## Configuration

| Variable | Where | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | `.env` or env var | Required for LLM query expansion |
| `GRADIO_SERVER_NAME` | env var | Default `127.0.0.1`; set to `0.0.0.0` in Docker |
| `GRADIO_SERVER_PORT` | env var | Default `7860` |

---

## Running Locally

```bash
pip install -r requirements.txt
cp .env.example .env      # add ANTHROPIC_API_KEY
python app.py             # open http://localhost:7860
```

## Running with Docker

```bash
docker build -t dimobi-search .
docker run -p 7860:7860 -e ANTHROPIC_API_KEY=sk-ant-... dimobi-search
# open http://localhost:7860
```

Or with an env file:
```bash
docker run -p 7860:7860 --env-file .env dimobi-search
```

---

## Retrieval Thresholds (tunable in `rag.py`)

| Constant | Value | Effect |
|---|---|---|
| `SYMBOL_THRESHOLD` | 1.20 | Max cosine distance for any individual symbol |
| `STRONG_THRESHOLD` | 1.12 | Max distance for a category to be included |
| `RELATIVE_SPREAD` | 0.06 | Within a category, only show symbols within this delta of the best match |
| `MAX_PER_CATEGORY` | 5 | Max symbols displayed per category |
| `PHRASE_FALLBACK_THRESHOLD` | 1.20 | Vector fallback threshold for phrase construction |
| `PHRASE_MAX` | 5 | Max symbols in the phrase strip |
