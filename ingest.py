#!/usr/bin/env python3
"""Ingest dyvogra symbols: convert SVGs to PNGs, embed and store in sqlite-vec."""

import argparse
import shutil
import sqlite3
import struct
from pathlib import Path

import cairosvg
import sqlite_vec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
SVG_DIR = BASE_DIR / "data" / "dyvogra"
PNG_DIR = BASE_DIR / "data" / "dyvogra-png"
DB_PATH = BASE_DIR / "pictograms.db"

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
BATCH_SIZE = 128
PNG_WIDTH = 256

AUDIENCE_MAP = {"deti": "children", "dorosli": "adults"}


def parse_records() -> list[dict]:
    """Parse symbol metadata from filenames: {category}_____{name}___{variant}.{ext}"""
    records = []
    for path in sorted(SVG_DIR.glob("*")):
        if path.suffix not in (".svg", ".png"):
            continue
        stem = path.stem  # e.g. "actions_____ask___deti"
        parts = stem.split("_____", 1)
        if len(parts) != 2:
            continue
        category_raw, rest = parts
        name_parts = rest.rsplit("___", 1)
        if len(name_parts) != 2:
            continue
        name_raw, variant = name_parts

        category = category_raw.replace("_", " ").strip()
        name = name_raw.replace("_", " ").strip()
        audience = AUDIENCE_MAP.get(variant, variant)

        records.append({
            "symbol_id": stem,
            "symbol_name": stem,
            "display_name": name,
            "category": category,
            "audience": audience,
            "grammar": "",
            "tags": "",
            "doc_text": f"{name}. Category: {category}. Audience: {audience}.",
            "source_path": path,
            "png_path": None,
        })
    return records


def convert_to_png(records: list[dict]):
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    for rec in tqdm(records, desc="Converting to PNG"):
        png_path = PNG_DIR / f"{rec['symbol_name']}.png"
        if png_path.exists():
            rec["png_path"] = str(png_path)
            continue
        src = rec["source_path"]
        try:
            if src.suffix == ".png":
                shutil.copy2(src, png_path)
            else:
                cairosvg.svg2png(url=str(src), write_to=str(png_path), output_width=PNG_WIDTH)
            rec["png_path"] = str(png_path)
        except Exception as e:
            print(f"  failed {rec['symbol_name']}: {e}")
            failed += 1
    if failed:
        print(f"  {failed} conversions failed and will be skipped")


def open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def load_ai_metadata() -> dict[str, dict]:
    if not DB_PATH.exists():
        return {}
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT symbol_id, ai_title, ai_tags, description FROM symbols "
            "WHERE ai_title IS NOT NULL AND ai_title != ''"
        ).fetchall()
        return {row[0]: {"title": row[1], "tags": row[2], "description": row[3]} for row in rows}
    except Exception:
        return {}
    finally:
        conn.close()


def embed_and_store(records: list[dict], force: bool):
    valid = [r for r in records if r["png_path"] is not None]
    ai = load_ai_metadata()
    for r in valid:
        meta = ai.get(r["symbol_id"])
        if meta:
            r["doc_text"] = (
                f"{r['display_name']}. Category: {r['category']}. Audience: {r['audience']}."
                f" Also described as: {meta['description']}"
            )
    print(f"Embedding {len(valid)} symbols ({len(ai)} with Opus vision metadata)...")

    conn = open_db()

    if force:
        conn.execute("DROP TABLE IF EXISTS symbol_vss")
        conn.execute("DROP TABLE IF EXISTS symbols")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS symbols (
            symbol_id    TEXT PRIMARY KEY,
            symbol_name  TEXT,
            display_name TEXT,
            category     TEXT,
            audience     TEXT,
            grammar      TEXT,
            tags         TEXT,
            png_path     TEXT,
            description  TEXT,
            ai_title     TEXT,
            ai_tags      TEXT
        )
    """)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS symbol_vss USING vec0(
            symbol_id TEXT PRIMARY KEY,
            embedding FLOAT[{EMBED_DIM}]
        )
    """)

    existing = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    if existing > 0 and not force:
        print(f"DB already has {existing} symbols. Use --force to re-embed.")
        conn.close()
        return

    model = SentenceTransformer(EMBED_MODEL)

    for i in tqdm(range(0, len(valid), BATCH_SIZE), desc="Embedding batches"):
        batch = valid[i:i + BATCH_SIZE]
        embeddings = model.encode([r["doc_text"] for r in batch], normalize_embeddings=True)
        conn.executemany(
            "INSERT OR REPLACE INTO symbols VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(r["symbol_id"], r["symbol_name"], r["display_name"],
              r["category"], r["audience"], r["grammar"], r["tags"], r["png_path"],
              ai.get(r["symbol_id"], {}).get("description"),
              ai.get(r["symbol_id"], {}).get("title"),
              ai.get(r["symbol_id"], {}).get("tags")) for r in batch],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO symbol_vss VALUES (?, ?)",
            [(r["symbol_id"], struct.pack(f"{EMBED_DIM}f", *emb))
             for r, emb in zip(batch, embeddings)],
        )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    conn.close()
    print(f"Done. DB now has {count} symbols at {DB_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Ingest dyvogra symbols into sqlite-vec")
    parser.add_argument("--force", action="store_true", help="Re-embed even if DB exists")
    parser.add_argument("--skip-png", action="store_true", help="Skip SVG→PNG conversion")
    args = parser.parse_args()

    if not SVG_DIR.exists():
        print(f"ERROR: symbol directory not found at {SVG_DIR}")
        return

    print("Parsing symbols from filenames...")
    records = parse_records()
    print(f"  Found {len(records)} symbols")

    if not args.skip_png:
        convert_to_png(records)
    else:
        for rec in records:
            png_path = PNG_DIR / f"{rec['symbol_name']}.png"
            rec["png_path"] = str(png_path) if png_path.exists() else None

    embed_and_store(records, force=args.force)


if __name__ == "__main__":
    main()
