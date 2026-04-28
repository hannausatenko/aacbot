"""Pictogram search UI — phrase constructor."""

import base64
from collections import defaultdict
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from llm import expand_query
from rag import retrieve, retrieve_phrase

PNG_DIR = Path(__file__).resolve().parent / "data" / "dyvogra-png"

LANGUAGES = ["English", "Українська"]

AUDIENCE_OPTIONS = {
    "English":    ["Any", "Children", "Adults"],
    "Українська": ["Будь-які", "Діти", "Дорослі"],
}

AUDIENCE_VALUE_MAP = {
    "Any": None, "Будь-які": None,
    "Children": "children", "Діти": "children",
    "Adults": "adults",     "Дорослі": "adults",
}

UI_STRINGS = {
    "English": {
        "placeholder": "Describe what you want to say — e.g. 'I am hungry'",
        "no_results": "No pictograms found. Try rephrasing what you want to say.",
        "children": "children", "adults": "adults",
        "phrase_label": "▶ SUGGESTED PHRASE",
        "categories_label": "▶ SUGGESTED CATEGORIES",
        "search_btn": "Search",
        "title": "Dimobi Search",
        "subtitle": "Dyvogra symbols",
    },
    "Українська": {
        "placeholder": "Опишіть, що хочете сказати — напр. 'я голодний'",
        "no_results": "Нічого не знайдено. Спробуйте переформулювати.",
        "children": "діти", "adults": "дорослі",
        "phrase_label": "▶ ЗАПРОПОНОВАНА ФРАЗА",
        "categories_label": "▶ ЗАПРОПОНОВАНІ КАТЕГОРІЇ",
        "search_btn": "Пошук",
        "title": "Dimobi Пошук",
        "subtitle": "Символи Dyvogra",
    },
}

EXAMPLES = {
    "English": [
        "child is sick and doesn't want to eat",
        "I am hungry",
        "I have a headache",
        "I am scared and want my mother",
        "I need to go to the toilet",
        "I feel sad and lonely",
        "I want to call an ambulance",
        "I cannot breathe well",
        "I want to wash my hands",
        "Yes, thank you",
    ],
    "Українська": [
        "дитина хвора і не хоче їсти",
        "я голодний",
        "у мене болить голова",
        "мені страшно, хочу до мами",
        "мені потрібно в туалет",
        "мені сумно і самотньо",
        "хочу викликати швидку",
        "мені важко дихати",
        "хочу помити руки",
        "так, дякую",
    ],
}


def _img_b64(png_path: str) -> str | None:
    try:
        return base64.b64encode(Path(png_path).read_bytes()).decode()
    except Exception:
        return None


def _render_constructor(parts: list[dict], language: str = "English") -> str:
    if not parts:
        return (
            '<p style="color:#4a5568;font-size:13px;padding:12px 0">'
            'Search above to build a phrase.</p>'
        )

    slots = ""
    for i, part in enumerate(parts):
        if i > 0:
            slots += '<span style="color:#a78bfa;font-size:24px;align-self:center;margin-top:14px">→</span>'

        b64 = _img_b64(part["png_path"])
        bg = f"url(data:image/png;base64,{b64}) center/contain no-repeat #fff" if b64 else "#1e2035"
        role = part.get("role", "")
        name = part.get("display_name_uk", part["display_name"]) if language == "Українська" else part["display_name"]

        slots += f"""
<div style="flex:0 0 auto;text-align:center">
  <div style="font-size:10px;font-weight:700;color:#4a5568;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:6px;height:14px">{role}</div>
  <div style="width:96px;height:96px;background:{bg};border-radius:12px;
              box-shadow:0 2px 12px rgba(0,0,0,0.4)"></div>
  <div style="font-size:12px;font-weight:600;margin-top:8px;color:#e2e8f0;
              max-width:96px;line-height:1.3;word-wrap:break-word">{name}</div>
</div>"""

    return (
        f'<div style="background:#12152a;border:1px solid #2d2f45;border-radius:14px;'
        f'padding:20px 24px 24px">'
        f'<div style="display:flex;align-items:flex-start;flex-wrap:wrap;gap:10px">{slots}</div>'
        f'</div>'
    )


def _render_results(results: list[dict], language: str) -> str:
    strings = UI_STRINGS.get(language, UI_STRINGS["English"])
    if not results:
        return f'<p style="color:#94a3b8;padding:24px 0">{strings["no_results"]}</p>'

    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    use_uk = language == "Українська"

    # Category nav bar
    nav_items = ""
    for category, syms in by_category.items():
        cat_id = category.replace(" ", "-")
        first = syms[0]
        cat_label = (first.get("category_uk") or category) if use_uk else category.replace("_", " ").title()
        nav_items += (
            f'<a href="#{cat_id}" style="display:inline-block;padding:4px 10px;margin:3px;'
            f'background:#1e2035;border:1px solid #2d2f45;border-radius:20px;'
            f'font-size:12px;color:#a78bfa;text-decoration:none;white-space:nowrap;'
            f'transition:background 0.15s" '
            f'onmouseenter="this.style.background=\'#2d2f45\'" '
            f'onmouseleave="this.style.background=\'#1e2035\'">'
            f'{cat_label}</a>'
        )
    cat_header_label = strings["categories_label"]
    nav = (
        f'<div style="margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid #2d2f45">'
        f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:8px">'
        f'<span style="font-size:10px;font-weight:700;color:#4a5568;letter-spacing:0.12em;text-transform:uppercase">{cat_header_label}</span>'
        f'<a href="https://dimobi.org.ua/categories" target="_blank" '
        f'style="font-size:11px;color:#a78bfa;text-decoration:none;opacity:0.7" '
        f'onmouseenter="this.style.opacity=\'1\'" onmouseleave="this.style.opacity=\'0.7\'">dimobi.org.ua/categories ↗</a>'
        f'</div>'
        f'{nav_items}</div>'
    )

    html = nav
    for category, symbols in by_category.items():
        cat_id = category.replace(" ", "-")
        first = symbols[0]
        cat_label = (first.get("category_uk") or category) if use_uk else category.replace("_", " ").title()
        items_html = ""
        for r in symbols:
            b64 = _img_b64(r["png_path"])
            if not b64:
                continue
            src = f"data:image/png;base64,{b64}"
            audience = strings.get(r.get("audience", ""), r.get("audience", ""))
            aud_color = "#4a90d9" if r.get("audience") == "children" else "#9b7fe8"
            label = r.get("display_name_uk", r["display_name"]) if language == "Українська" else r["display_name"]
            items_html += (
                f'<div style="display:inline-block;text-align:center;width:100px;margin:6px;vertical-align:top">'
                f'<div style="width:80px;height:80px;background:url({src}) center/contain no-repeat #fff;'
                f'border-radius:8px;margin:0 auto;box-shadow:0 1px 4px rgba(0,0,0,0.15)"></div>'
                f'<div style="font-size:12px;font-weight:600;margin-top:6px;line-height:1.3;color:#e2e8f0">{label}</div>'
                f'<div style="font-size:10px;margin-top:2px;color:{aud_color};text-transform:uppercase;letter-spacing:0.06em">{audience}</div>'
                f'</div>'
            )
        html += (
            f'<div id="{cat_id}" style="margin-bottom:32px">'
            f'<a href="#{cat_id}" style="text-decoration:none">'
            f'<h3 style="font-size:15px;font-weight:700;color:#a78bfa;margin:0 0 12px;'
            f'border-bottom:1px solid #2d2f45;padding-bottom:8px;letter-spacing:0.03em">'
            f'# {cat_label}</h3></a>'
            f'<div style="display:flex;flex-wrap:wrap">{items_html}</div>'
            f'</div>'
        )
    return html


def _phrase_label(language: str) -> str:
    s = UI_STRINGS.get(language, UI_STRINGS["English"])
    return (
        f'<div style="font-size:10px;font-weight:700;color:#4a5568;letter-spacing:0.12em;'
        f'text-transform:uppercase;margin:16px 0 6px">{s["phrase_label"]}</div>'
    )


def _search(query: str, language: str, audience_label: str):
    if not query.strip():
        return [], "", _render_constructor([]), ""
    manual_audience = AUDIENCE_VALUE_MAP.get(audience_label)
    expanded, phrase_parts_raw, detected_audience = expand_query(query.strip())
    audience = manual_audience if manual_audience is not None else detected_audience
    phrase = retrieve_phrase(phrase_parts_raw, audience)
    results = retrieve(expanded, n_results=40, audience=audience)
    label = _phrase_label(language) if phrase else ""
    return phrase, label, _render_constructor(phrase, language), _render_results(results, language)



_DEFAULT_LANG = "English"
_EX = EXAMPLES[_DEFAULT_LANG]

with gr.Blocks(title="Dimobi Search") as demo:
    phrase_state = gr.State([])
    lang_state   = gr.State(_DEFAULT_LANG)

    with gr.Row():
        gr.Markdown("# Dimobi Search <small style='font-weight:400;font-size:14px;color:#64748b'>Dyvogra symbols</small>")
        flag_en = gr.Button("🇬🇧", size="sm", scale=0, min_width=48)
        flag_ua = gr.Button("🇺🇦", size="sm", scale=0, min_width=48)

    # Example queries at the top — 10 buttons reused across languages
    example_btns = []
    with gr.Row():
        for ex in _EX[:5]:
            example_btns.append(gr.Button(ex, size="sm"))
    with gr.Row():
        for ex in _EX[5:]:
            example_btns.append(gr.Button(ex, size="sm"))

    with gr.Row():
        audience_select = gr.Dropdown(
            choices=AUDIENCE_OPTIONS[_DEFAULT_LANG], value=AUDIENCE_OPTIONS[_DEFAULT_LANG][0],
            show_label=False, scale=1, container=False,
        )
        query_box = gr.Textbox(
            placeholder=UI_STRINGS[_DEFAULT_LANG]["placeholder"],
            show_label=False, scale=5, container=False,
        )
        search_btn = gr.Button(UI_STRINGS[_DEFAULT_LANG]["search_btn"], variant="primary", scale=1)

    phrase_label = gr.HTML(value="")
    phrase_html  = gr.HTML(value="")
    results_html = gr.HTML()

    for i, btn in enumerate(example_btns):
        def _set_example(lang, idx=i):
            exs = EXAMPLES.get(lang, EXAMPLES[_DEFAULT_LANG])
            return exs[idx] if idx < len(exs) else ""
        btn.click(
            fn=_set_example, inputs=[lang_state], outputs=query_box,
        ).then(
            fn=_search, inputs=[query_box, lang_state, audience_select],
            outputs=[phrase_state, phrase_label, phrase_html, results_html],
        )

    def _update_lang(language: str):
        strings = UI_STRINGS.get(language, UI_STRINGS[_DEFAULT_LANG])
        opts = AUDIENCE_OPTIONS.get(language, AUDIENCE_OPTIONS[_DEFAULT_LANG])
        exs = EXAMPLES.get(language, EXAMPLES[_DEFAULT_LANG])
        btn_updates = [gr.update(value=exs[i] if i < len(exs) else "") for i in range(10)]
        return (
            language,
            gr.update(placeholder=strings["placeholder"]),
            gr.update(choices=opts, value=opts[0]),
            gr.update(value=strings["search_btn"]),
            *btn_updates,
        )

    flag_en.click(fn=lambda: "English",   outputs=lang_state).then(
        fn=_update_lang, inputs=lang_state,
        outputs=[lang_state, query_box, audience_select, search_btn, *example_btns],
    )
    flag_ua.click(fn=lambda: "Українська", outputs=lang_state).then(
        fn=_update_lang, inputs=lang_state,
        outputs=[lang_state, query_box, audience_select, search_btn, *example_btns],
    )

    search_btn.click(
        fn=_search, inputs=[query_box, lang_state, audience_select],
        outputs=[phrase_state, phrase_label, phrase_html, results_html],
    )
    query_box.submit(
        fn=_search, inputs=[query_box, lang_state, audience_select],
        outputs=[phrase_state, phrase_label, phrase_html, results_html],
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        theme=gr.themes.Soft(),
        allowed_paths=[str(PNG_DIR)],
        show_error=True,
        css="footer { display: none !important; }",
    )
