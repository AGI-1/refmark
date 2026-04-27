"""Build an offline browser Refmark search demo for the German BGB.

The source is the official Gesetze-im-Internet HTML page. This script creates:

- a portable Refmark-compatible index
- a compact browser BM25 index
- a JavaScript data file for file:// friendly demos
- a static HTML demo that jumps to BGB paragraphs/sections
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import html
import json
import re
import sys
from pathlib import Path
from urllib import request

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from refmark.search_index import RetrievalView, SearchRegion, export_browser_search_index


BGB_URL = "https://www.gesetze-im-internet.de/bgb/BJNR001950896.html"


@dataclass(frozen=True)
class BgbRegion:
    doc_id: str
    region_id: str
    para: str
    title: str
    text: str
    source_anchor: str
    ordinal: int

    @property
    def stable_ref(self) -> str:
        return f"{self.doc_id}:{self.region_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the BGB browser search demo.")
    parser.add_argument("--output-dir", default="examples/bgb_browser_search/output")
    parser.add_argument("--source-url", default=BGB_URL)
    parser.add_argument("--limit", type=int, default=None, help="Optional region limit for fast probes.")
    parser.add_argument("--force-fetch", action="store_true", help="Refresh the official source HTML cache.")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    html_path = output / "bgb_official.html"
    source_html = fetch_source(args.source_url, html_path, force=args.force_fetch)
    regions = parse_bgb(source_html)
    if args.limit is not None:
        regions = regions[: args.limit]

    portable_path = output / "bgb_refmark_index.json"
    write_portable_index(regions, portable_path, args.source_url)
    browser_path = output / "bgb_browser_index.json"
    browser_payload = export_browser_search_index(portable_path, browser_path, max_text_chars=1000)
    data_path = output / "bgb_demo_data.js"
    data_path.write_text(
        "window.BGB_REFMARK_INDEX = "
        + json.dumps(browser_payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    demo_path = output / "index.html"
    demo_path.write_text(render_demo_html(regions), encoding="utf-8")
    manifest = {
        "source_url": args.source_url,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "regions": len(regions),
        "portable_index": str(portable_path),
        "browser_index": str(browser_path),
        "demo": str(demo_path),
        "data_js": str(data_path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def fetch_source(url: str, destination: Path, *, force: bool = False) -> str:
    if destination.exists() and not force:
        return destination.read_text(encoding="utf-8")
    req = request.Request(url, headers={"User-Agent": "refmark-bgb-demo/0.1"})
    with request.urlopen(req, timeout=90) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    try:
        text = raw.decode(charset)
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    destination.write_text(text, encoding="utf-8")
    return text


def parse_bgb(source_html: str) -> list[BgbRegion]:
    soup = BeautifulSoup(source_html, "html.parser")
    regions: list[BgbRegion] = []
    ordinal = 0
    for norm in soup.select('div.jnnorm[title="Einzelnorm"]'):
        heading = norm.select_one("h3")
        if heading is None:
            continue
        para_node = heading.select_one(".jnenbez")
        title_node = heading.select_one(".jnentitel")
        para = clean_text(para_node.get_text(" ", strip=True) if para_node else "")
        title = clean_text(title_node.get_text(" ", strip=True) if title_node else "")
        if not para.startswith("§"):
            continue
        source_anchor = ""
        anchor_node = norm.select_one("a[name]")
        if anchor_node and anchor_node.get("name"):
            source_anchor = str(anchor_node["name"])
        paragraphs = [clean_text(item.get_text(" ", strip=True)) for item in norm.select(".jurAbsatz")]
        paragraphs = [item for item in paragraphs if item and not item.startswith("(+++") and item != "(weggefallen)"]
        if not paragraphs and title:
            paragraphs = [title]
        for local_index, text in enumerate(paragraphs, start=1):
            ordinal += 1
            region_id = region_id_for(para, local_index, len(paragraphs))
            full_text = f"{para} {title}\n{text}".strip()
            regions.append(
                BgbRegion(
                    doc_id="bgb",
                    region_id=region_id,
                    para=para,
                    title=title,
                    text=full_text,
                    source_anchor=source_anchor,
                    ordinal=ordinal - 1,
                )
            )
    return regions


def write_portable_index(regions: list[BgbRegion], path: Path, source_url: str) -> None:
    search_regions = []
    for index, region in enumerate(regions):
        search_regions.append(
            SearchRegion(
                doc_id=region.doc_id,
                region_id=region.region_id,
                text=region.text,
                hash=hash_text(region.text),
                source_path=source_url + (f"#{region.source_anchor}" if region.source_anchor else ""),
                ordinal=region.ordinal,
                prev_region_id=regions[index - 1].region_id if index > 0 else None,
                next_region_id=regions[index + 1].region_id if index + 1 < len(regions) else None,
                view=local_legal_view(region),
            )
        )
    payload = {
        "schema": "refmark.portable_search_index.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_corpus": source_url,
        "settings": {
            "view_source": "local-bgb-demo",
            "model": "local-bgb-demo",
            "marker_format": "browser_data_ref",
            "chunker": "bgb-paragraph",
            "include_source_in_index": True,
        },
        "stats": {
            "documents": 1,
            "regions": len(search_regions),
            "approx_input_tokens": sum(max(1, len(region.text) // 4) for region in search_regions),
            "approx_output_tokens": 0,
            "approx_openrouter_cost_usd_at_mistral_nemo": 0.0,
        },
        "regions": [region.to_dict() for region in search_regions],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def local_legal_view(region: BgbRegion) -> RetrievalView:
    title_words = keywords(region.title)
    text_words = keywords(region.text)
    terms = unique([*title_words, *text_words, *synonyms_for(region.text)])[:18]
    topic = region.title or "diese Vorschrift"
    para = region.para.replace("§", "Paragraph").strip()
    questions = [
        f"Wo steht etwas zu {topic}?",
        f"Welche Vorschrift regelt {topic}?",
        f"Was sagt das BGB über {topic}?",
        f"Finde {para} {topic}",
        *question_hints(region.text),
    ]
    return RetrievalView(
        summary=f"{region.para} {region.title}".strip(),
        questions=questions,
        keywords=terms,
    )


def render_demo_html(regions: list[BgbRegion]) -> str:
    sections = "\n".join(render_region(region) for region in regions)
    return f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Refmark BGB Browser Search</title>
  <style>
    :root {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17212f; background: #f7f5ef; }}
    body {{ margin: 0; }}
    header {{ position: sticky; top: 0; z-index: 4; padding: 18px 24px 14px; background: #f7f5ef; border-bottom: 1px solid #d8d0c3; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
    .note {{ margin: 0; color: #5f6b7a; font-size: 14px; }}
    .search-row {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; margin-top: 14px; }}
    input {{ min-width: 0; padding: 12px 13px; border: 1px solid #b8ad9e; border-radius: 7px; font: inherit; background: #fff; }}
    button {{ padding: 0 14px; border: 1px solid #2c5f80; border-radius: 7px; background: #2c5f80; color: #fff; font: inherit; cursor: pointer; }}
    #results {{ display: grid; gap: 8px; margin: 12px 0 0; padding: 0; list-style: none; }}
    #results button {{ display: grid; width: 100%; gap: 3px; padding: 9px 11px; border: 1px solid #d2c7b9; background: #fff; color: #17212f; text-align: left; }}
    #results button:hover {{ background: #edf4f8; }}
    #results small {{ color: #607080; }}
    section {{ scroll-margin-top: 210px; padding: 20px 0; border-bottom: 1px solid #dfd7ca; }}
    h2 {{ margin: 0 0 8px; font-size: 21px; letter-spacing: 0; }}
    p {{ margin: 0; font-size: 16px; line-height: 1.58; }}
    .ref {{ color: #51616f; font-size: 13px; }}
    .refmark-active-hit {{ background: #e7f4ff; outline: 3px solid #2f7ebc; outline-offset: 7px; }}
  </style>
</head>
<body>
  <header>
    <h1>BGB Semantic Find</h1>
    <p class="note">Offline browser search over the official BGB text from gesetze-im-internet.de. Navigation demo only, not legal advice.</p>
    <form id="search" class="search-row">
      <input id="query" type="search" autocomplete="off" placeholder="Ask: Kündigungsfrist Wohnung, Geschäftsfähigkeit Minderjährige, Erbschaft ausschlagen...">
      <button>Search</button>
    </form>
    <ol id="results"></ol>
  </header>
  <main>
{sections}
  </main>
  <script src="../../../refmark/browser_search.js"></script>
  <script src="bgb_demo_data.js"></script>
  <script>
    const input = document.getElementById("query");
    const results = document.getElementById("results");
    const form = document.getElementById("search");
    function render() {{
      const hits = RefmarkSearch.search(window.BGB_REFMARK_INDEX, input.value, {{ topK: 5 }});
      results.innerHTML = "";
      for (const hit of hits) {{
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.innerHTML = `<strong>${{hit.stable_ref.replace("bgb:", "")}}</strong><small>${{hit.summary || hit.text}}</small>`;
        button.addEventListener("click", () => RefmarkSearch.jumpTo(hit.stable_ref, {{ behavior: "smooth" }}));
        item.appendChild(button);
        results.appendChild(item);
      }}
    }}
    input.addEventListener("input", render);
    form.addEventListener("submit", (event) => {{
      event.preventDefault();
      const first = RefmarkSearch.search(window.BGB_REFMARK_INDEX, input.value, {{ topK: 1 }})[0];
      if (first) RefmarkSearch.jumpTo(first.stable_ref, {{ behavior: "smooth" }});
      render();
    }});
  </script>
</body>
</html>
"""


def render_region(region: BgbRegion) -> str:
    return (
        f'    <section data-refmark-ref="{html.escape(region.stable_ref)}" id="{html.escape(dom_id(region.stable_ref))}">\n'
        f"      <div class=\"ref\">{html.escape(region.stable_ref)}</div>\n"
        f"      <h2>{html.escape(region.para)} {html.escape(region.title)}</h2>\n"
        f"      <p>{html.escape(region.text.split(chr(10), 1)[-1])}</p>\n"
        f"    </section>"
    )


def region_id_for(para: str, local_index: int, count: int) -> str:
    base = para.replace("§§", "SS").replace("§", "S")
    base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")
    if count <= 1:
        return base
    return f"{base}_A{local_index:02d}"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def keywords(text: str) -> list[str]:
    stop = {
        "und", "oder", "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
        "eines", "mit", "für", "von", "zur", "zum", "bei", "nach", "sich", "nicht",
        "ist", "sind", "werden", "wird", "über", "absatz", "satz", "nummer",
    }
    words = re.findall(r"[A-Za-zÄÖÜäöüß0-9]{3,}", text.lower())
    output = []
    for word in words:
        if word not in stop and word not in output:
            output.append(word)
    return output


def synonyms_for(text: str) -> list[str]:
    lowered = text.lower()
    pairs = {
        "wohnraum": ["wohnung", "mietwohnung"],
        "mieter": ["miete", "wohnung"],
        "vermieter": ["wohnung", "miete"],
        "kündigung": ["kuendigung", "kündigen", "kuendigen", "kündigungsfrist"],
        "frist": ["deadline", "dauer", "zeitraum"],
        "mangel": ["defekt", "fehler", "mietminderung"],
        "minderjähr": ["kind", "jugendlich"],
        "geschäftsfähigkeit": ["geschaeftsfaehigkeit", "minderjährige"],
        "erbschaft": ["erbe", "nachlass"],
        "ausschlagung": ["ausschlagen", "ablehnen"],
        "widerruf": ["widerrufsrecht", "zurücktreten", "rückgabe"],
        "verbraucher": ["kunde", "privatperson"],
        "digitale produkte": ["software", "app", "download"],
    }
    output: list[str] = []
    for needle, values in pairs.items():
        if needle in lowered:
            output.extend(values)
    return output


def question_hints(text: str) -> list[str]:
    lowered = text.lower()
    hints = []
    if "kündigung" in lowered and "wohnraum" in lowered:
        hints.extend(["Welche Kündigungsfrist gilt für eine Wohnung?", "Wann kann ein Mieter oder Vermieter kündigen?"])
    if "mietminderung" in lowered or ("miete" in lowered and "mangel" in lowered):
        hints.append("Wann darf die Miete wegen eines Mangels gemindert werden?")
    if "erbschaft" in lowered and "ausschlag" in lowered:
        hints.append("Wie kann man eine Erbschaft ausschlagen?")
    if "geschäftsfähigkeit" in lowered or "minderjähr" in lowered:
        hints.append("Welche Regeln gelten für minderjährige Personen?")
    if "digitale" in lowered and "produkt" in lowered:
        hints.append("Welche Rechte gelten bei digitalen Produkten?")
    return hints


def unique(items: list[str]) -> list[str]:
    output = []
    for item in items:
        if item and item not in output:
            output.append(item)
    return output


def dom_id(stable_ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", stable_ref)


if __name__ == "__main__":
    main()
