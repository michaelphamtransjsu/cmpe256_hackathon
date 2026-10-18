"""Technology PDF/web adapter. Prepare explicitly: python -m shared.technology_retrieval."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from shared.hybrid_retrieval import ROOT, SETTINGS, Retriever, build_cisco_retriever, require_prepared_cache
from shared.html_evidence import extract_html_text
from shared.query_context import cache_only

WEB_DIR = ROOT / "data/technology/web"
CACHE = ROOT / "storage/technology/multivendor"
SOURCES = [
    {"id": "google-mcp-overview", "vendor": "Google Cloud", "title": "Google Cloud MCP servers overview",
     "url": "https://docs.cloud.google.com/mcp/overview"},
    {"id": "google-mcp-explainer", "vendor": "Google Cloud", "title": "What is Model Context Protocol?",
     "url": "https://cloud.google.com/discover/what-is-model-context-protocol"},
    {"id": "google-rag", "vendor": "Google Cloud", "title": "Retrieval-augmented generation",
     "url": "https://cloud.google.com/use-cases/retrieval-augmented-generation"},
    {"id": "ibm-rag-architecture", "vendor": "IBM", "title": "Retrieval-augmented generation architecture pattern",
     "url": "https://www.ibm.com/think/architectures/patterns/genai-rag"},
    {"id": "ibm-mcp", "vendor": "IBM", "title": "What is Model Context Protocol?",
     "url": "https://www.ibm.com/think/topics/model-context-protocol"},
    {'id': 'cisco-8300-overview', 'product_family': 'Catalyst 8300', 'vendor': 'Cisco', 'title': 'Cisco Catalyst 8300 Series Edge Platforms', 'url': 'https://www.cisco.com/c/en/us/products/routers/catalyst-8300-series-edge-platforms/index.html'},
    {'id': 'cisco-8200-overview', 'product_family': 'Catalyst 8200', 'vendor': 'Cisco', 'title': 'Cisco Catalyst 8200 Series Edge Platforms', 'url': 'https://www.cisco.com/c/en/us/products/routers/catalyst-8200-series-edge-platforms/index.html'},
    {'id': 'cisco-8000v-overview', 'product_family': 'Catalyst 8000V', 'vendor': 'Cisco', 'title': 'Cisco Catalyst 8000V Edge Software', 'url': 'https://www.cisco.com/c/en/us/products/routers/catalyst-8000v-edge-software/index.html'},
    {'id': 'cisco-4000-overview', 'product_family': 'ISR 4000', 'vendor': 'Cisco', 'title': 'Cisco 4000 Series Integrated Services Routers (ISR)', 'url': 'https://www.cisco.com/c/en/us/products/routers/4000-series-integrated-services-routers-isr/index.html'},
    {'id': 'cisco-asr1000-overview', 'product_family': 'ASR 1000', 'vendor': 'Cisco', 'title': 'Cisco ASR 1000 Series Aggregation Services Routers', 'url': 'https://www.cisco.com/c/en/us/products/routers/asr-1000-series-aggregation-services-routers/index.html'},
]
WEB_SETTINGS = {"version": 1, "chunk_tokens": 200, "overlap": 25, "extractor": "main-or-article"}


def article_text(html):
    from bs4 import BeautifulSoup

    # Validate the full response first so challenge-page titles cannot be discarded.
    extract_html_text(html)
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"})
    if main is not None:
        content = str(main)
    else:
        # Google marketing and IBM Think use content sections rather than <main>.
        sections = soup.select("section[id], div[id^='rich-text-'], "
                               "#container-product-family-highlights, #container-models-preview, "
                               "#container-product-offers, #container-add-ons, "
                               "div[id^='overview-'], div[id^='modelcomparison-']")
        selected = {id(node) for node in sections}
        content = "\n".join(str(node) for node in sections
                            if not any(id(parent) in selected for parent in node.parents))
        if not content:
            raise ValueError("No recognized article content; refusing to index page chrome")
    text = extract_html_text(content)
    if len(text.split()) < 100:
        raise ValueError("Insufficient article text; page may require JavaScript")
    return text


def download_sources(web_dir=WEB_DIR, *, refresh=False):
    """Fetch only the curated official URLs; normal retrieval never downloads."""
    import requests

    if cache_only.get():
        raise RuntimeError("Source preparation is disabled during answering")
    web_dir = Path(web_dir)
    web_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = web_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest, failures = {}, []
    for source in SOURCES:
        path = web_dir / (source["id"] + ".html")
        try:
            if path.exists() and not refresh:
                content = path.read_bytes()
                old = previous.get(source["id"], {})
                final_url = old.get("final_url", source["url"])
                retrieved_at = old.get("retrieved_at") or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            else:
                response = requests.get(source["url"], timeout=60)
                print(f"{source['id']}: HTTP {response.status_code}", flush=True)
                response.raise_for_status()
                if "html" not in response.headers.get("content-type", "").lower():
                    raise ValueError("Expected an HTML article")
                content = response.content
                final_url = response.url
                retrieved_at = datetime.now(timezone.utc).isoformat()
            article_text(content.decode("utf-8", errors="replace"))
            path.write_bytes(content)
            manifest[source["id"]] = {**source, "final_url": final_url, "retrieved_at": retrieved_at,
                "sha256": hashlib.sha256(content).hexdigest(), "filename": path.name}
            print(f"{source['id']}: validated article ({len(content)} bytes)", flush=True)
        except Exception as error:
            failures.append(source["id"])
            print(f"FAILED {source['id']} ({source['url']}): {type(error).__name__}: {error}", flush=True)
    if failures:
        raise RuntimeError("Web preparation incomplete: " + ", ".join(failures))
    manifest_path.write_text(json.dumps(manifest, indent=2))


class TechnologyRetriever:
    def __init__(self, index):
        self.index = index
        self.views = {}

    def retrieve(self, question, k=5, *, mode="hybrid", filter_family=True, vendor=None):
        names = set()
        if vendor:
            aliases = {"cisco": "Cisco", "google": "Google Cloud", "google cloud": "Google Cloud",
                       "gcp": "Google Cloud", "ibm": "IBM"}
            if vendor.casefold() not in aliases:
                raise ValueError("Supported vendors: Cisco, Google Cloud, IBM")
            names.add(aliases[vendor.casefold()])
        else:
            for pattern, name in [(r"\bCisco\b|\bCatalyst\b|\bASR\b|\bISR\b", "Cisco"),
                                  (r"\bGoogle(?: Cloud)?\b|\bGCP\b", "Google Cloud"),
                                  (r"\bIBM\b|\bwatsonx\b", "IBM")]:
                if re.search(pattern, question, re.I):
                    names.add(name)
        key = tuple(sorted(names))
        if key not in self.views:
            ids = [i for i, d in enumerate(self.index.documents) if not names or d.metadata["vendor"] in names]
            self.views[key] = Retriever([self.index.documents[i] for i in ids],
                                       self.index.embeddings[ids], self.index.model) if ids else None
        view = self.views[key]
        # A Cisco family restriction must not discard another explicitly requested vendor.
        return view.retrieve(question, k, mode=mode,
                             filter_family=filter_family and len(names) <= 1) if view else []


def build_technology_retriever(web_dir=WEB_DIR, cache_dir=CACHE):
    """Reuse Cisco PDF chunks and add token-bounded webpage chunks to a separate index."""
    web_dir = Path(web_dir)
    manifest = json.loads((web_dir / "manifest.json").read_text())
    files = {}
    for source in SOURCES:
        entry = manifest[source["id"]]
        digest = hashlib.sha256((web_dir / entry["filename"]).read_bytes()).hexdigest()
        if digest != entry["sha256"] or entry["url"] != source["url"]:
            raise ValueError(f"Source changed: {source['id']}; refresh source preparation")
        files[source["id"]] = digest
    pdf_paths = sorted((ROOT / "data/technology").glob("cisco_*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"Missing Cisco source PDFs: {ROOT / 'data/technology'}")
    pdf_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in pdf_paths}
    key = json.dumps({"pdfs": pdf_hashes, "web": files, "sources": SOURCES,
                      "settings": SETTINGS, "web_settings": WEB_SETTINGS}, sort_keys=True)

    require_prepared_cache(cache_dir, key)
    cached_path = Path(cache_dir) / "index.json"
    if (cached_path.exists() and (Path(cache_dir) / "embeddings.npy").is_file()
            and json.loads(cached_path.read_text()).get("key") == key):
        # The combined cache is self-contained; do not load the PDF-only index too.
        return TechnologyRetriever(Retriever.from_documents([], cache_dir, key))
    cisco = build_cisco_retriever()

    def load():
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        documents = [Document(page_content=d.page_content, metadata={**d.metadata, "vendor": "Cisco",
            "source_id": Path(d.metadata["source"]).stem, "format": "pdf"}) for d in cisco.documents]
        # Retain existing chunks verbatim when their content and processing settings match.
        old_path = Path(cache_dir) / "index.json"
        old = json.loads(old_path.read_text()) if old_path.exists() else {}
        old_key = json.loads(old["key"]) if old else {}
        compatible = (old_key.get("settings") == SETTINGS
                      and old_key.get("web_settings") == WEB_SETTINGS)
        old_sources = {s["id"]: s for s in old_key.get("sources", [])}
        for source in SOURCES:
            entry = manifest[source["id"]]
            unchanged = (compatible and old_sources.get(source["id"]) == source
                         and old_key.get("web", {}).get(source["id"]) == files[source["id"]])
            retained = [Document(**d) for d in old.get("documents", [])
                        if d["metadata"].get("source_id") == source["id"]] if unchanged else []
            if retained:
                documents.extend(retained)
                print(f"{source['id']}: retained {len(retained)} chunks", flush=True)
                continue
            text = article_text((web_dir / entry["filename"]).read_text(errors="replace"))
            prefix = source["vendor"] + " — " + source["title"] + "\n"
            budget = WEB_SETTINGS["chunk_tokens"] - len(cisco.model.tokenizer.encode(prefix))
            splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                cisco.model.tokenizer, chunk_size=budget, chunk_overlap=WEB_SETTINGS["overlap"])
            parts = splitter.split_text(text)
            for part in parts:
                documents.append(Document(page_content=prefix + part, metadata={
                    "source": entry["final_url"], "source_id": source["id"], "vendor": source["vendor"],
                    "title": source["title"], "original_url": source["url"], "format": "html", "pages": [], "page": None,
                    "sha256": entry["sha256"], "retrieved_at": entry["retrieved_at"],
                    **({"product_family": source["product_family"]} if "product_family" in source else {}),
                }))
            print(f"{source['id']}: {len(parts)} chunks", flush=True)
        return documents

    return TechnologyRetriever(Retriever.from_documents(load, cache_dir, key, cisco.model))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Explicitly refresh official webpage snapshots")
    args = parser.parse_args()
    download_sources(refresh=args.refresh)
    build_technology_retriever()
