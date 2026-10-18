"""Local SEC ingestion and hybrid retrieval: python -m shared.sec_retrieval."""

import hashlib
import json
import re
from functools import lru_cache
from collections import defaultdict
from pathlib import Path

import numpy as np

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from shared.hybrid_retrieval import MODEL, MODEL_REVISION, ROOT, Retriever

DATA = ROOT / "data/UnitedStatesSecuritiesandExchangeCommission(SEC)"
CACHE = ROOT / "storage/sec"
FILINGS = [
    {"company": "Cisco", "fiscal_year": 2025, "filing_type": "10-K",
     "period_end": "2025-07-26", "filename": "6c821d1a-cf4e-4310-9801-df9399aa701b.pdf"},
    {"company": "Agilent", "fiscal_year": 2025, "filing_type": "10-K",
     "period_end": "2025-10-31", "filename": "b5098e8e-5716-4b35-8d8e-84dbdab516b1.pdf"},
    {"company": "Cisco", "fiscal_year": 2026, "filing_type": "10-K",
     "period_end": "2026-07-25", "filename": "028fbac9-c5ba-409b-83ce-dfee9c17fdbc.pdf"},
    {"company": "Alphabet", "fiscal_year": 2024, "filing_type": "10-K",
     "period_end": "2024-12-31", "filename": "0ca59726-c36c-45fe-a114-e147f8cd7237.pdf"},
    {"company": "Alphabet", "fiscal_year": 2025, "filing_type": "10-K",
     "period_end": "2025-12-31", "filename": "8873e505-7403-4c46-b377-81d814482cbb.pdf"},
]
CONFIG = {"model": MODEL, "max_tokens": 220, "merge_peers": False,
          "repeat_table_header": True, "do_ocr": False, "version": 1,
          "model_revision": MODEL_REVISION}


def _artifact_name(source_id, digest):
    """Parsed-document reuse depends on conversion settings, not chunk size."""
    conversion = {"sha256": digest, "do_ocr": CONFIG["do_ocr"], "version": CONFIG["version"]}
    identity = hashlib.sha256(json.dumps(conversion, sort_keys=True).encode()).hexdigest()
    return f"{source_id}-{identity}.docling.json"


class _ReuseEmbeddings:
    """Reuse exact text vectors only with the identical pinned embedding model."""
    def __init__(self, model, previous, vectors):
        self.model = model
        self.vectors = defaultdict(list)
        for document, vector in zip(previous, vectors):
            self.vectors[document["page_content"]].append(vector)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return self.model.encode(texts, **kwargs)
        missing = list(dict.fromkeys(t for t in texts if t not in self.vectors))
        missing_set = set(missing)
        print(f"Embeddings: reused {len(texts) - sum(t in missing_set for t in texts)}; "
              f"encoding {len(missing)} new unique chunks", flush=True)
        if missing:
            for text, vector in zip(missing, self.model.encode(missing, **kwargs)):
                self.vectors[text] = [vector]
        occurrences = defaultdict(int)
        result = []
        for text in texts:
            vectors = self.vectors[text]
            result.append(vectors[min(occurrences[text], len(vectors) - 1)])
            occurrences[text] += 1
        return np.stack(result)


class SECRetriever:
    def __init__(self, index, contexts):
        self.index, self.contexts = index, contexts
        self.views = {}

    def retrieve(self, question, k=5, *, company=None, fiscal_year=None):
        """Filter before TF-IDF and semantic ranking; no embeddings are recomputed."""
        if k < 1:
            raise ValueError("k must be positive")
        companies = [company] if isinstance(company, str) else list(company or [])
        years = [fiscal_year] if isinstance(fiscal_year, int) else list(fiscal_year or [])
        names = tuple(sorted({name.casefold() for name in companies}))
        periods = tuple(sorted(set(years)))
        key = (names[0] if len(names) == 1 else names or None,
               periods[0] if len(periods) == 1 else periods or None)
        if key not in self.views:
            ids = [i for i, d in enumerate(self.index.documents)
                   if (not names or d.metadata["company"].casefold() in names)
                   and (not periods or d.metadata["fiscal_year"] in periods)
                   and (d.metadata["table_refs"] or len(d.page_content.split()) >= 40)]
            self.views[key] = Retriever([self.index.documents[i] for i in ids],
                self.index.embeddings[ids], self.index.model) if ids else None
        view = self.views[key]
        if view is None:
            return []
        query = question
        if company:
            # Metadata already scopes these entities; repeated company headers are not evidence.
            for filing in FILINGS:
                query = re.sub(r"\b" + re.escape(filing["company"]) + r"(?:['’]s)?\b", "", query, flags=re.I)
        for year in periods:
            query = re.sub(r"\bFY\s*" + str(year) + r"\b", "", query, flags=re.I)
        query = re.sub(r"\brevenues\b", "revenue", query, flags=re.I)
        results = view.retrieve(query, k=k, filter_family=False)
        for result in results:
            meta = dict(result["metadata"])
            context = self.contexts[meta["source_id"]]
            meta["table_context"] = {ref: context["tables"][ref] for ref in meta["table_refs"]}
            # Full physical pages preserve nearby unit labels and unattached footnotes.
            meta["page_context"] = {str(p): context["pages"][str(p)] for p in meta["pages"]}
            result["metadata"] = meta
        return results


def build_sec_retriever(data_dir=DATA, cache_dir=CACHE):
    from docling.chunking import HybridChunker
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from docling_core.types.doc import DoclingDocument
    import pypdfium2 as pdfium

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(data_dir) / f["company"] / f["filename"] for f in FILINGS]
    fingerprints = {p.relative_to(data_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    key = json.dumps({"files": fingerprints, "config": CONFIG, "filings": FILINGS}, sort_keys=True)
    model = SentenceTransformer(MODEL, revision=CONFIG["model_revision"], device="cpu")
    chunker = HybridChunker(tokenizer=HuggingFaceTokenizer(tokenizer=model.tokenizer, max_tokens=CONFIG["max_tokens"]),
                            merge_peers=CONFIG["merge_peers"], repeat_table_header=CONFIG["repeat_table_header"])
    contexts = {}
    previous_path = cache_dir / "index.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    previous_key = json.loads(previous["key"]) if previous else {}
    previous_filings = previous_key.get("filings", [])
    compatible = previous_key.get("config") == CONFIG

    def load():
        documents, failures = [], []
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(do_ocr=CONFIG["do_ocr"]))})
        for filing, path in zip(FILINGS, paths):
            source_id = f"{filing['company'].lower()}-{filing['fiscal_year']}-{filing['filing_type']}"
            digest = fingerprints[path.relative_to(data_dir).as_posix()]
            artifact = cache_dir / _artifact_name(source_id, digest)
            context_path = cache_dir / f"{source_id}-context.json"
            try:
                relative = path.relative_to(data_dir).as_posix()
                retained = [d for d in previous.get("documents", [])
                            if d["metadata"]["source_id"] == source_id]
                if (compatible and filing in previous_filings
                        and previous_key.get("files", {}).get(relative) == digest
                        and retained and context_path.exists()):
                    contexts[source_id] = json.loads(context_path.read_text())
                    documents.extend(Document(**d) for d in retained)
                    print(f"{source_id}: reused {len(retained)} cached chunks and context", flush=True)
                    continue
                if artifact.exists():
                    doc = DoclingDocument.load_from_json(artifact)
                else:
                    result = converter.convert(path)
                    if result.status != ConversionStatus.SUCCESS:
                        raise RuntimeError(f"{result.status}: {result.errors}")
                    doc = result.document
                    doc.save_as_json(artifact)
                tables = {t.self_ref: {"markdown": t.export_to_markdown(doc=doc),
                    "captions": [r.resolve(doc).text for r in t.captions],
                    "footnotes": [r.resolve(doc).text for r in t.footnotes]} for t in doc.tables}
                pages = {}
                with pdfium.PdfDocument(path) as pdf:
                    for i in range(len(pdf)):
                        page = pdf[i]
                        text = page.get_textpage()
                        pages[str(i + 1)] = text.get_text_range()
                        text.close()
                        page.close()
                contexts[source_id] = {"tables": tables, "pages": pages}
                context_path.write_text(json.dumps(contexts[source_id], ensure_ascii=False))
                count = 0
                for chunk in chunker.chunk(doc):
                    meta = chunk.meta.export_json_dict()
                    items = meta.get("doc_items", [])
                    physical_pages = sorted({p["page_no"] for item in items for p in item.get("prov", [])})
                    documents.append(Document(page_content=chunker.contextualize(chunk), metadata={
                        **filing, "source": "data/UnitedStatesSecuritiesandExchangeCommission(SEC)/" + path.relative_to(data_dir).as_posix(),
                        "source_id": source_id, "sha256": digest, "pages": physical_pages,
                        "page": physical_pages[0] if len(physical_pages) == 1 else None,
                        "table_refs": [item["self_ref"] for item in items if item["self_ref"] in tables],
                        "dl_meta": meta,
                    }))
                    count += 1
                if not count:
                    raise ValueError("No chunks produced")
                print(f"{source_id}: {count} chunks, {len(tables)} tables", flush=True)
            except Exception as error:
                failures.append(source_id)
                print(f"FAILED {source_id}: {type(error).__name__}: {error}", flush=True)
        if failures:
            raise RuntimeError(f"Incomplete SEC corpus: {', '.join(failures)}")
        return documents

    embedding_model = model
    vector_path = cache_dir / "embeddings.npy"
    previous_config = previous_key.get("config", {})
    if (previous and vector_path.exists()
            and all(previous_config.get(k) == CONFIG[k] for k in ("model", "model_revision"))):
        vectors = np.load(vector_path, allow_pickle=False)
        if vectors.shape != (len(previous["documents"]), model.get_sentence_embedding_dimension()):
            raise ValueError("Invalid previous SEC embedding shape")
        embedding_model = _ReuseEmbeddings(model, previous["documents"], vectors)
    index = Retriever.from_documents(load, cache_dir, key, embedding_model)
    index.model = model
    for filing in FILINGS:
        source_id = f"{filing['company'].lower()}-{filing['fiscal_year']}-{filing['filing_type']}"
        if source_id not in contexts:
            contexts[source_id] = json.loads((cache_dir / f"{source_id}-context.json").read_text())
    return SECRetriever(index, contexts)


@lru_cache(maxsize=1)
def _default():
    return build_sec_retriever()


def retrieve(question, k=5, *, company=None, fiscal_year=None):
    return _default().retrieve(question, k, company=company, fiscal_year=fiscal_year)


if __name__ == "__main__":
    for question, company in [
        ("What are Cisco's principal business and operational risks in FY2025?", "Cisco"),
        ("What are Agilent's principal risks in FY2025?", "Agilent"),
        ("What were Cisco's and Agilent's FY2025 revenues?", "Cisco"),
        ("What were Cisco's and Agilent's FY2025 revenues?", "Agilent"),
    ]:
        print(f"\nQUESTION: {question} COMPANY: {company}", flush=True)
        for rank, result in enumerate(retrieve(question, company=company, fiscal_year=2025), 1):
            meta = result["metadata"]
            print(f"\n{rank}. RRF={result['score']:.6f} " + json.dumps({
                k: meta[k] for k in ["company", "fiscal_year", "filing_type", "period_end", "filename", "pages"]}))
            print(result["text"], flush=True)
