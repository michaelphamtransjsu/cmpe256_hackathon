"""Local FY2025 SEC ingestion and hybrid retrieval: python -m shared.sec_retrieval."""

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from shared.hybrid_retrieval import MODEL, ROOT, Retriever

DATA = ROOT / "data/UnitedStatesSecuritiesandExchangeCommission(SEC)"
CACHE = ROOT / "storage/sec"
FILINGS = [
    {"company": "Cisco", "fiscal_year": 2025, "filing_type": "10-K",
     "period_end": "2025-07-26", "filename": "6c821d1a-cf4e-4310-9801-df9399aa701b.pdf"},
    {"company": "Agilent", "fiscal_year": 2025, "filing_type": "10-K",
     "period_end": "2025-10-31", "filename": "b5098e8e-5716-4b35-8d8e-84dbdab516b1.pdf"},
]
CONFIG = {"model": MODEL, "max_tokens": 220, "merge_peers": False,
          "repeat_table_header": True, "do_ocr": False, "version": 1}


class SECRetriever:
    def __init__(self, index, contexts):
        self.index, self.contexts = index, contexts
        self.views = {}

    def retrieve(self, question, k=5, *, company=None, fiscal_year=None):
        """Filter before TF-IDF and semantic ranking; no embeddings are recomputed."""
        if k < 1:
            raise ValueError("k must be positive")
        key = (company.casefold() if company else None, fiscal_year)
        if key not in self.views:
            ids = [i for i, d in enumerate(self.index.documents)
                   if (key[0] is None or d.metadata["company"].casefold() == key[0])
                   and (fiscal_year is None or d.metadata["fiscal_year"] == fiscal_year)
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
        if fiscal_year:
            query = re.sub(r"\bFY\s*" + str(fiscal_year) + r"\b", "", query, flags=re.I)
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
    fingerprints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    key = json.dumps({"files": fingerprints, "config": CONFIG, "filings": FILINGS}, sort_keys=True)
    model = SentenceTransformer(MODEL, device="cpu")
    chunker = HybridChunker(tokenizer=HuggingFaceTokenizer(tokenizer=model.tokenizer, max_tokens=220),
                            merge_peers=False, repeat_table_header=True)
    contexts = {}

    def load():
        documents, failures = [], []
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(do_ocr=False))})
        for filing, path in zip(FILINGS, paths):
            source_id = f"{filing['company'].lower()}-{filing['fiscal_year']}-{filing['filing_type']}"
            digest = fingerprints[str(path)]
            artifact = cache_dir / f"{source_id}-{digest}.docling.json"
            context_path = cache_dir / f"{source_id}-context.json"
            try:
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
                        **filing, "source": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
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

    index = Retriever.from_documents(load, cache_dir, key, model)
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
