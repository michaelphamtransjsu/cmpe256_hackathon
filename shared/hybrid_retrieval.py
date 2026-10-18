"""Local semantic/TF-IDF retrieval. Compare modes with python -m shared.hybrid_retrieval."""

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {"model": MODEL, "max_tokens": 220, "merge_peers": False,
            "repeat_table_header": True, "do_ocr": False, "cache_version": 1,
            "model_revision": MODEL_REVISION}


def _lexical_text(text):
    text = text.lower()
    text = re.sub(r"(\d+)\s*-?\s*(?:gigabit|gbps|g)\b", r"\1g", text)
    return re.sub(r"\b(ports|routers|interfaces)\b", lambda m: m[0][:-1], text)


def _source_family(metadata):
    if metadata.get("product_family"):
        family = str(metadata["product_family"])
        parsed = _question_families(family)
        return next(iter(parsed)) if parsed else re.sub(r"\W", "", family.lower())
    stem = Path(metadata.get("source", "")).stem.lower()
    return stem.removeprefix("cisco_") if stem.startswith("cisco_") else None


def _question_families(question):
    return {prefix.lower() + number.lower() if prefix.lower() == "asr" else number.lower()
            for prefix, number in re.findall(
                r"\b(Catalyst|ASR|ISR)\s*-?\s*(\d+[a-z]*)\b", question, re.I)}


class Retriever:
    """Accepts any loader's Documents. Hybrid scores are RRF, semantic scores cosine."""

    def __init__(self, documents, embeddings, model):
        self.documents = documents
        self.embeddings = embeddings
        self.model = model
        self.vectorizer = TfidfVectorizer(preprocessor=_lexical_text, stop_words="english",
                                          ngram_range=(1, 2), sublinear_tf=True)
        self.lexical = self.vectorizer.fit_transform([d.page_content for d in documents])

    def retrieve(self, question: str, k: int = 5, *, mode="hybrid",
                 filter_family=True) -> list[dict]:
        if k < 1:
            raise ValueError("k must be positive")
        if mode not in {"semantic", "hybrid"}:
            raise ValueError("mode must be 'semantic' or 'hybrid'")
        families = _question_families(question) if filter_family else set()
        candidates = np.array([i for i, d in enumerate(self.documents)
                               if not families or _source_family(d.metadata) in families], dtype=int)
        if not len(candidates):
            return []
        query = self.model.encode(question, normalize_embeddings=True)
        semantic_scores = self.embeddings[candidates] @ query
        semantic_order = np.argsort(-semantic_scores, kind="stable")
        scores = semantic_scores.copy()
        if mode == "hybrid":
            # Family has already selected the corpus; do not reward repeated branding.
            lexical_question = re.sub(r"\b(?:Catalyst|ASR|ISR)\s*-?\s*\d+[a-z]*\b", "",
                                      question, flags=re.I) if families else question
            lexical_scores = (self.lexical[candidates] @
                              self.vectorizer.transform([lexical_question]).T).toarray().ravel()
            lexical_order = np.argsort(-lexical_scores, kind="stable")
            scores = np.zeros(len(candidates))
            # Standard RRF: equal-weight lists, 1-based ranks, smoothing constant 60.
            scores[semantic_order] += 1.0 / (60 + np.arange(1, len(candidates) + 1))
            # Zero lexical overlap provides no ranking evidence.
            lexical_order = lexical_order[lexical_scores[lexical_order] > 0]
            scores[lexical_order] += 1.0 / (60 + np.arange(1, len(lexical_order) + 1))
        indices = np.argsort(-scores, kind="stable")[:k]
        return [{"text": self.documents[candidates[i]].page_content, "score": float(scores[i]),
                 "semantic_score": float(semantic_scores[i]), "score_type": mode,
                 "metadata": self.documents[candidates[i]].metadata} for i in indices]

    @classmethod
    def from_documents(cls, documents, cache_dir, cache_key, model=None):
        """Cache already-chunked documents and vectors; caller controls invalidation key."""
        model = model if model is not None else SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / "index.json"
        vectors_path = cache_dir / "embeddings.npy"
        if manifest_path.exists() and vectors_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["key"] == cache_key:
                docs = [Document(**item) for item in manifest["documents"]]
                vectors = np.load(vectors_path, allow_pickle=False)
                if vectors.shape != (len(docs), model.get_sentence_embedding_dimension()):
                    raise ValueError("Cached embeddings have an invalid shape")
                print(f"Loaded {len(docs)} cached chunks and embeddings", flush=True)
                return cls(docs, vectors, model)
        docs = list(documents() if callable(documents) else documents)
        if not docs:
            raise ValueError("No documents available to index")
        overflows = []
        for i, doc in enumerate(docs):
            count = len(model.tokenizer.encode(doc.page_content, truncation=False))
            doc.metadata["embedding_tokens"] = count
            if count > 256:
                overflows.append(i)
                print(f"OVER LIMIT: chunk {i}, {count} tokens, {doc.metadata}", flush=True)
        print(f"Token check: {len(docs)} chunks; {len(overflows)} exceed 256 tokens", flush=True)
        if overflows:
            raise ValueError("Refusing to silently truncate oversized chunks")
        vectors = model.encode([d.page_content for d in docs], normalize_embeddings=True,
                               show_progress_bar=True, convert_to_numpy=True)
        np.save(vectors_path, vectors, allow_pickle=False)
        manifest_path.write_text(json.dumps({"key": cache_key, "documents": [
            {"page_content": d.page_content, "metadata": d.metadata} for d in docs
        ]}, ensure_ascii=False))
        return cls(docs, vectors, model)


def build_cisco_retriever(pdf_dir=ROOT / "data/technology",
                          cache_dir=ROOT / "storage/technology"):
    """Docling is confined to this adapter; generic Retriever accepts any documents."""
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    paths = sorted(Path(pdf_dir).glob("cisco_*.pdf"))
    if not paths:
        raise FileNotFoundError(f"No Cisco PDFs in {pdf_dir}")
    model = SentenceTransformer(MODEL, revision=SETTINGS["model_revision"], device="cpu")
    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=model.tokenizer, max_tokens=SETTINGS["max_tokens"]),
        merge_peers=SETTINGS["merge_peers"], repeat_table_header=SETTINGS["repeat_table_header"],
    )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    key = json.dumps({"files": fingerprints, "settings": SETTINGS}, sort_keys=True)

    def load():
        # These digital PDFs already contain text; retain Docling table recognition.
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(do_ocr=SETTINGS["do_ocr"]))})
        failures = []
        documents = []
        for path in paths:
            fingerprint = hashlib.sha256((fingerprints[path.name] +
                                         path.name + json.dumps(SETTINGS, sort_keys=True)).encode()).hexdigest()
            chunk_cache = cache_dir / f"{path.stem}-{fingerprint}.json"
            try:
                if chunk_cache.exists():
                    chunks = [Document(**d) for d in json.loads(chunk_cache.read_text())]
                else:
                    chunks = DoclingLoader(str(path.resolve()), export_type=ExportType.DOC_CHUNKS,
                                           chunker=chunker, converter=converter).load()
                    if not chunks:
                        raise ValueError("Loader produced no chunks")
                    for chunk in chunks:
                        chunk.metadata["source"] = f"data/technology/{path.name}"
                        items = chunk.metadata.get("dl_meta", {}).get("doc_items", [])
                        chunk.metadata["pages"] = sorted({prov["page_no"] for item in items
                                                          for prov in item.get("prov", [])})
                    chunk_cache.write_text(json.dumps([
                        {"page_content": d.page_content, "metadata": d.metadata} for d in chunks
                    ], ensure_ascii=False))
                print(f"{path.name}: {len(chunks)} chunks", flush=True)
                documents.extend(chunks)
            except Exception as error:
                failures.append(path.name)
                print(f"FAILED {path.name}: {type(error).__name__}: {error}", flush=True)
        if failures:
            raise RuntimeError(f"Incomplete corpus; retry failed files: {', '.join(failures)}")
        return documents

    return Retriever.from_documents(load, cache_dir, key, model)


@lru_cache(maxsize=1)
def _default_retriever():
    return build_cisco_retriever()


def retrieve(question: str, k: int = 5, *, mode="hybrid", filter_family=True) -> list[dict]:
    return _default_retriever().retrieve(question, k, mode=mode, filter_family=filter_family)


if __name__ == "__main__":
    for question in ["Which Catalyst 8300 routers have two 10Gbps Ethernet ports?",
                     "Which Catalyst 8300 routers have no 10G ports?",
                     "I need a router with two high-speed WAN interfaces."]:
        print(f"\nQUESTION: {question}", flush=True)
        for mode in ["semantic", "hybrid"]:
            print(f"\nMODE: {mode} (family filter: {mode == 'hybrid'})")
            for rank, result in enumerate(retrieve(question, mode=mode,
                                                   filter_family=mode == "hybrid"), 1):
                meta = result["metadata"]
                print(f"\n{rank}. score={result['score']:.6f} source={meta.get('source')} "
                      f"pages={meta.get('pages')} tokens={meta.get('embedding_tokens')}")
                print(result["text"], flush=True)
