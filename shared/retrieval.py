"""Isolated, durable LangChain vector indexes. No pickle or model downloads."""

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .model import AgentName, Settings

STOP_WORDS = set("a an and are as at be by can do for from how in is it of on or that the this to was were what which with".split())


def meaningful_words(text: str) -> set[str]:
    return set(re.findall(r"\b[\w-]+\b", text.casefold())) - STOP_WORDS


def _lexical_terms(text: str) -> list[str]:
    words = re.findall(r"\b\w+\b", text.casefold())
    return [word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word
            for word in words if word not in STOP_WORDS]


def _bm25_rank(matches: list[tuple[Document, float]], question: str, min_score: float) -> list[Document]:
    """Rerank lexical vector candidates using discriminative corpus terms."""
    terms = set(_lexical_terms(question))
    counts = [Counter(_lexical_terms(doc.page_content)) for doc, _ in matches]
    if not counts or not terms:
        return []
    lengths = [sum(count.values()) for count in counts]
    average_length = sum(lengths) / len(lengths) or 1
    frequency = {term: sum(term in count for count in counts) for term in terms}
    ranked = []
    for (doc, cosine), count, length in zip(matches, counts, lengths, strict=True):
        if cosine < min_score:
            continue
        score = 0.0
        for term in terms & count.keys():
            idf = math.log(1 + (len(counts) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            tf = count[term]
            score += idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * length / average_length))
        if score > 0:
            ranked.append((score, cosine, doc))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [doc for _, _, doc in ranked]


class HashEmbeddings(Embeddings):
    """Normalized hashed word counts for reproducible, offline lexical retrieval.

    This is not a semantic language model. Stable hashes also make persisted
    vectors usable across Python processes (unlike Python's randomized hash).
    """

    dimensions = 2048

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in re.findall(r"\b[\w-]+\b", text.casefold()):
            if word in STOP_WORDS:
                continue
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def make_embeddings(settings: Settings) -> Embeddings:
    if settings.embeddings_provider == "hash":
        return HashEmbeddings()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embedding_model, max_retries=2, request_timeout=60)


class SpecialistIndex:
    """One atomic JSON snapshot per specialist, including embedding identity.

    InMemoryVectorStore supplies LangChain retrieval; its vectors and metadata
    are persisted together and loaded on restart. Rebuild replaces a whole
    specialist corpus, so removed documents cannot linger in its index.
    Intended for a local single-writer workflow, not concurrent ingestion.
    """

    def __init__(self, agent: AgentName, settings: Settings):
        if agent not in ("clinical", "corporate_responsibility"):
            raise ValueError("Unknown specialist")
        self.agent = agent
        self.settings = settings
        self.path = settings.data_dir / "indexes" / agent / "vectors.json"
        self.identity = {
            "format": 1,
            "agent": agent,
            "provider": settings.embeddings_provider,
            "model": "hash-word-v1-2048" if settings.embeddings_provider == "hash" else settings.embedding_model,
        }
        snapshot = None
        if self.path.exists():
            snapshot = json.loads(self.path.read_text(encoding="utf-8"))
            if snapshot.get("identity") != self.identity:
                raise ValueError("Index embedding configuration changed; restore the original settings or re-ingest using a new data directory")
        self.store = InMemoryVectorStore(make_embeddings(settings))
        if snapshot is not None:
            self.store.store = snapshot["vectors"]
            if any(row["metadata"].get("agent") != agent for row in self.store.store.values()):
                raise ValueError("Index contains a different specialist's sources")

    def replace(self, documents: list[Document], *, allow_empty: bool = True) -> int:
        if any(doc.metadata.get("agent") != self.agent for doc in documents):
            raise ValueError("Refusing to index sources assigned to a different specialist")
        for doc in documents:
            source_id = doc.metadata.get("source_id")
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError("Every document requires a nonempty source_id for citations")
            page = doc.metadata.get("page")
            if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
                raise ValueError("Document page numbers must be positive integers")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=160, add_start_index=True)
        chunks = splitter.split_documents([doc for doc in documents if doc.page_content.strip()])
        if self.settings.embeddings_provider == "hash":
            chunks = [doc for doc in chunks if meaningful_words(doc.page_content)]
        if not chunks and not allow_empty:
            raise ValueError("No indexable document chunks; index preserved. An empty corpus requires --allow-empty")
        fresh = InMemoryVectorStore(self.store.embedding)
        if chunks:
            ids = [hashlib.sha256((str(doc.metadata.get("source_id")) + ":" + str(doc.metadata.get("page")) + ":" + str(doc.metadata.get("start_index")) + ":" + doc.page_content).encode()).hexdigest() for doc in chunks]
            fresh.add_documents(chunks, ids=ids)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=".vectors-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"identity": self.identity, "vectors": fresh.store}, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            if self.path.exists():
                # Retain the last corpus even when a successful, smaller
                # manifest intentionally replaces it. Backups are content keyed.
                previous = self.path.read_bytes()
                directory = self.path.parent / "backups"
                directory.mkdir(exist_ok=True)
                backup = directory / f"vectors-{hashlib.sha256(previous).hexdigest()}.json"
                try:
                    with backup.open("xb") as stream:
                        stream.write(previous)
                        stream.flush()
                        os.fsync(stream.fileno())
                except FileExistsError:
                    if backup.read_bytes() != previous:
                        raise ValueError("Index backup differs; existing index preserved") from None
            os.replace(temp_name, self.path)
        finally:
            if Path(temp_name).exists():
                Path(temp_name).unlink()
        self.store = fresh
        return len(fresh.store)

    def search(self, question: str, limit: int | None = None, *, report_year: int | None = None) -> list[Document]:
        if not self.store.store:
            return []
        k = limit or self.settings.top_k
        # This store exposes cosine scores directly, but does not implement
        # LangChain's relevance-score normalization used by threshold retrievers.
        embedding = self.store.embedding.embed_query(question)
        if not any(embedding):
            return []
        lexical = self.settings.embeddings_provider == "hash"
        matches = self.store.similarity_search_with_score_by_vector(
            embedding, k=len(self.store.store) if lexical else k,
            filter=lambda doc: doc.metadata.get("agent") == self.agent
                and (report_year is None or doc.metadata.get("report_year") == report_year),
        )
        if lexical:
            return _bm25_rank(matches, question, self.settings.min_score)[:k]
        return [doc for doc, score in matches if score >= self.settings.min_score][:k]
