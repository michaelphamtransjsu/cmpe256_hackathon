"""Read-only semantic cache regressions without downloads or PDF ingestion."""

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from shared import hybrid_retrieval as hybrid
from shared.query_context import answering_only


@pytest.mark.parametrize("state", ["missing", "stale"])
def test_unprepared_query_fails_before_model_loading(tmp_path, monkeypatch, state):
    if state == "stale":
        (tmp_path / "index.json").write_text(json.dumps({"key": "old", "documents": []}))
        np.save(tmp_path / "embeddings.npy", np.zeros((0, 2)))
    monkeypatch.setattr(hybrid, "load_embedding_model", lambda: pytest.fail("Model must not load"))
    with answering_only(), pytest.raises(RuntimeError, match=f"cache is {state}"):
        hybrid.Retriever.from_documents(lambda: pytest.fail("Must not ingest"), tmp_path, "new")


def test_query_model_is_local_only_and_setup_errors_are_actionable(monkeypatch):
    calls = []

    def model(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=model))
    with answering_only():
        hybrid.load_embedding_model()
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["revision"] == hybrid.MODEL_REVISION
    with answering_only():
        hybrid.load_embedding_model(revision="explicit-prepared-revision")
    assert calls[1][1]["revision"] == "explicit-prepared-revision"
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ModuleNotFoundError, match="install '.\\[retrieval-runtime\\]'"):
        hybrid.load_embedding_model()


def test_missing_lexical_dependency_has_setup_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", None)
    with pytest.raises(ModuleNotFoundError, match="requires scikit-learn"):
        hybrid.Retriever([], np.zeros((0, 2)), object())


def test_missing_local_model_retains_original_cause(monkeypatch):
    def missing(*args, **kwargs):
        raise OSError("Model weights not cached")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=missing))
    with answering_only(), pytest.raises(RuntimeError, match="Prepare the model cache") as failure:
        hybrid.load_embedding_model()
    assert isinstance(failure.value.__cause__, OSError)


def test_prepared_cisco_query_needs_no_docling_and_does_not_rewrite_cache(tmp_path, monkeypatch):
    data, cache = tmp_path / "pdfs", tmp_path / "cache"
    data.mkdir()
    cache.mkdir()
    pdf = data / "cisco_8300.pdf"
    pdf.write_bytes(b"Fixture bytes used only to verify cache identity")
    key = json.dumps({"files": {pdf.name: hashlib.sha256(pdf.read_bytes()).hexdigest()},
                      "settings": hybrid.SETTINGS}, sort_keys=True)
    document = {"page_content": "C8300 fixture supports two 10Gbps ports.",
                "metadata": {"source": "data/technology/cisco_8300.pdf", "pages": [6]}}
    (cache / "index.json").write_text(json.dumps({"key": key, "documents": [document]}))
    np.save(cache / "embeddings.npy", np.array([[1., 0.]]))
    before = {p.name: p.read_bytes() for p in cache.iterdir()}

    class Model:
        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, question, **kwargs):
            assert isinstance(question, str), "Cached queries must not embed the corpus"
            return np.array([1., 0.])

    monkeypatch.setattr(hybrid, "load_embedding_model", lambda **kwargs: Model())
    for module in ("docling", "langchain_docling"):
        monkeypatch.setitem(sys.modules, module, None)
    with answering_only():
        index = hybrid.build_cisco_retriever(data, cache)
        results = index.retrieve("Which Catalyst 8300 has two 10Gbps ports?")
    assert results[0]["text"] == document["page_content"]
    assert results[0]["metadata"] == document["metadata"]
    assert before == {p.name: p.read_bytes() for p in cache.iterdir()}


def test_corrupt_cached_vector_shape_is_rejected(tmp_path):
    (tmp_path / "index.json").write_text(json.dumps({"key": "key", "documents": [
        {"page_content": "Evidence", "metadata": {}}]}))
    np.save(tmp_path / "embeddings.npy", np.zeros((2, 2)))
    model = SimpleNamespace(get_sentence_embedding_dimension=lambda: 2)
    with answering_only(), pytest.raises(ValueError, match="invalid shape"):
        hybrid.Retriever.from_documents(lambda: pytest.fail("Must not ingest"), tmp_path, "key", model)


def test_standalone_retrieval_helper_is_cache_only(monkeypatch):
    from shared.query_context import cache_only

    def build():
        assert cache_only.get(), "The public query helper must never rebuild a corpus"
        return SimpleNamespace(retrieve=lambda *args, **kwargs: [])

    hybrid._default_retriever.cache_clear()
    monkeypatch.setattr(hybrid, "build_cisco_retriever", build)
    try:
        assert hybrid.retrieve("Which Catalyst 8300 has two 10Gbps ports?") == []
        assert not cache_only.get()
    finally:
        hybrid._default_retriever.cache_clear()
