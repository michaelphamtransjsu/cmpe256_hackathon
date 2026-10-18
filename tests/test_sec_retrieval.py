"""SEC adapter tests use synthetic documents, never production filings or downloads."""

import numpy as np
import pytest
from langchain_core.documents import Document

from shared.hybrid_retrieval import Retriever
from shared.sec_retrieval import FILINGS, SECRetriever


@pytest.fixture
def sec():
    class Model:
        def encode(self, question, **kwargs):
            return np.array([1., 0.])

    docs = [Document(page_content=text, metadata={"company": company, "fiscal_year": year,
        "source_id": str(i), "source": f"fixture:{i}", "filing_type": "10-K",
        "filename": f"fixture-{i}.pdf", "pages": [8], "page": 8, "table_refs": ["#/tables/0"]})
        for i, (company, year, text) in enumerate([
            ("Cisco", 2025, "Revenue for current reporting period."),
            ("Cisco", 2024, "Revenue for prior reporting period."),
            ("Agilent", 2025, "Revenue for current reporting period."),
        ])]
    contexts = {str(i): {"tables": {"#/tables/0": {"markdown": "Year | Revenue\n2025 | 12",
        "captions": [], "footnotes": ["Synthetic units: millions"]}},
        "pages": {"8": "Synthetic table, dollars in millions; year ended 2025."}} for i in range(3)}
    return SECRetriever(Retriever(docs, np.array([[0.8, 0.6], [1., 0.], [1., 0.]]), Model()), contexts)


def test_filters_apply_before_top_k_and_preserve_context(sec):
    result = sec.retrieve("Revenue", k=1, company="CISCO", fiscal_year=2025)[0]
    assert result["metadata"]["source_id"] == "0"
    assert result["metadata"]["pages"] == [8]
    assert "millions" in result["metadata"]["page_context"]["8"]
    assert result["metadata"]["table_context"]["#/tables/0"]["footnotes"]
    assert "page_context" not in sec.index.documents[0].metadata


def test_combined_company_and_year_filters_and_empty_results(sec):
    assert len(sec.retrieve("Revenue", fiscal_year=2025)) == 2
    assert len(sec.retrieve("Revenue", company="Cisco")) == 2
    assert sec.retrieve("Revenue", company="Alphabet") == []
    assert sec.retrieve("Revenue", fiscal_year=2026) == []
    with pytest.raises(ValueError):
        sec.retrieve("Revenue", k=0)


def test_cached_views_reuse_existing_embeddings(sec):
    sec.retrieve("Revenue", company="Cisco", fiscal_year=2025)
    view = sec.views[("cisco", 2025)]
    sec.retrieve("Risk", company="Cisco", fiscal_year=2025)
    assert sec.views[("cisco", 2025)] is view
    assert view.model is sec.index.model


def test_ingestion_scope_is_only_the_two_requested_filings():
    assert {(f["company"], f["fiscal_year"], f["filing_type"]) for f in FILINGS} == {
        ("Cisco", 2025, "10-K"), ("Agilent", 2025, "10-K")}


def test_short_signature_fragments_are_not_ranked(sec):
    sec.index.documents[0].page_content = "COMPANY NAME By:"
    sec.index.documents[0].metadata["table_refs"] = []
    assert sec.retrieve("Revenue", company="Cisco", fiscal_year=2025) == []


def _prepared_sec_fixture(tmp_path):
    import hashlib
    import json
    from shared.sec_retrieval import CONFIG

    data_dir, cache_dir = tmp_path / "filings", tmp_path / "sec-cache"
    cache_dir.mkdir()
    files, documents = {}, []
    for filing in FILINGS:
        relative = filing["company"] + "/" + filing["filename"]
        path = data_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Synthetic source; parsing this fixture is forbidden.")
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        source_id = filing["company"].lower() + "-2025-10-K"
        documents.append({"page_content": "Synthetic risk disclosure and reporting context.",
                          "metadata": {**filing, "source_id": source_id, "table_refs": [],
                                       "pages": [1], "page": 1, "source": "fixture:" + source_id}})
        (cache_dir / (source_id + "-context.json")).write_text(json.dumps(
            {"tables": {}, "pages": {"1": "Synthetic page."}}))
    key = json.dumps({"files": files, "config": CONFIG, "filings": FILINGS}, sort_keys=True)
    (cache_dir / "index.json").write_text(json.dumps({"key": key, "documents": documents}))
    np.save(cache_dir / "embeddings.npy", np.array([[1., 0.], [0., 1.]]), allow_pickle=False)
    return data_dir, cache_dir


def test_prepared_sec_cache_does_not_import_ingestion_dependencies(tmp_path, monkeypatch):
    import builtins
    from shared import sec_retrieval
    from shared.query_context import answering_only

    data_dir, cache_dir = _prepared_sec_fixture(tmp_path)

    class Model:
        def get_sentence_embedding_dimension(self):
            return 2

    monkeypatch.setattr(sec_retrieval, "load_embedding_model", lambda **kwargs: Model())
    original_import = builtins.__import__

    def forbid_ingestion(name, *args, **kwargs):
        if name.startswith(("docling", "docling_core", "pypdfium2")):
            raise AssertionError("Prepared SEC query imported ingestion dependency: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_ingestion)
    before = {p.name: p.read_bytes() for p in cache_dir.iterdir()}
    with answering_only():
        retriever = sec_retrieval.build_sec_retriever(data_dir, cache_dir)
    assert len(retriever.index.documents) == 2
    assert set(retriever.contexts) == {"cisco-2025-10-K", "agilent-2025-10-K"}
    assert {p.name: p.read_bytes() for p in cache_dir.iterdir()} == before


def test_missing_sec_cache_fails_before_model_or_ingestion(tmp_path, monkeypatch):
    from shared import sec_retrieval
    from shared.query_context import answering_only

    data_dir, cache_dir = _prepared_sec_fixture(tmp_path)
    (cache_dir / "index.json").unlink()

    def forbidden_model():
        pytest.fail("Missing SEC cache must be rejected before model initialization")

    monkeypatch.setattr(sec_retrieval, "load_embedding_model", forbidden_model)
    with answering_only(), pytest.raises(RuntimeError, match="cache"):
        sec_retrieval.build_sec_retriever(data_dir, cache_dir)
    assert not (cache_dir / "index.json").exists()


def test_independent_sec_query_enables_cache_only(monkeypatch):
    from shared import sec_retrieval
    from shared.query_context import cache_only

    marker = object()
    def build():
        assert cache_only.get() is True
        return marker

    sec_retrieval._default.cache_clear()
    monkeypatch.setattr(sec_retrieval, "build_sec_retriever", build)
    try:
        assert sec_retrieval._default() is marker
        assert cache_only.get() is False
    finally:
        sec_retrieval._default.cache_clear()
