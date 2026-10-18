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


def test_ingestion_scope_covers_all_five_requested_filings():
    assert {(f["company"], f["fiscal_year"], f["filing_type"]) for f in FILINGS} == {
        ("Cisco", 2025, "10-K"), ("Agilent", 2025, "10-K"),
        ("Cisco", 2026, "10-K"), ("Alphabet", 2024, "10-K"), ("Alphabet", 2025, "10-K")}


def test_short_signature_fragments_are_not_ranked(sec):
    sec.index.documents[0].page_content = "COMPANY NAME By:"
    sec.index.documents[0].metadata["table_refs"] = []
    assert sec.retrieve("Revenue", company="Cisco", fiscal_year=2025) == []


def test_incremental_embedding_reuse_only_encodes_new_text():
    from shared.sec_retrieval import _ReuseEmbeddings
    class Model:
        def encode(self, texts, **kwargs):
            assert texts == ["new"]
            return np.array([[0., 1.]])
    model = _ReuseEmbeddings(Model(), [{"page_content": "old"}], np.array([[1., 0.]]))
    vectors = model.encode(["old", "new", "old"], normalize_embeddings=True)
    assert np.array_equal(vectors, [[1., 0.], [0., 1.], [1., 0.]])


def test_multiple_company_and_year_filters(sec):
    rows = sec.retrieve("Revenue", company=["Cisco", "Agilent"], fiscal_year=[2024, 2025])
    assert len(rows) == 3
    assert len(sec.retrieve("Revenue", company=["Cisco"], fiscal_year=[2024, 2025])) == 2
    assert sec.retrieve("Revenue", company=["Microsoft"], fiscal_year=[2025]) == []


def test_duplicate_text_keeps_each_original_vector_exactly():
    from shared.sec_retrieval import _ReuseEmbeddings
    class Model:
        def encode(self, *args, **kwargs):
            pytest.fail("Existing vectors must not be regenerated")
    previous = [{"page_content": "duplicate"}, {"page_content": "duplicate"}]
    vectors = np.array([[1., 0.], [1., 0.00000001]])
    model = _ReuseEmbeddings(Model(), previous, vectors)
    assert np.array_equal(model.encode(["duplicate", "duplicate"]), vectors)
