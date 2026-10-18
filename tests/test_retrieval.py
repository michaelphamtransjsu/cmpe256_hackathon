"""Retrieval tests use synthetic fixtures and never require source archives or APIs."""

import json

import pytest
from langchain_core.documents import Document

from shared import retrieval
from shared.model import Settings
from shared.retrieval import HashEmbeddings, SpecialistIndex


def document(agent, source_id, text, page=1):
    return Document(
        page_content=text,
        metadata={"agent": agent, "source_id": source_id, "source": f"fixture:{source_id}", "page": page},
    )


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, min_score=0.15)


def test_specialist_indexes_are_isolated_and_persist_across_instances(settings):
    clinical = SpecialistIndex("clinical", settings)
    corporate = SpecialistIndex("corporate_responsibility", settings)
    clinical.replace([document("clinical", "clinical-fixture", "Synthetic nutrition laboratory study.", page=2)])
    corporate.replace([document("corporate_responsibility", "sofi-fixture", "Synthetic nutrition food affordability analysis.", page=3)])

    for agent, expected_id, expected_page in [
        ("clinical", "clinical-fixture", 2),
        ("corporate_responsibility", "sofi-fixture", 3),
    ]:
        reopened = SpecialistIndex(agent, settings)
        matches = reopened.search("nutrition")
        assert len(matches) == 1
        assert matches[0].metadata["source_id"] == expected_id
        assert matches[0].metadata["page"] == expected_page
        assert matches[0].metadata["agent"] == agent
    assert clinical.path != corporate.path


def test_replace_is_idempotent_and_removes_old_documents(settings):
    index = SpecialistIndex("clinical", settings)
    old = document("clinical", "old", "Obsolete electrophoresis evidence.")
    current = document("clinical", "current", "Current spectroscopy evidence.")
    assert index.replace([old]) == 1
    assert index.search("electrophoresis")
    assert index.replace([current]) == 1
    first_snapshot = json.loads(index.path.read_text())
    assert index.replace([current]) == 1
    assert json.loads(index.path.read_text()) == first_snapshot

    reopened = SpecialistIndex("clinical", settings)
    assert reopened.search("electrophoresis") == []
    assert [doc.metadata["source_id"] for doc in reopened.search("spectroscopy")] == ["current"]


def test_empty_replace_persists_an_empty_corpus(settings):
    index = SpecialistIndex("clinical", settings)
    index.replace([document("clinical", "old", "Spectroscopy evidence.")])
    assert index.replace([document("clinical", "blank", " \n\t")]) == 0
    assert SpecialistIndex("clinical", settings).search("spectroscopy") == []


def test_foreign_documents_are_rejected_without_replacing_valid_corpus(settings):
    index = SpecialistIndex("clinical", settings)
    index.replace([document("clinical", "accepted", "Spectroscopy evidence.")])
    for foreign_agent in ("corporate_responsibility", "technology", "sec", "retail"):
        with pytest.raises(ValueError, match="different specialist"):
            index.replace([document(foreign_agent, "foreign", "Spectroscopy foreign evidence.")])
    assert [doc.metadata["source_id"] for doc in index.search("spectroscopy")] == ["accepted"]
    with pytest.raises(ValueError, match="Unknown specialist"):
        SpecialistIndex("technology", settings)


@pytest.mark.parametrize("metadata", [
    {"agent": "clinical"},
    {"agent": "clinical", "source_id": None},
    {"agent": "clinical", "source_id": " "},
    {"agent": "clinical", "source_id": 12},
    {"agent": "clinical", "source_id": "fixture", "page": 0},
    {"agent": "clinical", "source_id": "fixture", "page": -1},
    {"agent": "clinical", "source_id": "fixture", "page": True},
    {"agent": "clinical", "source_id": "fixture", "page": "1"},
    {"agent": "clinical", "source_id": "fixture", "page": 1.5},
])
def test_invalid_citation_metadata_cannot_replace_a_valid_index(settings, metadata):
    index = SpecialistIndex("clinical", settings)
    index.replace([document("clinical", "accepted", "Spectroscopy evidence.")])
    with pytest.raises(ValueError, match="source_id|page"):
        index.replace([Document(page_content="Nutrition evidence.", metadata=metadata)])
    assert [doc.metadata["source_id"] for doc in SpecialistIndex("clinical", settings).search("spectroscopy")] == ["accepted"]


def test_search_rejects_unrelated_queries_and_respects_limit(settings):
    index = SpecialistIndex("clinical", settings)
    assert index.search("spectroscopy") == []
    index.replace([
        document("clinical", f"fixture-{number}", f"Spectroscopy laboratory finding {number}.")
        for number in range(4)
    ])
    assert len(index.search("spectroscopy", limit=2)) == 2
    assert index.search("pumpernickel") == []
    assert index.search("the and of") == []


def test_stopword_only_corpus_does_not_crash_on_nonzero_query(settings):
    index = SpecialistIndex("clinical", settings)
    index.replace([document("clinical", "stopwords", "the and of")])
    assert index.search("nutrition") == []


def test_hash_collision_does_not_turn_unrelated_text_into_evidence(settings):
    collision = "zzunrelated4035"
    assert HashEmbeddings().embed_query(collision) == HashEmbeddings().embed_query("nutrition")
    index = SpecialistIndex("clinical", settings)
    index.replace([
        document("clinical", "unrelated-collision", collision),
        document("clinical", "relevant", "Synthetic nutrition evidence."),
    ])
    # The collision scores 1.0, above the relevant document: it must not consume
    # the limit and hide the actual evidence after relevance filtering.
    assert [doc.metadata["source_id"] for doc in index.search("nutrition", limit=1)] == ["relevant"]
    assert [doc.metadata["source_id"] for doc in index.search(collision, limit=1)] == ["unrelated-collision"]


def test_embedding_configuration_mismatch_cannot_silently_reuse_vectors(settings, monkeypatch):
    index = SpecialistIndex("clinical", settings)
    index.replace([document("clinical", "fixture", "Spectroscopy evidence.")])
    changed = settings.model_copy(update={"embeddings_provider": "openai", "embedding_model": "different-model"})
    # Detect configuration drift before constructing an API client or making a request.
    with monkeypatch.context() as patch:
        patch.setattr(retrieval, "make_embeddings", lambda _: pytest.fail("Do not construct embeddings for an incompatible index"))
        with pytest.raises(ValueError, match="embedding configuration changed"):
            SpecialistIndex("clinical", changed)
    assert SpecialistIndex("clinical", settings).search("spectroscopy")


def test_persisted_foreign_agent_metadata_is_rejected(settings):
    index = SpecialistIndex("clinical", settings)
    index.replace([document("clinical", "fixture", "Spectroscopy evidence.")])
    snapshot = json.loads(index.path.read_text())
    next(iter(snapshot["vectors"].values()))["metadata"]["agent"] = "corporate_responsibility"
    index.path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="different specialist"):
        SpecialistIndex("clinical", settings)


def test_hash_embeddings_are_deterministic_normalized_and_do_not_need_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    first, second = HashEmbeddings(), HashEmbeddings()
    assert first.embed_query("SPECTROSCOPY and laboratory") == second.embed_query("spectroscopy laboratory")
    vector = first.embed_query("spectroscopy laboratory")
    assert sum(value * value for value in vector) == pytest.approx(1)
    assert not any(first.embed_query("the and of"))
