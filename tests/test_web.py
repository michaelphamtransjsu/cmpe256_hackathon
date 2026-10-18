"""Exercise the HTTP boundary and real manager orchestration without providers."""

import pytest
from fastapi.testclient import TestClient

import manager
from backend.app import app
from shared.schemas import Analysis, SpecialistResponse

client = TestClient(app)


class Model:
    """Structured model fixture; the real ManagerAgent still routes/synthesizes."""

    def __init__(self, selected):
        self.selected = selected

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        return self

    def invoke(self, messages):
        if self.schema is not Analysis:
            return {"agents": self.selected, "limitations": []}
        return {
            "answer": "Two ports [technology:E1].",
            "findings": [{"text": "Two ports", "evidence_ids": ["technology:E1"]}],
            "recommendations": [{"action": "Review ports", "rationale": "Check fit", "evidence_ids": ["technology:E1"]}],
            "limitations": [],
        }


def test_http_uses_public_manager_entry_point_and_preserves_contract(monkeypatch, tmp_path):
    calls = []
    index = tmp_path / 'vectors.json'
    index.write_text('existing index')

    class Specialist:
        def invoke(self, request):
            calls.append(request.question)
            return SpecialistResponse(
                agent="technology", question=request.question, mode="extractive",
                status="evidence_found", answer="Two ports [E1].",
                findings=[{"text": "Two ports", "evidence_ids": ["E1"]}],
                recommendations=[], limitations=["Verify configuration"],
                citations=[{"id": "E1", "source_id": "router", "source": "/private/router.pdf",
                            "page": 2, "excerpt": "Two 10Gbps ports."}],
            )

    # Use the public function and real ManagerAgent, injecting only model/registry.
    monkeypatch.setattr(manager, "make_chat_model", lambda settings: Model(["technology"]))
    monkeypatch.setattr(manager, "AGENT_REGISTRY", {"technology": lambda settings: Specialist()})
    monkeypatch.setenv("SPECIALIST_DATA_DIR", str(tmp_path))
    import shared.sources
    monkeypatch.setattr(shared.sources, "load_documents", lambda *a, **k: pytest.fail("Chat must not ingest"))
    before = index.stat().st_mtime_ns
    response = client.post("/api/chat", json={"question": "  Ports?  "})
    assert response.status_code == 200
    result = manager.ManagerResponse.model_validate(response.json())
    assert result.question == "Ports?" and calls == ["Ports?"]
    assert result.answer == "Two ports [technology:E1]."
    assert result.agents_used == ["technology"]
    assert result.citation_labels == {"technology:E1": "router.pdf · p. 2"}
    assert result.citations[0].excerpt == "Two 10Gbps ports."
    assert result.citations[0].source == "/private/router.pdf"
    assert result.specialist_responses[0].citations[0].id == "E1"
    assert result.recommendations[0].evidence_ids == ["technology:E1"]
    assert "technology: Verify configuration" in result.limitations
    assert index.read_text() == 'existing index' and index.stat().st_mtime_ns == before


@pytest.mark.parametrize("payload", [
    {}, {"question": " "}, {"question": "x" * 8001}, {"question": 42},
    {"question": "Ports?", "history": []}, {"question": "Ports?", "agent": "technology"},
])
def test_invalid_requests_never_call_manager(monkeypatch, payload):
    monkeypatch.setattr(manager, "consult", lambda question: pytest.fail("Invalid input reached manager"))
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


def test_provider_failure_returns_safe_retryable_error(monkeypatch, caplog):
    def fail(question):
        raise RuntimeError("private-token sensitive-question provider-detail")

    monkeypatch.setattr(manager, "consult", fail)
    response = client.post("/api/chat", json={"question": "Question?"})
    assert response.status_code == 503
    assert "Try again" in response.json()["detail"]
    assert "Traceback" in caplog.text and " in fail" in caplog.text
    for secret in ["private-token", "sensitive-question", "provider-detail"]:
        assert secret not in response.text + caplog.text


def test_no_evidence_response_is_successful_and_limitations_survive(monkeypatch):
    monkeypatch.setattr(manager, "make_chat_model", lambda settings: Model([]))
    response = client.post("/api/chat", json={"question": "SEC filings?"})
    assert response.status_code == 200
    assert response.json()["citations"] == response.json()["agents_used"] == []
    assert response.json()["limitations"] == ["No registered specialist covers this question."]


def test_health_does_not_consult_or_construct_models(monkeypatch):
    monkeypatch.setattr(manager, "consult", lambda question: pytest.fail("Liveness called manager"))
    monkeypatch.setattr(manager, "make_chat_model", lambda settings: pytest.fail("Liveness loaded model"))
    assert client.get("/api/health").json() == {"status": "ok"}


def test_chat_sets_and_resets_request_local_cache_policy(monkeypatch):
    from shared.query_context import cache_only

    observed = []

    def consult(question):
        observed.append(cache_only.get())
        return manager.ManagerResponse(question=question, answer="No evidence",
            findings=[], recommendations=[], limitations=[], citations=[], agents_used=[])

    monkeypatch.setattr(manager, "consult", consult)
    assert cache_only.get() is False
    assert client.post("/api/chat", json={"question": "Question?"}).status_code == 200
    assert observed == [True]
    assert cache_only.get() is False


def test_cache_only_policy_blocks_ingestion_and_reuses_valid_cache(tmp_path):
    import json
    import numpy as np
    from shared.hybrid_retrieval import Retriever
    from shared.query_context import answering_only, cache_only

    class Embeddings:
        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, *args, **kwargs):
            pytest.fail("HTTP request must not embed a corpus")

    def documents():
        pytest.fail("HTTP request must not load or convert source documents")

    # Both absent and stale caches refuse rebuilding before document loading.
    with answering_only():
        with pytest.raises(RuntimeError, match="prepare indexes"):
            Retriever.from_documents(documents, tmp_path, "key", Embeddings())
        (tmp_path / "index.json").write_text(json.dumps({"key": "stale"}))
        np.save(tmp_path / "embeddings.npy", np.array([[1., 0.]]))
        with pytest.raises(RuntimeError, match="prepare indexes"):
            Retriever.from_documents(documents, tmp_path, "key", Embeddings())
        manifest = {"key": "key", "documents": [{"page_content": "Documented router ports", "metadata": {}}]}
        (tmp_path / "index.json").write_text(json.dumps(manifest))
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        result = Retriever.from_documents(documents, tmp_path, "key", Embeddings())
        assert len(result.documents) == 1
        assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert cache_only.get() is False


@pytest.mark.parametrize("selected", [
    ["technology", "sec"], ["clinical", "corporate_responsibility"],
])
def test_cross_domain_endpoint_preserves_namespaced_evidence(monkeypatch, selected):
    import json

    calls = []

    class CrossDomainModel(Model):
        def invoke(self, messages):
            if self.schema is not Analysis:
                return {"agents": selected, "limitations": []}
            payload = json.loads(messages[1].content)
            ids = [c["id"] for item in payload["specialists"] for c in item["citations"]]
            return {
                "answer": "Documented facts " + " ".join(f"[{identifier}]" for identifier in ids),
                "findings": [{"text": "Documented facts", "evidence_ids": ids}],
                "recommendations": [], "limitations": [],
            }

    def factory(name):
        class Specialist:
            def invoke(self, request):
                calls.append(name)
                return SpecialistResponse(
                    agent=name, question=request.question, mode="extractive",
                    status="evidence_found", answer=f"{name} fact [E1].",
                    findings=[{"text": f"{name} fact", "evidence_ids": ["E1"]}],
                    recommendations=[], limitations=[],
                    citations=[{"id": "E1", "source_id": name, "source": f"/documents/{name}.pdf",
                                "page": 1, "excerpt": f"Original {name} evidence."}],
                )
        return lambda settings: Specialist()

    instance = manager.ManagerAgent(model=CrossDomainModel(selected),
        registry={name: factory(name) for name in selected})
    monkeypatch.setattr(manager, "consult", instance.consult)
    response = client.post("/api/chat", json={"question": "Compare the evidence across both domains."})
    assert response.status_code == 200
    result = response.json()
    expected = [f"{name}:E1" for name in selected]
    assert calls == result["agents_used"] == selected
    assert [c["id"] for c in result["citations"]] == expected
    assert result["findings"][0]["evidence_ids"] == expected
    assert all(f"[{identifier}]" in result["answer"] for identifier in expected)
    assert [r["citations"][0]["id"] for r in result["specialist_responses"]] == ["E1", "E1"]


def test_clinical_dates_serialize_through_http(monkeypatch):
    from datetime import date
    from shared.trials import TrialRecord

    record = TrialRecord(trial_id="SYNTHETIC-001", title="Fixture trial",
                         source="fixture:trial", updated_at=date(2026, 10, 1))
    specialist = SpecialistResponse(
        agent="clinical", question="Clinical trial?", mode="extractive", status="evidence_found",
        answer="Fixture registration [E1].",
        findings=[{"text": "Fixture registration", "evidence_ids": ["E1"]}],
        recommendations=[], limitations=[], trial_records=[record],
        citations=[{"id": "E1", "source_id": "trial", "source": "fixture:trial",
                    "excerpt": "Fixture registration."}],
    )
    result = manager.ManagerResponse(
        question="Clinical trial?", answer="Fixture registration [clinical:E1].",
        findings=[{"text": "Fixture registration", "evidence_ids": ["clinical:E1"]}],
        recommendations=[], limitations=[], agents_used=["clinical"],
        citations=[specialist.citations[0].model_copy(update={"id": "clinical:E1"})],
        specialist_responses=[specialist],
    )
    monkeypatch.setattr(manager, "consult", lambda question: result)
    response = client.post("/api/chat", json={"question": "Clinical trial?"})
    assert response.status_code == 200
    assert response.json()["specialist_responses"][0]["trial_records"][0]["updated_at"] == "2026-10-01"


def test_partial_specialist_failure_is_visible_without_losing_other_evidence(monkeypatch, caplog):
    class Technology:
        def invoke(self, request):
            return SpecialistResponse(
                agent="technology", question=request.question, mode="extractive", status="evidence_found",
                answer="Two ports [E1].", findings=[{"text": "Two ports", "evidence_ids": ["E1"]}],
                recommendations=[], limitations=[],
                citations=[{"id": "E1", "source_id": "router", "source": "fixture:router",
                            "excerpt": "Two ports."}],
            )

    def unavailable(settings):
        raise ValueError("private-provider-token")

    instance = manager.ManagerAgent(model=Model(["technology", "sec"]), registry={
        "technology": lambda settings: Technology(), "sec": unavailable,
    })
    monkeypatch.setattr(manager, "consult", instance.consult)
    response = client.post("/api/chat", json={"question": "Compare router ports and SEC risks."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["agents_used"] == ["technology"]
    assert payload["citations"][0]["id"] == "technology:E1"
    assert "No combined cross-domain conclusion" in payload["answer"]
    assert any("sec: specialist failed (ValueError)" in item for item in payload["limitations"])
    assert "private-provider-token" not in response.text + caplog.text
