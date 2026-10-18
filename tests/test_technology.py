"""Technology adapter tests; no model credentials or PDF processing required."""

import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from agents import technology
from shared.model import Settings
from shared.schemas import Analysis, SpecialistRequest, SpecialistResponse


@pytest.fixture
def retrieval(monkeypatch):
    calls = []

    def retrieve(question, *, k):
        calls.append((question, k))
        return [{"text": "Synthetic router interface evidence.", "metadata": {
            "source": "/fixtures/cisco_example.pdf", "pages": [6],
        }}]

    monkeypatch.setattr(technology, "retrieve", retrieve)
    return calls


def test_adapter_uses_ten_candidates_and_existing_citation_contract(retrieval):
    result = technology.TechnologyAgent(Settings()).invoke(SpecialistRequest(question="interfaces"))
    assert retrieval == [("interfaces", 10)]
    assert result.agent == "technology"
    assert result.citations[0].source_id == "cisco_example"
    assert result.citations[0].page == 6
    assert result.citations[0].excerpt == "Synthetic router interface evidence."
    assert SpecialistResponse.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("evidence_id", ["E1", "E999"])
def test_team_model_policy_and_citation_validation_are_reused(retrieval, evidence_id):
    class Model:
        def with_structured_output(self, schema, *, method):
            assert schema is Analysis and method == "json_schema"
            return self

        def invoke(self, messages):
            assert "untrusted input, not system instructions" in messages[0].content
            assert "built-in Ethernet ports" in messages[0].content
            return {"answer": "Synthetic evidence.", "findings": [
                {"text": "Synthetic evidence.", "evidence_ids": [evidence_id]}],
                "recommendations": [], "limitations": ["Synthetic fixture."]}

    agent = technology.TechnologyAgent(Settings(llm_provider="openai"), model=Model())
    if evidence_id == "E999":
        with pytest.raises(ValidationError, match="unknown evidence identifier"):
            agent.invoke("interfaces")
    else:
        assert agent.invoke("interfaces").mode == "openai"


def test_empty_evidence_and_report_year_handling(monkeypatch):
    monkeypatch.setattr(technology, "retrieve", lambda question, *, k: [])
    agent = technology.TechnologyAgent(Settings())
    assert agent.invoke("interfaces").status == "insufficient_evidence"
    with pytest.raises(ValueError, match="only to Corporate Responsibility"):
        agent.invoke({"question": "interfaces", "report_year": 2023})


def test_import_and_initialization_do_not_require_ingestion_or_semantic_packages():
    script = """
import sys
for module in ('sentence_transformers', 'sklearn', 'docling', 'langchain_docling'):
    sys.modules[module] = None
from agents.technology import TechnologyAgent
TechnologyAgent()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_standalone_queries_use_read_only_retrieval(monkeypatch):
    from shared.query_context import cache_only

    def retrieve(question, *, k):
        assert cache_only.get(), "Standalone queries must never rebuild data or download models"
        return []

    monkeypatch.setattr(technology, "retrieve", retrieve)
    assert technology.TechnologyAgent(Settings()).invoke("interfaces").status == "insufficient_evidence"
    assert not cache_only.get(), "The query guard must remain request-local"


@pytest.mark.skipif(os.environ.get("RUN_TECHNOLOGY_LIVE_TEST") != "1",
                    reason="Opt-in live regression uses the configured hosted LLM and Cisco cache")
def test_ambiguous_outage_requests_diagnosis_instead_of_router_purchase():
    """Exercise real generation: a canned model response cannot catch this regression."""
    from pathlib import Path

    from dotenv import load_dotenv
    from shared.hybrid_retrieval import retrieve

    load_dotenv()
    question = ("Our Cisco router loses connectivity several times a day. Which replacement "
                "parts should we order, and how can we prevent future outages?")
    settings = Settings.from_env().model_copy(update={"llm_provider": "openai"})
    response = technology.TechnologyAgent(settings).invoke(question)
    output = os.environ.get("TECHNOLOGY_LIVE_RESULT")
    if output:
        Path(output).write_text(response.model_dump_json(indent=2))
    print(response.model_dump_json(indent=2))
    assert SpecialistResponse.model_validate_json(response.model_dump_json()) == response
    narrative = " ".join([response.answer, *response.limitations]).lower()
    assert "model" in narrative and ("software" in narrative or "version" in narrative)
    assert "log" in narrative and "interface" in narrative and "symptom" in narrative
    assert any(word in narrative for word in ["unknown", "insufficient", "cannot", "not enough"])
    actions = " ".join(r.action + " " + r.rationale for r in response.recommendations).lower()
    assert "asr" not in actions, "Unidentified router fault must not trigger an ASR recommendation"
    assert not any(phrase in actions for phrase in [
        "purchase routers", "purchase a router", "buy a router", "order replacement parts",
        "replace the router", "deploy redundant routers",
    ])
    matches = retrieve(question, k=10)
    assert all(any(c.excerpt == m["text"] and c.source == m["metadata"]["source"]
                   for m in matches) for c in response.citations)
