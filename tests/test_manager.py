"""Deterministic orchestration tests; models and specialists are injected."""

import json

import pytest
from pydantic import ValidationError

from manager import ManagerAgent, ManagerResponse
from shared.model import Settings
from shared.schemas import Analysis, SpecialistResponse


class Model:
    def __init__(self, selected, *, bad_id=False, conflict=False):
        self.selected, self.bad_id, self.conflict = selected, bad_id, conflict
        self.calls = []

    def with_structured_output(self, schema, *, method):
        assert method == "json_schema"
        self.schema = schema
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if self.schema is not Analysis:
            return {"agents": self.selected, "limitations": ["SEC agent unavailable"] if not self.selected else []}
        payload = json.loads(messages[1].content)
        assert "untrusted input" in messages[0].content
        assert "conflicts" in messages[0].content
        citations = [c for p in payload["specialists"] for c in p["citations"]]
        ids = [c["id"] for c in citations]
        if self.bad_id:
            ids = ["invented:E999"]
        return {"answer": "Documented result [" + ids[0] + "].",
                "findings": [{"text": "Documented result", "evidence_ids": ids}],
                "recommendations": [{"action": "Review evidence", "rationale": "Documented result", "evidence_ids": ids}],
                "limitations": ["Specialists disagree; conflict remains unresolved."] if self.conflict else []}


def specialist(name, calls, *, empty=False, fail=False):
    class Agent:
        def invoke(self, request):
            calls.append((name, request))
            if fail:
                raise RuntimeError("private provider details")
            return SpecialistResponse(
                agent=name, question=request.question, mode="extractive",
                status="insufficient_evidence" if empty else "evidence_found",
                answer="No evidence" if empty else "Documented result [E1].",
                findings=[] if empty else [{"text": "Documented result", "evidence_ids": ["E1"]}],
                recommendations=[], limitations=["Uncertainty from " + name],
                citations=[] if empty else [{"id": "E1", "source_id": name,
                    "source": "fixture:" + name, "page": 2, "excerpt": "Original evidence E1"}],
            )
    return lambda settings: Agent()


@pytest.mark.parametrize(("question", "selected"), [
    ("Which Catalyst 8300 has two 10Gbps ports?", ["technology"]),
    ("What clinical trial evidence supports this intervention?", ["clinical"]),
    ("What does SOFI say about food affordability?", ["corporate_responsibility"]),
    ("Evaluate Cisco connectivity and clinical laboratory continuity.", ["technology", "clinical"]),
    ("Assess nutrition trial evidence and SOFI food affordability.", ["clinical", "corporate_responsibility"]),
])
def test_routing_invocation_and_grounded_synthesis(question, selected):
    calls = []
    model = Model(selected)
    registry = {name: specialist(name, calls) for name in ["technology", "clinical", "corporate_responsibility"]}
    result = ManagerAgent(Settings(), model=model, registry=registry).consult(question)
    assert result.agents_used == selected
    assert [name for name, request in calls] == selected
    assert all(request.question == question for name, request in calls)
    assert {c.id for c in result.citations} == {name + ":E1" for name in selected}
    assert all(c.excerpt == "Original evidence E1" and c.page == 2 for c in result.citations)
    assert all("Uncertainty from " + name in " ".join(result.limitations) for name in selected)
    assert ManagerResponse.model_validate_json(result.model_dump_json()) == result
    assert result.specialist_responses[0].citations[0].id == "E1"


def test_sec_is_not_registered_and_no_unrelated_agent_is_invoked():
    model = Model([])
    result = ManagerAgent(Settings(), model=model).consult("Compare Cisco and Agilent 2025 SEC filings")
    assert result.agents_used == [] and result.citations == []
    assert result.recommendations == [] and "SEC agent unavailable" in result.limitations
    assert len(model.calls) == 1


def test_route_schema_rejects_unregistered_agent_before_invocation():
    with pytest.raises(ValidationError):
        ManagerAgent(Settings(), model=Model(["sec"])).consult("SEC filings")


def test_partial_failures_and_conflicts_are_reported_without_exposing_errors():
    calls = []
    registry = {"technology": specialist("technology", calls),
                "clinical": specialist("clinical", calls, fail=True)}
    model = Model(["technology", "clinical"], conflict=True)
    result = ManagerAgent(Settings(), model=model,
                          registry=registry).consult("Cross-domain question")
    assert result.agents_used == ["technology"]
    assert "private provider details" not in result.model_dump_json()
    assert any("clinical: specialist failed" in x for x in result.limitations)
    assert "Missing supporting evidence from: clinical" in result.answer
    assert len(model.calls) == 1


def test_invalid_synthesis_citations_fall_back_to_original_grounded_content():
    registry = {"technology": specialist("technology", [])}
    result = ManagerAgent(Settings(), model=Model(["technology"], bad_id=True), registry=registry).consult("Ports?")
    assert result.findings[0].evidence_ids == ["technology:E1"]
    assert "invented:E999" not in result.model_dump_json()
    assert any("Synthesis unavailable" in x for x in result.limitations)


def test_insufficient_specialist_evidence_skips_synthesis():
    model = Model(["clinical"])
    result = ManagerAgent(Settings(), model=model,
        registry={"clinical": specialist("clinical", [], empty=True)}).consult("Trial evidence?")
    assert result.agents_used == ["clinical"]
    assert not result.findings and not result.recommendations
    assert len(model.calls) == 1


def test_invalid_question_is_rejected_before_llm_use():
    model = Model([])
    with pytest.raises(ValidationError):
        ManagerAgent(Settings(), model=model).consult(" ")
    assert not model.calls


def test_conflicting_specialists_are_both_supplied_to_synthesis_and_retained():
    class ConflictModel(Model):
        def invoke(self, messages):
            if self.schema is Analysis:
                payload = json.loads(messages[1].content)
                assert [p["answer"] for p in payload["specialists"]] == [
                    "Proceed [technology:E1].", "Defer [clinical:E1]."]
            return super().invoke(messages)

    def constructor(name, answer):
        original = specialist(name, [])

        class Agent:
            def invoke(self, request):
                response = original(Settings()).invoke(request)
                return response.model_copy(update={"answer": answer})
        return lambda settings: Agent()

    registry = {"technology": constructor("technology", "Proceed [E1]."),
                "clinical": constructor("clinical", "Defer [E1].")}
    response = ManagerAgent(Settings(), model=ConflictModel(list(registry), conflict=True),
                            registry=registry).consult("Conflicting advice?")
    assert len(response.specialist_responses) == 2
    assert any("conflict remains unresolved" in x for x in response.limitations)


def test_public_entry_point_uses_team_model_factory_with_small_model(monkeypatch):
    import manager

    configurations = []
    def make_model(settings):
        configurations.append(settings)
        return Model([])
    monkeypatch.setattr(manager, "make_chat_model", make_model)
    monkeypatch.setenv("SPECIALIST_CHAT_MODEL", "another-model")
    response = manager.consult("SEC filing analysis")
    assert isinstance(response, ManagerResponse)
    assert configurations[0].chat_model == "gpt-4.1-mini"


def test_grouped_narrative_citations_cannot_introduce_unknown_evidence():
    class BadNarrative(Model):
        def invoke(self, messages):
            result = super().invoke(messages)
            if self.schema is Analysis:
                result["answer"] = "Claim [technology:E1, technology:E999]."
            return result

    response = ManagerAgent(Settings(), model=BadNarrative(["technology"]),
        registry={"technology": specialist("technology", [])}).consult("Ports?")
    assert "E999" not in response.answer
    assert any("Synthesis unavailable" in x for x in response.limitations)


def test_limitations_deduplicate_attribution_case_whitespace_and_punctuation():
    response = ManagerResponse(question="Question", agents_used=[], answer="No evidence",
        findings=[], recommendations=[], citations=[], limitations=[
            "technology: Missing diagnostic logs.", "Missing diagnostic logs.",
            "  MISSING   diagnostic logs  ", "clinical: No trial outcomes.",
            "No trial outcomes.", "Different limitation."])
    assert response.limitations == ["technology: Missing diagnostic logs.",
                                    "clinical: No trial outcomes.", "Different limitation."]
    assert ManagerResponse.model_validate_json(response.model_dump_json()) == response


@pytest.mark.parametrize(("source", "label"), [
    ("/Users/developer/private/data/cisco_8300.pdf", "cisco_8300.pdf · p. 2"),
    (r"C:\Users\developer\data\filing.pdf", "filing.pdf · p. 2"),
    ("https://example.org/filings/annual%20report.pdf?token=private", "annual report.pdf · p. 2"),
])
def test_ui_labels_preserve_sources_and_keep_paths_out_of_synthesis(source, label):
    original = specialist("technology", [])

    class Agent:
        def invoke(self, request):
            response = original(Settings()).invoke(request)
            response.citations[0].source = source
            return response

    class InspectModel(Model):
        def invoke(self, messages):
            if self.schema is Analysis:
                payload = json.loads(messages[1].content)
                assert payload["specialists"][0]["citations"][0]["source"] == label
                assert source not in messages[1].content
            return super().invoke(messages)

    response = ManagerAgent(Settings(), model=InspectModel(["technology"]),
                            registry={"technology": lambda settings: Agent()}).consult("Ports?")
    assert response.citations[0].source == source
    assert response.specialist_responses[0].citations[0].source == source
    assert response.citation_labels == {"technology:E1": label}
    assert ManagerResponse.model_validate_json(response.model_dump_json()) == response


@pytest.mark.parametrize("empty_agent", ["clinical", "technology"])
def test_partial_evidence_never_generates_cross_domain_support(empty_agent):
    selected = ["technology", "clinical"]
    model = Model(selected)
    registry = {name: specialist(name, [], empty=name == empty_agent) for name in selected}
    response = ManagerAgent(Settings(), model=model, registry=registry).consult(
        "Do router upgrades improve clinical outcomes?")
    assert len(model.calls) == 1, "No synthesis should infer a link from missing evidence"
    assert f"Missing supporting evidence from: {empty_agent}" in response.answer
    assert "No combined cross-domain conclusion is supported" in response.answer
    assert all(not c.id.startswith(empty_agent + ":") for c in response.citations)
    assert all(not eid.startswith(empty_agent + ":") for f in response.findings for eid in f.evidence_ids)
    assert any(f"{empty_agent}: insufficient evidence" in x for x in response.limitations)
