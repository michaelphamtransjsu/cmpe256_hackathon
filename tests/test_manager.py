"""Deterministic orchestration tests; models and specialists are injected."""

import json
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from manager import ManagerAgent, ManagerResponse
from shared.model import Settings
from shared.schemas import Analysis, SpecialistResponse
from shared.trials import TrialRecord


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


@pytest.fixture
def clinical_trial(tmp_path):
    from agents.clinical import ClinicalAgent

    settings = Settings(data_dir=tmp_path)
    clinical = ClinicalAgent(settings)
    record = TrialRecord(
        trial_id="SYNTHETIC-DIABETES-001", title="Synthetic diabetes registration",
        source="fixture:synthetic-trial", updated_at=date(2026, 10, 1),
        metadata={"provenance": {"source_id": "synthetic-csv", "record_number": 2}},
    )
    clinical.trials.upsert(record)
    return settings, clinical, record


@pytest.mark.parametrize("updated_at", [
    date(2026, 10, 1), datetime(2026, 10, 1, 12, 30, tzinfo=timezone.utc), None,
], ids=["date", "datetime", "missing-date"])
@pytest.mark.parametrize("structured_model", [False, True], ids=["dict", "pydantic"])
def test_clinical_trial_synthesis_serializes_dates_and_preserves_evidence(
        clinical_trial, updated_at, structured_model):
    settings, clinical, record = clinical_trial
    record.updated_at = updated_at
    clinical.trials.upsert(record)
    question = "What diabetes trial evidence is available?"
    original = clinical.invoke(question)
    original_payload = original.model_dump()

    class TrialModel(Model):
        def invoke(self, messages):
            result = super().invoke(messages)
            if self.schema is not Analysis:
                result["limitations"] = ["Routing scope: clinical trial evidence only."]
            else:
                payload = json.loads(messages[1].content)
                trial = payload["specialists"][0]["trial_records"][0]
                assert trial == record.model_dump(mode="json")
                assert payload["specialists"][0]["findings"][0]["evidence_ids"] == ["clinical:E1"]
                assert payload["specialists"][0]["citations"][0]["excerpt"] == original.citations[0].excerpt
                result["limitations"] = ["Registration alone does not establish efficacy."]
            return self.schema.model_validate(result) if structured_model else result

    model = TrialModel(["clinical"])
    response = ManagerAgent(settings, model=model, registry={"clinical": lambda settings: clinical}).consult(question)

    assert len(model.calls) == 2
    assert response.answer == "Documented result [clinical:E1]."
    assert response.recommendations[0].evidence_ids == ["clinical:E1"]
    assert response.findings[0].evidence_ids == ["clinical:E1"]
    assert response.agents_used == ["clinical"] and response.question == question
    assert response.citations[0].model_dump() == {**original.citations[0].model_dump(), "id": "clinical:E1"}
    assert response.specialist_responses[0].model_dump() == original_payload
    assert response.specialist_responses[0].trial_records == [record]
    assert response.citation_labels == {"clinical:E1": "fixture:synthetic-trial"}
    assert "Routing scope: clinical trial evidence only." in response.limitations
    assert "Registration alone does not establish efficacy." in response.limitations
    assert all(f"clinical: {item}" in response.limitations for item in original.limitations)
    assert not any("Synthesis unavailable" in item for item in response.limitations)
    assert ManagerResponse.model_validate_json(response.model_dump_json()) == response


@pytest.mark.parametrize(("failure", "error_type"), [
    ("provider", "RuntimeError"), ("provider-type-error", "TypeError"),
    ("malformed", "ValidationError"), ("unknown-evidence", "ValidationError"),
    ("unknown-narrative", "ValueError"),
])
def test_clinical_trial_synthesis_failures_keep_grounded_fallback(clinical_trial, failure, error_type):
    settings, clinical, record = clinical_trial
    question = "What diabetes trial evidence is available?"
    original = clinical.invoke(question)

    class FailingModel(Model):
        def invoke(self, messages):
            result = super().invoke(messages)
            if self.schema is Analysis:
                if failure in {"provider", "provider-type-error"}:
                    error = RuntimeError if failure == "provider" else TypeError
                    raise error("private provider details")
                if failure == "malformed":
                    return {"answer": "private malformed response"}
                if failure == "unknown-evidence":
                    result["findings"][0]["evidence_ids"] = ["clinical:E999"]
                if failure == "unknown-narrative":
                    result["answer"] = "Unsupported claim [clinical:E1, clinical:E999]."
            return result

    model = FailingModel(["clinical"])
    response = ManagerAgent(settings, model=model, registry={"clinical": lambda settings: clinical}).consult(question)

    reason = (f"Synthesis unavailable ({error_type}); specialist responses are shown separately "
              "and conflicts remain unresolved.")
    assert len(model.calls) == 2, "The dated trial must reach the synthesis provider"
    assert response.answer == f"{reason}\n\nclinical: {original.answer}"
    assert reason in response.limitations
    assert response.agents_used == ["clinical"] and response.question == question
    assert response.citations[0].model_dump() == {**original.citations[0].model_dump(), "id": "clinical:E1"}
    assert response.findings[0].evidence_ids == ["clinical:E1"]
    assert response.recommendations == original.recommendations
    assert response.specialist_responses == [original]
    assert response.specialist_responses[0].trial_records == [record]
    assert "private" not in response.model_dump_json() and "E999" not in response.model_dump_json()
    assert ManagerResponse.model_validate_json(response.model_dump_json()) == response


@pytest.mark.parametrize(("question", "selected"), [
    ("Which Catalyst 8300 has two 10Gbps ports?", ["technology"]),
    ("What clinical trial evidence supports this intervention?", ["clinical"]),
    ("What does SOFI say about food affordability?", ["corporate_responsibility"]),
    ("Evaluate Cisco connectivity and clinical laboratory continuity.", ["technology", "clinical"]),
    ("Assess nutrition trial evidence and SOFI food affordability.", ["clinical", "corporate_responsibility"]),
    ("What are Cisco's FY2025 financial risks?", ["sec"]),
    ("What was Agilent's FY2025 revenue?", ["sec"]),
    ("Assess Cisco router specifications and FY2025 financial risks.", ["technology", "sec"]),
])
def test_routing_invocation_and_grounded_synthesis(question, selected):
    calls = []
    model = Model(selected)
    registry = {name: specialist(name, calls) for name in ["technology", "clinical", "corporate_responsibility", "sec"]}
    result = ManagerAgent(Settings(), model=model, registry=registry).consult(question)
    assert result.agents_used == selected
    assert [name for name, request in calls] == selected
    assert all(request.question == question for name, request in calls)
    assert {c.id for c in result.citations} == {name + ":E1" for name in selected}
    assert all(c.excerpt == "Original evidence E1" and c.page == 2 for c in result.citations)
    assert all("Uncertainty from " + name in " ".join(result.limitations) for name in selected)
    assert ManagerResponse.model_validate_json(result.model_dump_json()) == result
    assert result.specialist_responses[0].citations[0].id == "E1"
    assert len(model.calls) == 2
    assert result.answer == f"Documented result [{selected[0]}:E1]."
    assert result.recommendations[0].evidence_ids == [name + ":E1" for name in selected]


def test_unavailable_sec_in_custom_registry_does_not_invoke_unrelated_agent():
    model = Model([])
    result = ManagerAgent(Settings(), model=model, registry={
        "technology": specialist("technology", [])}).consult("Compare Cisco and Agilent 2025 SEC filings")
    assert result.agents_used == [] and result.citations == []
    assert result.recommendations == [] and "SEC agent unavailable" in result.limitations
    assert len(model.calls) == 1


def test_route_schema_rejects_unregistered_agent_before_invocation():
    with pytest.raises(ValidationError):
        ManagerAgent(Settings(), model=Model(["retail"])).consult("Retail analysis")


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


def test_default_sec_registration_and_scope(monkeypatch):
    import manager
    from agents import create_specialist
    from agents.sec import SECAgent

    settings = Settings()
    assert isinstance(create_specialist("sec", settings), SECAgent)
    calls = []
    monkeypatch.setattr(manager, "create_specialist", lambda name, settings: specialist(name, calls)(settings))
    model = Model(["sec"])
    result = manager.ManagerAgent(settings, model=model).consult("Cisco FY2025 financial risks")
    assert result.agents_used == ["sec"]
    route_schema = model.calls[0][0].content
    assert "Agilent FY2025" in route_schema
    assert "Technology and SEC" in route_schema


def test_sec_periods_and_both_domains_evidence_survive_synthesis():
    class SEC:
        def invoke(self, request):
            return SpecialistResponse(agent="sec", question=request.question, mode="extractive",
                status="evidence_found", report_year=2025,
                answer="Cisco FY2025 total revenue was USD 56,654 million for the year ended July 26, 2025 [E1].",
                findings=[{"text": "Cisco FY2025 total revenue: USD 56,654 million; year ended July 26, 2025.", "evidence_ids": ["E1"]}],
                recommendations=[], limitations=["No causal link to router specifications is established."],
                citations=[{"id": "E1", "source_id": "cisco-2025-10-K", "source": "fixture:cisco-filing",
                    "page": 59, "report_year": 2025, "excerpt": "Cisco | FY2025 | year ended 2025-07-26 | total revenue USD 56,654 million."}])

    class InspectModel(Model):
        def invoke(self, messages):
            if self.schema is Analysis:
                payload = json.loads(messages[1].content)
                sec = next(p for p in payload["specialists"] if p["agent"] == "sec")
                assert sec["report_year"] == sec["citations"][0]["report_year"] == 2025
                assert "USD 56,654 million" in sec["answer"] and "July 26, 2025" in sec["answer"]
                assert "Do not infer that product features" in messages[0].content
            return super().invoke(messages)

    result = ManagerAgent(Settings(), model=InspectModel(["technology", "sec"]), registry={
        "technology": specialist("technology", []), "sec": lambda settings: SEC()}).consult(
            "Compare Cisco router specifications and FY2025 financial risk.")
    assert {eid.split(":")[0] for f in result.findings for eid in f.evidence_ids} == {"sec", "technology"}
    citation = next(c for c in result.citations if c.id == "sec:E1")
    assert citation.report_year == 2025 and citation.page == 59
    assert "2025-07-26" in citation.excerpt and "USD 56,654 million" in citation.excerpt
    assert "sec: No causal link to router specifications is established." in result.limitations
    assert ManagerResponse.model_validate_json(result.model_dump_json()) == result


def test_cross_domain_synthesis_policy_reconciles_supplied_specs_and_direct_support():
    """Regression of prompt/data wiring; semantic compliance is checked by the live test."""
    specs = "Synthetic router X has two built-in 10Gbps Ethernet ports."
    risk = "Synthetic issuer discloses acquisition integration risk."

    def constructor(name, text, recommendations):
        class Agent:
            def invoke(self, request):
                return SpecialistResponse(agent=name, question=request.question, mode="extractive",
                    status="evidence_found", answer=text,
                    findings=[{"text": text, "evidence_ids": ["E1"]}],
                    recommendations=recommendations, limitations=[],
                    citations=[{"id": "E1", "source_id": name, "source": "fixture:" + name,
                                "excerpt": text}])
        return lambda settings: Agent()

    class ReviewModel(Model):
        def invoke(self, messages):
            if self.schema is not Analysis:
                return super().invoke(messages)
            policy = messages[0].content
            assert "Consider all specialist findings before generating recommendations" in policy
            assert "Do not recommend obtaining information already supplied by another specialist" in policy
            assert "actual excerpts directly support the associated factual" in policy
            assert "do not blindly copy specialist recommendations" in policy
            assert "do not imply that the source prescribes the action or proves its effectiveness" in policy
            data = json.loads(messages[1].content)
            supplied = {p["agent"]: p for p in data["specialists"]}
            assert supplied["technology"]["findings"][0]["text"] == specs
            assert supplied["sec"]["citations"][0]["excerpt"] == risk
            assert supplied["sec"]["recommendations"][0]["action"] == "Obtain product specifications"
            return {"answer": specs + " [technology:E1] " + risk + " [sec:E1]",
                    "findings": [f for p in data["specialists"] for f in p["findings"]],
                    "recommendations": [], "limitations": ["No causal relationship is established."]}

    result = ManagerAgent(Settings(), model=ReviewModel(["technology", "sec"]), registry={
        "technology": constructor("technology", specs, []),
        "sec": constructor("sec", risk, [{"action": "Obtain product specifications",
            "rationale": "Not supplied by this specialist", "evidence_ids": ["E1"]}]),
    }).consult("Which router ports are documented and what are the issuer's risks?")
    assert result.recommendations == []
    assert result.findings[0].evidence_ids == ["technology:E1"]
    assert result.findings[1].evidence_ids == ["sec:E1"]
