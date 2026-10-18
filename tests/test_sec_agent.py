import json

import pytest
from pydantic import ValidationError

from agents import sec
from shared.model import Settings
from shared.schemas import Analysis, SpecialistResponse


@pytest.fixture
def evidence(monkeypatch):
    calls = []
    def retrieve(question, k, *, company, fiscal_year):
        calls.append((company, fiscal_year, k))
        if fiscal_year != 2025:
            return []
        return [{"text": "Synthetic risk disclosure.", "metadata": {
            "company": company, "filing_type": "10-K", "fiscal_year": 2025,
            "period_end": "2025-07-26" if company == "Cisco" else "2025-10-31",
            "source": "fixture:" + company, "source_id": company, "page": 5, "pages": [5],
            "table_context": {"table": {"markdown": "2025 | Revenue | 12", "footnotes": ["Synthetic footnote."]}},
            "page_context": {"5": "USD in millions, year ended 2025."},
        }}]
    monkeypatch.setattr(sec, "retrieve", retrieve)
    return calls


def test_filters_period_and_financial_context_are_preserved(evidence):
    r = sec.SECAgent(Settings()).invoke("Compare Cisco and Agilent FY2025 risks")
    assert evidence == [("Cisco", 2025, 5), ("Agilent", 2025, 5)]
    assert r.agent == "sec" and r.report_year == 2025
    assert len(r.citations) == 2
    assert "2025-07-26" in r.citations[0].excerpt
    assert "2025-10-31" in r.citations[1].excerpt
    assert all("USD in millions" in c.excerpt and "Synthetic footnote" in c.excerpt for c in r.citations)
    assert SpecialistResponse.model_validate_json(r.model_dump_json()) == r


def test_missing_year_is_not_silently_replaced_with_2025(evidence):
    r = sec.SECAgent(Settings()).invoke("Cisco FY2026 risks")
    assert evidence == [("Cisco", 2026, 5)]
    assert r.status == "insufficient_evidence" and not r.citations


def test_explicit_structured_year_takes_precedence(evidence):
    r = sec.SECAgent(Settings()).invoke({"question": "Cisco FY2026 risks", "report_year": 2025})
    assert evidence == [("Cisco", 2025, 5)]
    assert r.report_year == 2025


def test_unsupported_company_requires_clarification(evidence):
    with pytest.raises(ValueError, match="Specify Cisco or Agilent"):
        sec.SECAgent(Settings()).invoke("Alphabet risks")
    assert not evidence


@pytest.mark.parametrize("eid", ["E1", "E999"])
def test_team_model_policy_and_citation_validation(evidence, eid):
    class Model:
        def with_structured_output(self, schema, *, method):
            assert schema is Analysis and method == "json_schema"
            return self
        def invoke(self, messages):
            assert "untrusted input" in messages[0].content
            assert "comparative-year" in messages[0].content.lower()
            assert "Cisco" in json.loads(messages[1].content)["evidence"][0]["excerpt"]
            return {"answer": "Synthetic disclosure.", "findings": [
                {"text": "Synthetic disclosure.", "evidence_ids": [eid]}],
                "recommendations": [], "limitations": ["Synthetic evidence."]}
    agent = sec.SECAgent(Settings(llm_provider="openai"), model=Model())
    if eid == "E999":
        with pytest.raises(ValidationError, match="unknown evidence identifier"):
            agent.invoke("Cisco FY2025 risks")
    else:
        assert agent.invoke("Cisco FY2025 risks").mode == "openai"


def test_generic_available_filings_question_balances_both_companies(evidence):
    response = sec.SECAgent(Settings()).invoke(
        "What major financial risks are disclosed in the available SEC filings?"
    )
    assert evidence == [("Cisco", 2025, 5), ("Agilent", 2025, 5)]
    assert response.status == "evidence_found"
    assert [citation.source_id for citation in response.citations] == ["Cisco", "Agilent"]
    assert all(finding.evidence_ids for finding in response.findings)
    assert any("only Cisco and Agilent FY2025" in note for note in response.limitations)


def test_generic_question_with_unavailable_year_has_no_fabricated_evidence(evidence):
    response = sec.SECAgent(Settings()).invoke(
        {"question": "What risks appear in the available SEC filings?", "report_year": 2026}
    )
    assert evidence == [("Cisco", 2026, 5), ("Agilent", 2026, 5)]
    assert response.status == "insufficient_evidence"
    assert response.citations == []
    assert response.findings == []
