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
            "source": "fixture:" + company, "source_id": company.lower() + "-2025-10-K", "page": 5, "pages": [5],
            "table_context": {"table": {"markdown": "2025 | Revenue | 12", "footnotes": ["Synthetic footnote."]}},
            "page_context": {"5": "USD in millions, year ended 2025."},
        }}]
    monkeypatch.setattr(sec, "retrieve", retrieve)
    return calls


def test_filters_period_and_financial_context_are_preserved(evidence):
    r = sec.SECAgent(Settings()).invoke("Compare Cisco and Agilent FY2025 risks")
    assert evidence == [("Cisco", 2025, 5), ("Agilent", 2025, 5)]
    assert r.agent == "sec" and r.report_year == 2025
    assert not any("Hash embeddings" in item for item in r.limitations)
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


def test_unsupported_company_is_not_substituted(evidence):
    for question in ["Microsoft risks", "Microsoft FY2025 risks in the SEC reports", "Amazon revenue",
                     "Compare Microsoft and Cisco risks",
                     "Compare Cisco FY2025 and microsoft FY2026 risks"]:
        response = sec.SECAgent(Settings()).invoke(question)
        assert response.status == "insufficient_evidence"
        assert "not substituted" in response.limitations[0]
    assert not evidence


def test_alphabet_is_supported(evidence):
    response = sec.SECAgent(Settings()).invoke("Alphabet FY2025 risks")
    assert evidence == [("Alphabet", 2025, 5)]
    assert response.status == "evidence_found"


def test_comparison_balances_company_year_groups(evidence):
    response = sec.SECAgent(Settings()).invoke("Compare Cisco FY2025 and FY2026 risks")
    assert evidence == [("Cisco", 2025, 5), ("Cisco", 2026, 5)]
    assert response.report_year is None
    assert "Cisco FY2026" in response.limitations[-1]


def test_general_question_searches_all_filings(evidence):
    response = sec.SECAgent(Settings()).invoke(
        "What business risks and regulatory concerns are mentioned in the SEC reports?")
    assert len(evidence) == 5 and len(set(evidence)) == 5
    assert response.status == "evidence_found"
    assert {c.report_year for c in response.citations} == {2025}


def test_multi_year_citations_conform_to_shared_response(monkeypatch):
    def retrieve(question, k, *, company, fiscal_year):
        return [{"text": "Synthetic risk disclosure.", "metadata": {
            "company": company, "filing_type": "10-K", "fiscal_year": fiscal_year,
            "period_end": str(fiscal_year) + "-07-25", "pages": [5], "page": 5,
            "source": "fixture:cisco", "source_id": f"cisco-{fiscal_year}-10-K"}}]
    monkeypatch.setattr(sec, "retrieve", retrieve)
    response = sec.SECAgent(Settings()).invoke("Compare Cisco FY2025 and FY2026 risks")
    assert response.report_year is None
    assert {c.report_year for c in response.citations} == {2025, 2026}
    assert SpecialistResponse.model_validate_json(response.model_dump_json()) == response


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


def test_cross_company_periods_stay_attached_to_each_issuer():
    assert {(f['company'], f['fiscal_year']) for f in sec._filings(
        'Compare Cisco FY2026 and Alphabet FY2024 risks')} == {('Cisco', 2026), ('Alphabet', 2024)}
    assert {(f['company'], f['fiscal_year']) for f in sec._filings(
        'Compare Cisco FY2024 and Agilent FY2025 risks')} == {('Agilent', 2025)}


def test_repeated_company_mentions_retain_all_requested_years():
    selected = sec._filings('Compare Cisco FY2025, Agilent FY2025, and Cisco FY2026 risks')
    assert {(f['company'], f['fiscal_year']) for f in selected} == {
        ('Cisco', 2025), ('Cisco', 2026), ('Agilent', 2025)}


def test_unavailable_comparison_period_is_reported(evidence):
    response = sec.SECAgent(Settings()).invoke('Compare Cisco FY2024 and Agilent FY2025 risks')
    assert evidence == [('Agilent', 2025, 5)]
    assert any('not indexed: Cisco FY2024' in item for item in response.limitations)
