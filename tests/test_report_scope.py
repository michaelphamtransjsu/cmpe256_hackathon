"""Publication scope and lexical ranking use synthetic fixtures, not SOFI facts."""

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from agents.clinical import ClinicalAgent
from agents.corporate_responsibility import CorporateResponsibilityAgent, focused_corporate_query, infer_report_year
from shared.model import Settings
from shared.schemas import SpecialistRequest, SpecialistResponse
from shared import retrieval
from shared.retrieval import HashEmbeddings, SpecialistIndex


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return Settings(data_dir=tmp_path, min_score=0.1)


def report_document(year, text, source_id=None):
    return Document(page_content=text, metadata={
        "agent": "corporate_responsibility",
        "source_id": source_id or f"synthetic-sofi-{year}",
        "source": f"fixture: synthetic publication {year}, not an actual SOFI excerpt",
        "page": 2,
        "report_year": year,
    })


@pytest.mark.parametrize(("question", "expected"), [
    ("What does SOFI 2023 say about nutrition?", 2023),
    ("Review SOFI-2024 nutrition evidence.", 2024),
    ("Use the 2023 report for nutrition.", 2023),
    ("Use the report from 2024 for nutrition.", 2024),
    ("The State of Food Security and Nutrition in the World 2023: nutrition?", 2023),
    ("What was diet cost in 2022?", None),
    ("Compare SOFI 2023 and SOFI 2024 nutrition evidence.", None),
    ("Compare the 2023 report with the 2024 report.", None),
    ("Use SOFI 2023 and SOFI 2024 nutrition evidence.", None),
    ("What is known about nutrition?", None),
])
def test_report_inference_distinguishes_publications_from_observation_years(question, expected):
    assert infer_report_year(question) == expected


@pytest.mark.parametrize("embeddings_provider", ["hash", "openai"])
def test_requested_year_is_filtered_before_the_result_limit_and_survives_reopening(settings, monkeypatch, embeddings_provider):
    settings = settings.model_copy(update={"embeddings_provider": embeddings_provider})
    # Exercise both retrieval paths with deterministic vectors and no hosted API.
    monkeypatch.setattr(retrieval, "make_embeddings", lambda _: HashEmbeddings())
    index = SpecialistIndex("corporate_responsibility", settings)
    index.replace([
        report_document(2024, "Nutrition", "wrong-year-highest-score"),
        report_document(2023, "Synthetic nutrition affordability uncertainty evidence.", "requested-year"),
    ])
    reopened = SpecialistIndex("corporate_responsibility", settings)
    assert [doc.metadata["source_id"] for doc in reopened.search("nutrition", limit=1)] == ["wrong-year-highest-score"]
    selected = reopened.search("nutrition", limit=1, report_year=2023)
    assert [doc.metadata["source_id"] for doc in selected] == ["requested-year"]
    assert selected[0].metadata["report_year"] == 2023


@pytest.mark.parametrize("agent_request", [
    "Use SOFI 2023 nutrition evidence.",
    {"question": "nutrition", "report_year": 2023},
    {"question": "Use SOFI 2024 nutrition evidence.", "report_year": 2023},
])
def test_corporate_explicit_or_inferred_year_never_cites_another_report(settings, agent_request):
    agent = CorporateResponsibilityAgent(settings)
    agent.index.replace([
        report_document(2024, "Nutrition", "wrong-year"),
        report_document(2023, "Synthetic nutrition affordability evidence.", "requested-year"),
    ])
    # The explicit structured filter takes precedence over publication wording.
    result = agent.invoke(agent_request)
    assert result.status == "evidence_found"
    assert result.report_year == 2023
    assert {citation.source_id for citation in result.citations} == {"requested-year"}
    assert {citation.report_year for citation in result.citations} == {2023}


def test_missing_requested_publication_returns_insufficient_evidence(settings):
    agent = CorporateResponsibilityAgent(settings)
    agent.index.replace([report_document(2024, "Synthetic nutrition evidence.")])
    result = agent.invoke({"question": "nutrition", "report_year": 2023})
    assert result.report_year == 2023
    assert result.status == "insufficient_evidence"
    assert result.findings == result.recommendations == result.citations == []


@pytest.mark.parametrize("question", [
    "What was the synthetic diet cost in 2022?",
    "Compare SOFI 2023 and SOFI 2024 synthetic diet cost.",
])
def test_observation_year_and_comparison_questions_keep_both_publications(settings, question):
    agent = CorporateResponsibilityAgent(settings)
    agent.index.replace([
        report_document(2023, "Synthetic diet cost in 2022 is recorded in this fixture."),
        report_document(2024, "Synthetic diet cost in 2022 is revised in this fixture."),
    ])
    result = agent.invoke(question)
    assert result.report_year is None
    assert {citation.report_year for citation in result.citations} == {2023, 2024}


@pytest.mark.parametrize("year", [1899, 2101, 2023.5, "not-a-year", True])
def test_invalid_publication_years_are_rejected_by_the_request_contract(year):
    with pytest.raises(ValidationError):
        SpecialistRequest(question="nutrition", report_year=year)


def test_clinical_rejects_a_corporate_publication_filter(settings):
    with pytest.raises(ValueError, match="only to Corporate Responsibility"):
        ClinicalAgent(settings).invoke({"question": "nutrition", "report_year": 2023})


@pytest.mark.parametrize("citation_year", [2024, None])
def test_response_validation_rejects_citations_outside_the_selected_publication(settings, citation_year):
    agent = CorporateResponsibilityAgent(settings)
    agent.index.replace([report_document(2023, "Synthetic nutrition evidence.")])
    payload = agent.invoke({"question": "nutrition", "report_year": 2023}).model_dump()
    payload["citations"][0]["report_year"] = citation_year
    with pytest.raises(ValidationError, match="outside the selected report year"):
        SpecialistResponse.model_validate(payload)


def test_query_focus_removes_title_boilerplate_and_expands_retrieval_terms():
    question = (
        "From within the context of The State of Food Security and Nutrition in the World 2023, "
        "list major reasons for war zones and food prices."
    )
    focused = focused_corporate_query(question, 2023).casefold()
    assert "state of food security and nutrition in the world" not in focused
    assert "2023" not in focused
    assert "from within the context of" not in focused
    for term in ("drivers", "causes", "conflict", "humanitarian", "inflation", "affordability", "purchasing power"):
        assert term in focused


def test_full_publication_title_does_not_make_a_cover_page_the_answer(settings):
    title = "The State of Food Security and Nutrition in the World 2023"
    agent = CorporateResponsibilityAgent(settings)
    excerpt = "Conflict humanitarian crises disrupted synthetic household food access."
    agent.index.replace([
        report_document(2023, f"{title}. {title}.", "cover-boilerplate"),
        report_document(2023, excerpt, "substantive-fixture"),
    ])
    question = f"From within the context of {title}, explain war zones."
    response = agent.invoke({"question": question, "max_sources": 1})
    assert response.question == question
    assert response.report_year == 2023
    assert [citation.source_id for citation in response.citations] == ["substantive-fixture"]
    assert [finding.text for finding in response.findings] == [excerpt]


@pytest.mark.parametrize(("question", "excerpt"), [
    ("Explain war zones in SOFI 2023.", "Conflict humanitarian crises disrupted synthetic household food access."),
    ("Explain prices in SOFI 2023.", "Affordability and purchasing power affect the synthetic healthy diet cost."),
    ("List major reasons in SOFI 2023.", "Drivers and causes include synthetic economic disruptions."),
])
def test_query_expansion_retrieves_source_wording_without_turning_synonyms_into_claims(settings, question, excerpt):
    agent = CorporateResponsibilityAgent(settings)
    agent.index.replace([report_document(2023, excerpt)])
    response = agent.invoke(question)
    assert response.status == "evidence_found"
    assert response.question == question
    assert response.report_year == 2023
    assert [citation.excerpt for citation in response.citations] == [excerpt]
    assert [finding.text for finding in response.findings] == [excerpt]
    assert response.recommendations == []


def test_rare_query_term_outranks_repeated_cover_boilerplate(settings):
    index = SpecialistIndex("corporate_responsibility", settings)
    covers = [report_document(2023, "Nutrition " * 30 + f"Synthetic cover {number}.", f"cover-{number}") for number in range(10)]
    rare = report_document(2023, "Conflict displaced synthetic farming households and disrupted livelihoods.", "substantive-evidence")
    index.replace([*covers, rare])
    result = SpecialistIndex("corporate_responsibility", settings).search("nutrition conflict drivers", limit=1)
    assert [doc.metadata["source_id"] for doc in result] == ["substantive-evidence"]
