"""Optional local evidence checks. Originals are never committed or modified."""

import hashlib
from pathlib import Path

import pytest
from pypdf import PdfReader

from agents.clinical import ClinicalAgent
from agents.corporate_responsibility import CorporateResponsibilityAgent
from shared.model import Settings
from shared.retrieval import SpecialistIndex
from shared.sources import discover_sources, load_documents
from shared.trial_csv import load_trial_csvs
from shared.trials import TrialStore


def assert_verbatim_citations(response, entries):
    assert response.status == "evidence_found" and response.citations
    by_id = {e["source_id"]: e for e in entries}
    pages = {}
    for citation in response.citations:
        entry = by_id[citation.source_id]
        assert citation.source == entry["original_member_path"]
        path = Path(entry["local_path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        if path not in pages:
            pages[path] = PdfReader(path).pages
        text = pages[path][citation.page - 1].extract_text()
        assert " ".join(citation.excerpt.split()) in " ".join(text.split())


def test_real_sofi_directory_retrieval_and_report_years(real_data_dir, tmp_path):
    if not all((real_data_dir / f"FoodSecurity/SOFI-{y}.pdf").is_file() for y in (2023, 2024)):
        pytest.skip("Both real FoodSecurity/SOFI PDFs are required")
    entries = discover_sources(real_data_dir, local_only=True)
    docs, updated = load_documents(entries, "corporate_responsibility")
    assert docs and not any(e["status"] == "failed" for e in updated if e["agent"] == "corporate_responsibility")
    settings = Settings(data_dir=tmp_path)
    index = SpecialistIndex("corporate_responsibility", settings)
    index.replace(docs)
    originals = {(d.metadata["source_id"], d.metadata["page"]): d for d in docs}
    for row in index.store.store.values():
        original = originals[row["metadata"]["source_id"], row["metadata"]["page"]]
        assert row["text"] in original.page_content
        assert row["metadata"]["sha256"] == original.metadata["sha256"]
        assert row["metadata"]["original_member_path"] == original.metadata["source"]
    assert {row["metadata"]["report_year"] for row in index.store.store.values()} == {2023, 2024}
    for year in (2023, 2024):
        response = CorporateResponsibilityAgent(settings).invoke({"question": "food insecurity drivers conflict climate prices", "report_year": year})
        assert {c.report_year for c in response.citations} == {year}
        assert_verbatim_citations(response, updated)


def test_real_clinical_pdf_retrieval(real_data_dir, tmp_path):
    entries = discover_sources(real_data_dir, local_only=True)
    entries = [e for e in entries if e["agent"] == "clinical" and e["format"] == "pdf"]
    if not entries:
        pytest.skip("Real Clinical research PDFs are required")
    documents, updated = load_documents(entries, "clinical")
    assert documents and all(e["status"] == "indexed" for e in updated)
    settings = Settings(data_dir=tmp_path)
    index = SpecialistIndex("clinical", settings)
    index.replace(documents)
    assert {row["metadata"]["source_id"] for row in index.store.store.values()} == {entry["source_id"] for entry in updated}
    agent = ClinicalAgent(settings)
    # Use source words so this checks document retrieval and exact attribution,
    # independently of medical interpretations or a hosted model.
    import re

    question = " ".join(re.findall(r"[A-Za-z]{5,}", documents[0].page_content)[:20])
    assert_verbatim_citations(agent.invoke(question), updated)


def test_real_clinical_csv_import_and_trial_retrieval(real_data_dir, tmp_path):
    entries = discover_sources(real_data_dir, local_only=True)
    if not any(e["topic"] == "clinical_trials" for e in entries):
        pytest.skip("Real Clinical/ctg-studies_*.csv exports are required")
    records = load_trial_csvs(entries)
    settings = Settings(data_dir=tmp_path)
    store = TrialStore(tmp_path / "clinical/trials.sqlite3")
    assert store.import_records(records)["imported"] == len(records)
    assert store.import_records(records)["skipped"] == len(records)
    for record in records:
        assert store.get(record.trial_id) == record
    response = ClinicalAgent(settings).invoke(records[0].trial_id)
    assert response.status == "evidence_found"
    assert response.trial_records[0] == records[0]
    assert response.citations[0].source == records[0].source
