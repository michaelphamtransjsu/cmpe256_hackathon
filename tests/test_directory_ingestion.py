"""Explicit extracted-folder assignments and answering-only persistence."""

import hashlib
import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from shared.sources import discover_sources, load_documents
from shared.model import Settings
from shared.retrieval import SpecialistIndex
from test_cli import pdf_bytes, run_cli
from test_trial_csv import write_csv


def test_only_explicit_directory_sources_are_assigned(tmp_path):
    root = tmp_path / "data"
    allowed = ["Clinical/Covid/paper.pdf", "Clinical/Diabetes/paper.pdf", "Clinical/Heart_attack/paper.pdf",
               "Clinical/KneeInjuries/paper.pdf", "FoodSecurity/SOFI-2023.pdf", "FoodSecurity/SOFI-2024.pdf"]
    unrelated = ["Retail/Clinical/paper.pdf", "SEC/Clinical/paper.pdf", "Cisco/report.pdf",
                 "FoodSecurity/other.pdf", "FoodSecurity/nested/SOFI-2023.pdf", "other/SOFI-2024.pdf",
                 "Clinical/unassigned.txt", "Clinical/Cisco/report.pdf", "paper.pdf"]
    for relative in allowed + unrelated:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf_bytes("SYNTHETIC evidence fixture."))
        path.with_name(path.name + ":Zone.Identifier").write_text("Windows metadata")
    csv = write_csv(root / "Clinical/ctg-studies_fixture.csv")
    entries = discover_sources(root)
    ready = {e["original_member_path"].removeprefix("data/") for e in entries if e["status"] == "ready"}
    assert ready == set(allowed)
    assert not any(":Zone.Identifier" in e["original_member_path"] for e in entries)
    trial = next(e for e in entries if e["topic"] == "clinical_trials")
    assert trial["status"] == "ready_for_import"
    assert trial["sha256"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    for year in (2023, 2024):
        assert next(e for e in entries if e["original_member_path"] == f"data/FoodSecurity/SOFI-{year}.pdf")["report_year"] == year
    clinical, _ = load_documents(entries, "clinical")
    corporate, _ = load_documents(entries, "corporate_responsibility")
    assert len(clinical) == 4 and len(corporate) == 2
    assert all(Path(d.metadata["local_path"]).is_file() and d.metadata["sha256"] for d in clinical + corporate)


@pytest.mark.parametrize("name", ["Clinical", "FoodSecurity"])
def test_discovery_can_select_an_assigned_subdirectory(tmp_path, name):
    root = tmp_path / "data" / name
    root.mkdir(parents=True)
    (root / ("paper.pdf" if name == "Clinical" else "SOFI-2024.pdf")).write_bytes(pdf_bytes("SYNTHETIC fixture evidence."))
    assert discover_sources(root)[0]["status"] == "ready"


def test_directory_symlinks_cannot_admit_external_documents(tmp_path):
    root = tmp_path / "data/Clinical"
    root.mkdir(parents=True)
    outside = tmp_path / "external.pdf"
    outside.write_bytes(pdf_bytes("EXTERNAL fixture."))
    (root / "paper.pdf").symlink_to(outside)
    entries = discover_sources(root.parent)
    assert entries[0]["status"] == "failed"
    assert load_documents(entries, "clinical")[0] == []


@pytest.mark.parametrize("scope", ["Retail", "SEC", "Cisco"])
def test_selecting_excluded_subtree_does_not_erase_its_assignment(tmp_path, scope):
    path = tmp_path / "data" / scope / "Clinical" / "paper.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(pdf_bytes("EXCLUDED fixture."))
    for root in (path.parent.parent, path.parent, path):
        entries = discover_sources(root)
        assert entries[0]["status"] == "excluded"
        assert load_documents(entries, "clinical")[0] == []


def test_local_only_catalog_selection_is_explicit_and_keeps_references(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "Agilent.txt").write_text("https://www.agilent.com/synthetic-fixture")
    strict = discover_sources(root)
    local = discover_sources(root, local_only=True)
    assert any(e["status"] == "pending_download" for e in strict)
    url = next(e for e in local if e.get("url"))
    assert url["status"] == "reference_only" and "deferred" in url["reason"]
    assert url["parent_source_id"] in {e["source_id"] for e in local if e["topic"] == "source_catalog"}


def test_index_backup_failure_preserves_existing_snapshot(tmp_path):
    path = tmp_path / "data/Clinical/paper.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(pdf_bytes("SYNTHETIC fixture evidence."))
    documents, _ = load_documents(discover_sources(path.parent), "clinical")
    index = SpecialistIndex("clinical", Settings(data_dir=tmp_path / "state"))
    index.replace(documents)
    previous = index.path.read_bytes()
    (index.path.parent / "backups").write_text("fixture blocking backup creation")
    with pytest.raises(OSError):
        index.replace(documents)
    assert index.path.read_bytes() == previous


def test_cli_empty_guard_also_applies_after_chunk_filtering(tmp_path):
    index = SpecialistIndex("clinical", Settings(data_dir=tmp_path))
    metadata = {"agent": "clinical", "source_id": "synthetic-fixture", "page": 1}
    index.replace([Document(page_content="SYNTHETIC clinical evidence", metadata=metadata)])
    previous = index.path.read_bytes()
    with pytest.raises(ValueError, match="index preserved"):
        index.replace([Document(page_content="the and of", metadata=metadata)], allow_empty=False)
    assert index.path.read_bytes() == previous


def test_extracted_cli_year_filters_backup_and_answers_without_originals(tmp_path):
    root = tmp_path / "data"
    files = {"Clinical/Diabetes/paper.pdf": "SYNTHETIC diabetes recruitment workflow.",
             "FoodSecurity/SOFI-2023.pdf": "SYNTHETIC hunger prices regional indicators in 2023.",
             "FoodSecurity/SOFI-2024.pdf": "SYNTHETIC hunger prices affordability indicators in 2024."}
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf_bytes(text))
    (root / "Agilent.txt").write_text("https://www.agilent.com/synthetic-fixture")
    write_csv(root / "Clinical/ctg-studies_fixture.csv")
    state, manifest = tmp_path / "state", tmp_path / "manifest.json"
    run_cli("discover", root, "--manifest", manifest, "--local-only")
    for agent in ("clinical", "corporate_responsibility"):
        assert run_cli("--data-dir", state, "ingest", agent, "--manifest", manifest)["chunks"]
    snapshot = state / "indexes/clinical/vectors.json"
    original = snapshot.read_bytes()
    run_cli("--data-dir", state, "ingest", "clinical", "--manifest", manifest)
    backup = next((snapshot.parent / "backups").glob("vectors-*.json"))
    assert backup.read_bytes() == original
    # Query processes must use saved indexes even when original inputs move.
    root.rename(tmp_path / "moved-inputs")
    clinical = run_cli("--data-dir", state, "ask", "clinical", "diabetes recruitment")
    assert clinical["status"] == "evidence_found"
    assert clinical["citations"][0]["source"] == "data/Clinical/Diabetes/paper.pdf"
    for year in (2023, 2024):
        response = run_cli("--data-dir", state, "ask", "corporate_responsibility", "hunger prices", "--report-year", year)
        assert response["status"] == "evidence_found"
        assert {c["report_year"] for c in response["citations"]} == {year}
        assert all(c["source"] == f"data/FoodSecurity/SOFI-{year}.pdf" and c["page"] == 1 for c in response["citations"])
    assert snapshot.read_bytes() == original
    assert run_cli("--data-dir", state, "ask", "corporate_responsibility", "hunger", "--report-year", 2022)["status"] == "insufficient_evidence"
