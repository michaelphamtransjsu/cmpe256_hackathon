"""CSV fixtures mirror observed CTG headers; all row contents are synthetic."""

import csv
from datetime import date
import hashlib
from pathlib import Path
import sqlite3

import pytest

from shared.sources import discover_sources, inspect_archive, load_documents
from shared.trial_csv import load_trial_csvs
from shared.trials import TrialRecord, TrialStore
from test_cli import run_cli, zip_bytes


HEADERS = [
    "NCT Number", "Study Title", "Study URL", "Acronym", "Study Status", "Brief Summary",
    "Study Results", "Conditions", "Interventions", "Primary Outcome Measures",
    "Secondary Outcome Measures", "Other Outcome Measures", "Sponsor", "Collaborators",
    "Sex", "Age", "Phases", "Enrollment", "Funder Type", "Study Type", "Study Design",
    "Other IDs", "Start Date", "Primary Completion Date", "Completion Date", "First Posted",
    "Results First Posted", "Last Update Posted", "Locations", "Study Documents",
]


def write_csv(path: Path, rows=None, headers=HEADERS):
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {"NCT Number": "NCT00000001", "Study Title": "SYNTHETIC diabetes trial fixture",
            "Study URL": "https://clinicaltrials.gov/study/NCT00000001"}
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows([base] if rows is None else [{**base, **row} for row in rows])
    return path


def test_actual_header_mapping_missing_values_multiline_and_large_fields(tmp_path):
    path = write_csv(tmp_path / "data/Clinical/ctg-studies_fixture.csv", [{
        "Conditions": "Synthetic diabetes|Synthetic comorbidity",
        "Interventions": "DRUG: synthetic fixture|OTHER: synthetic control",
        "Study Status": "RECRUITING", "Phases": "PHASE1|PHASE2", "Sponsor": "Fixture sponsor",
        "Brief Summary": "SYNTHETIC first line\n" + "fixture " * 25000,
        "Last Update Posted": "2024-02-29", "Enrollment": "15", "Primary Outcome Measures": "Fixture outcome",
    }, {"NCT Number": "NCT00000002", "Study URL": "", "Last Update Posted": "2024-02"}])
    entries = discover_sources(path.parent.parent)
    assert load_documents(entries, "clinical")[0] == []
    first, missing = load_trial_csvs(entries)
    assert first.trial_id == "NCT00000001"
    assert first.title == "SYNTHETIC diabetes trial fixture"
    assert first.conditions == ["Synthetic diabetes", "Synthetic comorbidity"]
    assert first.interventions == ["DRUG: synthetic fixture", "OTHER: synthetic control"]
    assert first.phase == "PHASE1|PHASE2" and first.sponsor == "Fixture sponsor"
    assert first.status == "RECRUITING" and first.summary.startswith("SYNTHETIC first line\n")
    assert first.updated_at == date(2024, 2, 29)
    assert first.metadata["csv_columns"]["Enrollment"] == "15"
    origin = first.metadata["csv_provenance"][0]
    assert origin["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert origin["original_member_path"] == "data/Clinical/ctg-studies_fixture.csv"
    assert origin["record_number"] == 2 and origin["line_start"] == 2 and origin["line_end"] == 3
    assert str(path.absolute()) == origin["local_path"]
    assert first.citations[0].endswith("#record=2")
    assert missing.source.endswith("#record=3")
    assert missing.conditions == missing.interventions == []
    assert missing.phase is missing.sponsor is missing.status is missing.summary is missing.updated_at is None
    assert missing.metadata["csv_columns"]["Last Update Posted"] == "2024-02"
    store = TrialStore(tmp_path / "trials.sqlite3")
    store.import_records([first, missing])
    assert TrialStore(store.path).get(first.trial_id) == first


def test_duplicate_csv_rows_merge_origins_and_import_is_idempotent(tmp_path):
    root = tmp_path / "data/Clinical"
    write_csv(root / "ctg-studies_first.csv")
    write_csv(root / "ctg-studies_second.csv")
    records = load_trial_csvs(discover_sources(root))
    assert len(records) == 1 and len(records[0].metadata["csv_provenance"]) == 2
    assert len([c for c in records[0].citations if "#record=" in c]) == 2
    store = TrialStore(tmp_path / "trials.sqlite3")
    assert store.import_records(records)["imported"] == 1
    assert store.import_records(records)["skipped"] == 1
    assert store.get(records[0].trial_id) == records[0]


def test_separate_csv_imports_add_origins_without_losing_prior_attribution(tmp_path):
    first = write_csv(tmp_path / "Clinical/ctg-studies_first.csv")
    second = write_csv(first.with_name("ctg-studies_second.csv"))
    store = TrialStore(tmp_path / "trials.sqlite3")
    store.import_records(load_trial_csvs(discover_sources(first)))
    result = store.import_records(load_trial_csvs(discover_sources(second)))
    assert result["replaced"] == 0
    assert len(store.get("NCT00000001").metadata["csv_provenance"]) == 2
    assert store.import_records(load_trial_csvs(discover_sources(first)))["skipped"] == 1


def test_conflicting_csv_duplicates_are_rejected(tmp_path):
    first = write_csv(tmp_path / "Clinical/ctg-studies_first.csv")
    write_csv(first.with_name("ctg-studies_second.csv"), [{"Study Title": "Conflicting synthetic title"}])
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        load_trial_csvs(discover_sources(first.parent))


@pytest.mark.parametrize("changes", [
    {"NCT Number": "bad"}, {"Study Title": " "}, {"Study URL": "https://www.sec.gov/fixture"},
    {"Study URL": "https://clinicaltrials.gov/study/NCT00000002"},
    {"Last Update Posted": "2024-02-30"}, {"Last Update Posted": "yesterday"},
])
def test_invalid_rows_are_rejected_before_writes(tmp_path, changes):
    path = write_csv(tmp_path / "Clinical/ctg-studies_fixture.csv", [{}, changes])
    with pytest.raises(ValueError):
        load_trial_csvs(discover_sources(path))


@pytest.mark.parametrize("content", [
    "Trial ID,Title\nNCT00000001,fixture\n",
    "NCT Number,Study Title,Study Title\nNCT00000001,fixture,fixture\n",
    "NCT Number,Study Title\nNCT00000001\n",
    "NCT Number,Study Title\nNCT00000001,fixture,extra\n",
    'NCT Number,Study Title\nNCT00000001,"unclosed\n',
    "NCT Number,Study Title\n",
])
def test_invalid_csv_structure(tmp_path, content):
    path = tmp_path / "Clinical/ctg-studies_fixture.csv"
    path.parent.mkdir()
    path.write_text(content)
    with pytest.raises((ValueError, csv.Error)):
        load_trial_csvs(discover_sources(path))


def test_csv_hash_assignment_and_symlink_checks(tmp_path):
    path = write_csv(tmp_path / "Clinical/ctg-studies_fixture.csv")
    entries = discover_sources(path)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        load_trial_csvs(entries)
    entries = discover_sources(path)
    entries[0]["original_member_path"] = "Retail/ctg-studies_fixture.csv"
    with pytest.raises(ValueError, match="Unassigned"):
        load_trial_csvs(entries)
    link = path.with_name("ctg-studies_link.csv")
    link.symlink_to(path)
    entries = discover_sources(link)
    with pytest.raises(ValueError, match="Missing local|symlink|No assigned"):
        load_trial_csvs(entries)
    entries = discover_sources(path)
    entries[0]["local_path"] = str(link)
    with pytest.raises(ValueError, match="symlinks"):
        load_trial_csvs(entries)


def test_conflicts_are_atomic_and_explicit_replacement_has_recoverable_backup(tmp_path):
    store = TrialStore(tmp_path / "trials.sqlite3")
    old = TrialRecord(trial_id="FIXTURE-1", title="Old fixture", source="synthetic fixture")
    new = old.model_copy(update={"title": "New fixture"})
    other = old.model_copy(update={"trial_id": "FIXTURE-2"})
    store.import_records([old])
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        store.import_records([old, new])
    with pytest.raises(ValueError, match="database preserved"):
        store.import_records([other, new])
    assert store.get("FIXTURE-2") is None and store.get(old.trial_id) == old
    report = store.import_records([other, new], replace_existing=True)
    assert report["replaced"] == 1
    backup = TrialStore(report["backup"])
    assert backup.get(old.trial_id) == old and backup.get(other.trial_id) is None
    assert store.get(new.trial_id) == new and store.get(other.trial_id) == other


def test_write_failure_rolls_back_entire_batch(tmp_path):
    store = TrialStore(tmp_path / "trials.sqlite3")
    with sqlite3.connect(store.path) as connection:
        connection.execute("""CREATE TRIGGER fail_fixture BEFORE INSERT ON clinical_trials
                              WHEN NEW.trial_id = 'FAIL' BEGIN SELECT RAISE(ABORT, 'fixture failure'); END""")
    with pytest.raises(sqlite3.IntegrityError):
        store.import_records([TrialRecord(trial_id=id, title="Fixture", source="synthetic") for id in ["OK", "FAIL"]])
    assert store.get("OK") is None


def test_manifest_csv_cli_import_search_and_restart(tmp_path):
    path = write_csv(tmp_path / "data/Clinical/ctg-studies_fixture.csv")
    manifest = tmp_path / "manifest.json"
    target = tmp_path / "state"
    run_cli("discover", path.parent.parent, "--manifest", manifest, "--local-only")
    assert run_cli("--data-dir", target, "trials-import", "--manifest", manifest)["imported"] == 1
    assert run_cli("--data-dir", target, "trials-import", "--manifest", manifest)["skipped"] == 1
    result = run_cli("--data-dir", target, "ask", "clinical", "diabetes")
    assert result["trial_records"][0]["metadata"]["csv_provenance"][0]["sha256"]
    assert result["citations"][0]["source"].startswith("https://clinicaltrials.gov/study/")
    assert run_cli("--data-dir", target, "ask", "corporate_responsibility", "diabetes")["trial_records"] == []
    assert run_cli("--data-dir", target, "trials-search", "diabetes")[0]["trial_id"] == "NCT00000001"


def test_csv_zip_workflow_retains_archive_lineage(tmp_path):
    path = write_csv(tmp_path / "fixture/ctg-studies_fixture.csv")
    archive = tmp_path / "Clinical.zip"
    archive.write_bytes(zip_bytes({path.name: path.read_bytes(), path.name + ":Zone.Identifier": b"metadata"}))
    entries = inspect_archive(archive, tmp_path / "extracted")
    assert not any(":Zone.Identifier" in entry["original_member_path"] for entry in entries)
    records = load_trial_csvs(entries)
    assert records[0].citations[0].startswith("Clinical.zip!/")
