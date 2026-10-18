"""End-to-end CLI validation with explicitly synthetic evidence, never uploads."""

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def pdf_bytes(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    content = DecodedStreamObject()
    content.set_data(f"BT /F1 12 Tf 40 700 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = content
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def zip_bytes(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return output.getvalue()


def run_cli(*args: object, success: bool = True) -> dict | list | str:
    arguments = list(map(str, args))
    common = []
    if arguments[:1] == ["--data-dir"]:
        common, arguments = arguments[:2], arguments[2:]
    if arguments[:1] == ["ask"]:
        command = [sys.executable, "-m", f"agents.{arguments[1]}", *arguments[2:], *common]
    else:
        command = [sys.executable, "ingest.py", *common, *arguments]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if success:
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert result.returncode != 0
    return result.stderr


def test_both_agents_archive_ingestion_and_restart(tmp_path: Path) -> None:
    bundle = tmp_path / "Hackathon_export.zip"
    original = zip_bytes({
        "Clinical.zip": zip_bytes({"trials.txt": b"SYNTHETIC: Diabetes trial operations include recruitment monitoring."}),
        "FoodSecurity.zip": zip_bytes({
            "SOFI-2023.pdf": pdf_bytes("SYNTHETIC 2023: Hunger monitoring requires regional indicators."),
            "SOFI-2024.pdf": pdf_bytes("SYNTHETIC 2024: Hunger reporting includes affordability indicators."),
        }),
        "Retail.zip": zip_bytes({"transactions.txt": b"DO NOT INDEX MARKET BASKET"}),
    })
    bundle.write_bytes(original)
    manifest = tmp_path / "manifest.json"
    data = tmp_path / "data"
    run_cli("inspect", bundle, "--output", tmp_path / "extracted", "--manifest", manifest)
    assert run_cli("--data-dir", data, "ingest", "clinical", "--manifest", manifest)["chunks"] == 1
    assert run_cli("--data-dir", data, "ingest", "corporate_responsibility", "--manifest", manifest)["chunks"] == 2
    clinical = run_cli("--data-dir", data, "ask", "clinical", "diabetes recruitment")
    corporate = run_cli("--data-dir", data, "ask", "corporate_responsibility", "hunger indicators")
    assert clinical["status"] == corporate["status"] == "evidence_found"
    assert len(corporate["citations"]) == 2
    assert all(c["page"] == 1 and "FoodSecurity.zip" in c["source"] for c in corporate["citations"])
    assert all("Clinical.zip" in c["source"] for c in clinical["citations"])
    scoped = run_cli("--data-dir", data, "ask", "corporate_responsibility", "hunger indicators", "--report-year", 2023)
    assert scoped["report_year"] == 2023
    assert len(scoped["citations"]) == 1
    assert scoped["citations"][0]["report_year"] == 2023
    assert run_cli("--data-dir", data, "ask", "clinical", "hunger affordability")["status"] == "insufficient_evidence"
    assert bundle.read_bytes() == original
    assert run_cli("--data-dir", data, "ingest", "clinical", "--manifest", manifest)["chunks"] == 1
    # Each invocation is a new interpreter, exercising actual on-disk restoration.
    assert run_cli("--data-dir", data, "ask", "clinical", "diabetes")["citations"]
    assert "already exists" in run_cli("inspect", bundle, "--output", tmp_path / "extracted", "--manifest", manifest, success=False)


def test_trial_import_validation_and_clinical_query(tmp_path: Path) -> None:
    records = tmp_path / "trials.json"
    records.write_text(json.dumps([{"trial_id": "FIXTURE-1", "title": "Synthetic diabetes study", "source": "fixture-only.txt", "conditions": ["diabetes"]}]))
    data = tmp_path / "data"
    assert run_cli("--data-dir", data, "trials-import", records)["imported"] == 1
    assert run_cli("--data-dir", data, "ask", "clinical", "diabetes")["trial_records"][0]["trial_id"] == "FIXTURE-1"
    assert run_cli("--data-dir", data, "ask", "corporate_responsibility", "diabetes")["trial_records"] == []
    records.write_text(json.dumps([{"trial_id": "BAD", "title": "Missing source"}]))
    assert "source" in run_cli("--data-dir", data, "trials-import", records, success=False)
    assert len(run_cli("--data-dir", data, "trials-search")) == 1


def test_failed_parse_preserves_existing_index(tmp_path: Path) -> None:
    bundle = tmp_path / "Clinical.zip"
    bundle.write_bytes(zip_bytes({"trial.txt": b"Synthetic diabetes evidence."}))
    manifest = tmp_path / "manifest.json"
    data = tmp_path / "data"
    run_cli("inspect", bundle, "--output", tmp_path / "output", "--manifest", manifest)
    run_cli("--data-dir", data, "ingest", "clinical", "--manifest", manifest)
    before = (data / "indexes" / "clinical" / "vectors.json").read_bytes()
    entries = json.loads(manifest.read_text())["entries"]
    path = Path(next(row["local_path"] for row in entries if row["format"] == "txt"))
    path.write_text("Changed document must fail its saved checksum.")
    assert "preserved" in run_cli("--data-dir", data, "ingest", "clinical", "--manifest", manifest, success=False)
    assert (data / "indexes" / "clinical" / "vectors.json").read_bytes() == before


def test_agilent_attach_ingest_and_repair(tmp_path: Path) -> None:
    bundle = tmp_path / "Clinical.zip"
    bundle.write_bytes(zip_bytes({"Agilent.txt": b"https://example.org/agilent-study.txt"}))
    manifest = tmp_path / "manifest.json"
    data = tmp_path / "data"
    report = run_cli("inspect", bundle, "--output", tmp_path / "output", "--manifest", manifest)
    assert report["statuses"]["pending_download"] == 1
    assert "unresolved" in run_cli("--data-dir", data, "ingest", "clinical", "--manifest", manifest, success=False)
    entry = next(row for row in json.loads(manifest.read_text())["entries"] if row["status"] == "pending_download")
    local = tmp_path / "manually-downloaded.txt"
    local.write_text("SYNTHETIC spectroscopy evidence.")
    run_cli("attach-source", "--manifest", manifest, "--source-id", entry["source_id"], "--path", local)
    run_cli("--data-dir", data, "ingest", "clinical", "--manifest", manifest)
    answer = run_cli("--data-dir", data, "ask", "clinical", "spectroscopy")
    assert answer["citations"][0]["source"] == "https://example.org/agilent-study.txt"
    local.write_text("SYNTHETIC updated spectroscopy evidence.")
    assert "preserved" in run_cli("--data-dir", data, "ingest", "clinical", "--manifest", manifest, success=False)
    # Explicit re-association records the intentionally replaced download's hash.
    run_cli("attach-source", "--manifest", manifest, "--source-id", entry["source_id"], "--path", local)
    assert run_cli("--data-dir", data, "ingest", "clinical", "--manifest", manifest)["chunks"] == 1


def test_missing_archive_reports_nonzero_and_records_missing_input(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    result = subprocess.run([
        sys.executable, "ingest.py", "inspect", str(tmp_path / "missing.zip"),
        "--output", str(tmp_path / "output"), "--manifest", str(manifest),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert json.loads(result.stdout)["statuses"] == {"missing": 1}
    assert json.loads(manifest.read_text())["entries"][0]["status"] == "missing"


def test_unresolved_archive_or_assigned_format_blocks_partial_index(tmp_path: Path) -> None:
    from shared.sources import inspect_archive

    bundle = tmp_path / "Clinical.zip"
    bundle.write_bytes(zip_bytes({"trial.txt": b"SYNTHETIC diabetes evidence.", "unknown.docx": b"unsupported source"}))
    manifest = tmp_path / "manifest.json"
    entries = inspect_archive(bundle, tmp_path / "output")
    manifest.write_text(json.dumps({"version": 1, "entries": entries}))
    data = tmp_path / "data"
    assert "unresolved" in run_cli("--data-dir", data, "ingest", "clinical", "--manifest", manifest, success=False)
    assert not (data / "indexes" / "clinical" / "vectors.json").exists()
    # Even unattributed failed bundles may hide assigned documents.
    entries = [row for row in entries if row["format"] != "docx"]
    entries.append({"source_id": "bad-bundle", "agent": None, "format": "zip", "status": "failed", "topic": "source_bundle"})
    manifest.write_text(json.dumps({"version": 1, "entries": entries}))
    assert "unresolved" in run_cli("--data-dir", data, "ingest", "clinical", "--manifest", manifest, success=False)
    assert not (data / "indexes" / "clinical" / "vectors.json").exists()


def test_attached_html_is_passive_evidence_with_url_citation(tmp_path: Path) -> None:
    bundle = tmp_path / "Clinical.zip"
    bundle.write_bytes(zip_bytes({"Agilent.txt": b"https://www.agilent.com/example-page"}))
    manifest = tmp_path / "manifest.json"
    data = tmp_path / "data"
    run_cli("inspect", bundle, "--output", tmp_path / "output", "--manifest", manifest)
    entry = next(row for row in json.loads(manifest.read_text())["entries"] if row["status"] == "pending_download")
    html = tmp_path / "agilent.html"
    html.write_text('<html><head><title>Synthetic laboratory source</title></head><body><nav>irrelevant navigation</nav><main><h1>Spectroscopy</h1><p>SYNTHETIC spectroscopy evidence for a laboratory workflow.</p><script>document.write("EXECUTED");</script></main></body></html>')
    run_cli("attach-source", "--manifest", manifest, "--source-id", entry["source_id"], "--path", html)
    assert run_cli("--data-dir", data, "ingest", "clinical", "--manifest", manifest)["chunks"] == 1
    answer = run_cli("--data-dir", data, "ask", "clinical", "spectroscopy")
    citation = answer["citations"][0]
    assert citation["source"] == entry["url"] and citation["page"] is None
    assert "Spectroscopy" in citation["excerpt"]
    assert "EXECUTED" not in citation["excerpt"] and "navigation" not in citation["excerpt"]


def test_ingestion_cli_does_not_expose_an_answering_command() -> None:
    help_result = subprocess.run(
        [sys.executable, "ingest.py", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "inspect" in help_result.stdout and "trials-import" in help_result.stdout
    assert "ask" not in help_result.stdout
    answer_result = subprocess.run(
        [sys.executable, "ingest.py", "ask", "clinical", "diabetes"],
        capture_output=True, text=True, timeout=30,
    )
    assert answer_result.returncode == 2
    assert "invalid choice" in answer_result.stderr


@pytest.mark.parametrize("agent", ["clinical", "corporate_responsibility"])
def test_each_agent_has_a_standalone_query_cli(agent: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", f"agents.{agent}", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "question" in result.stdout.casefold()
    assert "--data-dir" in result.stdout
    assert "--max-sources" in result.stdout
    assert "inspect" not in result.stdout
    assert "RuntimeWarning" not in result.stderr


@pytest.mark.parametrize("agent", ["clinical", "corporate_responsibility"])
def test_answering_import_does_not_load_ingestion(agent: str) -> None:
    # A new interpreter prevents earlier ingestion tests from masking imports.
    code = (
        "import importlib, json, sys; "
        f"importlib.import_module('agents.{agent}'); "
        "print(json.dumps(sorted(set(sys.modules) & "
        "{'ingest', 'shared.sources', 'shared.html_evidence'})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []
