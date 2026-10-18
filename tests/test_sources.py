"""Security and assignment boundaries for source ingestion."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import stat
import zipfile

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from shared.sources import ArchiveLimits, discover_sources, inspect_archive, load_documents


def bundle(members: dict[str, bytes], *, compression: int = zipfile.ZIP_STORED) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return output.getvalue()


def test_nested_assignments_exclusions_catalog_and_originals(tmp_path: Path) -> None:
    original = bundle({
        "Clinical.zip": bundle({"trial.txt": b"Trial NCT123 is recruiting.", "Cisco/annual.txt": b"Excluded"}),
        "FoodSecurity.zip": bundle({"SOFI-2023.pdf": b"pdf evidence", "other.txt": b"unassigned"}),
        "assignment.pdf": b"project instructions are evidence only",
        "Agilent.txt": b"https://example.org/agilent.pdf\nhttps://example.org/agilent.pdf\n",
        "Retail.zip": bundle({"transactions.txt": b"excluded"}),
        "notes.txt": b"unknown",
    })
    archive = tmp_path / "Hackathon_export.zip"
    archive.write_bytes(original)
    entries = inspect_archive(archive, tmp_path / "extracted")
    clinical = next(item for item in entries if item["original_member_path"].endswith("trial.txt"))
    assert clinical["agent"] == "clinical"
    assert clinical["status"] == "ready"
    assert clinical["sha256"] == hashlib.sha256(b"Trial NCT123 is recruiting.").hexdigest()
    sofi = next(item for item in entries if item["original_member_path"].endswith("SOFI-2023.pdf"))
    assert sofi["agent"] == "corporate_responsibility"
    assert next(item for item in entries if item["original_member_path"].endswith("assignment.pdf"))["status"] == "reference_only"
    urls = [item for item in entries if item["status"] == "pending_download"]
    assert len(urls) == 1 and urls[0]["agent"] == "clinical"
    excluded = [item for item in entries if item["status"] == "excluded"]
    assert len(excluded) == 2 and all(item["local_path"] is None for item in excluded)
    assert next(item for item in entries if item["original_member_path"].endswith("other.txt"))["local_path"] is None
    assert next(item for item in entries if item["original_member_path"].endswith("notes.txt"))["local_path"] is None
    assert archive.read_bytes() == original
    assert all(key in clinical for key in ("source_id", "agent", "topic", "format", "status", "sha256", "original_member_path", "local_path"))


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "folder/../../escape.txt", "C:\\escape.txt", "folder\\escape.txt"])
def test_rejects_unsafe_archive_paths(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "Clinical.zip"
    archive.write_bytes(bundle({name: b"unsafe", "safe.txt": b"safe"}))
    entries = inspect_archive(archive, tmp_path / "output")
    assert any(item.get("reason") == "unsafe_archive_path" for item in entries)
    assert next(item for item in entries if item["original_member_path"].endswith("safe.txt"))["status"] == "ready"
    assert not (tmp_path / "escape.txt").exists()


def test_rejects_symlinks_and_unsupported_executables(tmp_path: Path) -> None:
    archive = tmp_path / "Clinical.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        link = zipfile.ZipInfo("link.txt")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        handle.writestr(link, "/etc/passwd")
        handle.writestr("run.sh", "echo should never execute")
    entries = inspect_archive(archive, tmp_path / "output")
    assert next(item for item in entries if item["original_member_path"].endswith("link.txt"))["status"] == "failed"
    assert next(item for item in entries if item["original_member_path"].endswith("run.sh"))["status"] == "unsupported"
    assert list((tmp_path / "output").rglob("*.sh")) == []


def test_size_ratio_depth_and_global_budget_limits(tmp_path: Path) -> None:
    archive = tmp_path / "Clinical.zip"
    archive.write_bytes(bundle({"large.txt": b"a" * 10000}, compression=zipfile.ZIP_DEFLATED))
    entries = inspect_archive(archive, tmp_path / "ratio", limits=ArchiveLimits(max_compression_ratio=2))
    assert any(item.get("reason") == "archive_compression_ratio_limit_exceeded" for item in entries)
    entries = inspect_archive(archive, tmp_path / "size", limits=ArchiveLimits(max_member_bytes=10))
    assert any(item.get("reason") == "archive_member_size_limit_exceeded" for item in entries)
    archive.write_bytes(bundle({"first.txt": b"a" * 10, "second.txt": b"b" * 10}))
    entries = inspect_archive(archive, tmp_path / "budget", limits=ArchiveLimits(max_total_bytes=15))
    assert any(item.get("reason") == "archive_total_size_limit_exceeded" for item in entries)
    archive.write_bytes(bundle({"nested.zip": bundle({"deeper.zip": bundle({"trial.txt": b"evidence"})})}))
    entries = inspect_archive(archive, tmp_path / "depth", limits=ArchiveLimits(max_depth=1))
    assert any(item.get("reason") == "archive_nesting_limit_exceeded" for item in entries)


def test_extraction_never_overwrites_different_files(tmp_path: Path) -> None:
    archive = tmp_path / "Clinical.zip"
    archive.write_bytes(bundle({"trial.txt": b"evidence"}))
    output = tmp_path / "output"
    first = inspect_archive(archive, output)
    path = Path(next(item["local_path"] for item in first if item["format"] == "txt"))
    assert next(item for item in inspect_archive(archive, output) if item["format"] == "txt")["status"] == "ready"
    path.write_text("user work", encoding="utf-8")
    repeated = inspect_archive(archive, output)
    assert next(item for item in repeated if item["format"] == "txt")["status"] == "failed"
    assert path.read_text() == "user work"


def test_symlink_extraction_parent_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "Clinical.zip"
    archive.write_bytes(bundle({"sub/trial.txt": b"evidence"}))
    outside = tmp_path / "outside"
    outside.mkdir()
    output = tmp_path / "output"
    (output / "Clinical.zip.contents").mkdir(parents=True)
    (output / "Clinical.zip.contents" / "sub").symlink_to(outside, target_is_directory=True)
    entries = inspect_archive(archive, output)
    assert any(item.get("reason") == "unsafe_output_directory" for item in entries)
    assert not (outside / "trial.txt").exists()


def test_discovery_does_not_assign_arbitrary_agent_directories(tmp_path: Path) -> None:
    (tmp_path / "clinical").mkdir()
    (tmp_path / "clinical" / "untrusted.txt").write_text("unassigned", encoding="utf-8")
    (tmp_path / "Clinical.zip.contents").mkdir()
    (tmp_path / "Clinical.zip.contents" / "trial.txt").write_text("assigned", encoding="utf-8")
    (tmp_path / "FoodSecurity.zip.contents").mkdir()
    (tmp_path / "FoodSecurity.zip.contents" / "SOFI-2024.pdf").write_bytes(b"pdf evidence")
    entries = discover_sources(tmp_path)
    assert next(item for item in entries if item["original_member_path"].endswith("untrusted.txt"))["agent"] is None
    assert next(item for item in entries if item["original_member_path"].endswith("trial.txt"))["agent"] == "clinical"
    assert next(item for item in entries if item["original_member_path"].endswith("SOFI-2024.pdf"))["agent"] == "corporate_responsibility"


def test_loading_text_preserves_source_and_agent_boundaries(tmp_path: Path) -> None:
    archive = tmp_path / "Clinical.zip"
    archive.write_bytes(bundle({"trial.txt": b"Trial evidence.", "Google/other.txt": b"excluded"}))
    entries = inspect_archive(archive, tmp_path / "output")
    documents, updated = load_documents(entries, "clinical")
    assert len(documents) == 1
    assert documents[0].metadata["source"].endswith("Clinical.zip!/trial.txt")
    assert documents[0].metadata["page"] == 1
    assert documents[0].metadata["agent"] == "clinical"
    assert next(item for item in updated if item["format"] == "txt" and item["agent"] == "clinical")["status"] == "indexed"
    assert next(item for item in entries if item["format"] == "txt" and item["agent"] == "clinical")["status"] == "ready"
    assert load_documents(entries, "corporate_responsibility")[0] == []


def test_hash_mismatch_and_empty_documents_do_not_enter_index(tmp_path: Path) -> None:
    archive = tmp_path / "Clinical.zip"
    archive.write_bytes(bundle({"changed.txt": b"original", "empty.txt": b"   "}))
    entries = inspect_archive(archive, tmp_path / "output")
    changed = next(item for item in entries if item["original_member_path"].endswith("changed.txt"))
    Path(changed["local_path"]).write_text("tampered", encoding="utf-8")
    documents, updated = load_documents(entries, "clinical")
    assert documents == []
    assert all(item["status"] == "failed" for item in updated if item["format"] == "txt")
    assert any("source_hash_mismatch" in item.get("reason", "") for item in updated)


def test_missing_input_is_manifested_without_inventing_sources(tmp_path: Path) -> None:
    entries = inspect_archive(tmp_path / "Hackathon_export.zip", tmp_path / "output")
    assert len(entries) == 1 and entries[0]["status"] == "missing"
    assert entries[0]["sha256"] is None
    assert discover_sources(tmp_path / "missing")[0]["status"] == "missing"


def test_pdf_pages_have_real_page_citations_and_scope_is_never_indexed(tmp_path: Path) -> None:
    writer = PdfWriter()
    for number in (1, 2):
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 72 720 Td (Food report page {number}) Tj ET".encode())
        page[NameObject("/Contents")] = content
    output = io.BytesIO()
    writer.write(output)
    archive = tmp_path / "FoodSecurity.zip"
    archive.write_bytes(bundle({"SOFI-2024.pdf": output.getvalue(), "Project Scope.pdf": output.getvalue()}))
    entries = inspect_archive(archive, tmp_path / "output")
    documents, updated = load_documents(entries, "corporate_responsibility")
    assert len(documents) == 2
    assert [document.metadata["page"] for document in documents] == [1, 2]
    assert documents[1].page_content == "Food report page 2"
    assert next(item for item in updated if item["topic"] == "project_scope")["status"] == "reference_only"


def test_agilent_catalog_cannot_admit_excluded_sec_sources(tmp_path: Path) -> None:
    catalog = tmp_path / "Agilent.txt"
    catalog.write_text("https://www.sec.gov/annual.pdf\nhttps://example.org/clinical.pdf", encoding="utf-8")
    entries = discover_sources(catalog)
    assert next(item for item in entries if item.get("url", "").startswith("https://www.sec.gov"))["status"] == "excluded"
    assert len([item for item in entries if item["status"] == "pending_download"]) == 1


def test_attached_agilent_reference_uses_original_url_and_verified_bytes(tmp_path: Path) -> None:
    catalog = tmp_path / "Agilent.txt"
    catalog.write_text("https://example.org/study.txt", encoding="utf-8")
    entries = discover_sources(catalog)
    source = next(item for item in entries if item["status"] == "pending_download")
    downloaded = tmp_path / "arbitrary-download-name.txt"
    downloaded.write_text("Verified Agilent study evidence.", encoding="utf-8")
    source.update(status="ready", local_path=str(downloaded), sha256=hashlib.sha256(downloaded.read_bytes()).hexdigest(), format="txt")
    documents, _ = load_documents(entries, "clinical")
    assert len(documents) == 1
    assert documents[0].metadata["source"] == "https://example.org/study.txt"


def test_malformed_archive_and_entry_count_are_reported(tmp_path: Path) -> None:
    archive = tmp_path / "Clinical.zip"
    archive.write_bytes(b"not a zip archive")
    assert inspect_archive(archive, tmp_path / "bad")[0]["status"] == "failed"
    archive.write_bytes(bundle({"first.txt": b"a", "second.txt": b"b"}))
    entries = inspect_archive(archive, tmp_path / "limited", limits=ArchiveLimits(max_entries=1))
    assert any(item.get("reason") == "archive_entry_limit_exceeded" for item in entries)


def test_malformed_nested_zip_is_not_duplicated_in_manifest(tmp_path: Path) -> None:
    archive = tmp_path / "Hackathon_export.zip"
    archive.write_bytes(bundle({"Clinical.zip": b"invalid nested archive"}))
    entries = inspect_archive(archive, tmp_path / "output")
    nested = [item for item in entries if item["original_member_path"].endswith("!/Clinical.zip")]
    assert len(nested) == 1 and nested[0]["status"] == "failed"


@pytest.mark.parametrize("original", [
    "unassigned.txt",
    "clinical/arbitrary.txt",
    "Clinical.zip!/assignment.pdf",
    "FoodSecurity.zip!/other.txt",
    "FoodSecurity.zip!/SOFI-2024.pdf",
    "Clinical.zip!/Cisco/annual.txt",
    "Clinical.zip!/Technology/annual.txt",
])
def test_loading_does_not_trust_claimed_ready_agent(tmp_path: Path, original: str) -> None:
    path = tmp_path / "download.txt"
    path.write_text("Evidence must still have an assigned source.", encoding="utf-8")
    entry = {
        "source_id": "test",
        "original_member_path": original,
        "agent": "clinical",
        "topic": "clinical_research",
        "format": "txt",
        "status": "ready",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "local_path": str(path),
    }
    documents, updated = load_documents([entry], "clinical")
    assert documents == []
    assert updated[0]["status"] in {"failed", "excluded"}


def test_forged_or_excluded_agilent_reference_is_blocked(tmp_path: Path) -> None:
    catalog = tmp_path / "Agilent.txt"
    catalog.write_text("https://example.org/study.txt", encoding="utf-8")
    entries = discover_sources(catalog)
    source = next(item for item in entries if item["status"] == "pending_download")
    downloaded = tmp_path / "download.txt"
    downloaded.write_text("Study evidence.", encoding="utf-8")
    source.update(status="ready", local_path=str(downloaded), sha256=hashlib.sha256(downloaded.read_bytes()).hexdigest(), format="txt")
    source["url"] = "https://www.sec.gov/study.txt"
    source["original_member_path"] = "Agilent.txt#url=https://www.sec.gov/study.txt"
    assert load_documents(entries, "clinical")[0] == []
    source["url"] = "https://example.org/not-listed.txt"
    source["original_member_path"] = "Agilent.txt#url=https://example.org/not-listed.txt"
    documents, updated = load_documents(entries, "clinical")
    assert documents == []
    assert next(item for item in updated if item.get("url"))["reason"] == "url_not_in_agilent_catalog"


def test_malformed_catalog_url_is_manifested_without_crashing(tmp_path: Path) -> None:
    catalog = tmp_path / "Agilent.txt"
    catalog.write_text("https://[invalid\nhttps:///missing-host.pdf\nhttps://example.org/valid.pdf", encoding="utf-8")
    entries = discover_sources(catalog)
    assert len([item for item in entries if item["status"] == "failed"]) == 2
    assert len([item for item in entries if item["status"] == "pending_download"]) == 1


def test_filesystem_symlink_directory_is_recorded_but_not_followed(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "secret.txt").write_text("never follow", encoding="utf-8")
    (root / "Clinical.zip.contents").symlink_to(elsewhere, target_is_directory=True)
    entries = discover_sources(root)
    assert len(entries) == 1
    assert entries[0]["status"] == "failed"
    assert entries[0]["local_path"] is None


def test_failed_checksum_can_recover_after_original_bytes_are_restored(tmp_path: Path) -> None:
    archive = tmp_path / "Clinical.zip"
    original = b"Original clinical trial evidence."
    archive.write_bytes(bundle({"trial.txt": original}))
    entries = inspect_archive(archive, tmp_path / "output")
    entry = next(item for item in entries if item["format"] == "txt")
    path = Path(entry["local_path"])
    path.write_text("Changed evidence.", encoding="utf-8")
    documents, failed = load_documents(entries, "clinical")
    assert documents == []
    assert next(item for item in failed if item["format"] == "txt")["status"] == "failed"
    path.write_bytes(original)
    documents, recovered = load_documents(failed, "clinical")
    assert len(documents) == 1
    assert next(item for item in recovered if item["format"] == "txt")["status"] == "indexed"
    assert documents[0].page_content == original.decode()


def test_missing_local_source_can_recover_after_file_is_restored(tmp_path: Path) -> None:
    archive = tmp_path / "Clinical.zip"
    original = b"Restorable source evidence."
    archive.write_bytes(bundle({"trial.txt": original}))
    entries = inspect_archive(archive, tmp_path / "output")
    path = Path(next(item["local_path"] for item in entries if item["format"] == "txt"))
    path.unlink()
    documents, missing = load_documents(entries, "clinical")
    assert documents == []
    assert next(item for item in missing if item["format"] == "txt")["status"] == "missing"
    path.write_bytes(original)
    documents, recovered = load_documents(missing, "clinical")
    assert len(documents) == 1
    assert next(item for item in recovered if item["format"] == "txt")["status"] == "indexed"


def test_failed_agilent_attachment_can_recover_with_original_bytes(tmp_path: Path) -> None:
    catalog = tmp_path / "Agilent.txt"
    catalog.write_text("https://example.org/study.txt", encoding="utf-8")
    entries = discover_sources(catalog)
    source = next(item for item in entries if item["status"] == "pending_download")
    downloaded = tmp_path / "download.txt"
    original = b"Verified Agilent study."
    downloaded.write_bytes(original)
    source.update(status="ready", local_path=str(downloaded), sha256=hashlib.sha256(original).hexdigest(), format="txt")
    downloaded.write_text("Changed download", encoding="utf-8")
    documents, failed = load_documents(entries, "clinical")
    assert documents == []
    assert next(item for item in failed if item.get("url"))["status"] == "failed"
    downloaded.write_bytes(original)
    documents, recovered = load_documents(failed, "clinical")
    assert len(documents) == 1
    assert documents[0].metadata["source"] == "https://example.org/study.txt"
    assert next(item for item in recovered if item.get("url"))["status"] == "indexed"


def test_retry_does_not_parse_failed_archives_or_bypass_exclusions(tmp_path: Path) -> None:
    archive = tmp_path / "Clinical.zip"
    archive.write_bytes(bundle({"trial.txt": b"Clinical evidence."}))
    entries = inspect_archive(archive, tmp_path / "output")
    archive_entry = entries[0]
    archive_entry.update(agent="clinical", status="failed", reason="malformed_archive")
    evidence = next(item for item in entries if item["format"] == "txt")
    evidence.update(original_member_path="Clinical.zip!/SEC/report.txt", status="failed")
    documents, updated = load_documents(entries, "clinical")
    assert documents == []
    assert updated[0]["status"] == "failed" and updated[0]["reason"] == "malformed_archive"
    assert next(item for item in updated if item["format"] == "txt")["status"] == "excluded"
