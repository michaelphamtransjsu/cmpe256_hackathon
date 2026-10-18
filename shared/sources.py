"""Bounded, non-executing source discovery and document ingestion.

The archive is evidence, never an installation script.  Extraction preserves the
original archive and writes only PDF/text evidence and source catalogs.  It never
fetches links, executes a member, or indexes an unassigned document.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any
from urllib.parse import urlparse
import zipfile


AGENTS = frozenset({"clinical", "corporate_responsibility"})
_TEXT_FORMATS = frozenset({"txt", "md"})
_EVIDENCE_FORMATS = _TEXT_FORMATS | {"pdf", "html", "htm"}
_EXCLUDED = frozenset({"cisco", "google", "sec", "retail", "technology", "tech"})
_SOFI_NAMES = frozenset({"sofi-2023.pdf", "sofi-2024.pdf"})
_URL = re.compile(r"https?://[^\s<>\"'\])}]+", re.IGNORECASE)


@dataclass(frozen=True)
class ArchiveLimits:
    """Safety limits apply across the entire nested archive, not per ZIP."""

    max_depth: int = 4
    max_entries: int = 10_000
    max_member_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 256 * 1024 * 1024
    max_compression_ratio: float = 250.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(original: str, *, data: bytes | None = None, **fields: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_id": hashlib.sha256(original.encode("utf-8")).hexdigest()[:24],
        "agent": None,
        "topic": "unassigned",
        "format": Path(original.split("!/")[-1]).suffix.lower().lstrip(".") or "unknown",
        "status": "reference_only",
        "sha256": hashlib.sha256(data).hexdigest() if data is not None else None,
        "original_member_path": original,
        "local_path": None,
    }
    result.update(fields)
    return result


def _classify(original: str) -> dict[str, Any]:
    """Only explicit assignment lineage grants membership in an agent index."""
    path = original.replace("!/", "/")
    tokens = set(re.findall(r"[a-z0-9]+", path.lower()))
    name = PurePosixPath(path).name.lower()
    parts = {part.lower() for part in PurePosixPath(path).parts}
    if name.endswith(":zone.identifier"):
        return {"status": "ignored", "topic": "windows_metadata"}
    if tokens & _EXCLUDED:
        return {"status": "excluded", "topic": "out_of_scope", "reason": "assigned_to_other_project_or_specialist"}
    if re.search(r"assignment|project[ _-]*scope|hackathon[ _-]*(brief|instructions|scope)", name):
        return {"status": "reference_only", "topic": "project_scope"}
    if name == "agilent.txt":
        return {"agent": "clinical", "status": "reference_only", "topic": "source_catalog"}
    relative = list(PurePosixPath(path).parts)
    if relative and relative[0].lower() == "data":
        relative = relative[1:]
    clinical_folder = len(relative) >= 2 and relative[0].lower() == "clinical"
    food_folder = len(relative) == 2 and relative[0].lower() == "foodsecurity"
    clinical_archive = bool(parts & {"clinical.zip", "clinical.zip.contents"})
    if ((clinical_folder and len(relative) == 2) or clinical_archive) and re.fullmatch(r"ctg-studies_[^/]+\.csv", name):
        return {"agent": "clinical", "status": "ready_for_import", "topic": "clinical_trials"}
    if clinical_folder and name.endswith(".pdf"):
        return {"agent": "clinical", "status": "ready", "topic": "clinical_research"}
    if food_folder and name in _SOFI_NAMES:
        return {"agent": "corporate_responsibility", "status": "ready", "topic": "food_security", "report_year": int(name[5:9])}
    if clinical_archive:
        return {"agent": "clinical", "status": "ready", "topic": "clinical_research"}
    if parts & {"foodsecurity.zip", "foodsecurity.zip.contents"}:
        if name in _SOFI_NAMES:
            return {"agent": "corporate_responsibility", "status": "ready", "topic": "food_security", "report_year": int(name[5:9])}
        return {"status": "reference_only", "topic": "unassigned", "reason": "not_an_assigned_sofi_report"}
    return {"status": "reference_only", "topic": "unassigned", "reason": "no_explicit_source_assignment"}


def _safe_member(name: str) -> bool:
    # Backslashes and drive delimiters are rejected even on POSIX: a later move
    # to another platform must not change the meaning of the manifest path.
    if not name or "\\" in name or "\x00" in name or ":" in name:
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in name.rstrip("/").split("/"))


def _safe_write(root: Path, relative: Path, data: bytes) -> Path:
    """Create a new file without following extraction-directory symlinks."""
    target = root / relative
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        if current.is_symlink() or not current.is_dir():
            raise ValueError("unsafe_output_directory")
    if target.is_symlink():
        raise ValueError("unsafe_output_symlink")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError:
        # Re-running inspection is safe and idempotent only for identical bytes.
        if target.is_file() and _sha256(target) == hashlib.sha256(data).hexdigest():
            return target
        raise ValueError("output_already_exists") from None
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    return target


def _url_entries(entry: dict[str, Any], text: str) -> list[dict[str, Any]]:
    links = dict.fromkeys(match.rstrip(".,;:") for match in _URL.findall(text))
    result = []
    for url in links:
        original = f"{entry['original_member_path']}#url={url}"
        try:
            parsed_url = urlparse(url)
            if not parsed_url.netloc or not parsed_url.hostname:
                raise ValueError("missing_url_host")
        except ValueError:
            result.append(_entry(original, agent="clinical", topic="agilent_reference", format="url", status="failed", url=url, parent_source_id=entry["source_id"], reason="invalid_source_url"))
            continue
        url_path = parsed_url.path
        excluded = set(re.findall(r"[a-z0-9]+", url.lower())) & _EXCLUDED
        result.append(_entry(
            original,
            agent=None if excluded else "clinical",
            topic="out_of_scope" if excluded else "agilent_reference",
            format=Path(url_path).suffix.lower().lstrip(".") or "url",
            status="excluded" if excluded else "pending_download",
            url=url,
            parent_source_id=entry["source_id"],
            reason="assigned_to_other_project_or_specialist" if excluded else "external_reference_requires_explicit_download_and_validation",
        ))
    return result


def inspect_archive(
    archive: Path,
    output_dir: Path,
    *,
    limits: ArchiveLimits | None = None,
) -> list[dict[str, Any]]:
    """Inspect nested ZIPs and safely extract assigned evidence/scope PDFs.

    Each manifest entry records its archive lineage, hash (when bytes were read),
    assignment and status. Failed/unsafe entries remain visible in the manifest.
    Out-of-scope files are never extracted. Existing differing files are never
    replaced. Unknown text/PDFs are recorded but only assignment/scope evidence,
    assigned documents and Agilent.txt catalogs are extracted.
    """
    archive, output_dir = Path(archive), Path(output_dir)
    limits = limits or ArchiveLimits()
    if any(value <= 0 for value in (limits.max_depth, limits.max_entries, limits.max_member_bytes, limits.max_total_bytes, limits.max_compression_ratio)):
        raise ValueError("Archive limits must be positive")
    if not archive.is_file():
        return [_entry(archive.name, status="missing", reason="archive_not_found")]
    if archive.is_symlink():
        return [_entry(archive.name, status="failed", reason="archive_symlink_rejected")]
    if output_dir.is_symlink() or any(parent.is_symlink() for parent in output_dir.absolute().parents):
        raise ValueError("Extraction root must not contain a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    archive_resolved = archive.resolve()
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    total_entries = 0
    root_entry = _entry(archive.name, sha256=_sha256(archive), format="zip", topic="source_bundle", status="inspected", local_path=str(archive_resolved))
    entries.append(root_entry)

    def visit(source: Path | io.BytesIO, prefix: str, destination: Path, depth: int) -> None:
        nonlocal total_bytes, total_entries
        with zipfile.ZipFile(source) as bundle:
            for member in bundle.infolist():
                total_entries += 1
                original = f"{prefix}!/{member.filename}"
                item = _entry(original, **_classify(original))
                if total_entries > limits.max_entries:
                    entries.append(_entry(original, status="failed", reason="archive_entry_limit_exceeded"))
                    return
                if item["status"] == "ignored":
                    continue
                if not _safe_member(member.filename):
                    item.update(status="failed", reason="unsafe_archive_path")
                    entries.append(item)
                    continue
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                    item.update(status="failed", reason="unsafe_archive_member_type")
                    entries.append(item)
                    continue
                if member.is_dir():
                    continue
                if item["status"] == "excluded":
                    entries.append(item)
                    continue
                extension = item["format"]
                if extension not in _EVIDENCE_FORMATS | {"zip"} and item["topic"] != "clinical_trials":
                    item.update(status="unsupported", reason="unsupported_document_format")
                    entries.append(item)
                    continue
                if member.flag_bits & 1:
                    item.update(status="failed", reason="encrypted_archive_member")
                    entries.append(item)
                    continue
                if member.file_size > limits.max_member_bytes:
                    item.update(status="failed", reason="archive_member_size_limit_exceeded")
                    entries.append(item)
                    continue
                if member.file_size > limits.max_compression_ratio * max(member.compress_size, 1):
                    item.update(status="failed", reason="archive_compression_ratio_limit_exceeded")
                    entries.append(item)
                    continue
                if total_bytes + member.file_size > limits.max_total_bytes:
                    item.update(status="failed", reason="archive_total_size_limit_exceeded")
                    entries.append(item)
                    continue
                if extension == "zip" and depth >= limits.max_depth:
                    item.update(status="failed", reason="archive_nesting_limit_exceeded")
                    entries.append(item)
                    continue
                # Unknown documents need no decompression or extraction. ZIPs
                # still need inspection to locate the assigned nested bundles.
                should_extract = item["status"] in {"ready", "ready_for_import"} or item["topic"] in {"source_catalog", "project_scope"}
                if extension != "zip" and not should_extract:
                    entries.append(item)
                    continue
                try:
                    with bundle.open(member) as stream:
                        data = stream.read(min(limits.max_member_bytes, limits.max_total_bytes - total_bytes) + 1)
                    if len(data) != member.file_size or len(data) > limits.max_member_bytes:
                        raise ValueError("archive_member_size_mismatch")
                    total_bytes += len(data)
                    item["sha256"] = hashlib.sha256(data).hexdigest()
                    if extension == "zip":
                        item.update(status="inspected", topic="source_bundle")
                        entries.append(item)
                        visit(io.BytesIO(data), original, destination / f"{member.filename}.contents", depth + 1)
                    else:
                        local = _safe_write(output_dir, destination / member.filename, data)
                        item["local_path"] = str(local)
                        entries.append(item)
                        if item["topic"] == "source_catalog":
                            entries.extend(_url_entries(item, data.decode("utf-8-sig", errors="replace")))
                except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as error:
                    item.update(status="failed", reason=str(error))
                    if not any(existing is item for existing in entries):
                        entries.append(item)

    try:
        visit(archive, archive.name, Path(f"{archive.name}.contents"), 0)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        root_entry.update(status="failed", reason=str(error))
    return entries


def discover_sources(root: Path, *, local_only: bool = False) -> list[dict[str, Any]]:
    """Inventory local inputs without fetching URLs or unpacking ZIPs.

    The selected root is a data directory: only its Clinical PDFs/trial CSVs and
    two top-level FoodSecurity SOFI PDFs are assigned. ZIP extraction lineage is
    also supported. local_only records catalog URLs as deferred references, so
    local evidence can be ingested without claiming external URLs were read.
    """
    root = Path(root)
    if not root.exists():
        return [_entry(str(root), format="directory", status="missing", reason="source_directory_not_found")]
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        return [_entry(str(root), status="failed", reason="source_symlink_rejected")]
    candidates = [root] if root.is_file() else sorted(root.rglob("*"))
    data_root = next((ancestor for ancestor in [root, *root.parents] if ancestor.name.lower() == "data"), None)
    entries: list[dict[str, Any]] = []
    for path in candidates:
        if path.name.lower().endswith(":zone.identifier"):
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        original = str(path.relative_to(root)) if root.is_dir() else path.name
        # Include a deliberately named extraction root in the assignment lineage.
        if data_root is not None:
            original = f"{data_root.name}/{path.relative_to(data_root).as_posix()}"
        elif root.is_dir() and root.name.lower() in {"clinical", "foodsecurity", "clinical.zip.contents", "foodsecurity.zip.contents"}:
            original = f"{root.name}/{original}"
        elif root.is_file() and root.parent.name.lower() == "clinical":
            original = f"Clinical/{original}"
        if data_root is None:
            # Selecting an excluded subtree directly must not erase its scope.
            excluded_parent = next((p.name for p in [root, *root.parents]
                                    if set(re.findall(r"[a-z0-9]+", p.name.lower())) & _EXCLUDED), None)
            if excluded_parent:
                original = f"{excluded_parent}/{original}"
        entry = _entry(original, **_classify(original), local_path=str(path.absolute()))
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            entry.update(status="failed", local_path=None, reason="source_symlink_rejected")
        elif entry["status"] != "excluded":
            try:
                entry["sha256"] = _sha256(path)
                if entry["format"] == "zip":
                    entry.update(status="pending_inspection", topic="source_bundle")
                elif entry["format"] not in _EVIDENCE_FORMATS and entry["topic"] != "clinical_trials":
                    entry.update(status="unsupported", reason="unsupported_document_format")
                elif entry["topic"] == "source_catalog":
                    urls = _url_entries(entry, path.read_text(encoding="utf-8-sig", errors="replace"))
                    if local_only:
                        for url in urls:
                            if url["status"] == "pending_download":
                                url.update(status="reference_only", reason="external_reference_deferred_local_only")
                    entries.extend(urls)
            except OSError as error:
                entry.update(status="failed", reason=str(error))
        entries.append(entry)
    return entries


def _assignment_error(entry: dict[str, Any], entries: list[dict[str, Any]], agent: str) -> str | None:
    """Recheck the manifest's claimed assignment at the actual ingestion boundary."""
    original = str(entry.get("original_member_path", ""))
    url = entry.get("url")
    if set(re.findall(r"[a-z0-9]+", f"{original} {url or ''}".lower())) & _EXCLUDED:
        return "excluded_source_assignment"
    if url is not None:
        catalog_original, separator, listed_url = original.partition("#url=")
        try:
            parsed_url = urlparse(url)
        except (TypeError, ValueError):
            return "invalid_agilent_reference"
        if (
            agent != "clinical"
            or not separator
            or listed_url != url
            or parsed_url.scheme.lower() not in {"http", "https"}
            or not parsed_url.netloc
            or _classify(catalog_original).get("topic") != "source_catalog"
        ):
            return "invalid_agilent_reference"
        parent = next((candidate for candidate in entries if candidate.get("source_id") == entry.get("parent_source_id")), None)
        if (
            parent is None
            or parent.get("original_member_path") != catalog_original
            or parent.get("topic") != "source_catalog"
            or parent.get("agent") != "clinical"
        ):
            return "missing_agilent_catalog"
        # The parent and URL were recorded by discovery, not inferred from a
        # download's arbitrary filename. Recheck the preserved catalog when it
        # is locally available, including its original content hash.
        parent_path = parent.get("local_path")
        if not parent_path:
            return "missing_agilent_catalog"
        catalog = Path(parent_path)
        try:
            if catalog.is_symlink() or any(component.is_symlink() for component in catalog.parents):
                return "unsafe_agilent_catalog"
            if parent.get("sha256") and _sha256(catalog) != parent["sha256"]:
                return "agilent_catalog_hash_mismatch"
            urls = _url_entries(parent, catalog.read_text(encoding="utf-8-sig", errors="replace"))
            if not any(candidate.get("url") == url and candidate["status"] == "pending_download" for candidate in urls):
                return "url_not_in_agilent_catalog"
        except OSError:
            return "missing_agilent_catalog"
        return None
    assignment = _classify(original)
    if assignment.get("status") != "ready" or assignment.get("agent") != agent:
        return "unassigned_source_manifest"
    return None


def load_documents(entries: list[dict[str, Any]], agent: str) -> tuple[list[Any], list[dict[str, Any]]]:
    """Parse assigned evidence into LangChain Documents with page citations.

    Returns new manifest dictionaries; input entries are not mutated. A file's
    hash is rechecked before reading so changed evidence cannot silently inherit
    an earlier manifest identity. Missing, encrypted, empty and malformed inputs
    remain visible as failed entries. URLs are not downloaded.
    """
    from langchain_core.documents import Document
    from pypdf import PdfReader

    if agent not in AGENTS:
        raise ValueError(f"Unknown specialist: {agent}")
    updated = [dict(entry) for entry in entries]
    documents: list[Document] = []
    for entry in updated:
        if entry.get("topic") == "clinical_trials":
            continue  # Structured trial facts have a separate validated importer.
        retry_local_evidence = (
            entry.get("status") in {"failed", "missing"}
            and entry.get("local_path")
            and entry.get("format") in _EVIDENCE_FORMATS
            and entry.get("topic") != "source_catalog"
        )
        if entry.get("agent") != agent or (entry.get("status") not in {"ready", "indexed"} and not retry_local_evidence):
            continue
        assignment_error = _assignment_error(entry, updated, agent)
        if assignment_error:
            entry.update(status="excluded" if assignment_error == "excluded_source_assignment" else "failed", reason=assignment_error)
            continue
        path_string = entry.get("local_path")
        if not path_string:
            entry.update(status="missing", reason="local_source_not_available")
            continue
        path = Path(path_string)
        try:
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
                raise ValueError("source_symlink_rejected")
            if not path.is_file():
                entry.update(status="missing", reason="local_source_not_found")
                continue
            digest = _sha256(path)
            if entry.get("sha256") and digest != entry["sha256"]:
                raise ValueError("source_hash_mismatch")
            entry["sha256"] = digest
            metadata = {
                "source_id": entry["source_id"],
                "source": entry.get("url") or entry["original_member_path"],
                "topic": entry["topic"],
                "agent": agent,
                "sha256": digest,
                "original_member_path": entry["original_member_path"],
                "local_path": str(path.absolute()),
            }
            if agent == "corporate_responsibility":
                match = re.search(r"SOFI-(2023|2024)\.pdf(?:$|!/)", entry["original_member_path"], re.I)
                if match:
                    metadata["report_year"] = int(match.group(1))
                    entry["report_year"] = int(match.group(1))
            pages: list[tuple[int | None, str]] = []
            if entry["format"] == "pdf":
                reader = PdfReader(path)
                if reader.is_encrypted:
                    raise ValueError("encrypted_pdf_not_supported")
                pages = [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]
            elif entry["format"] in _TEXT_FORMATS:
                pages = [(1, path.read_text(encoding="utf-8-sig"))]
            elif entry["format"] in {"html", "htm"}:
                from .html_evidence import extract_html_text

                pages = [(None, extract_html_text(path.read_text(encoding="utf-8-sig")))]
            else:
                entry.update(status="unsupported", reason="unsupported_document_format")
                continue
            parsed = [Document(page_content=text.strip(), metadata={**metadata, "page": number}) for number, text in pages if text.strip()]
            if not parsed:
                raise ValueError("no_extractable_text_ocr_may_be_required")
            documents.extend(parsed)
            entry.update(status="indexed", page_count=len(pages), indexed_page_count=len(parsed))
            entry.pop("reason", None)
        except Exception as error:  # Parsing failures belong in the per-source manifest.
            entry.update(status="failed", reason=f"{type(error).__name__}: {error}")
    return documents, updated
