"""Ingestion-only mapping of ClinicalTrials.gov CSV exports to supplied facts."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
import re
from urllib.parse import urlparse

from .sources import _classify, _sha256
from .trials import TrialRecord


def same_csv_facts(left: TrialRecord, right: TrialRecord) -> bool:
    """Only merge origins when both the mapped facts and raw columns agree."""
    if not left.metadata.get("csv_columns") or not right.metadata.get("csv_columns"):
        return False
    def facts(record):
        fields = record.model_dump(exclude={"citations", "metadata"})
        fields["metadata"] = {k: v for k, v in record.metadata.items() if k != "csv_provenance"}
        # Without a registry URL, the source is the CSV row's original path.
        if not record.metadata["csv_columns"].get("Study URL", "").strip():
            fields.pop("source")
        return fields
    return facts(left) == facts(right)


def consolidate_records(records: list[TrialRecord]) -> list[TrialRecord]:
    """Reject contradictory duplicate IDs instead of choosing an arbitrary row."""
    unique: dict[str, TrialRecord] = {}
    for record in records:
        previous = unique.get(record.trial_id)
        if previous is None:
            unique[record.trial_id] = record.model_copy(deep=True)
        elif previous == record:
            continue
        elif same_csv_facts(previous, record):
            previous.citations = list(dict.fromkeys([*previous.citations, *record.citations]))
            origins = previous.metadata["csv_provenance"]
            for origin in record.metadata["csv_provenance"]:
                if origin not in origins:
                    origins.append(origin)
        else:
            raise ValueError(f"Conflicting duplicate trial ID {record.trial_id}; no records imported")
    return list(unique.values())


def load_trial_csvs(entries: list[dict]) -> list[TrialRecord]:
    """Validate all assigned CSVs and hashes before the caller opens SQLite.

    CSV record numbers start at 2 (the header is record 1); physical line ranges
    are also retained for quoted multiline fields. Unmapped columns remain raw
    strings in metadata. Missing values never become guessed registry facts.
    """
    assigned = [entry for entry in entries if entry.get("topic") == "clinical_trials"]
    if not assigned:
        raise ValueError("No assigned ctg-studies_*.csv inputs found")
    records = []
    previous_limit = csv.field_size_limit(16 * 1024 * 1024)
    try:
        for entry in assigned:
            original = entry.get("original_member_path", "")
            assignment = _classify(original)
            if assignment.get("topic") != "clinical_trials" or entry.get("agent") != "clinical":
                raise ValueError("Unassigned clinical CSV manifest entry")
            if not entry.get("local_path"):
                raise ValueError(f"Missing local clinical CSV: {original}")
            path = Path(entry["local_path"])
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
                raise ValueError("Clinical CSV symlinks are rejected")
            digest = _sha256(path)
            if not entry.get("sha256") or digest != entry["sha256"]:
                raise ValueError(f"Clinical CSV source_hash_mismatch: {original}")
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream, strict=True)
                headers = reader.fieldnames or []
                if (not {"NCT Number", "Study Title"}.issubset(headers)
                        or len(set(headers)) != len(headers) or any(not header.strip() for header in headers)):
                    raise ValueError(f"Invalid clinical CSV headers in {original}; NCT Number and Study Title are required")
                line = reader.line_num
                for number, row in enumerate(reader, 2):
                    start_line, line = line + 1, reader.line_num
                    location = f"{original}#record={number}"
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"Malformed clinical CSV row at {location}")
                    def value(column):
                        return row.get(column, "").strip() or None
                    trial_id = value("NCT Number") or ""
                    if not re.fullmatch(r"NCT[0-9]{8}", trial_id):
                        raise ValueError(f"Invalid NCT Number at {location}")
                    url = value("Study URL")
                    if url:
                        parsed = urlparse(url)
                        if (parsed.scheme != "https" or parsed.netloc.lower() not in {"clinicaltrials.gov", "www.clinicaltrials.gov"}
                                or parsed.path.rstrip("/") != f"/study/{trial_id}" or parsed.query or parsed.fragment):
                            raise ValueError(f"Invalid or mismatched Study URL at {location}")
                    updated = value("Last Update Posted")
                    # Preserve source precision: dates have no invented time or
                    # timezone. Partial ISO dates remain raw metadata only.
                    updated_at = None
                    if updated:
                        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", updated):
                            updated_at = date.fromisoformat(updated)
                        elif not re.fullmatch(r"\d{4}(-(?:0[1-9]|1[0-2]))?", updated):
                            raise ValueError(f"Invalid Last Update Posted at {location}")
                    records.append(TrialRecord(
                        trial_id=trial_id, title=value("Study Title"), source=url or location,
                        conditions=[s.strip() for s in (value("Conditions") or "").split("|") if s.strip()],
                        interventions=[s.strip() for s in (value("Interventions") or "").split("|") if s.strip()],
                        status=value("Study Status"), phase=value("Phases"), sponsor=value("Sponsor"),
                        summary=value("Brief Summary"), updated_at=updated_at,
                        citations=[location, *([url] if url else [])],
                        metadata={"csv_columns": row, "csv_provenance": [{
                            "source_id": entry["source_id"], "sha256": digest,
                            "original_member_path": original, "local_path": str(path.absolute()),
                            "record_number": number, "line_start": start_line, "line_end": line,
                        }]},
                    ))
                if line == 1:
                    raise ValueError(f"No clinical trial rows in {original}")
            if _sha256(path) != digest:
                raise ValueError(f"Clinical CSV changed while reading: {original}")
    finally:
        csv.field_size_limit(previous_limit)
    return consolidate_records(records)
