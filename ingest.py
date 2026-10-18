"""Document preparation and index ingestion, separate from specialist answering."""

import argparse
from collections import Counter
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from shared.cli import safe_error_message
from shared.model import Settings
from shared.retrieval import SpecialistIndex
from shared.trials import TrialRecord, TrialStore


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def _read_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not isinstance(manifest.get("entries"), list):
        raise ValueError("Expected a version-1 manifest with an entries list")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Prepare documents and trial records for the two specialists")
    result.add_argument("--data-dir", type=Path, help="Override SPECIALIST_DATA_DIR")
    commands = result.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Safely inspect/extract a ZIP and record its sources")
    inspect.add_argument("archive", type=Path)
    inspect.add_argument("--output", type=Path, required=True, help="New extraction directory")
    inspect.add_argument("--manifest", type=Path, required=True, help="New manifest path; existing files are preserved")
    discover = commands.add_parser("discover", help="Inventory existing extracted sources without indexing them")
    discover.add_argument("root", type=Path)
    discover.add_argument("--manifest", type=Path, required=True)
    attach = commands.add_parser("attach-source", help="Associate a manually downloaded Agilent reference with its manifest entry")
    attach.add_argument("--manifest", type=Path, required=True)
    attach.add_argument("--source-id", required=True)
    attach.add_argument("--path", type=Path, required=True)
    ingest = commands.add_parser("ingest", help="Replace one specialist index from its manifest entries")
    ingest.add_argument("agent", choices=["clinical", "corporate_responsibility"])
    ingest.add_argument("--manifest", type=Path, required=True)
    ingest.add_argument("--allow-empty", action="store_true", help="Explicitly permit clearing an index when no readable documents exist")
    trials = commands.add_parser("trials-import", help="Validate and upsert a JSON array of source-attributed trial records")
    trials.add_argument("path", type=Path)
    search = commands.add_parser("trials-search", help="Query saved clinical trial records")
    search.add_argument("query", nargs="?", default="")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.data_dir is not None:
            settings = Settings(**{**settings.model_dump(), "data_dir": args.data_dir})
        if args.command in ("inspect", "discover"):
            from shared.sources import discover_sources, inspect_archive

            if args.manifest.exists():
                raise ValueError("Manifest already exists; choose a new path to preserve the existing inventory")
            entries = inspect_archive(args.archive, args.output) if args.command == "inspect" else discover_sources(args.root)
            _write_json(args.manifest, {"version": 1, "entries": entries})
            statuses = dict(Counter(row["status"] for row in entries))
            print(json.dumps({"manifest": str(args.manifest), "entries": len(entries), "statuses": statuses}))
            if statuses.get("failed") or statuses.get("missing"):
                return 1
        elif args.command == "attach-source":
            manifest = _read_manifest(args.manifest)
            entry = next((row for row in manifest["entries"] if row["source_id"] == args.source_id), None)
            if entry is None or entry.get("status") not in {"pending_download", "failed", "missing"} or entry.get("agent") != "clinical" or not entry.get("url"):
                raise ValueError("Only an existing pending/failed/missing Agilent URL reference can be attached")
            path = args.path.resolve(strict=True)
            if not path.is_file() or path.suffix.lower() not in (".pdf", ".txt", ".md", ".html", ".htm"):
                raise ValueError("The local source must be a PDF, TXT, Markdown, or HTML file")
            entry.update(local_path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), format=path.suffix.lower().lstrip("."), status="ready")
            _write_json(args.manifest, manifest)
            print(json.dumps({"source_id": args.source_id, "status": "ready"}))
        elif args.command == "ingest":
            from shared.sources import load_documents

            manifest = _read_manifest(args.manifest)
            documents, entries = load_documents(manifest["entries"], args.agent)
            problems = [row for row in entries if
                        (row.get("agent") == args.agent and row.get("status") in {"failed", "missing", "unavailable", "pending_download", "unsupported"})
                        or (row.get("format") == "zip" and row.get("status") in {"failed", "missing", "pending_inspection"})]
            if problems:
                for row in entries:
                    if row.get("agent") == args.agent and row.get("status") == "indexed":
                        row.update(status="ready", reason="index_replacement_aborted_due_to_other_unresolved_sources")
                manifest["entries"] = entries
                _write_json(args.manifest, manifest)
                raise ValueError(f"{len(problems)} assigned source(s) unresolved; index preserved. Resolve manifest failures/pending downloads before retrying")
            if not documents and not args.allow_empty:
                # Keep parse failures visible without deleting a previously valid index.
                manifest["entries"] = entries
                _write_json(args.manifest, manifest)
                raise ValueError("No readable assigned documents; index preserved. See manifest statuses")
            count = SpecialistIndex(args.agent, settings).replace(documents)
            manifest["entries"] = entries
            _write_json(args.manifest, manifest)
            print(json.dumps({"agent": args.agent, "documents": len(documents), "chunks": count, "manifest": str(args.manifest)}))
        elif args.command == "trials-import":
            raw = json.loads(args.path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("Trial input must be a JSON array")
            records = [TrialRecord.model_validate(row) for row in raw]
            store = TrialStore(settings.data_dir / "clinical" / "trials.sqlite3")
            for record in records:
                store.upsert(record)
            print(json.dumps({"imported": len(records)}))
        elif args.command == "trials-search":
            store = TrialStore(settings.data_dir / "clinical" / "trials.sqlite3")
            print(json.dumps([row.model_dump(mode="json") for row in store.search(args.query)], indent=2))
        return 0
    except Exception as error:
        print(f"error: {safe_error_message(error)}", file=sys.stderr)
        return 1



if __name__ == "__main__":
    raise SystemExit(main())
