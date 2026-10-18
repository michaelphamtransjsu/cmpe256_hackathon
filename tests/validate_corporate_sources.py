#!/usr/bin/env python3
"""Validate the local Corporate Responsibility index against supplied SOFI PDFs.

Run after inspect/ingest. This performs offline retrieval and verifies source
hashes, year isolation, verbatim PDF-page citations and restart persistence.
Documents remain external inputs; none are included in the repository.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

# Allow this standalone validator to run from a checkout without installation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PdfReader

from agents.corporate_responsibility import CorporateResponsibilityAgent
from shared.model import Settings


def validate(manifest_path: Path, data_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assigned = [entry for entry in manifest["entries"] if entry.get("agent") == "corporate_responsibility"]
    if len(assigned) != 2:
        raise ValueError("Expected exactly two assigned SOFI reports in this manifest")
    pages = {}
    sources = []
    for entry in assigned:
        path = Path(entry["local_path"])
        match = re.search(r"SOFI-(2023|2024)\.pdf$", entry["original_member_path"], re.I)
        if not match:
            raise ValueError("Manifest contains a corporate source other than the assigned SOFI PDFs")
        year = int(match.group(1))
        if year in pages:
            raise ValueError("Manifest contains duplicate SOFI report years")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Encrypted source PDFs are unsupported")
        text = [page.extract_text() or "" for page in reader.pages]
        pages[year] = text
        sources.append({
            "source_id": entry["source_id"], "source": entry["original_member_path"],
            "report_year": year, "sha256": actual_hash,
            "recorded_sha256": entry.get("sha256"), "pdf_pages": len(text),
            "nonempty_pages": sum(bool(page.strip()) for page in text),
            "characters": sum(len(page) for page in text),
            "status": entry.get("status"),
        })
    # Deliberately use deterministic offline mode even if provider variables exist.
    settings = Settings(data_dir=data_dir, embeddings_provider="hash", llm_provider="extractive")
    agent = CorporateResponsibilityAgent(settings)
    rows = list(agent.index.store.store.values())
    source_ids = {entry["source_id"] for entry in assigned}
    source_by_id = {source["source_id"]: source for source in sources}

    def attributed_chunk(row: dict) -> bool:
        metadata = row["metadata"]
        source = source_by_id.get(metadata.get("source_id"))
        if source is None:
            return False
        return (metadata.get("source") == source["source"]
                and metadata.get("report_year") == source["report_year"]
                and metadata.get("sha256") == source["sha256"])

    def verbatim_chunk(row: dict) -> bool:
        metadata = row["metadata"]
        year, page = metadata.get("report_year"), metadata.get("page")
        return bool(year in pages and isinstance(page, int) and 1 <= page <= len(pages[year])
                    and row["text"] in pages[year][page - 1])

    checks = {
        "nonempty_index": bool(rows),
        "both_assigned_years": set(pages) == {2023, 2024},
        "sources_indexed_in_manifest": all(source["status"] == "indexed" for source in sources),
        "original_pdf_hashes_preserved": all(source["sha256"] == source["recorded_sha256"] for source in sources),
        "corporate_metadata_only": all(row["metadata"].get("agent") == "corporate_responsibility" for row in rows),
        "assigned_sofi_only": {row["metadata"]["source_id"] for row in rows} == source_ids,
        "publication_year_metadata": all(row["metadata"].get("report_year") in pages for row in rows),
        "all_index_metadata_attributed": all(attributed_chunk(row) for row in rows),
        "all_chunks_verbatim_on_pdf_pages": all(verbatim_chunk(row) for row in rows),
    }
    queries = [
        ("assignment_2023", {"question": "List major Food insecurity reason in 2023 from within the context of State of Food Security and Nutrition in the World 2023 report."}, 2023),
        ("assignment_2024_explicit", {"question": "List major Food insecurity reason in 2024", "report_year": 2024}, 2024),
        ("assignment_war_2023", {"question": "Explain malnutrition in war zones from within the context of State of Food Security and Nutrition in the World 2023"}, 2023),
        ("assignment_war", {"question": "Explain malnutrition in war zones"}, None),
        ("assignment_prices", {"question": "Explain increase prices impact on food security"}, None),
        ("focused_drivers_2023", {"question": "major drivers conflict climate extremes economic slowdowns downturns", "report_year": 2023}, 2023),
        ("focused_drivers_2024", {"question": "food insecurity drivers conflict climate economic inequality", "report_year": 2024}, 2024),
        ("focused_conflict_2023", {"question": "humanitarian crises acute food insecurity conflict", "report_year": 2023}, 2023),
        ("focused_conflict_2024", {"question": "humanitarian crises acute food insecurity conflict", "report_year": 2024}, 2024),
        ("focused_prices_2024", {"question": "cost healthy diet food prices 3.96", "report_year": 2024}, 2024),
    ]
    results = {}
    for name, request, expected_year in queries:
        response = agent.invoke(request)
        results[name] = response.model_dump(mode="json")
        checks[name + "_has_evidence"] = response.status == "evidence_found" and bool(response.citations)
        checks[name + "_assigned_sources"] = all(c.source_id in source_ids for c in response.citations)
        if expected_year:
            checks[name + "_requested_report_only"] = response.report_year == expected_year and all(
                c.report_year == expected_year and f"SOFI-{expected_year}.pdf" in c.source for c in response.citations
            )
        for citation in response.citations:
            year = citation.report_year
            page_exists = year in pages and isinstance(citation.page, int) and 1 <= citation.page <= len(pages[year])
            checks[name + "_" + citation.id + "_verbatim_on_pdf_page"] = bool(
                page_exists and citation.excerpt in pages[year][citation.page - 1]
            )
    for year in (2023, 2024):
        checks[f"focused_{year}_driver_evidence"] = any(
            all(term in c["excerpt"].casefold() for term in ("conflict", "climate", "economic"))
            for c in results[f"focused_drivers_{year}"]["citations"]
        )
    checks["focused_2023_conflict_evidence"] = any(
        c["page"] == 44 and "acute" in c["excerpt"].casefold()
        for c in results["focused_conflict_2023"]["citations"]
    )
    checks["focused_2024_conflict_evidence"] = any(
        c["page"] == 43 and any(term in c["excerpt"].casefold() for term in ("gaza", "sudan", "humanitarian"))
        for c in results["focused_conflict_2024"]["citations"]
    )
    checks["focused_2024_price_evidence"] = any(
        c["page"] == 54 and "3.96" in c["excerpt"]
        for c in results["focused_prices_2024"]["citations"]
    )
    process_env = {**os.environ, "SPECIALIST_EMBEDDINGS_PROVIDER": "hash", "SPECIALIST_LLM_PROVIDER": "extractive"}
    persisted = json.loads(subprocess.check_output([
        sys.executable, "-m", "agents.corporate_responsibility",
        queries[-1][1]["question"], "--data-dir", str(data_dir), "--report-year", "2024",
    ], text=True, env=process_env, timeout=60))
    checks["new_process_same_response"] = persisted == results["focused_prices_2024"]
    counts = Counter(row["metadata"]["source_id"] for row in rows)
    for source in sources:
        source["chunks"] = counts[source["source_id"]]
    return {
        "validation_mode": "offline_extractive_hash_embeddings_with_bm25_ranking",
        "sources": sources, "indexed_chunks": len(rows), "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "queries": results,
        "limitations": [
            "Extractive mode returns source quotes, not synthesized consulting advice.",
            "Lexical retrieval can include tangential excerpts; correct source year alone does not establish answer quality.",
            "Publication years differ from observation years. Bare query years are not automatically report filters.",
            "Hosted model synthesis is not exercised by this offline validation.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/real-corporate/manifest.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, help="Save full JSON results, including evidence excerpts, to this local file")
    args = parser.parse_args()
    try:
        report = validate(args.manifest, args.data_dir)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": len(report["checks"]), "passed": sum(report["checks"].values()),
                      "failed_checks": report["failed_checks"], "indexed_chunks": report["indexed_chunks"]}))
    return 1 if report["failed_checks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
