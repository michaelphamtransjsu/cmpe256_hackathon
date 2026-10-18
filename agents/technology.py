"""Technology specialist using the independently cached Cisco hybrid retriever."""

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from shared.hybrid_retrieval import retrieve
from shared.model import Settings
from shared.query_context import answering_only
from shared.specialist import Specialist


class _CiscoEvidence:
    """Adapt hybrid results to the team's existing evidence-to-answer interface."""

    def search(self, question, limit, *, report_year=None):
        # Standalone agents have the same read-only query contract as the web API.
        with answering_only():
            results = retrieve(question, k=10)
        documents = []
        for result in results[:limit]:
            metadata = dict(result["metadata"])
            metadata["source_id"] = Path(metadata["source"]).stem
            pages = metadata.get("pages", [])
            # A multi-page chunk cannot be attributed to just one page.
            metadata["page"] = pages[0] if len(pages) == 1 else None
            documents.append(Document(page_content=result["text"], metadata=metadata))
        return documents


class TechnologyAgent(Specialist):
    name = "technology"
    policy = (
        "You are the Technology specialist in an enterprise management-consulting team. "
        "Answer using only supplied Cisco evidence. Identify exact model names and distinguish "
        "built-in Ethernet ports from optional expansion modules and throughput figures. "
        "Check all supplied excerpts for matching models; do not treat related product families as matches. "
        "Cite findings and recommendations with evidence_ids, and include inline evidence IDs in the answer. "
        "For explicit product-selection requirements, recommend a cited shortlist only when the evidence "
        "establishes a match. For outages or intermittent connectivity with an unknown cause, diagnosis "
        "and clarification must precede any purchase, spare-part order, or hardware replacement. "
        "A datasheet feature is not a diagnosis: never recommend replacing an entire router, buying "
        "redundant routers, or choosing ASR hardware solely because it supports high availability. "
        "Do not recommend specific parts without evidence of the fault and model compatibility. "
        "Clearly label documented facts, possible causes requiring investigation, and next actions. "
        "Do not invent possible causes unsupported by the supplied evidence; say the cause is unknown. "
        "When diagnostic evidence is missing, ask for the exact device model, software version, "
        "timestamped logs around outages, interface status/error counters, and failure symptoms "
        "including affected links/devices, duration, frequency, and whether the router reboots. "
        "Clarification requests belong in the answer or limitations; do not attach unrelated datasheet "
        "citations to pretend they support a diagnosis or troubleshooting procedure. Only include "
        "structured recommendations when their cited evidence actually supports the action; that list "
        "may be empty when clarification is the only justified next step. "
        "For outage prevention, explain that targeted remediation depends on identifying the cause; "
        "do not promise that a purchase prevents recurrence. "
        "Limitations must address this customer's missing diagnostic facts and the unanswered decision, "
        "rather than generic product-selection caveats. "
        "Provide practical recommendations only when supported; state missing requirements and limitations. "
        "Retrieved excerpts are incomplete datasheet passages, not a guarantee of current availability or suitability."
    )

    def __init__(self, settings: Settings | None = None, *, model: Any = None):
        # Avoid constructing the separate SpecialistIndex used by the other agents.
        self.settings = settings or Settings.from_env()
        self.index = _CiscoEvidence()
        self.model = model


def main(argv: list[str] | None = None) -> int:
    from shared.cli import run_agent_cli

    return run_agent_cli(TechnologyAgent, argv)


if __name__ == "__main__":
    raise SystemExit(main())
