"""Corporate Responsibility specialist for assigned SOFI food-security evidence."""

import re

from shared.schemas import SpecialistRequest
from shared.specialist import Specialist


REPORT_TITLE = r"(?:the\s+)?state\s+of\s+food\s+security\s+and\s+nutrition\s+in\s+the\s+world"


def infer_report_year(question: str) -> int | None:
    """Only explicit publication references constrain the corpus.

    A bare observation year (for example, 'diet cost in 2022') is deliberately
    not interpreted as a report year. Comparisons can use both reports.
    """
    if re.search(r"compar", question, re.I) and len(set(re.findall(r"\b20\d{2}\b", question))) > 1:
        return None
    years = set()
    for pattern in (
        r"\bSOFI\s*[- ]?\s*(20\d{2})\b",
        r"\b(20\d{2})\s+reports?\b",
        r"\breport\s+(?:(?:from|for|of)\s+)?(20\d{2})\b",
        REPORT_TITLE + r"\s+(20\d{2})\b",
    ):
        years.update(int(value) for value in re.findall(pattern, question, re.I))
    return next(iter(years)) if len(years) == 1 else None


def focused_corporate_query(question: str, report_year: int | None) -> str:
    """Remove report-title boilerplate and add retrieval synonyms, not assertions.

    These terms help lexical retrieval match question wording to report wording.
    Only retrieved source excerpts may become response evidence.
    """
    query = re.sub(REPORT_TITLE + r"(?:\s+20\d{2})?", " ", question, flags=re.I)
    query = re.sub(r"\bfrom\s+within\s+the\s+context\s+of\b|\b(?:list|explain|major|report|within|context)\b", " ", query, flags=re.I)
    query = re.sub(r"\bSOFI\s*[- ]?\s*20\d{2}\b", " ", query, flags=re.I)
    if report_year is not None:
        query = re.sub(rf"\b{report_year}\b", " ", query)
    expansions = []
    if re.search(r"\b(reason|cause|driver)s?\b", query, re.I):
        expansions.append("drivers causes")
    if re.search(r"\bwar(?:s)?\b|war[- ]zones?", query, re.I):
        expansions.append("conflict humanitarian crises acute food insecurity")
    if re.search(r"\bprices?\b|\binflation\b", query, re.I):
        expansions.append("inflation affordability purchasing power cost healthy diet")
    return " ".join((query + " " + " ".join(expansions)).split())


class CorporateResponsibilityAgent(Specialist):
    name = "corporate_responsibility"
    policy = (
        "You are the Corporate Responsibility specialist in an enterprise management-consulting team. "
        "Use assigned SOFI food-security evidence to assess hunger, nutrition, affordability, responsible investment and business implications. "
        "Preserve indicator definitions, units, geography, uncertainty, publication year and observation year. "
        "When comparing SOFI 2023 and 2024 distinguish revisions from trends; never infer comparable figures without evidence. "
        "Do not invent corporate commitments or use SEC, Technology, Cisco, Google or Retail sources."
    )

    def _prepare_query(self, request: SpecialistRequest) -> tuple[str, int | None]:
        report_year = request.report_year
        if report_year is None:
            report_year = infer_report_year(request.question)
        query = request.question
        if self.settings.embeddings_provider == "hash":
            query = focused_corporate_query(query, report_year)
        return query, report_year


def main(argv: list[str] | None = None) -> int:
    from shared.cli import run_agent_cli

    return run_agent_cli(CorporateResponsibilityAgent, argv, report_year=True)


if __name__ == "__main__":
    raise SystemExit(main())
