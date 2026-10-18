"""SEC specialist for the five locally indexed Cisco, Agilent, and Alphabet 10-Ks."""

import re

from langchain_core.documents import Document

from shared.model import Settings
from shared.schemas import SpecialistRequest, SpecialistResponse
from shared.sec_retrieval import FILINGS, retrieve
from shared.specialist import Specialist


ALIASES = {"Cisco": r"Cisco(?: Systems)?|CSCO", "Agilent": r"Agilent(?: Technologies)?",
           "Alphabet": r"Alphabet|Google|GOOGL|GOOG"}
# General corpus questions are allowed conservatively. Unknown issuer names must
# never fall through to an unrestricted search of unrelated companies.
_GENERAL_WORDS = set((
    "what which how are is were was do does did the a an and or of in on for from to "
    "these those all available provided professor local indexed sec reports report filings filing "
    "form forms 10 k annual fiscal year years fy business businesses risks risk regulatory "
    "regulation regulations concerns concern mentioned discussed disclosed describe summarize "
    "summary main principal operational financial operations information about contain contains "
    "compare comparison companies company issuers issuer their they face facing have has "
    "revenues revenue sales performance trends cybersecurity legal governance challenges "
    "tell me please across between show list give us most significant factors "
    "antitrust privacy data international trade restrictions supply chain competition competitive "
    "growth profitability margins capital allocation cash flow debt liquidity interest rates currency "
    "tax taxes tariffs economic market markets uncertainties compliance regulation impacts effects "
    "identify explain discuss outline key major potential challenges technology ai artificial intelligence"
).split())


def _companies(question):
    return [company for company, aliases in ALIASES.items()
            if re.search(r"\b(?:" + aliases + r")\b", question, re.I)]


def _scopes(question, report_year=None):
    companies = _companies(question)
    comparison_names = re.findall(
        r"\b(?:compare|and|versus|vs\.?|with|against)\s+([a-z][\w-]*)"
        r"(?=\s+(?:FY\s*20\d{2}|20\d{2}|risks?\b|revenues?\b|and\b|versus\b)|[’']s|[?.!]*$)",
        question, re.I)
    if any(not _companies(name) and name.casefold() not in _GENERAL_WORDS
           and not re.fullmatch(r"FY20\d{2}", name, re.I)
           for name in comparison_names):
        return None
    if not companies:
        words = re.findall(r"[a-z]+", question.casefold())
        if any(word not in _GENERAL_WORDS for word in words):
            return None
    years = {report_year} if report_year is not None else {
        int(year) for year in re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)}
    scopes = {company: set(years) for company in companies}
    # Keep years attached to their named issuer in cross-company comparisons.
    if len(companies) > 1 and report_year is None:
        pattern = r"\b(?:" + "|".join(ALIASES.values()) + r")\b"
        mentions = list(re.finditer(pattern, question, re.I))
        local_scopes = {}
        for i, mention in enumerate(mentions):
            end = mentions[i + 1].start() if i + 1 < len(mentions) else len(question)
            local_years = {int(y) for y in re.findall(r"(?<!\d)(20\d{2})(?!\d)", question[mention.end():end])}
            if local_years:
                local_scopes.setdefault(_companies(mention.group())[0], set()).update(local_years)
        scopes.update(local_scopes)
    return scopes, years


def _filings(question, report_year=None):
    selection = _scopes(question, report_year)
    if selection is None:
        return []
    scopes, years = selection
    return [f for f in FILINGS if (not scopes or f["company"] in scopes)
            and (not (scopes.get(f["company"], years))
                 or f["fiscal_year"] in scopes.get(f["company"], years))]


class _SECEvidence:
    def search(self, question, limit, *, report_year=None):
        groups = [retrieve(question, k=limit, company=f["company"], fiscal_year=f["fiscal_year"])
                  for f in _filings(question, report_year)]
        # Balance each requested company/year before the shared evidence limit.
        results = [group[i] for i in range(limit) for group in groups if i < len(group)][:limit]
        documents = []
        for result in results:
            meta = dict(result["metadata"])
            meta["report_year"] = meta["fiscal_year"]
            text = (f"{meta['company']} | Form {meta['filing_type']} | FY{meta['fiscal_year']} | "
                    f"Year ended {meta['period_end']} | physical PDF pages {meta['pages']}\n" + result["text"])
            if meta.get("table_context"):
                text += "\nFull table context:\n" + "\n".join(
                    t["markdown"] + "\n" + "\n".join(t.get("captions", []) + t.get("footnotes", []))
                    for t in meta["table_context"].values())
                text += "\nOriginal physical-page context (unit labels and footnotes):\n" + "\n".join(
                    meta.get("page_context", {}).values())
            documents.append(Document(page_content=text, metadata=meta))
        return documents


class SECAgent(Specialist):
    name = "sec"
    policy = (
        "You are the SEC specialist in a management consulting team. Analyze only supplied Cisco "
        "FY2025/FY2026, Agilent FY2025, and Alphabet FY2024/FY2025 Form 10-K evidence. Cite each finding and recommendation with evidence_ids "
        "and include evidence IDs in the answer. Separate documented financial facts and risk disclosures "
        "from your interpretations and suggested actions. A disclosed possible risk is not an observed loss "
        "or a forecast. Do not invent figures, probabilities, severity rankings, or company circumstances. "
        "If a requested issuer/year is absent, report that gap; never substitute another issuer or year. "
        "For every financial figure preserve company, fiscal year, exact reporting period, currency, scale, "
        "and whether it is a consolidated or segment amount. Read table column headers, units and footnotes; "
        "comparative-year figures must retain their own column periods, not the filing's fiscal year. "
        "Cisco's periods end July 26, 2025 and July 25, 2026; Agilent FY2025 ends October 31, 2025; "
        "Alphabet FY2024/FY2025 end December 31 of their respective years. Do not treat these periods "
        "as identical. For comparisons, cite each company/year separately; absence of retrieved "
        "evidence for a requested period is a limitation, never evidence of no change. "
        "Never infer currency or units when absent; report missing context instead. "
        "Explain financial and operational implications only where supported, label recommendations "
        "as proposed actions, and state limitations specific to this question. Retrieved risk excerpts "
        "are not an exhaustive assessment and do not establish which risk is most likely or most severe."
    )

    def __init__(self, settings: Settings | None = None, *, model=None):
        self.settings = settings or Settings.from_env()
        self.index = _SECEvidence()
        self.model = model

    def _prepare_query(self, request):
        years = set(re.findall(r"(?<!\d)(20\d{2})(?!\d)", request.question))
        year = request.report_year if request.report_year is not None else (
            int(next(iter(years))) if len(years) == 1 else None)
        return request.question, year

    def invoke(self, request: SpecialistRequest | dict | str) -> SpecialistResponse:
        response = super().invoke(request)
        if response.mode == "extractive":
            response.limitations = [item.replace(" Hash embeddings use lexical similarity.", "")
                                    for item in response.limitations]
        if response.status == "insufficient_evidence":
            response.limitations = [
                "No matching evidence was found for the requested issuer, fiscal year, or question. "
                "Available Form 10-Ks: Cisco FY2025/FY2026, Agilent FY2025, Alphabet FY2024/FY2025. "
                "Specify an available issuer/year or ask a general question about these SEC filings; "
                "unrecognized issuer requests are not substituted with other companies."]
        selection = _scopes(response.question, response.report_year)
        if selection:
            scopes, years = selection
            available = {(f["company"], f["fiscal_year"]) for f in FILINGS}
            unavailable = [f"{company} FY{year}" for company, periods in scopes.items()
                           for year in sorted(periods) if (company, year) not in available]
            if unavailable:
                response.limitations.append("Requested filings are not indexed: " + ", ".join(unavailable))
        selected = _filings(response.question, response.report_year)
        covered = {c.source_id for c in response.citations}
        missing = [f"{f['company']} FY{f['fiscal_year']}" for f in selected
                   if f"{f['company'].lower()}-{f['fiscal_year']}-{f['filing_type']}" not in covered]
        if missing and response.citations:
            response.limitations.append("No evidence included for: " + ", ".join(missing)
                                        + "; the combined excerpts do not support conclusions for these filings.")
        return response


def main(argv=None):
    from shared.cli import run_agent_cli

    return run_agent_cli(SECAgent, argv, report_year=True)


if __name__ == "__main__":
    raise SystemExit(main())
