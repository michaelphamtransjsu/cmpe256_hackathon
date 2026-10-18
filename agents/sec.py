"""SEC specialist for the locally indexed Cisco and Agilent FY2025 10-Ks."""

import re

from langchain_core.documents import Document

from shared.model import Settings
from shared.sec_retrieval import FILINGS, retrieve
from shared.specialist import Specialist


def _companies(question):
    return [f["company"] for f in FILINGS
            if re.search(r"\b" + re.escape(f["company"]) + r"\b", question, re.I)]


class _SECEvidence:
    def search(self, question, limit, *, report_year=None):
        groups = [retrieve(question, k=limit, company=company, fiscal_year=report_year)
                  for company in _companies(question)]
        # Balance named companies before applying the shared evidence limit.
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
        "and Agilent FY2025 Form 10-K evidence. Cite each finding and recommendation with evidence_ids "
        "and include evidence IDs in the answer. Separate documented financial facts and risk disclosures "
        "from your interpretations and suggested actions. A disclosed possible risk is not an observed loss "
        "or a forecast. Do not invent figures, probabilities, severity rankings, or company circumstances. "
        "For every financial figure preserve company, fiscal year, exact reporting period, currency, scale, "
        "and whether it is a consolidated or segment amount. Read table column headers, units and footnotes; "
        "comparative-year figures in a FY2025 filing are not FY2025 figures. Cisco's fiscal year ends "
        "July 26, 2025 and Agilent's ends October 31, 2025; do not treat those periods as identical. "
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
        if not _companies(request.question):
            raise ValueError("Specify Cisco or Agilent; only their FY2025 Form 10-Ks are indexed")
        years = set(re.findall(r"(?<!\d)(20\d{2})(?!\d)", request.question))
        year = request.report_year if request.report_year is not None else (
            int(next(iter(years))) if len(years) == 1 else 2025)
        return request.question, year


def main(argv=None):
    from shared.cli import run_agent_cli

    return run_agent_cli(SECAgent, argv, report_year=True)


if __name__ == "__main__":
    raise SystemExit(main())
