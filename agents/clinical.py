"""Clinical specialist: assigned research evidence and persistent trial records."""

from typing import Any

from shared.model import Settings
from shared.specialist import Specialist
from shared.trials import TrialStore


class ClinicalAgent(Specialist):
    name = "clinical"
    policy = (
        "You are the Clinical specialist in an enterprise management-consulting team. "
        "Analyze clinical research, trials, laboratory capabilities, and operational implications using only supplied evidence. "
        "Distinguish trial registration/status from demonstrated efficacy; report missing phase, population, outcomes and dates. "
        "Do not provide patient-specific treatment advice or invent trial records. Agilent materials support only claims they actually contain."
    )

    def __init__(self, settings: Settings | None = None, *, model: Any = None):
        super().__init__(settings, model=model)
        self.trials = TrialStore(self.settings.data_dir / "clinical" / "trials.sqlite3")

    def _trials(self, question: str, limit: int) -> list:
        return self.trials.search(question, limit=limit)


def main(argv: list[str] | None = None) -> int:
    from shared.cli import run_agent_cli

    return run_agent_cli(ClinicalAgent, argv)


if __name__ == "__main__":
    raise SystemExit(main())
