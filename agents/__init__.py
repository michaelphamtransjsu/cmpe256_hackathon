"""Independent specialist constructors; no orchestration or eager model setup."""

from shared.model import AgentName, Settings
from shared.specialist import Specialist


def create_specialist(name: AgentName, settings: Settings | None = None) -> Specialist:
    """Construct only the selected specialist, without delegation."""
    if name == "clinical":
        from .clinical import ClinicalAgent

        return ClinicalAgent(settings)
    if name == "corporate_responsibility":
        from .corporate_responsibility import CorporateResponsibilityAgent

        return CorporateResponsibilityAgent(settings)
    if name == "sec":
        from .sec import SECAgent

        return SECAgent(settings)
    raise ValueError("Unknown specialist")
