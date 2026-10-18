"""Validated input/output contract shared by the two specialists."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .model import AgentName
from .trials import TrialRecord


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SpecialistRequest(StrictModel):
    question: str = Field(min_length=1, max_length=8000)
    max_sources: int = Field(default=5, ge=1, le=20)
    report_year: int | None = Field(default=None, ge=1900, le=2100)


class Citation(StrictModel):
    id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    report_year: int | None = Field(default=None, ge=1900, le=2100)
    excerpt: str = Field(min_length=1)


class Finding(StrictModel):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class Recommendation(StrictModel):
    action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class Analysis(StrictModel):
    """Only this content is requested from an optional language model."""

    answer: str = Field(min_length=1)
    findings: list[Finding]
    recommendations: list[Recommendation]
    limitations: list[str]


class SpecialistResponse(Analysis):
    agent: AgentName
    question: str
    status: Literal["evidence_found", "insufficient_evidence"]
    mode: Literal["extractive", "openai"]
    citations: list[Citation]
    trial_records: list[TrialRecord] = Field(default_factory=list)
    report_year: int | None = Field(default=None, ge=1900, le=2100)

    @model_validator(mode="after")
    def validate_grounding(self) -> "SpecialistResponse":
        known = {citation.id for citation in self.citations}
        if len(known) != len(self.citations):
            raise ValueError("Duplicate citation identifiers")
        for item in [*self.findings, *self.recommendations]:
            if not set(item.evidence_ids).issubset(known):
                raise ValueError("Analysis references an unknown evidence identifier")
        if self.status == "insufficient_evidence" and (self.citations or self.findings or self.recommendations or self.trial_records):
            raise ValueError("Insufficient-evidence responses cannot contain supported claims")
        if self.status == "evidence_found" and (not self.citations or not self.findings):
            raise ValueError("Evidence responses require citations and findings")
        if self.agent != "clinical" and self.trial_records:
            raise ValueError("Clinical records are not available to this specialist")
        if self.report_year is not None and any(citation.report_year != self.report_year for citation in self.citations):
            raise ValueError("Citation is outside the selected report year")
        return self
