"""Validated, source-attributed clinical trial records stored in local SQLite.

This module stores records supplied by a caller; it does not infer trial facts or
retrieve registry data. ``source`` must identify the document or registry entry
from which the record was obtained. ``citations`` can hold page, section, or URL
references when those are available. Unknown trial attributes remain ``None`` or
empty lists rather than being filled with guessed values.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, field_validator


NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class TrialRecord(BaseModel):
    """A caller-supplied clinical trial fact record with explicit provenance.

    Registry IDs such as ``NCT01234567`` and explicit project-local identifiers
    are accepted. ``source`` is required even when finer citations are unknown.
    List fields contain only known values, with empty lists representing unknowns.
    ``updated_at`` is the source's update time, not a fabricated retrieval time.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)

    trial_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
    title: NonemptyText
    source: NonemptyText
    conditions: list[NonemptyText] = Field(default_factory=list)
    interventions: list[NonemptyText] = Field(default_factory=list)
    status: NonemptyText | None = None
    phase: NonemptyText | None = None
    sponsor: NonemptyText | None = None
    summary: NonemptyText | None = None
    updated_at: datetime | None = None
    citations: list[NonemptyText] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("trial_id")
    @classmethod
    def identifier_has_no_control_characters(cls, value: str) -> str:
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("trial_id cannot contain control characters")
        return value


class TrialStore:
    """A persistent trial store using parameterized SQLite statements.

    Each operation opens and closes its own connection, allowing independent
    agents and processes to use the same local file. Schema initialization is
    idempotent. Record JSON keeps optional fields extensible without adding
    columns for each new clinical attribute.
    """

    MAX_SEARCH_RESULTS = 100
    _STOPWORDS = frozenset(
        "a about all an and are as at available be by can clinical could describe "
        "details did do does find for from get give had has have how i in information "
        "into is it its list me my of on or our please provide records show stored "
        "studies study tell than that the their them there these they this those to "
        "trial trials us was were what when where which who why will with would you your".split()
    )

    def __init__(self, path: Path | str):
        if str(path) == ":memory:":
            raise ValueError("TrialStore requires a persistent SQLite file path")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS clinical_trials (
                    trial_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS clinical_trials_title "
                "ON clinical_trials (title COLLATE NOCASE)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def _search_text(record: TrialRecord) -> str:
        fields = [
            record.trial_id,
            record.title,
            *record.conditions,
            *record.interventions,
            record.status or "",
            record.phase or "",
            record.sponsor or "",
            record.summary or "",
        ]
        # Python's casefold supports non-ASCII medical and sponsor names, while
        # SQLite's default LIKE folding only handles ASCII reliably.
        return "\n".join(fields).casefold()

    def upsert(self, record: TrialRecord) -> None:
        """Insert or replace facts for the same ID, preserving one record per ID."""
        if not isinstance(record, TrialRecord):
            raise TypeError("record must be a validated TrialRecord")
        # Revalidate mutable nested lists/dicts before persisting. Pydantic's
        # assignment validation cannot observe an in-place list mutation.
        validated = TrialRecord.model_validate(record.model_dump())
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """INSERT INTO clinical_trials (trial_id, title, search_text, record_json)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(trial_id) DO UPDATE SET
                       title = excluded.title,
                       search_text = excluded.search_text,
                       record_json = excluded.record_json""",
                (
                    validated.trial_id,
                    validated.title,
                    self._search_text(validated),
                    validated.model_dump_json(),
                ),
            )

    def get(self, trial_id: str) -> TrialRecord | None:
        """Read one record by its exact ID, or return ``None`` when absent."""
        if not isinstance(trial_id, str):
            raise TypeError("trial_id must be a string")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT record_json FROM clinical_trials WHERE trial_id = ?", (trial_id.strip(),)
            ).fetchone()
        return TrialRecord.model_validate_json(row[0]) if row else None

    def search(self, query: str, limit: int = 5) -> list[TrialRecord]:
        """Find records by meaningful query terms, ranked by matching term count.

        Natural-language filler and generic clinical-trial words are ignored.
        An exact phrase contributes extra weight; ties use stable title/ID order.
        Percent signs, underscores, and SQL metacharacters are always literal.
        An empty query lists records. Limits must be integers from 1 through 100.
        Search only returns saved facts and never synthesizes missing trial data.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if not 1 <= limit <= self.MAX_SEARCH_RESULTS:
            raise ValueError(f"limit must be between 1 and {self.MAX_SEARCH_RESULTS}")
        normalized = query.strip().casefold()
        if not normalized:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT record_json FROM clinical_trials "
                    "ORDER BY title COLLATE NOCASE, trial_id LIMIT ?",
                    (limit,),
                ).fetchall()
            return [TrialRecord.model_validate_json(row[0]) for row in rows]

        # Preserve identifier separators (including underscores) and ignore bare
        # numbers: an irrelevant 'OR 1=1' must not match every numeric trial ID.
        tokens = list(
            dict.fromkeys(
                token
                for token in re.findall(r"\w+(?:[-.:/]\w+)*", normalized)
                if len(token) > 1 and not token.isdecimal() and token not in self._STOPWORDS
            )
        )[:32]
        if not tokens:
            # Literal punctuation/numeric searches remain useful for project IDs
            # and percentages. A question consisting only of filler returns no
            # records rather than an arbitrary list of trial facts.
            if normalized.isdecimal() or any(character in normalized for character in "%_\\"):
                tokens = [normalized]
            else:
                return []

        terms = [normalized, *[token for token in tokens if token != normalized]]
        scoring = " + ".join(
            f"{2 if index == 0 else 1} * (instr(search_text, ?) > 0)"
            for index in range(len(terms))
        )
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT record_json FROM ("
                "SELECT record_json, title, trial_id, "
                f"({scoring}) AS relevance FROM clinical_trials"
                ") WHERE relevance > 0 "
                "ORDER BY relevance DESC, title COLLATE NOCASE, trial_id LIMIT ?",
                (*terms, limit),
            ).fetchall()
        return [TrialRecord.model_validate_json(row[0]) for row in rows]
