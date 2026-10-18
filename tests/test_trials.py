"""SQLite tests use explicitly synthetic records, never invented source evidence."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from shared.trials import TrialRecord, TrialStore


def synthetic_trial(**overrides):
    fields = {
        "trial_id": "SYNTHETIC-001",
        "title": "Synthetic trial fixture",
        "conditions": ["Synthetic condition"],
        "interventions": ["Synthetic intervention"],
        "source": "test fixture; not a real clinical trial",
        "citations": ["tests/test_trials.py"],
    }
    fields.update(overrides)
    return TrialRecord(**fields)


def test_records_survive_new_store_instance_and_round_trip_types(tmp_path):
    path = tmp_path / "nested" / "trials.sqlite3"
    trial = synthetic_trial(
        updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        metadata={"synthetic": True, "count": 2, "nested": {"missing": None}},
    )
    TrialStore(path).upsert(trial)

    reopened = TrialStore(path)
    assert reopened.get(trial.trial_id) == trial
    assert reopened.get("unknown") is None
    assert reopened.search("CONDITION") == [trial]


def test_repeated_upsert_updates_one_record_and_search_index(tmp_path):
    store = TrialStore(tmp_path / "trials.sqlite3")
    store.upsert(synthetic_trial(title="Former synthetic title"))
    replacement = synthetic_trial(title="Replacement synthetic title", status="Synthetic status")
    store.upsert(replacement)
    store.upsert(replacement)

    assert store.get(replacement.trial_id) == replacement
    assert store.search("") == [replacement]
    assert store.search("former") == []
    assert store.search("status") == [replacement]


def test_search_is_parameterized_and_sql_wildcards_are_literal(tmp_path):
    store = TrialStore(tmp_path / "trials.sqlite3")
    ordinary = synthetic_trial()
    literal = synthetic_trial(trial_id="SYNTHETIC-002", title="Synthetic 50%_fixture\\test")
    store.upsert(ordinary)
    store.upsert(literal)

    assert store.search("' OR 1=1; DROP TABLE clinical_trials; --") == []
    assert store.search("%_fixture\\") == [literal]
    assert store.search("%") == [literal]
    assert store.get("' OR 1=1 --") is None
    assert len(store.search("")) == 2


def test_search_is_stable_bounded_and_covers_clinical_fields(tmp_path):
    store = TrialStore(tmp_path / "trials.sqlite3")
    second = synthetic_trial(trial_id="SYNTHETIC-B", title="B synthetic", sponsor="Straße fixture")
    first = synthetic_trial(trial_id="SYNTHETIC-A", title="A synthetic", phase="Synthetic phase")
    store.upsert(second)
    store.upsert(first)

    assert store.search("synthetic", limit=1) == [first]
    assert store.search("synthetic") == [first, second]
    assert store.search("strasse") == [second]
    assert store.search("phase") == [first]
    for limit in (0, -1, 101):
        with pytest.raises(ValueError, match="limit"):
            store.search("", limit=limit)
    for limit in (True, 1.0, "5"):
        with pytest.raises(TypeError, match="limit"):
            store.search("", limit=limit)


def test_natural_language_search_ranks_relevant_trials_and_matches_registry_ids(tmp_path):
    store = TrialStore(tmp_path / "trials.sqlite3")
    recruiting = synthetic_trial(
        trial_id="NCT00000001",
        title="Z synthetic diabetes trial",
        conditions=["diabetes"],
        status="Recruiting",
    )
    completed = synthetic_trial(
        trial_id="NCT00000002",
        title="A synthetic diabetes trial",
        conditions=["diabetes"],
        status="Completed",
    )
    unrelated = synthetic_trial(
        trial_id="NCT00000003", title="Unrelated synthetic trial", conditions=["asthma"]
    )
    for trial in (completed, unrelated, recruiting):
        store.upsert(trial)

    assert store.search("What trials for diabetes are recruiting?") == [recruiting, completed]
    assert store.search("Tell me about trial nct00000002") == [completed]
    assert store.search("What clinical trials are available?") == []
    assert store.search("' OR 1=1; DROP TABLE clinical_trials; --") == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"trial_id": " "},
        {"trial_id": "invalid\nidentifier"},
        {"title": ""},
        {"source": " "},
        {"conditions": [""]},
        {"citations": [" "]},
        {"unexpected_field": "not part of the schema"},
    ],
)
def test_record_rejects_missing_provenance_and_invalid_facts(overrides):
    with pytest.raises(ValidationError):
        synthetic_trial(**overrides)


def test_record_does_not_invent_unknown_trial_attributes():
    trial = TrialRecord(trial_id="PROJECT-001", title="Only known title", source="synthetic fixture")
    assert trial.conditions == []
    assert trial.interventions == []
    assert trial.status is None
    assert trial.phase is None
    assert trial.sponsor is None
    assert trial.summary is None
    assert trial.updated_at is None


def test_store_revalidates_mutated_nested_values_before_persisting(tmp_path):
    trial = synthetic_trial()
    trial.citations.append(" ")
    store = TrialStore(tmp_path / "trials.sqlite3")
    with pytest.raises(ValidationError):
        store.upsert(trial)
    assert store.search("") == []


def test_store_requires_validated_records_and_persistent_path(tmp_path):
    store = TrialStore(tmp_path / "trials.sqlite3")
    with pytest.raises(TypeError, match="TrialRecord"):
        store.upsert({"trial_id": "not validated"})
    with pytest.raises(ValueError, match="persistent"):
        TrialStore(":memory:")
