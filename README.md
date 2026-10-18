# Clinical and Corporate Responsibility specialists

Two independent Python specialists for an enterprise management-consulting team. This package contains no manager, UI, SEC agent, or Technology agent.

- **Clinical** retrieves clinical/Agilent evidence and searches source-attributed trial records in SQLite. Its model prompt distinguishes trial registration from efficacy and identifies missing clinical evidence.
- **Corporate Responsibility** retrieves assigned SOFI 2023/2024 evidence. Its model prompt preserves geography, units, publication/observation years, and differences between revisions and trends.

Both expose `invoke(question | request) -> SpecialistResponse`, share a validated JSON contract, and have separate persistent vector indexes. No background service is required. Document ingestion is a separate command; answering queries reads the existing indexes and does not ingest documents.

## Structure

```text
agents/
  __init__.py
  clinical.py                   # Clinical agent, policy, and query CLI
  corporate_responsibility.py   # Corporate agent, report scope, and query CLI
shared/
  __init__.py
  retrieval.py                  # Persistent LangChain retrieval
  schemas.py                    # Validated requests, responses, and evidence
  model.py                      # Settings and chat-model configuration
  specialist.py                 # Reusable query/response runtime
  cli.py                        # Shared answering-only CLI helper
  trials.py                     # SQLite clinical-trial helper
  trial_csv.py                  # Validated CTG CSV mapping and provenance
  sources.py                    # Safe inventory, manifests, and ingestion
  html_evidence.py              # Passive HTML text extraction
tests/                         # Unit/integration tests and real-source validator
ingest.py                      # Document/trial ingestion and inventory CLI
requirements.txt               # Tested dependency pins
.env.example
.gitignore
README.md
```

The code uses a flat Python layout with one primary module per specialist and reusable helpers in `shared/`. `pyproject.toml` provides optional editable installation and pytest configuration; `sources/` holds the source manifest and compact validation evidence. Original documents, local indexes, and SQLite records stay in ignored `data/`.

## Local inputs

The pipeline accepts the existing extracted tree directly:

```text
data/
  Clinical/
    Covid/*.pdf
    Diabetes/*.pdf
    Heart_attack/*.pdf
    KneeInjuries/*.pdf
    ctg-studies_*.csv
  FoodSecurity/
    SOFI-2023.pdf
    SOFI-2024.pdf
  Agilent.txt
```

Only Clinical PDFs, the named clinical-trial CSV exports, and these two SOFI reports are assigned from this tree. Retail, SEC, Cisco, Google, and other unrelated sources are excluded or remain unassigned. Windows `:Zone.Identifier` sidecars are ignored. The CSV headers were inspected before implementing the mapping: the four supplied exports contain 32,611 rows, including 219 identical duplicate registrations, yielding 32,392 unique trial IDs.

Original inputs, manifests, local indexes, backups, and SQLite records belong in ignored `data/` or `work/`. Ignore rules do not untrack datasets already present in repository history; this change adds no datasets to version control.

## Install and test

Python 3.11+; validated here with 3.12. From the existing checkout:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

`requirements.txt` pins the tested runtime and test dependencies. Commands run directly from the checkout; editable installation is optional (`.venv/bin/python -m pip install --no-deps -e .`). `pyproject.toml` declares compatible update ranges and configures pytest to import from the repository root. Each cloud task already has an isolated checkout; no worktree is needed.

## Ingest the existing extracted data

Run from the repository root after installing dependencies. These commands create a new manifest, index local PDFs for each agent, and import all assigned clinical CSVs into the existing SQLite store. They never fetch URLs or execute document instructions.

```bash
.venv/bin/python ingest.py discover data \
  --manifest data/extracted-manifest.json --local-only
.venv/bin/python ingest.py ingest clinical \
  --manifest data/extracted-manifest.json
.venv/bin/python ingest.py ingest corporate_responsibility \
  --manifest data/extracted-manifest.json
.venv/bin/python ingest.py trials-import \
  --manifest data/extracted-manifest.json

# Query saved evidence after ingestion; these commands do not ingest inputs.
.venv/bin/python -m agents.clinical 'diabetes recruiting' --max-sources 3
.venv/bin/python -m agents.corporate_responsibility \
  'food insecurity drivers conflict climate prices' --report-year 2023
.venv/bin/python -m agents.corporate_responsibility \
  'food insecurity drivers conflict climate prices' --report-year 2024
.venv/bin/python ingest.py trials-search 'diabetes recruiting'
```

`discover` refuses to overwrite an existing manifest. For a new inventory, choose a new manifest filename. Reuse an existing manifest to repeat ingestion/import; trial imports skip identical records. `--local-only` explicitly defers the URLs listed in Agilent.txt to `reference_only` entries, preserving each URL and its catalog attribution while selecting local evidence as the corpus. Omit the flag when external references must be resolved before Clinical index replacement. Deferred URLs can be attached later using the command below. CSV import is separate from PDF ingestion: CSVs are never embedded as prose, and document queries never import CSVs.

| Input | Assignment |
| --- | --- |
| `data/Clinical/**/*.pdf` | Clinical research evidence |
| `data/Clinical/ctg-studies_*.csv` | Clinical SQLite trial records, separately validated |
| `data/FoodSecurity/SOFI-2023.pdf`, `SOFI-2024.pdf` | Corporate Responsibility; publication years 2023 and 2024 |
| URLs actually listed in `Agilent.txt` | Clinical catalog references; explicit download/association required |
| `Clinical.zip` / `Clinical.zip.contents` descendants | Existing ZIP evidence workflow; named CTG CSV exports supported |
| `FoodSecurity.zip` / `FoodSecurity.zip.contents` SOFI reports | Existing ZIP evidence workflow |
| Assignment/project-scope PDFs | Reference only |
| Cisco, Google, SEC, Technology/Tech, Retail | Excluded from both indexes |
| Other files and Windows metadata | Unassigned/unsupported or ignored; never indexed |

Discovery accepts a data root or its `Clinical/` or `FoodSecurity/` subdirectory. Nested lookalikes such as `Retail/Clinical/` and arbitrary SOFI PDFs elsewhere do not grant assignment. Extracted Clinical TXT/HTML files are unassigned; the original ZIP evidence formats remain supported.

Manifest entries retain `source_id`, `agent`, `topic`, `format`, `status`, `original_member_path`, `local_path`, and SHA-256 for inspected local inputs. SOFI entries include `report_year` from discovery onward. Document metadata preserves both paths and the verified hash. Citations retain original source paths, one-based PDF pages, and report publication years. Scanned/empty PDFs fail visibly and may need separate OCR preparation.

Agilent URLs are not fetched implicitly. Download a listed document, verify its identity, then associate it with its catalog entry:

```bash
.venv/bin/python ingest.py attach-source \
  --manifest data/extracted-manifest.json --source-id SOURCE_ID \
  --path /path/to/downloaded-source.html
.venv/bin/python ingest.py ingest clinical --manifest data/extracted-manifest.json
```

### ZIP alternative

```bash
.venv/bin/python ingest.py inspect /path/to/Hackathon_export.zip \
  --output data/extracted --manifest data/zip-manifest.json
.venv/bin/python ingest.py ingest clinical --manifest data/zip-manifest.json
.venv/bin/python ingest.py ingest corporate_responsibility --manifest data/zip-manifest.json
# If the ZIP contains assigned ctg-studies_*.csv exports:
.venv/bin/python ingest.py trials-import --manifest data/zip-manifest.json
```

Original archives remain intact. Archive limits: depth 4, 10,000 entries, 64 MiB per member, 256 MiB cumulative decompression, ratio 250. Traversal, symlinks, unsafe types, and differing-file overwrites are rejected. Failures remain visible in the manifest.

### Existing index safeguards

Missing, failed, pending, or unsupported assigned document sources block index replacement; trial CSV validation belongs to the separate importer. Archive inspection failures also block replacement because their contents are unknown. No readable documents preserves the existing index unless `--allow-empty` is explicit. Source hashes are rechecked before ingestion; changed bytes require a new inventory or deliberate URL reattachment.

Successful ingestion atomically replaces one specialist's complete corpus. Use a complete manifest rather than successive partial manifests. Before replacing any existing snapshot, ingestion writes a durable, content-keyed backup to `data/indexes/AGENT/backups/vectors-SHA256.json`. Backup failure preserves the existing snapshot. To restore, stop ingestion and copy the selected backup to that agent's `vectors.json`; use the original embedding settings. Changing providers/models requires a new data directory or a deliberate backup/migration.

## Run either specialist

Defaults are **offline lexical retrieval and extractive responses**: no key or model download. Hashed vector candidates are reranked using BM25 term relevance. Corporate question wording receives small retrieval-only expansions (for example, war → conflict) and report-title boilerplate is removed from lexical scoring. Only retrieved text becomes evidence. Answers quote exact evidence and provide no synthesized recommendations.

```bash
.venv/bin/python -m agents.clinical 'What evidence describes recruitment?'
.venv/bin/python -m agents.corporate_responsibility \
  'What supports food-affordability initiatives?' --report-year 2024 --max-sources 4
```

Without relevant sources, responses have `status: insufficient_evidence` and empty findings/citations. Model knowledge never substitutes for a missing corpus.

```python
from agents.clinical import ClinicalAgent
from agents.corporate_responsibility import CorporateResponsibilityAgent
from shared.model import Settings

settings = Settings.from_env()
clinical = ClinicalAgent(settings)
corporate = CorporateResponsibilityAgent(settings)
result = clinical.invoke({"question": "What evidence describes recruitment?", "max_sources": 3})
payload = result.model_dump(mode="json")
# corporate.invoke(...) has the same input/output shape.
```

Responses contain `agent`, `question`, `status`, `mode`, `answer`, `findings`, `recommendations`, `limitations`, `citations`, `trial_records`, and the selected `report_year`. Every finding/recommendation requires known evidence IDs. Citation excerpts come from retrieval, never the model; SOFI citations include their publication year. Unknown IDs, wrong-year citations, and malformed model outputs raise validation errors. Reference validation does not prove that a model interpretation follows from a citation; review synthesized advice against originals.

Corporate requests can specify `report_year` (CLI: `--report-year`). Explicit wording such as “SOFI-2023” or “2023 report” also selects that publication before retrieval, so 2024 chunks cannot displace 2023 evidence. A bare observation year such as “diet costs in 2022” does not select a report; comparisons may use both. A missing requested report returns insufficient evidence. Clinical rejects SOFI report-year filters.

`max_sources` bounds combined document/trial evidence; Clinical prioritizes matching trial records. String/dictionary requests without a limit use `SPECIALIST_TOP_K`. Explicit `SpecialistRequest` objects use their validated value.

## Trial records

SQLite remains at `data/clinical/trials.sqlite3` (lowercase `clinical`, as in the existing implementation). The validated trial-record schema requires `trial_id`, `title`, and `source`; optional fields remain empty when absent. CSV records use these observed columns:

| CSV column | Trial record field |
| --- | --- |
| `NCT Number` | `trial_id`; validated as NCT plus eight digits |
| `Study Title` | `title`; required, nonempty |
| `Study URL` | `source` and citation; if empty, use the original CSV path and record number |
| `Conditions`, `Interventions` | `conditions`, `interventions`; split only on `|` |
| `Study Status`, `Phases`, `Sponsor`, `Brief Summary` | `status`, `phase`, `sponsor`, `summary` |
| `Last Update Posted` | `updated_at` for a complete ISO date; no invented time/timezone |
| All original columns | `metadata.csv_columns`, preserving raw strings and unmapped fields |

The importer preserves a date-only update as a date. Partial ISO dates stay in raw metadata with `updated_at` unset. It never uses file modification time, infers outcomes from registration status, or invents a registry URL. A supplied Study URL must be the matching HTTPS ClinicalTrials.gov study URL. Each record retains file SHA-256, source ID, original/local paths, CSV record number, physical line range, and citations under `metadata.csv_provenance`. Record numbers start at 2 after the header and differ from physical lines for multiline values.

UTF-8 BOMs, quoted multiline values, and fields up to 16 MiB are supported. Invalid headers, malformed rows, missing required facts, invalid IDs/dates/URLs, symlinks, changed hashes, and contradictory duplicate IDs fail before writes. Identical duplicate CSV registrations merge all file/row attributions. Every file and record is validated before one SQLite transaction writes the batch; write failures roll back the batch. Records absent from an import are never deleted.

Additional supported inputs:

```bash
.venv/bin/python ingest.py trials-import data/Clinical
.venv/bin/python ingest.py trials-import data/Clinical/ctg-studies_diabetes.csv
.venv/bin/python ingest.py trials-import /path/to/verified-trials.json
```

JSON input remains an array of trial-record objects. Existing IDs with changed content fail by default and preserve the database. To deliberately replace changed records, use the same import command with `--replace-existing`. This creates a consistent SQLite backup at `data/clinical/backups/trials-*.sqlite3` before any replacements and prints its path. Extra verified CSV origins can be added without discarding existing origins. Conflicting IDs inside one input batch always fail. To restore, stop writers and restore the selected SQLite backup. `TrialStore.upsert` remains the explicit single-record replacement API; `import_records` provides the safeguarded batch import.

Corporate Responsibility never reads the trial store. Clinical queries retrieve saved records and cite their supplied registry URL or CSV location.

## Configuration

Use [.env.example](.env.example) as a reference; `.env` files are not automatically executed. Set secrets securely in the process environment.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SPECIALIST_DATA_DIR` | `./data` | Index/SQLite location |
| `SPECIALIST_LLM_PROVIDER` | `extractive` | `extractive` or `openai` |
| `SPECIALIST_EMBEDDINGS_PROVIDER` | `hash` | Offline lexical `hash` or semantic `openai` |
| `SPECIALIST_CHAT_MODEL` | `gpt-4.1-mini` | Opt-in chat model |
| `SPECIALIST_EMBEDDING_MODEL` | `text-embedding-3-small` | Opt-in embedding model |
| `SPECIALIST_TOP_K` | `5` | Default evidence limit, 1–20 |
| `SPECIALIST_MIN_SCORE` | `0.08` | Minimum cosine score, 0–1 |
| `OPENAI_API_KEY` | unset | Required only for OpenAI components |

For synthesized analysis, set `SPECIALIST_LLM_PROVIDER=openai`. For semantic retrieval, also set `SPECIALIST_EMBEDDINGS_PROVIDER=openai` **before ingestion**. Hosted mode requires access to `api.openai.com` and an approved model deployment for your evidence. No credential values are written to source or snapshots. Hosted requests were not executed during current validation.

The durable LangChain vector JSON snapshot records its embedding identity and rejects incompatible loads. To change providers/models, ingest into a new data directory or back up and deliberately remove only the old vector snapshot. No pickle deserialization is used. Vectors and SQLite survive process restarts. Local ingestion assumes one writer; this is not a distributed vector service. Hash embeddings are lexical, not semantic.

## Validation

Run the automated suite without private data:

```bash
.venv/bin/python -m pytest -q
```

Tests cover the original ZIP workflow, extracted-folder discovery, exclusions, Windows sidecars, CSV mapping/validation, missing values, long/multiline fields, duplicate provenance, atomic writes, conflict safeguards/backups, SQLite persistence, both agents, report-year filtering, source hashes, page citations, index backups, and answering without re-reading source documents. Synthetic fixtures remain in temporary test directories.

Optional real-data tests use temporary indexes/databases and leave existing stores untouched. They skip with an explicit reason unless enabled and the corresponding originals are present:

```bash
.venv/bin/python -m pytest -q -rs tests/test_real_data.py \
  --run-real-data --real-data-dir data
```

The real CSV test validates and imports every supplied trial, checks database round trips and idempotence, and retrieves a cited record through Clinical. The clinical PDF test requires actual `Clinical/**/*.pdf` files. The SOFI test requires both actual reports, builds a temporary directory-based index, checks retrieval for each publication year, and compares citations against original PDF pages and hashes.

For a deeper check of a saved corporate index after ingestion:

```bash
.venv/bin/python tests/validate_corporate_sources.py \
  --manifest data/extracted-manifest.json --data-dir data \
  --output data/corporate-validation.json
```

That validator checks every corporate chunk against its original PDF page, source/year/hash attribution, focused questions, and retrieval after a process restart. Real Clinical PDF retrieval requires local originals; Windows metadata sidecars do not contain document text. Hosted model/embedding behavior requires configured credentials and network access and is not part of offline validation.
