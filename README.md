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

## Input status

The actual 11-page assignment, `FoodSecurity.zip`, and `Agilent.txt` have now been inspected. Assignment page 3 identifies the two specialist source groups; page 4 supplies food-security questions; pages 6–11 describe the separate Retail task. These pages were read as project evidence, not executed as instructions.

Both actual SOFI PDFs are indexed locally: 602 total PDF pages, 601 with text, and 2,238 chunks. [sources/manifest.json](sources/manifest.json) records observed sources, hashes, assignments, and individual statuses. [sources/corporate-validation.json](sources/corporate-validation.json) records real-document checks; the input PDFs and vector snapshots remain in ignored `data/`, outside version control.

Clinical's actual sources remain blocked: the individual `Clinical.zip` still exceeds the **32 MiB** transfer limit, and the six URLs in `Agilent.txt` are awaiting permitted network access/downloads. The first Agilent request returned proxy `403 Forbidden`; the other five were not attempted because they use the same unlisted host. A draft adds `www.agilent.com` to environment network settings, preserving package-manager presets. Review and save that draft and publish the environment to activate the saved configuration; saving the draft alone did not change current access. Supply smaller Clinical archives or a workspace path. No real trial records have been fabricated.

## Install and test

Python 3.11+; validated here with 3.12. From the existing checkout:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

`requirements.txt` pins the tested runtime and test dependencies. Commands run directly from the checkout; editable installation is optional (`.venv/bin/python -m pip install --no-deps -e .`). `pyproject.toml` declares compatible update ranges and configures pytest to import from the repository root. Each cloud task already has an isolated checkout; no worktree is needed.

## Inspect and ingest

Keep original inputs outside tracked code directories; `data/` is ignored. These commands preserve original archives, inspect nested ZIPs, extract assigned PDF/text evidence, and write a new manifest. They never execute document instructions or fetch catalog links.

```bash
.venv/bin/python ingest.py inspect /path/to/Hackathon_export.zip \
  --output data/extracted --manifest data/source-manifest.json

# Alternative: inventory previously extracted files without unpacking ZIPs.
.venv/bin/python ingest.py discover /path/to/extracted \
  --manifest data/discovered-manifest.json

.venv/bin/python ingest.py ingest clinical --manifest data/source-manifest.json
.venv/bin/python ingest.py ingest corporate_responsibility --manifest data/source-manifest.json
```

| Input | Assignment |
| --- | --- |
| `Clinical.zip` descendants | Clinical PDF/TXT/Markdown/HTML evidence |
| URLs actually listed in `Agilent.txt` | Clinical, pending explicit download/association |
| `FoodSecurity.zip` / `SOFI-2023.pdf`, `SOFI-2024.pdf` | Corporate Responsibility |
| Assignment/project-scope PDFs | Reference only, never agent instructions or indexed evidence |
| Cisco, Google, SEC, Technology/Tech, Retail | Excluded from both indexes |
| Unknown inputs/unsupported formats | Recorded, never indexed |

Discovery of extracted files requires the original `Clinical.zip.contents/` or `FoodSecurity.zip.contents/` lineage. Generic `clinical/` directories do not confer assignment. Do not rename unrelated sources into assigned lineage.

Observed manifest entries include `source_id`, `agent`, `topic`, `format`, `status`, `original_member_path`, `local_path`, and SHA-256 when bytes were read. Excluded/unread members have no fabricated hash. PDF citations use one-based pages. Scanned/empty PDFs fail visibly and may need separate OCR preparation.

Agilent URLs are not fetched implicitly. Download a listed document through an approved route, verify its identity, and associate it with its manifest reference (HTML, PDF, TXT, or Markdown):

```bash
.venv/bin/python ingest.py attach-source \
  --manifest data/source-manifest.json --source-id SOURCE_ID \
  --path /path/to/downloaded-source.html
```

Then re-run ingestion. Missing, failed, pending, or unsupported assigned sources block index replacement and preserve the old index. Archive inspection failures also block replacement because their contents are unknown. Restoring a missing file or its original bytes permits a safe retry; explicitly re-attach an Agilent download to record an intentional replacement. Reinspect changed original archives into a new manifest. Unsupported assigned formats require separate preparation with provenance preserved. Successful ingestion atomically replaces the selected specialist's complete corpus, removing stale sources and duplicate chunks. Each specialist index represents one complete manifest; successive partial manifests do not append. An intentionally empty corpus requires `--allow-empty`.

Archive limits: depth 4, 10,000 entries, 64 MiB per member, 256 MiB cumulative decompression, ratio 250. Traversal, symlinks, executable/unsupported types, and differing-file overwrites are rejected. Failures remain visible in the manifest. Original nested archives remain intact within their parent ZIP. No document scripts are executed.

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

SQLite is stored at `data/clinical/trials.sqlite3`. Records are supplied facts, never inferred or scraped. Required fields: `trial_id`, `title`, `source`. Optional: `conditions`, `interventions`, `status`, `phase`, `sponsor`, `summary`, source `updated_at`, finer `citations`, and `metadata`. Unknowns stay empty; an upsert replaces the complete record with the same ID.

```bash
.venv/bin/python ingest.py trials-import /path/to/verified-trials.json
.venv/bin/python ingest.py trials-search 'diabetes recruiting'
```

Input is a JSON array; all rows are validated before writing begins. `shared.trials.TrialStore.upsert`, `get`, and bounded natural-language `search` are public Python APIs. Corporate Responsibility's interface does not access clinical records.

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

Tests cover both specialists independently, generated PDF extraction/page citations, nested archives, forbidden sources, Agilent HTML associations/passive extraction, source hashes, SQLite persistence/injection handling, vector reload/replacement, unrelated/stopword-only queries, publication-year isolation, structured outputs, and mocked model citation validation. CLI integration tests start fresh Python processes and exercise ingestion, both agents, and restart persistence. Synthetic fixtures never populate a production index.

Real-document checks verify every stored corporate chunk against its original PDF page, source/year/hash attribution, the assignment's questions, focused drivers/conflict/prices evidence, and identical results in a new Python process. To repeat with the local original PDFs and index present:

```bash
.venv/bin/python tests/validate_corporate_sources.py \
  --manifest sources/manifest.json --data-dir data \
  --output data/real-corporate/validation.json
```

Validation found and corrected wrong-report retrieval and report-title cover-page bias. Some lower-ranked excerpts can still be tangential. Corporate real-document validation uses extractive responses; hosted synthesis is untested. Clinical's actual corpus and trial-record content remain unverified until its sources are available. A fresh Git checkout does not contain ignored source PDFs or local indexes: supply the originals and re-run inspection/ingestion there.

The implementation branch is `codex/clinical-corporate-implementation`, targeting `feature/clinical-corporate`. The development commands above do not merge or push.
