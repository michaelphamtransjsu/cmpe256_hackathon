# Multi-agent compatibility audit and repair

Audit date: October 17, 2026 (America/Los_Angeles). Repository: `michaelphamtransjsu/cmpe256_hackathon`. Feature branch: `fix/multi-agent-system-audit`, based on `main` at `08048dec83957f5216c98d0bc4e91ec2f9d7c245`.

## Outcome and operational status

Targeted repairs preserve the public `invoke(...) -> SpecialistResponse`, `consult(...) -> ManagerResponse`, and HTTP contracts. Existing datasets, databases and vector indexes were not deleted or replaced. No production corpus was re-ingested, no model weights were downloaded, and no paid provider calls were made.

| Component | Verified locally | Production evidence / provider status |
| --- | --- | --- |
| Clinical | Import, initialization, independent extractive fixture retrieval, trial precedence, dates, schemas, citations, hosted-model stubs and missing-data abstention pass | Prepared Clinical vector index and trial database were not supplied. CSV originals exist; Clinical PDF originals are absent. Real grounded consultation is blocked until existing artifacts are restored. |
| Corporate Responsibility | Import, initialization, independent extractive fixture retrieval, SOFI year filters, citation validation, hosted-model stubs and missing-data abstention pass | Prepared index is absent. Unpacked SOFI PDFs are absent; an attached ZIP contains both originals but was not extracted/ingested. The reported food-insecurity question correctly abstains with the supplied data. |
| Technology | Import/init without semantic or ingestion extras, independent adapter calls, cached fixture retrieval, family filtering, citations and read-only safeguards pass | Existing 830 chunks and `(830, 384)` vectors match all five Cisco PDF fingerprints. Real semantic queries remain blocked by absent SentenceTransformers and pinned local MiniLM weights. |
| SEC | Import/init, generic available-filings questions, both-company balancing, company/year filters, financial table context, cached fixture retrieval and schemas pass | Existing 2,967 chunks and `(2967, 384)` vectors match both original filing fingerprints/configuration. Real semantic queries remain blocked by the same missing local semantic runtime/model. |
| Manager | All four registered agents, injected routing/synthesis, cross-domain coordination, conflict handling, partial failures, citation namespaces and date JSON pass | Hosted routing and synthesis require OpenAI even with extractive specialists, as in the existing architecture. No key was configured here; actual hosted routing/answer quality is blocked. |
| FastAPI / React | Build, deterministic HTTP integration and browser rendering/error checks pass | Live provider-backed grounded end-to-end answers are blocked by the prerequisites above. A safe setup/error state is not a grounded-answer pass. |

`SPECIALIST_DATA_DIR` locates Clinical/SOFI indexes and Clinical trials. Technology and SEC retain their existing repository-relative `data/...` and `storage/...` paths; changing `SPECIALIST_EMBEDDINGS_PROVIDER` does not change their MiniLM/TF-IDF implementation. Clinical initialization may create an empty SQLite schema when no trial store exists; it never imports CSV records during a query.

## Reproduced failures and root causes

### Technology: reported `ModuleNotFoundError`

Confirmed by importing the original module in the prepared base environment. `agents/technology.py` eagerly imported `shared/hybrid_retrieval.py`, which eagerly imported `sentence_transformers`. That optional package is absent from the base dependency install. Both reported Technology questions fail before retrieval for the same reason.

Complete original traceback:

```text
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
  File "/workspace/cmpe256_hackathon/agents/technology.py", line 8, in <module>
    from shared.hybrid_retrieval import retrieve
  File "/workspace/cmpe256_hackathon/shared/hybrid_retrieval.py", line 11, in <module>
    from sentence_transformers import SentenceTransformer
ModuleNotFoundError: No module named 'sentence_transformers'
```

Further inspection confirmed that even valid prepared caches unnecessarily required Docling imports and chunker setup. Independent specialist queries could also rebuild missing/stale indexes because the HTTP-only guard did not cover every standalone path.

Repair: lazy semantic/TF-IDF imports; defer Docling/PDF conversion/chunking to the explicit ingestion callback; validate prepared caches before model initialization; enforce cache-only/local-model-only behavior for independent queries. Missing dependencies and local weights still fail explicitly with setup guidance. No fallback invents semantic results or replaces the cache with different embeddings.

### SEC: reported generic-question `ValueError`

Confirmed original input-validation failure for:

`What major financial risks are disclosed in the available SEC filings?`

The original `_prepare_query` unconditionally required the literal issuer Cisco or Agilent. This question contains neither, so it raised before retrieval. Because the original retrieval module also had missing optional imports, the original SEC source was executed with only that import stubbed. The failing path never invokes the stub. This reproduces the actual original validation code, not successful semantic retrieval.

Complete dependency-isolated traceback:

```text
Traceback (most recent call last):
  File "<stdin>", line 16, in <module>
  File "/workspace/cmpe256_hackathon/shared/specialist.py", line 44, in invoke
    query, report_year = self._prepare_query(request)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "baseline/agents/sec.py", line 64, in _prepare_query
ValueError: Specify Cisco or Agilent; only their FY2025 Form 10-Ks are indexed
```

Repair: explicit questions about available/indexed/supplied/provided/local/these documents or filings search both existing issuers, with balanced evidence and an explicit Cisco/Agilent FY2025 limitation. Explicit supported company/year filters remain. Requests such as `Alphabet risks` still fail with a clear supported-corpus error. The Manager routing prompt now permits a scoped available-filings survey. SEC shares the lazy, cache-only retrieval fixes.

### Corporate / Manager: original `Synthesis unavailable (ValueError)`

**The exact original incident remains unreproduced and its root cause is unresolved.** The supplied checkout has no Corporate index, no incident traceback, and no saved specialist/synthesis provider output. The original Manager swallowed the exception details. The exception type alone cannot distinguish a provider `ValueError` from the Manager's unknown-narrative-citation error. Corporate namespaced IDs containing underscores are handled by the existing parser; there is no evidence of a Corporate-specific namespace incompatibility.

A controlled regression supplies valid Corporate evidence and an injected synthesis referencing `corporate_responsibility:E999`. The Manager's existing `ValueError("Unknown narrative citation")` reproduces the same UI fallback and preserves the original evidence. This establishes that failure path; it does not establish the cause of the original live incident. Safe traceback logging now preserves every chained stack frame and exception type for future diagnosis, excluding provider messages, source lines and locals.

### Additional confirmed defects

| Defect | Reproduction / repair |
| --- | --- |
| Invented specialist narrative citations accepted | An injected Corporate response cited `[E999]` in its answer while structured evidence used valid `E1`; it passed. Generated hosted analysis now validates bracketed narrative references in addition to structured IDs. Original excerpts and extractive quotations are preserved even if they contain bracketed text. |
| One specialist failure disappeared from UI limitations | Technology and SEC both raised `ModuleNotFoundError`; Manager deduplication stripped agent prefixes and collapsed the identical messages. Deduplication now preserves separately attributed limitations and removes bare synthesis repeats. Regression covers identical two-agent failures and bare-before-attributed ordering. |
| Misleading embedding attribution | Shared extractive text claimed hash embeddings for Technology/SEC because the Clinical/SOFI configuration selected hash. The lexical/hash statement now applies only to the actual shared `SpecialistIndex`. |
| Configured chat model ignored by Manager | Manager forcibly replaced `SPECIALIST_CHAT_MODEL` with `gpt-4.1-mini`. It now passes the configured settings to the existing factory; the default remains `gpt-4.1-mini`. |
| Failures lacked useful stack information | Specialist/synthesis errors were caught without logging; the HTTP boundary only logged the type. A shared safe logger now records complete chained frame locations and types without exception messages or submitted/provider content. |

## Files modified and purpose

| Files | Purpose |
| --- | --- |
| `agents/technology.py` | Independent queries enter the existing request-local read-only retrieval guard. |
| `agents/sec.py` | Scoped generic surveys, balanced issuer retrieval, explicit corpus limitation and clearer unsupported-input error. |
| `shared/hybrid_retrieval.py` | Lazy optional dependencies, prepared-cache validation, pinned local-only query model loading, actionable setup errors; preserve configured revision and existing vectors. |
| `shared/sec_retrieval.py` | Defer ingestion-only imports/construction, guard standalone initialization, avoid creating a cache directory before a missing-cache error. |
| `shared/schemas.py`, `shared/specialist.py` | Shared narrative citation parser, generated-analysis citation validation and correct embedding attribution. Public response schemas remain compatible. |
| `manager.py` | Scoped SEC routing prompt, configured model, safe failure stacks, separately attributed limitations and shared citation parser. Existing hosted routing/synthesis and grounded fallback remain. |
| `shared/diagnostics.py`, `backend/app.py` | Safe chained tracebacks at Manager and HTTP error boundaries; safe public errors unchanged. |
| `pyproject.toml` | `retrieval-runtime` extra for serving prepared semantic caches without Docling ingestion dependencies; declare `python-dotenv` used by the existing opt-in live test. Existing full `retrieval` extra remains available. |
| `.env.example`, `README.md`, `frontend/README.md`, this report | Explain provider/model/data requirements, no automatic `.env` loading, no query ingestion, exact setup and evidence-backed limitations. |
| `tests/test_agents.py`, `tests/test_technology.py`, `tests/test_sec_agent.py`, `tests/test_sec_retrieval.py` | Independent invocation, generic SEC regression, narrative grounding and read-only/cache behavior. |
| `tests/test_hybrid_retrieval.py`, `tests/test_cache_portability.py` | Prepared-cache serving without ingestion imports, no downloads/writes, corrupt/stale cache failures, explicit real-model prerequisite skip, source/config invalidation checks retained. |
| `tests/test_manager.py`, `tests/test_diagnostics.py`, `tests/test_web.py` | Manager configuration/attribution/fallback stacks, sanitized errors, cross-domain HTTP evidence and clinical date serialization. |

No React component or visual redesign was needed. Existing API validation, loading/retry behavior, source labels, participating-agent rendering and citation disclosure already matched the contract.

## Automated validation

Final full suite: **292 passed, 5 skipped**, exit 0, in 56.67 seconds. Frontend TypeScript/Vite production build passed. The 16 FastAPI tests passed, including both cross-domain HTTP cases, date serialization, partial failures and sanitized stack checks. All four specialist imports and constructors also passed in an isolated empty-data smoke check.

The final Chromium smoke passed the expected safe no-key HTTP 503/retry flow and nine separately mocked UI checks, with zero browser page errors; all four captured request bodies contained only `question`. Positive production retrieval was not claimed. Commands:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q -ra
cd frontend
npm run build
```

The original `main` was independently materialized under ignored `work/baseline/`. In the same base runtime, its full pytest run stops at collection with three missing-`sentence_transformers` import errors: SEC agent, SEC retrieval and Technology tests. The repaired modules import without those optional packages; unavailable real-model tests are explicitly blocked/skipped instead of reported as passed.

Five expected full-suite skips:

1. Real semantic cache relocation/query equivalence: SentenceTransformers and the pinned local weights are unavailable. Cache fingerprint/config invalidation tests still execute with controlled model construction.
2. Real SOFI source retrieval: opt-in data test; unpacked PDFs/index are absent.
3. Real Clinical PDF retrieval: opt-in data test; PDF originals/index are absent.
4. Real Clinical CSV import/retrieval: opt-in test intentionally not completed; production CSV records were not imported for this audit.
5. Technology live hosted diagnostic-answer regression: explicit opt-in requires a key, provider access and semantic runtime/model.

OpenAI-compatible structured output is tested with injected models. These tests verify contracts, policy/data wiring, validation and orchestration; they do not establish real provider availability or semantic compliance of generated prose. Real provider routing/synthesis was blocked, not passed.

The managed command sandbox initially denied local networking needed by TestClient/event-loop wakeups, causing hangs before request handling. HTTP and full-suite runs use the supported network-enabled execution permission. The same endpoint tests then complete. This is an execution-sandbox requirement, not an application dependency workaround.

## End-to-end integration coverage

- All four individual routes are exercised with injected routing decisions, validated specialist responses and namespaced synthesis citations.
- Cross-domain cases include Technology + SEC, Clinical + Corporate Responsibility, and Technology + Clinical. Tests preserve both domains' evidence, reporting periods, trial dates, limitations and original source excerpts.
- Missing/failed specialists prevent an unsupported combined conclusion. Existing separate-advice fallback preserves grounded content when synthesis fails validation or provider access.
- Conflicting specialist content is supplied to synthesis with attribution and retained in the response; deterministic tests verify explicit unresolved-conflict handling. Real model conflict resolution remains unverified.
- HTTP tests cover actual Manager execution with deterministic model/specialist fixtures, JSON date serialization, strict question-only requests, invalid inputs, safe 503 errors, partial failures, source IDs and namespace preservation.
- Final browser smoke verifies the production React/Vite/FastAPI no-key error flow. Positive citation/agent/loading/retry/history/mobile rendering uses clearly identified browser fixtures, not fabricated production evidence. Full hosted grounded browser success remains blocked.
- Keys remain backend environment variables. No key is sent in Vite variables, browser requests or saved deliverables. Source links allow HTTP(S) only; local paths display portable labels. No arbitrary document ingestion runs in `/api/chat` or specialist query paths.

## Dependencies, evidence setup and remaining limits

Use Python 3.11+ and Node 22.12+. The audit used Python 3.12 with FastAPI 0.143.3, Starlette 1.8.0, Pydantic 2.14.1, LangChain Core 1.6.12, LangChain OpenAI 1.7.2, NumPy 2.5.4, scikit-learn 1.9.1, pytest 9.1.1, httpx 0.28.1 and uvicorn 0.54.0. The web/test extras installed successfully. The semantic/ingestion stack was not fully installed or validated here; its absence is an explicit blocker.

Restore existing prepared artifacts before full grounded operation:

- Clinical/SOFI: `data/indexes/clinical/vectors.json`, `data/indexes/corporate_responsibility/vectors.json`, and the existing `data/clinical/trials.sqlite3`, with the embedding identity used when those indexes were created.
- Technology/SEC: preserve `storage/technology/`, `storage/sec/` and their existing source PDFs. Cache identities depend on source fingerprints/configuration; do not edit them to force a match.
- Semantic model: `sentence-transformers/all-MiniLM-L6-v2`, pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Query mode uses local files only. Model acquisition is separate from document ingestion.
- Hosted model: backend `OPENAI_API_KEY` and access to the OpenAI endpoint/model. No credential or outbound OpenAI identity was configured in this environment; Hugging Face is not in its supported HTTP allowlist.

The SEC corpus is only Cisco FY2025 (ended July 26, 2025) and Agilent FY2025 (ended October 31, 2025). Generic surveys disclose this scope. Company detection is conservative and is not general issuer recognition: `Alphabet risks` is rejected, but adding `in the available SEC filings` selects the scoped Cisco/Agilent corpus. Those excerpts cannot answer an Alphabet-specific question; hosted analysis must state that gap, and extractive output remains only labeled excerpts from the indexed issuers. Explicit unsupported-issuer questions should be clarified rather than interpreted as issuer coverage. Retrieved risks are incomplete excerpts, not probability/severity estimates.

If the missing Clinical/SOFI artifacts cannot be restored, ingestion would be a separate necessary setup operation requiring approval. It was not performed. The exact original Corporate synthesis cause still requires the original output/traceback or a rerun against the restored index and configured provider.

## Exact WSL commands

Assume the existing checkout is `~/cmpe256_hackathon`; substitute only that directory if yours differs. These commands install query dependencies and start both processes. They do not rebuild document indexes. Reuse restored prepared evidence as described above.

One-time setup (Ubuntu WSL, Python 3.11+, Node 22.12+):

```bash
cd ~/cmpe256_hackathon
git fetch origin
git switch fix/multi-agent-system-audit
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[web,retrieval-runtime,test]'
cd frontend
npm ci
npm run build
```

If the exact model is not already cached, this explicit setup command downloads only model assets (requires Hugging Face access; no PDFs are parsed/indexed):

```bash
cd ~/cmpe256_hackathon
.venv/bin/python - <<'PY'
from sentence_transformers import SentenceTransformer
from shared.hybrid_retrieval import MODEL, MODEL_REVISION
SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
PY
```

Terminal 1 — backend, preserving the embedding provider that matches your restored indexes:

```bash
cd ~/cmpe256_hackathon
read -rsp 'OpenAI API key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY
export SPECIALIST_DATA_DIR="$PWD/data"
export SPECIALIST_CHAT_MODEL=gpt-4.1-mini
export SPECIALIST_LLM_PROVIDER=openai
# Set this to openai instead if your restored Clinical/SOFI vectors used OpenAI.
export SPECIALIST_EMBEDDINGS_PROVIDER=hash
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2 — frontend:

```bash
cd ~/cmpe256_hackathon/frontend
npm run dev
```

Open `http://localhost:5173` in Windows. Vite proxies `/api` to WSL FastAPI on port 8000. `.env` files are not loaded automatically. Never set `VITE_OPENAI_API_KEY`. For local specialist extracts use `SPECIALIST_LLM_PROVIDER=extractive`; the Manager still requires hosted routing/synthesis.

Verification, in a third WSL terminal:

```bash
cd ~/cmpe256_hackathon
curl --fail http://127.0.0.1:8000/api/health
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q -ra
.venv/bin/python -m agents.clinical 'What clinical trial evidence is available?'
.venv/bin/python -m agents.corporate_responsibility 'What are the major causes of global food insecurity?'
.venv/bin/python -m agents.technology 'Which Catalyst 8300 has two 10Gbps ports?'
.venv/bin/python -m agents.sec 'What major financial risks are disclosed in the available SEC filings?'
curl --fail-with-body http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Compare Cisco router ports and risks disclosed in the available SEC filings."}'
```

The final five consultation commands use hosted generation with the environment above and may incur API usage. Tests use deterministic/injected providers unless explicitly opted in. Health is liveness only and does not establish that indexes/models/provider access are ready. Stop each server with Ctrl+C.
