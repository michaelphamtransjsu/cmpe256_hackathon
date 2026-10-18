# Manager Chat

React + TypeScript + Vite frontend for the existing Python Manager Agent. Run both
processes from WSL; open **http://localhost:5173** in the Windows browser.

## WSL setup (Ubuntu)

Requires Python 3.11+ with `venv`, and Node 22.12+ (Node 24 LTS recommended).
Use the existing checkout; the commands below assume it is `~/cmpe256_hackathon`.
If it lives elsewhere, replace that `cd` path in both terminals.

```bash
cd ~/cmpe256_hackathon
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# CPU PyTorch avoids installing unnecessary CUDA packages on typical WSL laptops.
.venv/bin/python -m pip install 'torch>=2.6,<3' torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[web,retrieval,test]'
cd frontend
npm ci
```

The `retrieval` extra declares the dependencies already used by the Technology and
SEC agents. Clinical and Corporate Responsibility still use the base dependencies.

### Terminal 1 — backend

```bash
cd ~/cmpe256_hackathon
# Enter the key without showing it or putting it in shell history.
read -rsp 'OpenAI API key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY
# Keep specialist defaults unless your team has already configured other values.
export SPECIALIST_DATA_DIR="$PWD/data"
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

The manager always uses the existing `gpt-4.1-mini` routing/synthesis factory, so it
needs backend access to OpenAI even when specialists use extractive mode. `.env`
files are not loaded automatically. Preserve any existing specialist/provider
configuration. Never set an API key in a `VITE_*` variable, browser code, or Git.

### Terminal 2 — frontend

```bash
cd ~/cmpe256_hackathon/frontend
npm run dev
```

Visit **http://localhost:5173**. Vite forwards `/api` to FastAPI at
`http://127.0.0.1:8000`, so no permissive CORS configuration is needed. Keep both
processes running; stop each with Ctrl+C. Use these commands inside WSL rather than
running one process in Windows and the other in WSL. If Python venv support is
missing on Ubuntu, install `python3-venv` using your normal system setup.

## Prepare evidence before chatting

Reuse the team's prepared indexes. Chat never calls ingestion: it runs
`manager.consult(question)` with a request-local cache-only safeguard. Missing or
stale Technology/SEC caches produce the manager's existing specialist-failure
limitations instead of document conversion or corpus embedding. Existing agent CLI
and ingestion behavior is unchanged outside the HTTP request.

For Technology/SEC setup, run these existing builders **separately before starting
the servers**, with the team's source PDFs and model dependencies available:

```bash
cd ~/cmpe256_hackathon
.venv/bin/python -m shared.hybrid_retrieval
.venv/bin/python -m shared.sec_retrieval
```

They reuse valid portable caches, but can build/replace caches when sources or
settings differ. The first setup may download the pinned MiniLM model and Docling
assets; it requires access to their model hosts. For Clinical/SOFI indexes and
trial import, follow the ingestion instructions in the root README. Do not put
these setup commands in an HTTP handler. Query-time model loading and query
embeddings can still occur; the safeguard prohibits corpus ingestion/re-embedding.

## Verified manager contract

Inspected `manager.py` and `tests/test_manager.py` before implementation, then
incorporated the latest `main` (including SEC registration and date serialization).

- `manager.consult(question: str) -> ManagerResponse` is synchronous. The public
  function constructs `ManagerAgent` with process settings. FastAPI runs the
  synchronous endpoint in a worker thread.
- `POST /api/chat` accepts **only** `{"question":"…"}`. Questions are stripped and
  must contain 1–8,000 characters, matching `SpecialistRequest`.
- The response is the actual `ManagerResponse`, preserving `answer`, `question`,
  `agents_used`, `findings`, `recommendations`, `limitations`, `citations`,
  `citation_labels`, and `specialist_responses`.
- Recommendations are objects with `action`, `rationale`, and `evidence_ids`.
  Citations contain `id`, `source_id`, `source`, `page`, `report_year`, and `excerpt`.
  Manager citation IDs are namespaced, e.g. `technology:E1`. Specialist response
  IDs remain unchanged. Display labels come from the manager's `citation_labels`.
- The UI displays Markdown answers, consulted specialists, evidence limitations,
  findings/recommendations, and expandable original citation excerpts. Only HTTP(S)
  sources become external links; local documents show their portable labels.
- Routing/provider exceptions return a safe HTTP 503 message with a retry action.
  Specialist/synthesis failures remain successful manager responses with limitations.
  Raw provider exception text is never sent to the browser or application logs.
- `GET /api/health` is liveness only and does not invoke providers or read indexes.

## History and interaction

Conversations live in `sessionStorage` for the current browser tab, including page
refreshes. New conversation starts an empty history; previous session conversations
remain selectable in the sidebar. No history is sent to the manager. Every question
is independent: include all needed context in each question. Reloading an in-flight
request marks it interrupted and offers retry; it does not resume the backend work.
If session storage is unavailable/full, the UI reports that history cannot survive
refresh. Enter submits; Shift+Enter inserts a line break. The UI supports mobile
navigation, keyboard focus, loading status, escaped Markdown, and reduced motion.

## Checks

```bash
cd ~/cmpe256_hackathon
.venv/bin/python -m pytest -q
cd frontend
npm run build
```

`tests/test_web.py` covers the HTTP adapter with the real manager and deterministic
model/specialist fixtures, response serialization, no-evidence results, invalid
input, safe errors, cache-only behavior, and liveness without provider calls.
Existing live/real-data test opt-ins remain unchanged. A successful build produces
`frontend/dist`; the documented Vite proxy applies to `npm run dev`, not to
`npm run preview`. A production deployment needs a same-origin `/api` reverse proxy
and appropriate access control; deployment is outside this PR's scope.

Validation in this implementation environment: `npm run build` and desktop/mobile
browser smoke checks passed. All 49 manager + HTTP integration tests passed. The
full suite (`HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q`) produced **255 passed,
4 skipped, 3 failed**. The three failures are existing cache-portability tests
requiring the pinned MiniLM model, which was not cached here. Running those tests
against untouched `main` reproduced the same three failures (and one pass).
Live OpenAI consultation was not exercised.
