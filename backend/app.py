"""Run from the repository root: python -m uvicorn backend.app:app."""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field

import manager
from shared.schemas import StrictModel
from shared.query_context import answering_only

logger = logging.getLogger(__name__)
app = FastAPI(title="CMPE 256 Manager Chat", version="1.0.0")


class ChatRequest(StrictModel):
    # Match SpecialistRequest's question validation; history is deliberately absent.
    question: str = Field(min_length=1, max_length=8000)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError):
    # Avoid returning submitted input or internal validation context to the browser.
    return JSONResponse(status_code=422, content={
        "detail": "Send a question containing 1–8,000 characters. Only the question field is accepted."
    })


@app.get("/api/health")
def health():
    """Liveness only; no provider calls, model construction, or index access."""
    return {"status": "ok"}


@app.post("/api/chat", response_model=manager.ManagerResponse)
def chat(request: ChatRequest) -> manager.ManagerResponse:
    # FastAPI runs this synchronous route in its thread pool. The manager owns all
    # routing, retrieval, and synthesis; no ingestion or specialist calls live here.
    try:
        with answering_only():
            return manager.consult(request.question)
    except Exception as error:
        # Provider messages/tracebacks can contain credentials and private content.
        logger.warning("Manager consultation failed (%s)", type(error).__name__)
        raise HTTPException(status_code=503, detail=(
            "The Manager Agent could not complete this request. Try again. "
            "If it continues, check the backend API key, provider access, and agent configuration."
        )) from None
