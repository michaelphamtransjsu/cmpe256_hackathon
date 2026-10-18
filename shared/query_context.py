"""Request-local safeguards; command-line ingestion keeps its existing behavior."""

from contextlib import contextmanager
from contextvars import ContextVar

# A ContextVar keeps concurrent HTTP requests and separate CLI operations isolated.
cache_only = ContextVar("cache_only", default=False)


@contextmanager
def answering_only():
    token = cache_only.set(True)
    try:
        yield
    finally:
        cache_only.reset(token)
