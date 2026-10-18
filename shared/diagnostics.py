"""Useful exception stacks without provider messages, source text, or locals."""

import traceback


def safe_traceback(error: BaseException) -> str:
    """Retain every frame and chained exception type; omit potentially secret content.

    Standard ``exc_info`` formatting includes exception messages and source lines,
    both of which can contain submitted documents, questions, or API credentials.
    """
    sections = []
    seen = set()

    def append(current):
        if id(current) in seen:
            return
        seen.add(id(current))
        cause = current.__cause__
        if cause is None and not current.__suppress_context__:
            cause = current.__context__
        if cause is not None:
            append(cause)
            sections.append("Chained exception:")
        sections.append("Traceback (most recent call last; messages and source omitted):")
        for frame in traceback.extract_tb(current.__traceback__):
            sections.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
        sections.append(type(current).__name__)

    append(error)
    return "\n".join(sections)


def log_failure(logger, context: str, error: BaseException) -> None:
    """Context must be a fixed application label, never submitted/provider text."""
    logger.error("%s\n%s", context, safe_traceback(error))
