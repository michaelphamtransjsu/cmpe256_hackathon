import logging

from shared.diagnostics import log_failure, safe_traceback


def test_chained_traceback_retains_frames_without_messages_or_source(caplog):
    try:
        try:
            raise ValueError("sk-private-api-key and private question")
        except ValueError as cause:
            raise RuntimeError("private provider body") from cause
    except RuntimeError as error:
        with caplog.at_level(logging.ERROR):
            log_failure(logging.getLogger("audit"), "Specialist failed", error)
        formatted = safe_traceback(error)
    assert "ValueError" in caplog.text and "RuntimeError" in caplog.text
    assert "test_chained_traceback_retains_frames" in formatted
    assert "test_diagnostics.py" in formatted and "line " in formatted
    assert "Chained exception" in formatted
    assert "sk-private" not in caplog.text
    assert "private question" not in caplog.text
    assert "private provider body" not in caplog.text
    assert "raise RuntimeError" not in formatted
