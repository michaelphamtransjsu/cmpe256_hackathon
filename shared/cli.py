"""Reusable query CLI support; document ingestion lives in the root ingest.py."""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from .model import Settings


def safe_error_message(error: Exception) -> str:
    """Report local validation errors without logging provider request details."""
    if isinstance(error, ValidationError):
        return str(error.errors(include_input=False, include_url=False))
    if isinstance(error, (ValueError, OSError)):
        return str(error)
    return f"{type(error).__name__}; check provider configuration or input format"


def run_agent_cli(agent_type, argv: list[str] | None = None, *, report_year: bool = False) -> int:
    """Run exactly the selected specialist against an already prepared index."""
    parser = argparse.ArgumentParser(description=f"Query the {agent_type.__name__} specialist")
    parser.add_argument("question")
    parser.add_argument("--data-dir", type=Path, help="Override SPECIALIST_DATA_DIR")
    parser.add_argument("--max-sources", type=int)
    if report_year:
        parser.add_argument("--report-year", type=int, help="Select a SOFI publication year")
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.data_dir is not None:
            settings = Settings(**{**settings.model_dump(), "data_dir": args.data_dir})
        request = {"question": args.question}
        if args.max_sources is not None:
            request["max_sources"] = args.max_sources
        if report_year and args.report_year is not None:
            request["report_year"] = args.report_year
        response = agent_type(settings).invoke(request)
        print(response.model_dump_json(indent=2))
        return 0
    except Exception as error:
        print(f"error: {safe_error_message(error)}", file=sys.stderr)
        return 1
