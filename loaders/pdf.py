"""Download Cisco PDFs, or one custom PDF: python -m loaders.pdf [URL] [OUTPUT]."""

import argparse
from pathlib import Path

import requests


URL = "https://www.cisco.com/c/en/us/products/collateral/routers/catalyst-8300-series-edge-platforms/datasheet-c78-744088.pdf"
OUTPUT = Path("data/technology/cisco_8300.pdf")
PDFS = {
    "cisco_8300.pdf": URL,
    "cisco_8200.pdf": "https://www.cisco.com/c/en/us/products/collateral/routers/catalyst-8200-series-edge-platforms/nb-06-cat8200-series-edge-plat-ds-cte-en.pdf",
    "cisco_8000v.pdf": "https://www.cisco.com/c/en/us/products/collateral/routers/catalyst-8000v-edge-software/catalyst-8000v-edge-software-ds.pdf",
    "cisco_4000.pdf": "https://www.cisco.com/c/en/us/products/collateral/routers/4000-series-integrated-services-routers-isr/data_sheet-c78-732542.pdf",
    "cisco_asr1000.pdf": "https://www.cisco.com/c/en/us/products/collateral/routers/asr-1000-series-aggregation-services-routers/datasheet-c78-731632.pdf",
}


def validate_pdf(content: bytes) -> int:
    """Parse the PDF and every page, returning its page count."""
    import pypdfium2 as pdfium

    if not content.lstrip().startswith(b"%PDF-"):
        raise ValueError("Content does not contain PDF bytes")
    with pdfium.PdfDocument(content) as document:
        pages = len(document)
        if not pages:
            raise ValueError("PDF has no pages")
        for index in range(pages):
            document[index].close()
    return pages


def download_pdf(url: str, output: Path) -> bool:
    if output.exists():
        try:
            pages = validate_pdf(output.read_bytes())
            print(f"{output.name}: skipped valid existing PDF ({pages} pages); HTTP status: not requested", flush=True)
            return True
        except Exception as error:
            print(f"{output.name}: existing file failed validation: {error}", flush=True)

    try:
        response = requests.get(url, timeout=60)
        print(f"{output.name}: requests HTTP status: {response.status_code}", flush=True)
        response.raise_for_status()
        content = response.content
        pages = validate_pdf(content)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        print(f"Saved validated PDF ({pages} pages): {output}")
        return True
    except Exception as error:
        print(f"{output.name}: requests attempt failed: {type(error).__name__}: {error}", flush=True)

    try:
        from docling.datamodel.base_models import ConversionStatus
        from docling.document_converter import DocumentConverter

        result = DocumentConverter().convert(url)
        print(f"{output.name}: Docling HTTP status: not exposed by conversion result")
        if result.status != ConversionStatus.SUCCESS:
            raise RuntimeError(f"Conversion status: {result.status}; errors: {result.errors}")
        print(f"{output.name}: Docling loaded {len(result.pages)} pages from the URL.")
        print(f"{output.name}: download failed: Docling exposes no PDF bytes to save.")
        return False
    except Exception as error:
        print(f"{output.name}: Docling attempt failed: {type(error).__name__}: {error}")
        print(f"{output.name}: both attempts failed; no PDF saved by this run.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?")
    parser.add_argument("output", nargs="?", type=Path, default=OUTPUT)
    args = parser.parse_args()
    documents = {args.output: args.url} if args.url else {
        OUTPUT.parent / filename: url for filename, url in PDFS.items()
    }
    failures = []
    for output, url in documents.items():
        if not download_pdf(url, output):
            failures.append(output.name)
    print(f"Complete: {len(documents) - len(failures)}/{len(documents)} PDFs available.")
    if failures:
        print(f"Failed: {', '.join(failures)}")
    raise SystemExit(1 if failures else 0)
