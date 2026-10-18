import numpy as np
import pytest
from langchain_core.documents import Document

from shared.hybrid_retrieval import Retriever
from shared.technology_retrieval import SOURCES, TechnologyRetriever, article_text


def test_web_sources_are_unique_official_and_tracking_free():
    assert len(SOURCES) == len({s["url"] for s in SOURCES}) == 10
    assert sum(s["vendor"] == "Google Cloud" for s in SOURCES) == 3
    assert sum(s["vendor"] == "IBM" for s in SOURCES) == 2
    assert all(s["url"].startswith(("https://cloud.google.com/", "https://docs.cloud.google.com/",
                                   "https://www.ibm.com/", "https://www.cisco.com/")) and "?" not in s["url"] for s in SOURCES)


@pytest.mark.parametrize("container", ["main", "article", "section id='topic'", "div id='rich-text-123'"])
def test_html_extraction_supports_vendor_templates_without_navigation(container):
    tag = container.split()[0]
    content = "Relevant architecture and retrieval documentation. " * 30
    html = f"<html><title>Article</title><nav>Navigation noise</nav><{container}>{content}<script>secretScript()</script></{tag}><footer>Footer noise</footer></html>"
    text = article_text(html)
    assert "Relevant architecture" in text
    assert "Navigation noise" not in text and "Footer noise" not in text and "secretScript" not in text


def test_access_challenges_and_empty_javascript_pages_are_rejected():
    for html in ["<title>Access denied</title><main>blocked</main>", "<body><script>render()</script></body>"]:
        with pytest.raises(ValueError):
            article_text(html)


@pytest.fixture
def retriever():
    class Model:
        def encode(self, text, **kwargs):
            return np.array([1., 0.])
    docs = [Document(page_content=f"{vendor} supports technical architecture documentation.",
        metadata={"vendor": vendor, "source": source, "pages": pages})
        for vendor, source, pages in [("Cisco", "data/technology/cisco_8300.pdf", [6]),
                                     ("Google Cloud", "https://docs.cloud.google.com/mcp/overview", []),
                                     ("IBM", "https://www.ibm.com/think/topics/model-context-protocol", [])]]
    return TechnologyRetriever(Retriever(docs, np.array([[1., 0.]] * 3), Model()))


def test_vendor_queries_and_web_citations_remain_isolated(retriever):
    for vendor in ["Cisco", "Google Cloud", "IBM"]:
        rows = retriever.retrieve(f"What does {vendor} support?")
        assert rows and all(r["metadata"]["vendor"] == vendor for r in rows)
    google = retriever.retrieve("MCP", vendor="gcp")[0]
    assert google["metadata"]["source"].startswith("https://docs.cloud.google.com/")
    assert google["metadata"]["pages"] == []


def test_cisco_family_filter_does_not_remove_other_requested_vendors(retriever):
    rows = retriever.retrieve("Compare Cisco Catalyst 8300 with Google Cloud documentation", k=3)
    assert {r["metadata"]["vendor"] for r in rows} == {"Cisco", "Google Cloud"}


def test_technology_agent_keeps_web_source_id_and_does_not_invent_page(monkeypatch):
    from agents import technology
    from shared.model import Settings

    url = "https://docs.cloud.google.com/mcp/overview"
    monkeypatch.setattr(technology, "retrieve", lambda question, *, k: [{
        "text": "Synthetic Google MCP documentation.", "metadata": {
            "source": url, "source_id": "google-mcp-overview", "vendor": "Google Cloud", "pages": []}}])
    response = technology.TechnologyAgent(Settings()).invoke("Google Cloud MCP")
    assert response.citations[0].source == url
    assert response.citations[0].source_id == "google-mcp-overview"
    assert response.citations[0].page is None


def test_manager_technology_scope_includes_google_and_ibm():
    from manager import AGENT_SCOPES

    assert all(vendor in AGENT_SCOPES["technology"] for vendor in ["Cisco", "Google Cloud", "IBM"])
    assert "MCP" in AGENT_SCOPES["technology"] and "RAG" in AGENT_SCOPES["technology"]


def test_cisco_product_containers_exclude_header_and_footer():
    content = 'Product interfaces and platform capabilities. ' * 30
    for container in ["container-product-family-highlights", "overview-abc", "modelcomparison-abc"]:
        text = article_text(f'<div id="fw-c-header">Header noise</div><div id="{container}">{content}</div><footer>Footer noise</footer>')
        assert 'Product interfaces' in text
        assert 'Header noise' not in text and 'Footer noise' not in text


def test_unchanged_web_chunks_are_preserved(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace
    from shared import technology_retrieval as tr

    source = tr.SOURCES[0]
    content = b'unchanged snapshot'
    digest = hashlib.sha256(content).hexdigest()
    (tmp_path / 'page.html').write_bytes(content)
    (tmp_path / 'manifest.json').write_text(json.dumps({source['id']: {
        **source, 'filename': 'page.html', 'sha256': digest}}))
    retained = {'page_content': 'Original processed text', 'metadata': {
        'source_id': source['id'], 'source': source['url'], 'retrieved_at': 'original-date'}}
    (tmp_path / 'index.json').write_text(json.dumps({'key': json.dumps({
        'settings': tr.SETTINGS, 'web_settings': tr.WEB_SETTINGS,
        'sources': [source], 'web': {source['id']: digest}}), 'documents': [retained]}))
    monkeypatch.setattr(tr, 'SOURCES', [source])
    monkeypatch.setattr(tr, 'ROOT', tmp_path)
    (tmp_path / 'data/technology').mkdir(parents=True)
    (tmp_path / 'data/technology/cisco_fixture.pdf').write_bytes(b'fixture')
    monkeypatch.setattr(tr, 'build_cisco_retriever', lambda: SimpleNamespace(documents=[], model=None))
    monkeypatch.setattr(tr, 'article_text', lambda _: pytest.fail('Existing source was reprocessed'))
    def build(documents, *args):
        docs = documents()
        assert len(docs) == 1
        assert docs[0].page_content == retained['page_content']
        assert docs[0].metadata == retained['metadata']
        return SimpleNamespace(documents=docs)
    monkeypatch.setattr(tr.Retriever, 'from_documents', build)
    tr.build_technology_retriever(tmp_path, tmp_path)


def test_cisco_landing_pages_have_filterable_product_families():
    from shared.hybrid_retrieval import _source_family, _question_families

    pages = [s for s in SOURCES if s['vendor'] == 'Cisco']
    assert len(pages) == 5
    for source in pages:
        assert _source_family(source) in _question_families(source['product_family'])
