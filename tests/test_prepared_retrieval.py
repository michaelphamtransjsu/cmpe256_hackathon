"""Query loading must remain offline and independent of document preparation."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.documents import Document

from shared import hybrid_retrieval as hybrid
from shared.query_context import answering_only, cache_only


def test_agent_imports_do_not_import_optional_semantic_or_conversion_dependencies():
    script = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'docling','docling_core','langchain_docling','pypdfium2',
                                      'sentence_transformers','sklearn','bs4'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, Block())
from agents.technology import TechnologyAgent
from agents.sec import SECAgent
from shared.model import Settings
TechnologyAgent(Settings())
SECAgent(Settings())
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_model_loading_pins_revision_and_is_offline_only_during_queries(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(
        SentenceTransformer=lambda *a, **kw: calls.append((a, kw))))
    with answering_only():
        hybrid.load_embedding_model()
    hybrid.load_embedding_model()
    assert [kw['local_files_only'] for _, kw in calls] == [True, False]
    assert all(a == (hybrid.MODEL,) and kw['revision'] == hybrid.MODEL_REVISION
               and kw['device'] == 'cpu' for a, kw in calls)


def test_missing_local_weights_report_a_preparation_error(monkeypatch):
    def absent(*args, **kwargs):
        assert kwargs['local_files_only'] is True
        raise OSError('No local model snapshot')
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=absent))
    with answering_only(), pytest.raises(RuntimeError, match='Queries do not download'):
        hybrid.load_embedding_model()


@pytest.mark.parametrize('state', ['missing', 'stale', 'invalid-vectors'])
def test_generic_cache_errors_precede_model_loading(tmp_path, monkeypatch, state):
    def forbidden(*a, **kw):
        pytest.fail('Model initialization or document loading preceded cache validation')
    monkeypatch.setattr(hybrid, 'load_embedding_model', forbidden)
    cache = tmp_path / 'index'
    if state != 'missing':
        cache.mkdir()
        (cache / 'index.json').write_text(json.dumps({
            'key': 'old' if state == 'stale' else 'key',
            'documents': [{'page_content': 'Evidence', 'metadata': {}}]}))
        np.save(cache / 'embeddings.npy', np.zeros((2, 2)))
    with answering_only(), pytest.raises(RuntimeError, match='prepare indexes separately'):
        hybrid.Retriever.from_documents(forbidden, cache, 'key')
    assert not cache.exists() if state == 'missing' else True


@pytest.mark.parametrize('domain', ['technology', 'sec'])
@pytest.mark.parametrize('stale', [False, True])
def test_domain_cache_validation_precedes_models(tmp_path, monkeypatch, domain, stale):
    from shared import technology_retrieval as technology, sec_retrieval as sec
    def forbidden(*a, **kw):
        pytest.fail('A missing/stale domain cache initialized models or the Cisco PDF index')
    cache = tmp_path / 'cache'
    if stale:
        cache.mkdir()
        (cache / 'index.json').write_text(json.dumps({'key': 'old'}))
    if domain == 'technology':
        monkeypatch.setattr(technology, 'ROOT', tmp_path)
        monkeypatch.setattr(technology, 'build_cisco_retriever', forbidden)
        monkeypatch.setattr(hybrid, 'load_embedding_model', forbidden)
        pdfs = tmp_path / 'data/technology'
        pdfs.mkdir(parents=True)
        (pdfs / 'cisco_fixture.pdf').write_bytes(b'PDF')
        web = tmp_path / 'web'
        web.mkdir()
        source = technology.SOURCES[0]
        monkeypatch.setattr(technology, 'SOURCES', [source])
        (web / 'page.html').write_bytes(b'HTML')
        (web / 'manifest.json').write_text(json.dumps({source['id']: {
            **source, 'filename': 'page.html', 'sha256': hashlib.sha256(b'HTML').hexdigest()}}))
        build = lambda: technology.build_technology_retriever(web, cache)
    else:
        monkeypatch.setattr(sec, 'load_embedding_model', forbidden)
        filing = sec.FILINGS[0]
        monkeypatch.setattr(sec, 'FILINGS', [filing])
        data = tmp_path / 'data'
        (data / filing['company']).mkdir(parents=True)
        (data / filing['company'] / filing['filename']).write_bytes(b'PDF')
        build = lambda: sec.build_sec_retriever(data, cache)
    with answering_only(), pytest.raises(RuntimeError, match='prepare indexes separately'):
        build()


def test_public_query_entry_points_enforce_cache_only_independently(monkeypatch):
    from shared import technology_retrieval as technology, sec_retrieval as sec
    observed = []
    def builder():
        observed.append(cache_only.get())
        return SimpleNamespace(retrieve=lambda *a, **kw: [])
    monkeypatch.setattr(technology, 'build_technology_retriever', builder)
    monkeypatch.setattr(sec, 'build_sec_retriever', builder)
    hybrid._default_retriever.cache_clear()
    sec._default.cache_clear()
    try:
        hybrid.retrieve('Cisco')
        sec.retrieve('Cisco')
        assert observed == [True, True] and not cache_only.get()
    finally:
        hybrid._default_retriever.cache_clear()
        sec._default.cache_clear()


def test_explicit_preparation_can_create_an_index(tmp_path):
    class Model:
        class tokenizer:
            @staticmethod
            def encode(text, **kwargs):
                return text.split()
        def get_sentence_embedding_dimension(self):
            return 2
        def encode(self, texts, **kwargs):
            assert texts == ['Prepared evidence']
            return np.array([[1., 0.]])
    calls = []
    def documents():
        calls.append('prepare')
        return [Document(page_content='Prepared evidence')]
    hybrid.Retriever.from_documents(documents, tmp_path, 'key', Model())
    with answering_only():
        result = hybrid.Retriever.from_documents(documents, tmp_path, 'key', Model())
    assert len(result.documents) == 1 and calls == ['prepare']


def test_sec_context_errors_precede_model_loading(tmp_path, monkeypatch):
    from shared import sec_retrieval as sec
    filing = sec.FILINGS[0]
    monkeypatch.setattr(sec, 'FILINGS', [filing])
    def forbidden(*a, **kw):
        pytest.fail('Model loaded before SEC context validation')
    monkeypatch.setattr(sec, 'load_embedding_model', forbidden)
    source = tmp_path / filing['company'] / filing['filename']
    source.parent.mkdir()
    source.write_bytes(b'PDF')
    key = json.dumps({'files': {source.relative_to(tmp_path).as_posix(): hashlib.sha256(b'PDF').hexdigest()},
                      'config': sec.CONFIG, 'filings': [filing]}, sort_keys=True)
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'index.json').write_text(json.dumps({'key': key, 'documents': [{'page_content': 'Evidence', 'metadata': {}}]}))
    np.save(cache / 'embeddings.npy', np.zeros((1, 2)))
    with answering_only(), pytest.raises(RuntimeError, match='SEC context is missing or invalid'):
        sec.build_sec_retriever(tmp_path, cache)


def test_real_agents_use_prepared_caches_without_processing_or_network():
    root = Path(__file__).resolve().parents[1]
    if not all((root / p).exists() for p in ['storage/technology/multivendor/index.json', 'storage/sec/index.json']):
        pytest.skip('Prepared multivendor and SEC caches are required')
    script = '''
import sys, importlib.abc, socket, hashlib
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'docling','docling_core','langchain_docling','pypdfium2'}:
            raise AssertionError('Processing dependency imported: ' + fullname)
sys.meta_path.insert(0, Block())
def forbidden(*a, **kw): raise AssertionError('Network or processing attempted')
socket.socket.connect = forbidden
socket.create_connection = forbidden
from sentence_transformers import SentenceTransformer
encode = SentenceTransformer.encode
init = SentenceTransformer.__init__
def local_init(self, *a, **kw):
    assert kw['local_files_only'] is True
    return init(self, *a, **kw)
def queries_only(self, text, *a, **kw):
    assert isinstance(text, str), 'Corpus embedding attempted'
    return encode(self, text, *a, **kw)
SentenceTransformer.__init__ = local_init
SentenceTransformer.encode = queries_only
from shared import technology_retrieval as technology, sec_retrieval as sec
from shared.hybrid_retrieval import ROOT
from agents.technology import TechnologyAgent
from agents.sec import SECAgent
from shared.model import Settings
from shared.schemas import SpecialistResponse
technology.build_cisco_retriever = forbidden
technology.download_sources = forbidden
technology.article_text = forbidden
def hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['technology', 'sec'] for p in (ROOT/'storage'/folder).rglob('*') if p.is_file()}
before = hashes()
for vendor, question, expected in [
    ('Cisco','Which Catalyst 8300 routers have two 10Gbps Ethernet ports?','cisco_8300'),
    ('Google','What do Google Cloud remote MCP servers let AI applications do?','google-mcp-overview'),
    ('IBM',"What are the main components in IBM's retrieval-augmented generation architecture?",'ibm-rag-architecture')]:
    response = TechnologyAgent(Settings()).invoke(question)
    assert response.citations[0].source_id == expected
    SpecialistResponse.model_validate_json(response.model_dump_json())
for filing in sec.FILINGS:
    question = f"What are {filing['company']}'s business risks in FY{filing['fiscal_year']}?"
    response = SECAgent(Settings()).invoke(question)
    assert response.citations and all(c.report_year == filing['fiscal_year'] for c in response.citations)
    assert all(c.source_id == f"{filing['company'].lower()}-{filing['fiscal_year']}-10-K" for c in response.citations)
assert before == hashes(), 'Production caches changed'
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=root, capture_output=True, text=True,
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_technology_preparation_does_not_mistake_missing_vectors_for_a_cache_hit(tmp_path, monkeypatch):
    from shared import technology_retrieval as technology
    source = technology.SOURCES[0]
    monkeypatch.setattr(technology, 'SOURCES', [source])
    monkeypatch.setattr(technology, 'ROOT', tmp_path)
    pdfs = tmp_path / 'data/technology'
    pdfs.mkdir(parents=True)
    (pdfs / 'cisco_fixture.pdf').write_bytes(b'PDF')
    web = tmp_path / 'web'
    web.mkdir()
    (web / 'page.html').write_bytes(b'HTML')
    web_hash = hashlib.sha256(b'HTML').hexdigest()
    (web / 'manifest.json').write_text(json.dumps({source['id']: {
        **source, 'filename': 'page.html', 'sha256': web_hash}}))
    key = json.dumps({'pdfs': {'cisco_fixture.pdf': hashlib.sha256(b'PDF').hexdigest()},
                      'web': {source['id']: web_hash}, 'sources': [source],
                      'settings': technology.SETTINGS, 'web_settings': technology.WEB_SETTINGS}, sort_keys=True)
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'index.json').write_text(json.dumps({'key': key, 'documents': []}))
    class Preparing(Exception):
        pass
    def prepare():
        raise Preparing('Explicit PDF preparation reached')
    monkeypatch.setattr(technology, 'build_cisco_retriever', prepare)
    with pytest.raises(Preparing):
        technology.build_technology_retriever(web, cache)
    with answering_only(), pytest.raises(RuntimeError, match='prepare indexes separately'):
        technology.build_technology_retriever(web, cache)
