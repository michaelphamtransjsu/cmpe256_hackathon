"""Relocate the actual prebuilt caches; prohibit conversion and corpus embedding."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkout(tmp_path):
    if not (ROOT / "storage/sec/index.json").exists() or not (ROOT / "storage/technology/index.json").exists():
        pytest.skip("Prebuilt caches are required for the relocation integration test")
    target = tmp_path / "different-checkout"
    shutil.copytree(ROOT / "shared", target / "shared", ignore=shutil.ignore_patterns("__pycache__"))
    for domain in ["technology", "sec"]:
        shutil.copytree(ROOT / "storage" / domain, target / "storage" / domain)
        manifest = json.loads((ROOT / "storage" / domain / "index.json").read_text())
        for source in {d["metadata"]["source"] for d in manifest["documents"]}:
            destination = target / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, destination)
    return target


def test_relocated_default_builders_reuse_exact_prebuilt_data(checkout):
    script = '''
import hashlib,json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
def forbidden(*a,**kw): raise AssertionError("Expensive processing was attempted")
DocumentConverter.convert = forbidden
DoclingLoader.load = forbidden
HybridChunker.chunk = forbidden
encode = SentenceTransformer.encode
def queries_only(self,text,*a,**kw):
    assert isinstance(text,str), "Corpus embeddings must not be regenerated"
    return encode(self,text,*a,**kw)
SentenceTransformer.encode = queries_only
from shared.hybrid_retrieval import ROOT, build_cisco_retriever
from shared.sec_retrieval import build_sec_retriever
def digests():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'storage').rglob('*') if p.is_file()}
before=digests()
tech=build_cisco_retriever()
sec=build_sec_retriever()
assert len(tech.documents)==830 and len(sec.index.documents)==2967
results = [tech.retrieve('Which Catalyst 8300 routers have two 10Gbps Ethernet ports?'),
           sec.retrieve('What were Cisco and Agilent FY2025 revenues?',company='Cisco',fiscal_year=2025)]
assert all(not Path(x['metadata']['source']).is_absolute() and (ROOT/x['metadata']['source']).is_file() for group in results for x in group)
assert before==digests(), "Cache loading must not rewrite persisted data"
summary=[[(x['text'],x['score'],x['metadata']['source'],x['metadata']['pages']) for x in group] for group in results]
Path('results.json').write_text(json.dumps(summary))
print('PASS: relocated/default cache load, real query embeddings, no conversion/chunking/corpus embedding')
'''
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    # Compare original and relocated default ROOT values using the same guarded runner.
    original = subprocess.run([sys.executable, "-c", script.replace("Path('results.json')", "Path(" + repr(str(checkout / "baseline.json")) + ")")],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert original.returncode == 0, original.stdout + original.stderr
    relocated = subprocess.run([sys.executable, "-c", script], cwd=checkout, env=env,
                               capture_output=True, text=True, timeout=120)
    assert relocated.returncode == 0, relocated.stdout + relocated.stderr
    assert json.loads((checkout / "baseline.json").read_text()) == json.loads((checkout / "results.json").read_text())


@pytest.mark.parametrize("domain", ["technology", "sec"])
def test_content_and_processing_changes_invalidate_index(checkout, monkeypatch, domain):
    from shared import hybrid_retrieval as tech, sec_retrieval as sec
    module = tech if domain == "technology" else sec
    data = checkout / "data" / ("technology" if domain == "technology" else "UnitedStatesSecuritiesandExchangeCommission(SEC)")
    cache = checkout / "storage" / domain
    manifest = json.loads((cache / "index.json").read_text())
    old_key = manifest["key"]
    captured = []
    models = []

    class Stop(Exception):
        pass

    def capture(documents, cache_dir, cache_key, model=None):
        captured.append(cache_key)
        models.append(model)
        raise Stop

    monkeypatch.setattr(tech.Retriever, "from_documents", capture)
    build = module.build_cisco_retriever if domain == "technology" else module.build_sec_retriever
    kwargs = {"pdf_dir" if domain == "technology" else "data_dir": data, "cache_dir": cache}
    with pytest.raises(Stop):
        build(**kwargs)
    assert captured[-1] == old_key
    monkeypatch.setattr(module, "SentenceTransformer", lambda *a, **kw: models[0])
    source = next(data.rglob("*.pdf"))
    original = source.read_bytes()
    source.write_bytes(original + b"\n")
    with pytest.raises(Stop):
        build(**kwargs)
    assert captured[-1] != old_key
    source.write_bytes(original)
    config = tech.SETTINGS if domain == "technology" else sec.CONFIG
    for field, value in [("max_tokens", 219), ("model_revision", "different-revision")]:
        with monkeypatch.context() as patch:
            patch.setitem(config, field, value)
            with pytest.raises(Stop):
                build(**kwargs)
            assert captured[-1] != old_key


def test_sec_conversion_cache_depends_on_content_and_conversion_settings(monkeypatch):
    from shared import sec_retrieval as sec
    original = sec._artifact_name("issuer", "hash1")
    assert original != sec._artifact_name("issuer", "hash2")
    monkeypatch.setitem(sec.CONFIG, "do_ocr", not sec.CONFIG["do_ocr"])
    assert original != sec._artifact_name("issuer", "hash1")
