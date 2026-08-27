from __future__ import annotations
import pytest
from app.orchestration.cache import PipelineCache

def test_pipeline_cache_put_get():
    cache = PipelineCache(enabled=True)
    cache.put("src_hash_1", "phase2", "cfg_hash_1", {"snr": 20.0})

    res = cache.get("src_hash_1", "phase2", "cfg_hash_1")
    assert res == {"snr": 20.0}

    miss = cache.get("src_hash_2", "phase2", "cfg_hash_1")
    assert miss is None

def test_pipeline_cache_disabled():
    cache = PipelineCache(enabled=False)
    cache.put("src_hash_1", "phase2", "cfg_hash_1", {"snr": 20.0})
    assert cache.get("src_hash_1", "phase2", "cfg_hash_1") is None

def test_pipeline_cache_clear():
    cache = PipelineCache(enabled=True)
    cache.put("src_1", "p1", "cfg_1", "data")
    assert cache.size() == 1
    cache.clear()
    assert cache.size() == 0
