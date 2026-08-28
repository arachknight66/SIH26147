from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hashlib
import json

@dataclass
class CacheEntry:
    key: str
    stage_name: str
    result: Any
    config_hash: str
    source_hash: str

class PipelineCache:
    """In-memory deterministic pipeline stage cache."""
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._store: dict[str, CacheEntry] = {}

    def make_key(self, source_hash: str, stage_name: str, config_hash: str) -> str:
        raw = f"{source_hash}:{stage_name}:{config_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, source_hash: str, stage_name: str, config_hash: str) -> Any | None:
        if not self.enabled:
            return None
        key = self.make_key(source_hash, stage_name, config_hash)
        entry = self._store.get(key)
        return entry.result if entry else None

    def put(self, source_hash: str, stage_name: str, config_hash: str, result: Any) -> None:
        if not self.enabled:
            return
        key = self.make_key(source_hash, stage_name, config_hash)
        self._store[key] = CacheEntry(
            key=key,
            stage_name=stage_name,
            result=result,
            config_hash=config_hash,
            source_hash=source_hash,
        )

    def clear(self) -> None:
        self._store.clear()

    def size(self) -> int:
        return len(self._store)
