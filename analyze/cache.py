from __future__ import annotations
import orjson as json
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib

class SimpleCache:
    def __init__(self, path: str = ".llm_cache.json"):
        self.path = Path(path)
        self.mem: Dict[str, Any] = {}
        if self.path.exists():
            try:
                self.mem = json.loads(self.path.read_bytes())
            except Exception:
                self.mem = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.mem.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self.mem[key] = value

    def save(self):
        self.path.write_bytes(json.dumps(self.mem, option=json.OPT_INDENT_2))

def make_cache_key(model: str, temperature: float, body: str, ancestors: str) -> str:
    h = hashlib.sha256()
    txt = f"{model}|{temperature}|{body.strip()}||{ancestors.strip()}"
    h.update(txt.encode("utf-8", errors="ignore"))
    return h.hexdigest()
