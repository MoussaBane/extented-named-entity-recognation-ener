"""Lightweight experiment logger that writes JSON metadata and timings."""
from typing import Dict, Any
import json
import os
import time


class ExperimentLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.meta: Dict[str, Any] = {"events": []}

    def log(self, key: str, value: Any) -> None:
        ts = time.time()
        self.meta.setdefault("events", []).append({"ts": ts, "key": key, "value": value})

    def save(self, fname: str = "experiment.json") -> None:
        path = os.path.join(self.out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
