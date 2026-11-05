"""Ettin-150M request router: a long-context replacement for the Laya route head.

Laya routes from at most 1,600 characters (opening and closing halves of long pastes), so it misroutes
long requests (41% vs 91% on the long-context benchmark in ~/AI/repos/cdm/cdm/route_bench.py).
This sequence classifier reads up to `max_tokens` tokens (head and tail kept beyond that) and returns
the same {"route", "scores"} shape as SecurityScanner.route_request. Runs on CPU so it never takes GPU
memory from the chat models.
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROUTES = ("general", "swift", "reasoner")
_CPU_THREADS = 4
_CACHE_LIMIT = 2_048


class EttinRouter:
    def __init__(self, path: str | Path, max_tokens: int = 4096):
        self.path = str(path)
        self.max_tokens = max_tokens
        self._model: Any = None
        self._tok: Any = None
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, dict[str, float]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_THREADS))
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            torch.set_num_threads(_CPU_THREADS)
            self._tok = AutoTokenizer.from_pretrained(self.path)
            model = AutoModelForSequenceClassification.from_pretrained(self.path).eval()
            labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
            if tuple(labels) != ROUTES:
                raise ValueError(f"route checkpoint labels {labels} != {list(ROUTES)}")
            self._model = model

    def _ids(self, text: str) -> list[int]:
        tok = self._tok
        ids = tok(text, add_special_tokens=False)["input_ids"]
        room = self.max_tokens - 2
        if len(ids) > room:  # keep the opening request and the closing question of long pastes
            ids = ids[: room // 2] + ids[-(room - room // 2):]
        return [tok.cls_token_id, *ids, tok.sep_token_id]

    def route_request(self, text: str) -> dict[str, Any]:
        """Classify a user request into a route class: general, swift or reasoner."""
        self.load()
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._cache_lock:
            scores = self._cache.get(key)
            if scores is not None:
                self._cache.move_to_end(key)
        if scores is None:
            import torch

            with self._lock, torch.no_grad():
                logits = self._model(input_ids=torch.tensor([self._ids(text)])).logits[0]
            probs = logits.float().softmax(-1).tolist()
            scores = {r: round(p, 4) for r, p in zip(ROUTES, probs)}
            with self._cache_lock:
                self._cache[key] = scores
                if len(self._cache) > _CACHE_LIMIT:
                    self._cache.popitem(last=False)
        return {"route": max(scores, key=scores.get), "scores": dict(scores)}
