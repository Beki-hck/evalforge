"""Runs a suite against a model and collects scored results."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .models import Model
from .scorers import Score, ScoringContext, get_scorer
from .suite import Item, Suite


@dataclass
class ItemResult:
    id: str
    category: str
    scorer: str
    prompt: str
    answer: str
    passed: bool
    value: float
    detail: str
    latency_s: float
    cached: bool = False
    error: str | None = None


@dataclass
class RunResult:
    suite: str
    model: str
    started_at: str
    duration_s: float
    results: list[ItemResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    def by_category(self) -> dict[str, dict[str, float]]:
        groups: dict[str, list[ItemResult]] = defaultdict(list)
        for r in self.results:
            groups[r.category].append(r)
        return {
            cat: {"n": len(rs), "passed": sum(r.passed for r in rs), "accuracy": sum(r.passed for r in rs) / len(rs)}
            for cat, rs in sorted(groups.items())
        }

    def summary(self) -> dict:
        lat = sorted(r.latency_s for r in self.results if not r.cached)
        return {
            "suite": self.suite,
            "model": self.model,
            "n": len(self.results),
            "passed": sum(r.passed for r in self.results),
            "accuracy": round(self.accuracy, 4),
            "errors": sum(r.error is not None for r in self.results),
            "median_latency_s": round(lat[len(lat) // 2], 3) if lat else None,
            "by_category": self.by_category(),
        }

    def to_dict(self) -> dict:
        return {**self.summary(), "started_at": self.started_at, "duration_s": self.duration_s,
                "results": [asdict(r) for r in self.results]}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


class ResponseCache:
    """Stores model answers on disk so re-running a suite (or re-scoring it) is free."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._data: dict[str, str] = {}
        if self.path and self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def key(model: str, system: str | None, prompt: str) -> str:
        return hashlib.sha256(json.dumps([model, system, prompt]).encode()).hexdigest()

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value

    def flush(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")


def _run_item(item: Item, suite: Suite, model: Model, ctx: ScoringContext, cache: ResponseCache) -> ItemResult:
    key = ResponseCache.key(model.name, suite.system, item.prompt)
    answer, cached, error = cache.get(key), True, None
    start = time.perf_counter()
    if answer is None:
        cached = False
        try:
            answer = model.generate(item.prompt, system=suite.system)
            cache.put(key, answer)
        except Exception as e:  # a failed call counts as a failed item, not a crashed run
            answer, error = "", f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - start

    if error:
        score = Score(False, 0.0, "model call failed")
    else:
        try:
            score = get_scorer(item.scorer)(item, answer, ctx)
        except Exception as e:
            score, error = Score(False, 0.0, "scorer failed"), f"{type(e).__name__}: {e}"

    return ItemResult(
        id=item.id, category=item.category, scorer=item.scorer, prompt=item.prompt, answer=answer,
        passed=score.passed, value=score.value, detail=score.detail, latency_s=round(latency, 3),
        cached=cached, error=error,
    )


def run_suite(
    suite: Suite,
    model: Model,
    *,
    judge: Model | None = None,
    workers: int = 4,
    cache_path: str | Path | None = None,
    limit: int | None = None,
    on_result=None,
) -> RunResult:
    items = suite.items[:limit] if limit else suite.items
    ctx = ScoringContext(judge=judge)
    cache = ResponseCache(cache_path)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    results: list[ItemResult] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for res in pool.map(lambda it: _run_item(it, suite, model, ctx, cache), items):
            results.append(res)
            if on_result:
                on_result(res)
    cache.flush()
    return RunResult(suite=suite.name, model=model.name, started_at=started,
                     duration_s=round(time.perf_counter() - t0, 2), results=results)
