"""Task suites are JSONL files: one test item per line."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from .scorers import get_scorer

_CORE = {"id", "prompt", "expected", "scorer", "category"}


@dataclass
class Item:
    id: str
    prompt: str
    scorer: str
    expected: Any = None
    category: str = "general"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Suite:
    name: str
    items: list[Item]
    system: str | None = None


def builtin_suites() -> list[str]:
    root = resources.files("evalforge") / "suites"
    return sorted(p.name.removesuffix(".jsonl") for p in root.iterdir() if p.name.endswith(".jsonl"))


def load_suite(source: str | Path) -> Suite:
    """Load a suite from a .jsonl path, or by name for the built-in suites."""
    path = Path(source)
    if path.suffix == ".jsonl" and path.exists():
        text, name = path.read_text(encoding="utf-8"), path.stem
    else:
        res = resources.files("evalforge") / "suites" / f"{source}.jsonl"
        if not res.is_file():
            raise FileNotFoundError(f"No suite file or built-in suite named {source!r}. Built-in: {builtin_suites()}")
        text, name = res.read_text(encoding="utf-8"), str(source)

    items, system, seen = [], None, set()
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{name}: line {n} is not valid JSON: {e}") from e
        if "system" in row and len(row) == 1:
            system = row["system"]  # optional header line: {"system": "..."}
            continue
        for key in ("id", "prompt", "scorer"):
            if key not in row:
                raise ValueError(f"{name}: line {n} is missing {key!r}")
        if row["id"] in seen:
            raise ValueError(f"{name}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        get_scorer(row["scorer"])  # fail early on typos
        items.append(
            Item(
                id=row["id"],
                prompt=row["prompt"],
                scorer=row["scorer"],
                expected=row.get("expected"),
                category=row.get("category", "general"),
                meta={k: v for k, v in row.items() if k not in _CORE},
            )
        )
    if not items:
        raise ValueError(f"{name}: suite has no items")
    return Suite(name=name, items=items, system=system)
