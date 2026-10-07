"""Scorers turn a model's answer into a pass/fail result with a short reason."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .models import Model
    from .suite import Item


@dataclass
class Score:
    passed: bool
    value: float  # 0.0 to 1.0
    detail: str = ""


ScorerFn = Callable[["Item", str, "ScoringContext"], Score]


@dataclass
class ScoringContext:
    judge: "Model | None" = None
    code_timeout: float = 10.0


_REGISTRY: dict[str, ScorerFn] = {}


def scorer(name: str):
    def wrap(fn: ScorerFn) -> ScorerFn:
        _REGISTRY[name] = fn
        return fn

    return wrap


def get_scorer(name: str) -> ScorerFn:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown scorer {name!r}. Available: {', '.join(sorted(_REGISTRY))}")


def available_scorers() -> list[str]:
    return sorted(_REGISTRY)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .")


def _expected_list(item: "Item") -> list[str]:
    exp = item.expected
    if exp is None:
        return []
    return [str(e) for e in exp] if isinstance(exp, list) else [str(exp)]


@scorer("exact_match")
def exact_match(item, answer, ctx) -> Score:
    got = _norm(answer)
    ok = any(got == _norm(e) for e in _expected_list(item))
    return Score(ok, float(ok), f"got {got[:60]!r}")


@scorer("contains")
def contains(item, answer, ctx) -> Score:
    low = answer.lower()
    hit = next((e for e in _expected_list(item) if e.lower() in low), None)
    return Score(hit is not None, float(hit is not None), f"found {hit!r}" if hit else "no expected text found")


@scorer("regex")
def regex(item, answer, ctx) -> Score:
    flags = re.IGNORECASE if item.meta.get("ignore_case") else 0
    ok = any(re.search(p, answer.strip(), flags) for p in _expected_list(item))
    return Score(ok, float(ok), "pattern matched" if ok else "no pattern matched")


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final_number(answer: str) -> float | None:
    """Prefer a number after 'answer:' / '####'; otherwise take the last number in the text."""
    for marker in (r"####\s*", r"answer\s*(?:is|:)\s*\**\$?"):
        m = re.search(marker + r"(-?\d[\d,]*(?:\.\d+)?)", answer, re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", ""))
    nums = _NUM.findall(answer)
    if not nums:
        return None
    return float(nums[-1].replace(",", "").rstrip("."))


@scorer("numeric")
def numeric(item, answer, ctx) -> Score:
    got = extract_final_number(answer)
    if got is None:
        return Score(False, 0.0, "no number in answer")
    tol = float(item.meta.get("tolerance", 1e-6))
    ok = any(abs(got - float(e)) <= tol for e in _expected_list(item))
    return Score(ok, float(ok), f"got {got:g}")


_CHOICE = re.compile(r"answer\s*(?:is|:)?\s*\(?([A-E])\b", re.IGNORECASE)


@scorer("choice")
def choice(item, answer, ctx) -> Score:
    m = _CHOICE.search(answer) or re.search(r"\b([A-E])\b(?!.*\b[A-E]\b)", answer.strip(), re.S)
    if not m:
        return Score(False, 0.0, "no option letter found")
    got = m.group(1).upper()
    ok = got in {e.upper() for e in _expected_list(item)}
    return Score(ok, float(ok), f"picked {got}")


def extract_code(answer: str) -> str:
    """Pull Python code out of a markdown answer, or use the whole answer if there is no fence."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", answer, re.S)
    return max(blocks, key=len) if blocks else answer


@scorer("python_tests")
def python_tests(item, answer, ctx) -> Score:
    """Run the model's code with the item's hidden asserts in a separate Python process.

    This runs untrusted code. It uses a temp directory, isolated mode and a timeout,
    but it is not a security sandbox: run it in a container if that matters to you.
    """
    tests = item.meta.get("tests")
    if not tests:
        raise ValueError(f"Item {item.id} uses python_tests but has no 'tests'")
    program = extract_code(answer) + "\n\n" + tests + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=ctx.code_timeout,
            )
        except subprocess.TimeoutExpired:
            return Score(False, 0.0, f"timed out after {ctx.code_timeout:g}s")
    if proc.returncode == 0:
        return Score(True, 1.0, "all tests passed")
    last = (proc.stderr.strip().splitlines() or ["failed"])[-1]
    return Score(False, 0.0, last[:200])


JUDGE_PROMPT = """You are grading an AI assistant's answer.

Question:
{prompt}

Rubric:
{rubric}

Answer to grade:
{answer}

Score the answer from 1 (fails the rubric) to 5 (fully meets it).
Reply with only JSON like {{"score": 4, "reason": "one sentence"}}."""


@scorer("llm_judge")
def llm_judge(item, answer, ctx) -> Score:
    if ctx.judge is None:
        raise ValueError("llm_judge items need a judge model (--judge provider:model)")
    rubric = item.meta.get("rubric", "The answer is correct, complete and clear.")
    raw = ctx.judge.generate(JUDGE_PROMPT.format(prompt=item.prompt, rubric=rubric, answer=answer))
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        verdict = json.loads(m.group(0)) if m else {}
        score = int(verdict["score"])
    except (ValueError, KeyError, TypeError):
        return Score(False, 0.0, f"judge reply unreadable: {raw[:80]!r}")
    score = max(1, min(5, score))
    threshold = int(item.meta.get("pass_score", 4))
    return Score(score >= threshold, (score - 1) / 4, f"judge {score}/5: {verdict.get('reason', '')}"[:200])
