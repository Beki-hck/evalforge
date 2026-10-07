import json
from pathlib import Path

import pytest

from evalforge.models import FunctionModel
from evalforge.runner import run_suite
from evalforge.suite import builtin_suites, load_suite

HERE = Path(__file__).parent


def test_builtin_suites_load():
    assert set(builtin_suites()) >= {"math", "code", "instructions"}
    for name in builtin_suites():
        assert load_suite(name).items


def test_code_suite_reference_solutions_score_100_percent():
    refs = json.loads((HERE / "code_reference.json").read_text())
    oracle = FunctionModel(lambda p: f"```python\n{lookup[p]}\n```", label="oracle")
    suite = load_suite("code")
    lookup = {it.prompt: refs[it.id] for it in suite.items}
    run = run_suite(suite, oracle, workers=4)
    assert run.accuracy == 1.0, [r for r in run.results if not r.passed]


def test_instruction_suite_reference_answers_score_100_percent():
    refs = json.loads((HERE / "instructions_reference.json").read_text())
    suite = load_suite("instructions")
    lookup = {it.prompt: refs[it.id] for it in suite.items}
    run = run_suite(suite, FunctionModel(lambda p: lookup[p]))
    assert run.accuracy == 1.0, [r.id for r in run.results if not r.passed]


def test_math_suite_expected_answers_score_100_percent():
    suite = load_suite("math")
    lookup = {it.prompt: f"Working...\nAnswer: {it.expected}" for it in suite.items}
    assert run_suite(suite, FunctionModel(lambda p: lookup[p])).accuracy == 1.0


def test_wrong_model_scores_zero():
    run = run_suite(load_suite("math"), FunctionModel(lambda p: "Answer: -999999"))
    assert run.accuracy == 0.0


@pytest.mark.parametrize(
    "content,msg",
    [
        ('{"id": "a", "prompt": "p"}', "missing 'scorer'"),
        ('{"id": "a", "prompt": "p", "scorer": "nope"}', "Unknown scorer"),
        ('{"id": "a", "prompt": "p", "scorer": "contains"}\n{"id": "a", "prompt": "p", "scorer": "contains"}', "duplicate id"),
        ("not json", "not valid JSON"),
    ],
)
def test_bad_suite_files_fail_clearly(tmp_path, content, msg):
    f = tmp_path / "bad.jsonl"
    f.write_text(content)
    with pytest.raises(ValueError, match=msg):
        load_suite(f)
