import pytest

from evalforge.scorers import ScoringContext, extract_code, extract_final_number, get_scorer
from evalforge.suite import Item
from evalforge.models import FunctionModel

CTX = ScoringContext()


def item(scorer, expected=None, **meta):
    return Item(id="t", prompt="q", scorer=scorer, expected=expected, meta=meta)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("So the total is 42.", 42),
        ("First 10, then 20. Answer: 1,250", 1250),
        ("#### -3.5", -3.5),
        ("The answer is **$17,595.00**", 17595),
        ("no numbers here", None),
    ],
)
def test_extract_final_number(text, expected):
    assert extract_final_number(text) == expected


def test_numeric_with_tolerance():
    s = get_scorer("numeric")
    assert s(item("numeric", 17595, tolerance=0.01), "Answer: 17595.00", CTX).passed
    assert not s(item("numeric", 95), "Answer: 105", CTX).passed


def test_exact_and_contains():
    assert get_scorer("exact_match")(item("exact_match", "Addis Ababa"), "  addis ababa. ", CTX).passed
    assert get_scorer("contains")(item("contains", ["blue", "azure"]), "It is Azure today", CTX).passed
    assert not get_scorer("contains")(item("contains", "red"), "blue", CTX).passed


def test_regex_is_case_sensitive_by_default():
    s = get_scorer("regex")
    assert not s(item("regex", [r"^[^A-Z]+$"]), "The sky", CTX).passed
    assert s(item("regex", [r"^the"], ignore_case=True), "The sky", CTX).passed


def test_choice():
    s = get_scorer("choice")
    assert s(item("choice", "B"), "Let me think... Answer: (B)", CTX).passed
    assert not s(item("choice", "B"), "I pick C", CTX).passed


def test_extract_code_prefers_fenced_block():
    ans = "Here you go:\n```python\ndef f():\n    return 1\n```\nHope it helps"
    assert extract_code(ans).strip() == "def f():\n    return 1"


def test_python_tests_pass_fail_and_timeout():
    s = get_scorer("python_tests")
    tests = "assert add(2, 3) == 5"
    assert s(item("python_tests", tests=tests), "```python\ndef add(a, b):\n    return a + b\n```", CTX).passed
    bad = s(item("python_tests", tests=tests), "def add(a, b):\n    return a - b", CTX)
    assert not bad.passed and "AssertionError" in bad.detail
    slow = s(item("python_tests", tests=tests), "while True: pass", ScoringContext(code_timeout=1))
    assert not slow.passed and "timed out" in slow.detail


def test_llm_judge_parses_json_and_threshold():
    s = get_scorer("llm_judge")
    judge = FunctionModel(lambda p: 'Sure: {"score": 4, "reason": "good"}')
    res = s(item("llm_judge", rubric="be nice"), "hello", ScoringContext(judge=judge))
    assert res.passed and res.value == 0.75
    low = FunctionModel(lambda p: '{"score": 2, "reason": "meh"}')
    assert not s(item("llm_judge"), "x", ScoringContext(judge=low)).passed
    junk = FunctionModel(lambda p: "I refuse")
    assert "unreadable" in s(item("llm_judge"), "x", ScoringContext(judge=junk)).detail


def test_llm_judge_requires_judge():
    with pytest.raises(ValueError):
        get_scorer("llm_judge")(item("llm_judge"), "x", CTX)
