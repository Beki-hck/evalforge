# evalforge

![tests](https://github.com/Beki-hck/evalforge/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![deps](https://img.shields.io/badge/dependencies-none-brightgreen)

A small, dependency-free harness for evaluating large language models. You point it at a model and a suite of tasks, and it gives you an accuracy score per category, a JSON file with every answer, and an HTML report you can open in a browser.

It runs **free local models through [Ollama](https://ollama.com)** by default, and also works with any OpenAI-compatible API or Anthropic's Claude API if you have a key.

```bash
evalforge run code -m ollama:llama3.2:3b -m ollama:qwen2.5:3b
```

## Why I built it

Most "which model is better?" claims are vibes. Evaluating models well means writing clear tasks, scoring them in a way that can't be gamed, and looking at the failures, not just the number. evalforge is the tool I use for that, and later projects in this portfolio (an Amharic benchmark, a preference-labeling app) build on it.

## Features

- **Pluggable models:** `ollama:*`, `openai:*` (any OpenAI-compatible endpoint, including Groq and LM Studio), `anthropic:*`, or any Python function.
- **Seven scorers:**
  - `numeric` extracts the final number from a worked answer
  - `python_tests` runs model-written code against hidden unit tests in a separate process with a timeout
  - `regex`, `exact_match` and `contains` for format and instruction-following checks
  - `choice` for multiple-choice questions
  - `llm_judge` grades open-ended answers against a rubric using a second model
- **Built-in suites:** 20 math word problems, 15 coding tasks, and 10 instruction-following tasks. Every expected answer is verified by the test suite (reference solutions must score 100%).
- **Answer cache:** re-running or re-scoring a suite doesn't call the model again.
- **Robust runs:** a failed API call is recorded as a failed item with the error, not a crashed run.
- **Reports:** a Markdown table for READMEs and a standalone HTML report with every prompt and answer.
- **No dependencies:** standard library only. `pytest` is needed only for development.

## Quick start

```bash
git clone https://github.com/Beki-hck/evalforge
cd evalforge
pip install -e .

# 1. Install Ollama from https://ollama.com, then pull a small model
ollama pull llama3.2:3b

# 2. Run a suite
evalforge run math -m ollama:llama3.2:3b --html

# 3. Compare models (writes results/report.html)
evalforge run code -m ollama:llama3.2:3b -m ollama:qwen2.5:3b

# See what's available
evalforge list
```

Using a hosted model instead:

```bash
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY=...
evalforge run instructions -m anthropic:claude-haiku-4-5
```

## Writing your own suite

A suite is a JSONL file with one task per line. An optional first line sets the system prompt.

```jsonl
{"system": "End with 'Answer: <number>'."}
{"id": "q1", "category": "percent", "prompt": "What is 15% of 240?", "expected": 36, "scorer": "numeric"}
{"id": "q2", "category": "code", "prompt": "Write `add(a, b)`.", "scorer": "python_tests", "tests": "assert add(2, 3) == 5"}
{"id": "q3", "category": "writing", "prompt": "Explain DNS simply.", "scorer": "llm_judge", "rubric": "Accurate, under 80 words."}
```

```bash
evalforge run my_suite.jsonl -m ollama:llama3.2:3b --judge ollama:qwen2.5:7b
```

See [`examples/`](examples/) for an LLM-as-judge suite.

## Results

<!-- results:start -->
_Results from local runs will be added here._
<!-- results:end -->

## How it works

```
suite.jsonl ──► load_suite ──► run_suite ──► model.generate (cached, parallel)
                                   │
                                   └──► scorer(item, answer) ──► Score(passed, value, detail)
                                                                      │
                                             results/*.json ◄─────────┘
                                                   │
                                     markdown_table / html_report
```

| File | Purpose |
|---|---|
| `src/evalforge/models.py` | Model backends over plain HTTP |
| `src/evalforge/scorers.py` | Scorer registry and the seven scorers |
| `src/evalforge/suite.py` | JSONL suite loading and validation |
| `src/evalforge/runner.py` | Parallel runs, answer cache, summaries |
| `src/evalforge/report.py` | Markdown and HTML reports |
| `src/evalforge/cli.py` | `evalforge run`, `report` and `list` |

## Safety note

`python_tests` runs code written by a model. It uses a temporary folder, Python's isolated mode and a timeout, but it is not a security sandbox. Run untrusted models in a container.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The test suite covers every scorer, suite validation, caching, error handling, the CLI, and the Ollama backend against a fake local HTTP server.

## Roadmap

- [ ] pass@k with multiple samples per task
- [ ] Bootstrap confidence intervals on accuracy
- [ ] Amharic suite (see the upcoming `amharic-llm-bench`)
- [ ] Judge-agreement checks: compare `llm_judge` scores with human labels

## License

MIT
