"""Command line: `evalforge run`, `evalforge report`, `evalforge list`."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .models import get_model
from .report import html_report, load_runs, markdown_table
from .runner import run_suite
from .scorers import available_scorers
from .suite import builtin_suites, load_suite


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def cmd_run(args) -> int:
    suite = load_suite(args.suite)
    judge = get_model(args.judge) if args.judge else None
    out_dir = Path(args.out)
    saved = []
    for spec in args.model:
        model = get_model(spec)
        print(f"\n{model.name} on {suite.name} ({len(suite.items[:args.limit] if args.limit else suite.items)} items)")

        def show(res):
            mark = "✓" if res.passed else "✗"
            print(f"  {mark} {res.id:<16} {res.detail[:70]}", flush=True)

        run = run_suite(suite, model, judge=judge, workers=args.workers, limit=args.limit,
                        cache_path=out_dir / ".cache.json", on_result=None if args.quiet else show)
        path = run.save(out_dir / f"{_slug(suite.name)}__{_slug(model.name)}.json")
        saved.append(path)
        print(f"  accuracy {run.accuracy:.1%}  ({sum(r.passed for r in run.results)}/{len(run.results)})  saved {path}")
    if len(saved) > 1 or args.html:
        runs = load_runs(saved)
        report = out_dir / "report.html"
        report.write_text(html_report(runs, f"evalforge · {suite.name}"), encoding="utf-8")
        print(f"\n{markdown_table(runs)}\n\nHTML report: {report}")
    return 0


def cmd_report(args) -> int:
    runs = load_runs(args.results)
    if args.markdown:
        print(markdown_table(runs))
    out = Path(args.output)
    out.write_text(html_report(runs, args.title), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_list(args) -> int:
    print("Built-in suites:")
    for name in builtin_suites():
        s = load_suite(name)
        print(f"  {name:<14} {len(s.items)} items")
    print("Scorers: " + ", ".join(available_scorers()))
    print("Model providers: ollama:<model>, openai:<model>, anthropic:<model>, echo:<any>")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evalforge", description="Evaluate LLMs on task suites.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a suite against one or more models")
    r.add_argument("suite", help="built-in suite name or path to a .jsonl file")
    r.add_argument("-m", "--model", action="append", required=True, help="provider:model, repeatable")
    r.add_argument("--judge", help="judge model for llm_judge items, e.g. ollama:qwen2.5:7b")
    r.add_argument("-o", "--out", default="results", help="output folder (default: results)")
    r.add_argument("-w", "--workers", type=int, default=2, help="parallel requests (default: 2)")
    r.add_argument("-n", "--limit", type=int, help="only run the first N items")
    r.add_argument("--html", action="store_true", help="also write an HTML report for a single model")
    r.add_argument("-q", "--quiet", action="store_true")
    r.set_defaults(fn=cmd_run)

    rep = sub.add_parser("report", help="build an HTML report from saved results")
    rep.add_argument("results", nargs="+")
    rep.add_argument("-o", "--output", default="report.html")
    rep.add_argument("--title", default="evalforge report")
    rep.add_argument("--markdown", action="store_true", help="also print a Markdown table")
    rep.set_defaults(fn=cmd_report)

    ls = sub.add_parser("list", help="show built-in suites, scorers and providers")
    ls.set_defaults(fn=cmd_list)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
