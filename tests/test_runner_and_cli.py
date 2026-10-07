import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from evalforge.cli import main
from evalforge.models import FunctionModel, ModelError, OllamaModel, get_model
from evalforge.report import html_report, markdown_table
from evalforge.runner import run_suite
from evalforge.suite import load_suite


def test_cache_avoids_repeat_calls(tmp_path):
    calls = []

    def fn(p):
        calls.append(p)
        return "Answer: 1"

    suite = load_suite("math")
    run_suite(suite, FunctionModel(fn), cache_path=tmp_path / "c.json", limit=3)
    again = run_suite(suite, FunctionModel(fn), cache_path=tmp_path / "c.json", limit=3)
    assert len(calls) == 3
    assert all(r.cached for r in again.results)


def test_model_errors_become_failed_items():
    def boom(p):
        raise ModelError("server down")

    run = run_suite(load_suite("math"), FunctionModel(boom), limit=2)
    assert run.accuracy == 0 and all("server down" in r.error for r in run.results)
    assert run.summary()["errors"] == 2


def test_summary_and_reports():
    run = run_suite(load_suite("math"), FunctionModel(lambda p: "Answer: 95", label="m1"), limit=4)
    d = run.to_dict()
    assert d["n"] == 4 and d["passed"] == 1
    md = markdown_table([d])
    assert "| m1 | math | **25.0%** (1/4)" in md
    page = html_report([d])
    assert "<!doctype html>" in page and "m1" in page


class _FakeOllama(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["model"] == "tiny" and body["stream"] is False
        reply = {"message": {"role": "assistant", "content": f"echo: {body['messages'][-1]['content']}"}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_ollama():
    srv = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_backend_over_http(fake_ollama):
    m = OllamaModel("tiny", host=fake_ollama)
    assert m.generate("hi", system="be brief") == "echo: hi"


def test_ollama_unreachable_raises_model_error():
    with pytest.raises(ModelError, match="Could not reach"):
        OllamaModel("x", host="http://127.0.0.1:9", timeout=2).generate("hi")


def test_get_model_specs():
    assert get_model("ollama:llama3.2").name == "ollama:llama3.2"
    assert get_model("anthropic:claude-haiku-4-5").name == "anthropic:claude-haiku-4-5"
    with pytest.raises(ValueError):
        get_model("llama3")
    with pytest.raises(ValueError):
        get_model("foo:bar")


def test_cli_run_and_report(tmp_path, capsys):
    out = tmp_path / "res"
    assert main(["run", "math", "-m", "echo:a", "-m", "echo:b", "-o", str(out), "-n", "3", "-q"]) == 0
    files = sorted(out.glob("math__*.json"))
    assert len(files) == 2 and (out / "report.html").exists()
    assert main(["report", *map(str, files), "-o", str(tmp_path / "r.html"), "--markdown"]) == 0
    assert "| Model | Suite |" in capsys.readouterr().out


def test_cli_list_and_bad_suite(capsys):
    assert main(["list"]) == 0
    assert "math" in capsys.readouterr().out
    assert main(["run", "nope", "-m", "echo:x"]) == 2
