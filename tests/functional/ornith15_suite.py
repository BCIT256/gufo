#!/usr/bin/env python3
"""Ornith-1.5 HTTP checks, suite ornith15-functional-v1.

Starts `gufo serve llm` for one APEX GGUF, or scores an already running
OpenAI server when `--base-url` is set. Five fixed English tasks. Each task
is pass/fail. Sampling is the model card (temperature 0.6, top_p 0.95,
top_k 20, seed 20261028) and, unless `--sampling` says otherwise, the same
run again at temperature 0. This file is the task definition; score it only
after it is committed. It is not part of `--suite all`.

Run under the engine nix shell so the gpu-test binary finds its loader:

  nix develop -c python3 tests/functional/ornith15_suite.py \
    --binary ./build/gpu-test/gufo \
    --model /path/to/Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf \
    --out /path/to/artifacts/m2/functional-v1
"""

import argparse
from http.client import HTTPConnection
import json
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

SUITE_ID = "ornith15-functional-v1"
SEED = 20261028
MAX_TOKENS = 2048
CONTEXT = 8192
CARD = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": SEED}
GREEDY = {"temperature": 0.0, "top_p": 0.95, "top_k": 20, "seed": SEED}
PYTHON_FENCE = re.compile(r"```python[^\n]*\n(.*?)```", re.I | re.S)

MERGE_PROMPT = """Write a Python function merge_half_open(intervals).

intervals is a list of [start, end) pairs of integers. Merge pairs that overlap. Touching endpoints do not merge: [0, 2) and [2, 4) stay separate. Unsorted input is allowed. An empty list returns an empty list. If any pair has end < start, raise ValueError. Return the merged pairs sorted by start.

Reply with one ```python block that contains only the function."""

MERGE_TESTS = r"""
def check():
    if merge_half_open([]) != []:
        raise SystemExit("empty")
    try:
        merge_half_open([[1, 0]])
    except ValueError:
        pass
    else:
        raise SystemExit("missing ValueError")
    if merge_half_open([[0, 2], [2, 4]]) != [[0, 2], [2, 4]]:
        raise SystemExit("touching")
    if merge_half_open([[0, 3], [1, 2]]) != [[0, 3]]:
        raise SystemExit("nested")
    if merge_half_open([[5, 7], [1, 3], [2, 6]]) != [[1, 7]]:
        raise SystemExit("unsorted overlap")
    if merge_half_open([[0, 1], [3, 4], [1, 2]]) != [[0, 2], [3, 4]]:
        raise SystemExit("adjacent after sort")
    if merge_half_open([[0, 5], [3, 8], [10, 12]]) != [[0, 8], [10, 12]]:
        raise SystemExit("overlap and gap")
check()
"""

MEDIAN_PROMPT = """This function raises IndexError on an empty list. The even branch also averages the wrong pair.

```python
def median_ms(samples):
    ordered = sorted(samples)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid] + ordered[mid + 1]) / 2
```

Traceback (most recent call last):
  File "bench.py", line 4, in <module>
    print(median_ms([]))
  File "stats.py", line 6, in median_ms
    return (ordered[mid] + ordered[mid + 1]) / 2
IndexError: list index out of range

Fix median_ms. An empty list must raise ValueError. An odd count returns the middle value. An even count averages the two central values. Keep the name median_ms. Reply with one ```python block that contains only the fixed function."""

MEDIAN_TESTS = r"""
def check():
    try:
        median_ms([])
    except ValueError:
        pass
    else:
        raise SystemExit("empty")
    if median_ms([10]) != 10:
        raise SystemExit("one")
    if median_ms([1, 3, 2]) != 2:
        raise SystemExit("odd")
    if median_ms([1, 2, 3, 4]) != 2.5:
        raise SystemExit("even4")
    if median_ms([4, 1]) != 2.5:
        raise SystemExit("even2")
check()
"""

CLAMP_TURN1 = """Write a Python function clamp(value, lo, hi). Return value limited to the inclusive range [lo, hi]. Assume lo <= hi. Reply with one ```python block that contains only the function."""

CLAMP_TURN2 = """Change the function from your previous answer so that when lo > hi the bounds are swapped before clamping. Keep every other behavior. Reply with the full function in one ```python block."""

CLAMP_LO_HI = r"""
def check():
    if clamp(5, 0, 10) != 5 or clamp(-1, 0, 10) != 0 or clamp(11, 0, 10) != 10:
        raise SystemExit("lo <= hi")
check()
"""

CLAMP_SWAP = r"""
def check():
    if clamp(5, 0, 10) != 5 or clamp(-1, 0, 10) != 0 or clamp(11, 0, 10) != 10:
        raise SystemExit("kept range")
    if clamp(3, 8, 2) != 3 or clamp(1, 8, 2) != 2 or clamp(9, 8, 2) != 8:
        raise SystemExit("swap")
check()
"""

TOOL_PROMPT = """The service port and the latest test summary are in notes/status.txt. Use the read_file and run_tests tools. Then answer with the port and the number of passed tests as digits."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the project tests and return the summary.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

AREA_PROMPT = """A rectangle is 13 units wide and 8 units tall. Reply with the area as the whole number only, with no units and no extra words."""


def reserve_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, binary, model, log_path):
        self.port = reserve_port()
        self.log_path = Path(log_path)
        self.command = [
            str(binary), "serve", "--host", "127.0.0.1", "--port", str(self.port),
            "llm", "--model", str(model), "--context", str(CONTEXT), "--sessions", "1",
            "--served-model-name", SUITE_ID,
        ]
        self.process = None
        self.reader = None

    def start(self, timeout):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w")
        self.process = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace")
        def read():
            for line in self.process.stdout:
                self.log.write(line)
                self.log.flush()
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"server exited {self.process.returncode}; see {self.log_path}")
                if self._health():
                    return
                time.sleep(0.2)
            raise TimeoutError(f"server startup timed out; see {self.log_path}")
        except Exception:
            self.stop()
            raise

    def _health(self):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            conn.request("GET", "/health")
            return conn.getresponse().status == 200
        except OSError:
            return False
        finally:
            conn.close()

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(2)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.reader:
            self.reader.join(timeout=10)
            self.reader = None
        log = getattr(self, "log", None)
        if log is not None and not log.closed:
            log.close()


def model_id(host, port):
    conn = HTTPConnection(host, port, timeout=30)
    try:
        conn.request("GET", "/v1/models")
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"/v1/models HTTP {response.status}: {body[:400]!r}")
        payload = json.loads(body)
    finally:
        conn.close()
    data = payload.get("data") or []
    if not data or not data[0].get("id"):
        raise RuntimeError(f"/v1/models has no id: {payload!r}"[:400])
    return data[0]["id"]


def metrics_of(usage, choice):
    usage = usage or {}
    gufo = usage.get("gufo") or {}
    return {
        "ttft_ms": gufo.get("ttft_ms"),
        "completion_tokens_per_second": usage.get("completion_tokens_per_second"),
        "prefill_ms": gufo.get("prefill_ms"),
        "decode_ms": gufo.get("decode_ms"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": (choice or {}).get("finish_reason"),
    }


def post_json(host, port, path, body, timeout):
    raw = json.dumps(body).encode()
    conn = HTTPConnection(host, port, timeout=timeout)
    started = time.perf_counter()
    try:
        conn.request("POST", path, body=raw, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = response.read()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return response.status, payload, elapsed_ms
    finally:
        conn.close()


def chat(host, port, model, sampling, messages, timeout, tools=None, stream=False):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        **sampling,
    }
    if tools is not None:
        body["tools"] = tools
    if not stream:
        status, payload, elapsed_ms = post_json(
            host, port, "/v1/chat/completions", body, timeout)
        parsed = json.loads(payload) if payload else {}
        if status != 200:
            raise RuntimeError(f"HTTP {status}: {payload[:800]!r}")
        choice = parsed["choices"][0]
        record = metrics_of(parsed.get("usage"), choice)
        record["client_total_ms"] = elapsed_ms
        message = choice["message"]
        return message, record, parsed
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    raw = json.dumps(body).encode()
    conn = HTTPConnection(host, port, timeout=timeout)
    started = time.perf_counter()
    content, reasoning = [], []
    usage = None
    finish = None
    client_ttft_ms = None
    try:
        conn.request("POST", "/v1/chat/completions", body=raw,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            detail = response.read()[:800]
            raise RuntimeError(f"HTTP {response.status}: {detail!r}")
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece and client_ttft_ms is None:
                client_ttft_ms = (time.perf_counter() - started) * 1000.0
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
    finally:
        conn.close()
    message = {
        "role": "assistant",
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
    }
    record = metrics_of(usage, {"finish_reason": finish})
    record["client_ttft_ms"] = client_ttft_ms
    record["client_total_ms"] = (time.perf_counter() - started) * 1000.0
    return message, record, {"stream": True, "message": message, "usage": usage}


def python_block(content):
    if not isinstance(content, str) or not content.strip():
        return None, "content has no python block"
    blocks = PYTHON_FENCE.findall(content)
    if not blocks:
        return None, "content has no ```python block"
    return blocks[-1], ""


def run_python(source, tests):
    script = source + "\n" + tests
    with tempfile.TemporaryDirectory(prefix="ornith15-fn-") as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", script], cwd=tmp,
                capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired:
            return False, "python check timed out"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "python check failed").strip()
        return False, detail.splitlines()[-1][:400]
    return True, ""


def task_merge(host, port, model, sampling, timeout):
    message, record, raw = chat(
        host, port, model, sampling,
        [{"role": "user", "content": MERGE_PROMPT}], timeout)
    source, reason = python_block(message.get("content"))
    passed, detail = (False, reason) if source is None else run_python(source, MERGE_TESTS)
    return passed, detail, [record], [raw]


def task_median(host, port, model, sampling, timeout):
    message, record, raw = chat(
        host, port, model, sampling,
        [{"role": "user", "content": MEDIAN_PROMPT}], timeout)
    source, reason = python_block(message.get("content"))
    passed, detail = (False, reason) if source is None else run_python(source, MEDIAN_TESTS)
    return passed, detail, [record], [raw]


def task_clamp(host, port, model, sampling, timeout):
    first, first_record, first_raw = chat(
        host, port, model, sampling,
        [{"role": "user", "content": CLAMP_TURN1}], timeout)
    source, reason = python_block(first.get("content"))
    if source is None:
        return False, reason, [first_record], [first_raw]
    ok, detail = run_python(source, CLAMP_LO_HI)
    if not ok:
        return False, "turn 1: " + detail, [first_record], [first_raw]
    assistant = {
        "role": "assistant",
        "content": first.get("content") or "",
    }
    if first.get("reasoning_content"):
        assistant["reasoning_content"] = first["reasoning_content"]
    second, second_record, second_raw = chat(
        host, port, model, sampling,
        [{"role": "user", "content": CLAMP_TURN1}, assistant,
         {"role": "user", "content": CLAMP_TURN2}], timeout)
    source, reason = python_block(second.get("content"))
    if source is None:
        return False, "turn 2: " + reason, [first_record, second_record], [first_raw, second_raw]
    ok, detail = run_python(source, CLAMP_SWAP)
    if not ok:
        detail = "turn 2: " + detail
    return ok, detail, [first_record, second_record], [first_raw, second_raw]


def tool_result(call):
    function = call["function"]
    arguments = json.loads(function["arguments"])
    if function["name"] == "read_file":
        if arguments.get("path") == "notes/status.txt":
            return "listen port 8741\n"
        return "error: file not found\n"
    return "3 passed\n"


def task_tools(host, port, model, sampling, timeout):
    messages = [{"role": "user", "content": TOOL_PROMPT}]
    records, raws = [], []
    saw_read = saw_tests = False
    for _ in range(4):
        message, record, raw = chat(
            host, port, model, sampling, messages, timeout, tools=TOOLS)
        records.append(record)
        raws.append(raw)
        calls = message.get("tool_calls") or []
        if not calls:
            content = message.get("content") or ""
            if not saw_read or not saw_tests:
                return False, "final answer did not use both tools", records, raws
            if "8741" not in content or re.search(r"\b3\b", content) is None:
                return False, "final answer missed the port or the pass count", records, raws
            return True, "", records, raws
        assistant = {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": calls,
        }
        if message.get("reasoning_content"):
            assistant["reasoning_content"] = message["reasoning_content"]
        messages.append(assistant)
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if call.get("type") != "function" or name not in ("read_file", "run_tests"):
                return False, "tool call name is not read_file or run_tests", records, raws
            if not isinstance(call.get("id"), str) or not call["id"]:
                return False, "tool call id is missing", records, raws
            if not isinstance(arguments, str):
                return False, "tool arguments are not a string", records, raws
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return False, "tool arguments are not JSON", records, raws
            if not isinstance(parsed, dict):
                return False, "tool arguments are not a JSON object", records, raws
            if name == "read_file":
                saw_read = True
            else:
                saw_tests = True
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": tool_result(call),
            })
    return False, "tool round trip did not finish", records, raws


def task_area(host, port, model, sampling, timeout):
    message, record, raw = chat(
        host, port, model, sampling,
        [{"role": "user", "content": AREA_PROMPT}], timeout, stream=True)
    content = (message.get("content") or "").strip()
    passed = re.fullmatch(r"104\.?", content) is not None
    detail = "" if passed else "content is not the whole number 104"
    return passed, detail, [record], [raw]


TASKS = (
    ("write_merge", task_merge),
    ("fix_median", task_median),
    ("tool_round_trip", task_tools),
    ("two_turn_clamp", task_clamp),
    ("rectangle_area", task_area),
)


def run_sampling(host, port, model, name, sampling, out, timeout):
    answers = out / "answers" / name
    answers.mkdir(parents=True, exist_ok=True)
    tasks = []
    for task_id, function in TASKS:
        try:
            passed, detail, records, raws = function(host, port, model, sampling, timeout)
        except Exception as error:
            passed, detail, records, raws = False, str(error), [], []
        for index, raw in enumerate(raws):
            (answers / f"{task_id}-{index}.json").write_text(
                json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
        tasks.append({
            "id": task_id,
            "pass": passed,
            "detail": detail,
            "requests": records,
        })
        print(f"{name} {task_id}: {'pass' if passed else 'fail'} {detail}", flush=True)
    report = {
        "suite": SUITE_ID,
        "sampling": name,
        "model": model,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "tasks": tasks,
    }
    (out / f"{name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--base-url", help="Score this OpenAI server instead of starting gufo")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sampling", choices=("card", "greedy", "both"), default="both")
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    owned = None
    if args.base_url:
        from urllib.parse import urlparse
        parsed = urlparse(args.base_url)
        host, port = parsed.hostname, parsed.port
        if not host or not port:
            parser.error("--base-url needs a host and port")
    else:
        if args.binary is None or args.model is None:
            parser.error("--binary and --model are required without --base-url")
        owned = Server(args.binary, args.model, args.out / "server.log")
        owned.start(args.startup_timeout)
        host, port = "127.0.0.1", owned.port
    try:
        served = model_id(host, port)
        selected = [("card", CARD)] if args.sampling == "card" else (
            [("greedy", GREEDY)] if args.sampling == "greedy" else
            [("card", CARD), ("greedy", GREEDY)])
        reports = [
            run_sampling(host, port, served, name, sampling, args.out, args.request_timeout)
            for name, sampling in selected
        ]
    finally:
        if owned:
            owned.stop()
    failed = [task["id"] for report in reports for task in report["tasks"] if not task["pass"]]
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
