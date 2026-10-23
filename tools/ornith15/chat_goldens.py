#!/usr/bin/env python3
"""Regenerate tests/fixtures/ornith15_chat_goldens.json with Hugging Face.

Renders each case with transformers' apply_chat_template and Ornith's
chat_template.jinja, then tokenizes the result with the Hugging Face
tokenizer. The fixture carries the inputs too, so the C++ test replays exactly
these messages. Run with the flake's pinned transformers (see README.md):

    python3 tools/ornith15/chat_goldens.py HF_DIR tests/fixtures/ornith15_chat_goldens.json
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import transformers

REPO = "ornith-ai/Ornith-1.5-35B-A3B"
REVISION = "10fbf86fed7ecee4a061f8b499a618f46001cac1"
TEMPLATE_SHA256 = "182e77dd83bd8e9ca818b240b82e28f243762cd5dda32e6eef327df7b1cd107e"

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get weather",
        "parameters_json": '{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}',
    },
    {
        "name": "shell",
        "description": "Run a shell command",
        "parameters_json": '{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}',
    },
]

USER = {"role": "user", "content": "Name one color."}
HISTORY = [
    USER,
    {"role": "assistant", "content": "Red", "thought": "I should answer with one color."},
    {"role": "user", "content": "Name another."},
]
TOOL_LOOP = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "thought": "I need the weather tool.",
        "tool_calls": [{"name": "get_weather", "arguments": [["city", "Paris"]]}],
    },
    {"role": "tool", "content": "Sunny, 21 C"},
]
MULTI_TOOL = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Compare Paris and Rome, then list src."},
    {
        "role": "assistant",
        "content": "Let me check both cities.",
        "thought": "Two weather calls and one shell call.",
        "tool_calls": [
            {"name": "get_weather", "arguments": [["city", "Paris"]]},
            {"name": "get_weather", "arguments": [["city", "Rome"]]},
            {"name": "shell", "arguments": [["command", "ls src\nwc -l src/*"]]},
        ],
    },
    {"role": "tool", "content": "Sunny, 21 C"},
    {"role": "tool", "content": "Cloudy, 18 C"},
    {"role": "tool", "content": "main.cpp\nutil.hpp"},
    {"role": "assistant", "content": "Paris is warmer.", "thought": "Paris 21, Rome 18."},
    {"role": "user", "content": "Thanks. Anything else?"},
]

# name, messages, tools offered, enable_thinking
CASES = [
    ("chat", [USER], False, False),
    ("thinking", [USER], False, True),
    ("history", HISTORY, False, True),
    ("history_chat", HISTORY, False, False),
    ("tools", [USER], True, False),
    ("system_tools", [{"role": "system", "content": "Be concise."}, USER], True, True),
    ("tool_loop", TOOL_LOOP, True, True),
    ("multi_tool", MULTI_TOOL, True, True),
]

TOKENIZATION = [
    ("spaces", "hello  world   return x;    indented\t\tword"),
    ("code", "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"),
    ("unicode", "naïve café — 東京 🚀 ½"),
    ("numbers", "3.14159 1,000,000 0x1F 2026-10-23"),
]


def hf_messages(messages):
    """Gufo's message fields mapped to the Hugging Face chat format."""
    out = []
    for m in messages:
        hf = {"role": m["role"], "content": m["content"]}
        if "thought" in m:
            hf["reasoning_content"] = m["thought"]
        if "tool_calls" in m:
            hf["tool_calls"] = [
                {"type": "function", "function": {"name": c["name"], "arguments": dict(c["arguments"])}}
                for c in m["tool_calls"]
            ]
        out.append(hf)
    return out


def hf_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": json.loads(t["parameters_json"]),
            },
        }
        for t in TOOLS
    ]


def token_digest(ids):
    return hashlib.sha256(b"".join(struct.pack("<I", i) for i in ids)).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hf_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    template = (args.hf_dir / "chat_template.jinja").read_text(encoding="utf-8")
    if hashlib.sha256(template.encode()).hexdigest() != TEMPLATE_SHA256:
        raise SystemExit("chat_template.jinja is not the pinned Ornith template")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.hf_dir)
    tokenizer_sha = hashlib.sha256((args.hf_dir / "tokenizer.json").read_bytes()).hexdigest()

    cases = []
    for name, messages, with_tools, thinking in CASES:
        rendered = tokenizer.apply_chat_template(
            hf_messages(messages),
            tools=hf_tools() if with_tools else None,
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        cases.append({
            "name": name,
            "messages": messages,
            "tools": with_tools,
            "enable_thinking": thinking,
            "rendered": rendered,
            "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "token_count": len(ids),
            "token_ids_le_u32_sha256": token_digest(ids),
        })
    tokenization = []
    for name, text in TOKENIZATION:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        tokenization.append({
            "name": name,
            "text": text,
            "token_count": len(ids),
            "token_ids_le_u32_sha256": token_digest(ids),
        })
    fixture = {
        "schema": "gufo.ornith15-chat-goldens.v1",
        "repo": REPO,
        "revision": REVISION,
        "template_sha256": TEMPLATE_SHA256,
        "tokenizer_sha256": tokenizer_sha,
        "transformers": transformers.__version__,
        "tools": TOOLS,
        "cases": cases,
        "tokenization": tokenization,
    }
    args.output.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases and {len(tokenization)} tokenization examples")


if __name__ == "__main__":
    main()
