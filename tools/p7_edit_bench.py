"""P7 edit/agent-class decode bench: a file-rewrite prompt whose continuation repeats long
spans of the prompt (the case prompt-lookup is meant to win). Greedy (temp 0), 128 tokens.

  python3 tools/p7_edit_bench.py http://127.0.0.1:8080 MODEL [--tokens 128] [--reps 3] [--label L]
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request

FILE = "\n".join([
    "def add(a, b):",
    "    return a + b",
    "",
    "def sub(a, b):",
    "    return a - b",
    "",
    "def mul(a, b):",
    "    return a * b",
    "",
    "def div(a, b):",
    "    return a / b",
    "",
    "def mod(a, b):",
    "    return a % b",
    "",
    "def power(a, b):",
    "    return a ** b",
    "",
    "def floor_div(a, b):",
    "    return a // b",
    "",
    "def max_of(a, b):",
    "    return a if a > b else b",
    "",
    "def min_of(a, b):",
    "    return a if a < b else b",
    "",
    "def abs_of(a):",
    "    return a if a >= 0 else -a",
    "",
    "def neg(a):",
    "    return -a",
])

PROMPTS = [
    {"name": "edit-full-file",
     "prompt": ("你是一个代码编辑助手。下面是文件 utils.py 的全部内容：\n\n```python\n" + FILE +
                "\n```\n\n请给每个函数加上类型注解，并完整输出整个文件（不要省略任何行，不要解释）。")},
    {"name": "edit-continue",
     "prompt": ("严格续写下面的文件内容，逐字复现已有函数，然后追加一个新函数 double(a) 返回 a * 2。"
                "只输出代码，不要解释：\n\n```python\n" + FILE)},
]


def stream(base: str, model: str, item: dict, tokens: int, temperature: float) -> dict:
    body = {"model": model, "max_tokens": tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}
    if temperature > 0:
        body.update(top_k=20, top_p=0.95)
    url = base + "/v1/chat/completions"
    body["messages"] = [{"role": "user", "content": item["prompt"]}]
    body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    usage = None
    text = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                if piece:
                    now = time.perf_counter()
                    first = first if first is not None else now
                    last = now
                    text.append(piece)
    n = int(usage["completion_tokens"]) if usage else None
    return {"ttft_s": first - start, "decode_s": last - first, "tokens": n,
            "decode_tps": (n - 1) / (last - first) if n and last > first else None, "text": "".join(text)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--label", default="")
    args = p.parse_args()
    for item in PROMPTS:
        stream(args.base, args.model, item, args.tokens, 0.0)              # warm-up
        runs = [stream(args.base, args.model, item, args.tokens, 0.0) for _ in range(args.reps)]
        tps = [r["decode_tps"] for r in runs if r["decode_tps"]]
        row = {"label": args.label, "prompt": item["name"], "decode_tps_median": statistics.median(tps),
               "decode_tps_all": [round(x, 2) for x in tps],
               "ttft_s_median": statistics.median(r["ttft_s"] for r in runs), "sample": runs[0]["text"][:120]}
        print(json.dumps({k: row[k] for k in ("label", "prompt", "decode_tps_median", "decode_tps_all", "ttft_s_median")}), flush=True)


if __name__ == "__main__":
    main()
