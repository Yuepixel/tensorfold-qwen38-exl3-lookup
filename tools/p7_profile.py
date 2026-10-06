#!/usr/bin/env python3
"""P7.1.2 comprehensive profile: per class, per stream count.

  decode mode (non-streaming): prefill_s, decode_tps, rounds, acceptance, prompt/cached tokens
  ttft mode (SSE): client-measured time to first content token, and total wall

  python3 tools/p7_profile.py http://127.0.0.1:8080 MODEL --label l7:2 --streams 1,2 --rounds 4
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request
import uuid

FILE = "\n".join([
    "def add(a, b):", "    return a + b", "", "def sub(a, b):", "    return a - b", "",
    "def mul(a, b):", "    return a * b", "", "def div(a, b):", "    return a / b", "",
    "def mod(a, b):", "    return a % b", "", "def power(a, b):", "    return a ** b", "",
    "def floor_div(a, b):", "    return a // b", "", "def max_of(a, b):", "    return a if a > b else b", "",
    "def min_of(a, b):", "    return a if a < b else b", "", "def abs_of(a):",
    "    return a if a >= 0 else -a", "", "def neg(a):", "    return -a",
])
CLASSES = {
    "chat": "用中文简要解释什么是 GPU 并行计算，用一段话说明，不超过 120 字。",
    "code": "用 Python 写一个函数 fib(n)，返回第 n 个斐波那契数，要求用迭代实现。只输出代码，不要解释。",
    "prose": "法国首都是哪座城市？只答城市名。",
    "repeat": "严格按下面的模式续写，逐字复现，只输出序列，不要解释：\n\napple, banana, cherry, apple, banana, cherry, apple, banana, cherry, apple,",
    "edit": "你是一个代码编辑助手。下面是文件 utils.py 的全部内容：\n\n```python\n" + FILE +
            "\n```\n\n请给每个函数加上类型注解，并完整输出整个文件（不要省略任何行，不要解释）。",
    "continue": "严格续写下面的文件内容，逐字复现已有函数，然后追加一个新函数 double(a) 返回 a * 2。"
                "只输出代码，不要解释：\n\n```python\n" + FILE,
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def body(model, prompt, tokens, stream):
    return {"model": model, "max_tokens": tokens, "temperature": 0, "ignore_eos": True, "draft": True,
            "messages": [{"role": "user", "content": prompt}], "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}


def call_decode(base, model, prompt, tokens, out, idx, barrier):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body(model, prompt, tokens, False)).encode(),
                                 headers={"Content-Type": "application/json"})
    barrier.wait()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    tf = d.get("tensorfold", {}) or {}
    u = d.get("usage", {}) or {}
    out[idx] = {"prompt": u.get("prompt_tokens"),
                "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "prefill_s": tf.get("prefill_s"), "decode_s": tf.get("decode_s"),
                "tps": tf.get("decode_tps"), "rounds": tf.get("rounds"),
                "drafted": tf.get("drafted"), "accepted": tf.get("accepted"),
                "ld": tf.get("lookup_drafted"), "la": tf.get("lookup_accepted")}


def call_ttft(base, model, prompt, tokens, out, idx, barrier):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body(model, prompt, tokens, True)).encode(),
                                 headers={"Content-Type": "application/json"})
    barrier.wait()
    t0 = time.perf_counter()
    ttft = None
    tlast = t0
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            tlast = time.perf_counter()
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            if ttft is None:
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                for ch in obj.get("choices", []):
                    if (ch.get("delta") or {}).get("content"):
                        ttft = time.perf_counter() - t0
                        break
    out[idx] = {"ttft": ttft, "total": tlast - t0}


def run_cell(base, model, klass, streams, rounds, mode, tokens, unique=False):
    base_prompt = CLASSES[klass]
    fn = call_decode if mode == "decode" else call_ttft
    rows = []
    for _ in range(rounds):
        prompts = [("[%s] " % uuid.uuid4().hex) + base_prompt for _ in range(streams)] if unique else [base_prompt] * streams
        out = {}
        barrier = threading.Barrier(streams)
        threads = [threading.Thread(target=fn, args=(base, model, prompts[i], tokens, out, i, barrier))
                   for i in range(streams)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        rows.append([out[i] for i in sorted(out)])
    steady = rows[1:] if len(rows) > 1 else rows
    flat = [v for row in steady for v in row]
    return flat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--classes", default="chat,code,edit,continue")
    p.add_argument("--streams", default="1,2")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--ttft-tokens", type=int, default=64)
    p.add_argument("--label", default="")
    p.add_argument("--unique", action="store_true", help="prepend a per-request nonce so every prompt is a cache miss (cold prefill)")
    a = p.parse_args()
    classes = [c for c in a.classes.split(",") if c]
    streams = [int(s) for s in a.streams.split(",") if s]
    for s in streams:
        for klass in classes:
            d = run_cell(a.base, a.model, klass, s, a.rounds, "decode", a.tokens, a.unique)
            acc = [c["accepted"] / c["drafted"] for c in d if c.get("drafted")]
            print("DEC %s class=%s streams=%d cold=%d prompt=%.0f cached=%.0f prefill=%.3f decode_tps=%.1f rounds=%.1f acc=%.3f"
                  % (a.label, klass, s, int(a.unique), mean([c["prompt"] for c in d]), mean([c["cached"] for c in d]),
                     mean([c["prefill_s"] for c in d]), mean([c["tps"] for c in d]),
                     mean([c["rounds"] for c in d]), mean(acc)), flush=True)
        for klass in classes:
            t = run_cell(a.base, a.model, klass, s, a.rounds, "ttft", a.ttft_tokens, a.unique)
            print("TTFT %s class=%s streams=%d cold=%d ttft=%.3f total=%.3f"
                  % (a.label, klass, s, int(a.unique), mean([c["ttft"] for c in t]), mean([c["total"] for c in t])), flush=True)


if __name__ == "__main__":
    main()
