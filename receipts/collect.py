#!/usr/bin/env python3
"""P8 receipts for TensorFold #444: byte-exact token-hash receipts across draft arms.

Run inside the tf-exl3 container against the local server (127.0.0.1:8080):

  python3 collect.py probe                  # dump response schema + the first token_sha
  python3 collect.py single   <label>       # serial vs drafted hashes for every probe prompt
  python3 collect.py parallel <label> [k]   # k concurrent requests; every stream's sha must match serial
  python3 collect.py cancel   <label>       # abort a stream mid-flight, then prove the next request is unchanged

`<label>` records which server state was under test, e.g. `lookup-l7:2`, `mtponly`, `auto`.
Each command writes receipt_<cmd>_<label>.json next to this file and prints one line per case.
Pure stdlib; no dependencies beyond the running server.
"""

import http.client
import json
import os
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
MODEL = "Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5"
HERE = os.path.dirname(os.path.abspath(__file__))

_FILE = "\n".join([
    "def add(a, b):", "    return a + b", "", "def sub(a, b):", "    return a - b", "",
    "def mul(a, b):", "    return a * b", "", "def div(a, b):", "    return a / b", "",
    "def mod(a, b):", "    return a % b", "", "def power(a, b):", "    return a ** b", "",
    "def floor_div(a, b):", "    return a // b", "", "def max_of(a, b):", "    return a if a > b else b", "",
    "def min_of(a, b):", "    return a if a < b else b", "", "def abs_of(a):",
    "    return a if a >= 0 else -a", "", "def neg(a):", "    return -a",
])

PROMPTS = {
    "prose": "法国首都是哪座城市？只答城市名。",
    "code": "用 Python 写一个函数 fib(n) 返回前 n 个斐波那契数，只输出代码。",
    "repeat": "严格按模式续写，每行只写一个词：\n1. apple\n2. apple\n3. apple\n4. apple\n5.",
    "chat": "用中文简要解释什么是 GPU 并行计算，用一段话说明，不超过 120 字。",
    "edit_full": ("你是一个代码编辑助手。下面是文件 utils.py 的全部内容：\n\n```python\n" + _FILE +
                  "\n```\n\n请给每个函数加上类型注解，并完整输出整个文件（不要省略任何行，不要解释）。"),
    "edit_continue": ("严格续写下面的文件内容，逐字复现已有函数，然后追加一个新函数 double(a) 返回 a * 2。"
                      "只输出代码，不要解释：\n\n```python\n" + _FILE),
    # a span that repeats (lookup rounds) then must diverge (MTP rounds): exercises the return lookup -> MTP
    "pattern_novel": (
        "严格逐字续写下面这一行（保持在同一种颜色循环里），然后在同一行末尾追加一个与前面任何颜色都"
        "不同的、你随机选出的颜色单词，再换行写 DONE，不要解释：\n\n"
        "red, green, blue, red, green, blue, red, green, blue, red, green, blue, red, green, blue, "
        "red, green, blue,"
    ),
}


def payload(prompt, draft, max_tokens=96, stream=False):
    return {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "draft": bool(draft), "stream": bool(stream),
            "chat_template_kwargs": {"enable_thinking": False}}


def find(o, key):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = find(v, key)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v, key)
            if r is not None:
                return r
    return None


def post(prompt, draft, max_tokens=96):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload(prompt, draft, max_tokens)).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def rec(d):
    return {"token_sha": find(d, "token_sha"),
            "completion_tokens": find(d, "completion_tokens"),
            "rounds": find(d, "rounds"),
            "decode_s": find(d, "decode_s"),
            "prefill_s": find(d, "prefill_s"),
            "lookup_drafted": find(d, "lookup_drafted"),
            "lookup_accepted": find(d, "lookup_accepted")}


def save(cmd, label, out):
    path = os.path.join(HERE, "receipt_%s_%s.json" % (cmd, label.replace("/", "_").replace(":", "-")))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("-> wrote", path)


def probe(label):
    d = post(PROMPTS["prose"], False, 16)
    print("top-level keys:", sorted(d.keys()))
    print("token_sha:", find(d, "token_sha"))
    tfkeys = sorted(d.get("tensorfold", {}).keys()) if isinstance(d.get("tensorfold"), dict) else None
    print("tensorfold keys:", tfkeys)


def single(label):
    out = {}
    for name, p in PROMPTS.items():
        s = rec(post(p, False))
        d = rec(post(p, True))
        ok = s["token_sha"] == d["token_sha"]
        out[name] = {"serial": s, "drafted": d, "match": ok}
        print(label, name, "serial", s["token_sha"], "drafted", d["token_sha"], "match", ok,
              "ld", d["lookup_drafted"], "la", d["lookup_accepted"], "rounds", d["rounds"])
    save("single", label, out)


def parallel(label, k=2):
    out = {}
    for name, p in PROMPTS.items():
        res = {}
        for draft in (False, True):
            vals = {}
            bar = threading.Barrier(k)

            def w(i, draft=draft):
                bar.wait()
                vals[i] = rec(post(p, draft))

            ts = [threading.Thread(target=w, args=(i,)) for i in range(k)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            res["drafted" if draft else "serial"] = [vals[i] for i in range(k)]
        shas = {v["token_sha"] for v in res["serial"]} | {v["token_sha"] for v in res["drafted"]}
        ok = len(shas) == 1
        out[name] = {"serial": res["serial"], "drafted": res["drafted"], "parallel_match": ok}
        print(label, name, "streams", k, "sha_set", sorted(str(s) for s in shas), "match", ok,
              "ld", [v["lookup_drafted"] for v in res["drafted"]])
    save("parallel", label, out)


def cancel(label):
    ref = rec(post(PROMPTS["repeat"], False))
    body = json.dumps(payload(PROMPTS["repeat"], True, 512, stream=True)).encode()
    conn = http.client.HTTPConnection("127.0.0.1", 8080, timeout=60)
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    r = conn.getresponse()
    got = 0
    t0 = time.time()
    while time.time() - t0 < 30:
        line = r.readline()
        if not line:
            break
        got += 1
        if got >= 5:
            break
    conn.close()
    time.sleep(2.0)
    h = json.loads(urllib.request.urlopen(BASE + "/health", timeout=30).read())
    after_s = rec(post(PROMPTS["repeat"], False))
    after_d = rec(post(PROMPTS["repeat"], True))
    ok = ref["token_sha"] == after_s["token_sha"] == after_d["token_sha"]
    out = {"aborted_after_lines": got,
           "ref_serial": ref,
           "post_cancel_serial": after_s,
           "post_cancel_drafted": after_d,
           "ref_match": ok,
           "health_after": {"busy": h.get("busy"), "requests_running": h.get("requests_running"),
                            "streams": h.get("streams")}}
    print(label, "aborted_lines", got, "ref_match", ok, "health", out["health_after"])
    save("cancel", label, out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    label = sys.argv[2] if len(sys.argv) > 2 else "unlabeled"
    if cmd == "probe":
        probe(label)
    elif cmd == "single":
        single(label)
    elif cmd == "parallel":
        parallel(label, int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    elif cmd == "cancel":
        cancel(label)
    else:
        raise SystemExit("unknown command %r" % cmd)
