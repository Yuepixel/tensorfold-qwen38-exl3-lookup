"""P21 disk-spill multi-session stress test (run against the served container).

usage: python3 p21_stress.py <burn|resend> <out.json> [M]

burn   : send M distinct long single-turn chats once (populates RAM kept slots;
         under --parallel 2 the older ones get evicted -> spilled to disk).
resend : send the SAME M chats again; report per-doc cached / token_sha / wall.
         After a cold restart with the spill dir intact, the docs that were
         evicted earlier must hit the disk (cached>0) and be much faster.
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODE = sys.argv[1]
OUT = sys.argv[2]
M = int(sys.argv[3]) if len(sys.argv) > 3 else 10


def call(system, user):
    body = json.dumps({
        "model": "tensorfold",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
        "return_token_ids": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.loads(r.read())
    tf = j.get("tensorfold", {}) or {}
    u = j.get("usage", {}) or {}
    d = u.get("prompt_tokens_details", {}) or {}
    return {
        "sha": tf.get("token_sha"),
        "ids": tf.get("token_ids"),
        "usage_cached": d.get("cached_tokens"),
        "tf_cached": tf.get("cached"),
        "prompt_tokens": u.get("prompt_tokens"),
        "wall": round(time.time() - t0, 3),
    }


unit = ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
        "eiusmod tempor incididunt ut labore et dolore magna aliqua. ")
N = 36000


def doc(seed):
    head = "Session %d marker %d-KWS. " % (seed, seed * 104729)
    s = (head + unit * (N // len(unit) + 2))[:N]
    return s


USER = "In one short sentence, state which letter of the Greek alphabet follows gamma."
DOCS = [doc(i) for i in range(M)]

results = {"mode": MODE, "m": M, "cases": {}}
total = 0.0
for i, s in enumerate(DOCS):
    r = call(s, USER)
    results["cases"]["doc%d" % i] = r
    total += r["wall"]
    print("doc%d ptok=%s cached=%s tf_cached=%s sha=%s wall=%.2fs" % (
        i, r["prompt_tokens"], r["usage_cached"], r["tf_cached"], r["sha"], r["wall"]),
        flush=True)

results["total_wall"] = round(total, 3)
results["hits"] = sum(1 for r in results["cases"].values() if (r["tf_cached"] or 0) > 0)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print("mode=%s total_wall=%.2fs hits=%d/%d wrote %s" % (
    MODE, total, results["hits"], M, OUT))
