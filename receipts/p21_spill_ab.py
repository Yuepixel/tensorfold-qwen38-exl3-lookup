"""P21 disk-spill end-to-end A/B (real machine).

Verifies that a kept prefix EVICTED to disk and later COLD-resumed (fresh process,
empty RAM) replays the SAME reply tokens as a full cold prefill.

Method: send N distinct long single-turn chats (system+user) so that, under
`--parallel 2` (2 slots), the later ones evict earlier kept prefixes to disk.
  warm  : spill ON, fresh dir  -> case0 is a full prefill (cached=0) = REFERENCE sha
  cold  : process restarted (RAM empty), spill ON -> case0 must come from DISK
          (cached>0) and its token_sha must equal the warm reference.

Run (in-container):
    python3 p21_spill_ab.py <warm|cold> <out.json>
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODE = sys.argv[1] if len(sys.argv) > 1 else "warm"
OUT = sys.argv[2] if len(sys.argv) > 2 else "p21_%s.json" % MODE


def call(system, user):
    body = json.dumps({
        "model": "tensorfold",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": False,
        "return_token_ids": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
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
    head = "Document %d private marker %d-QXZ. " % (seed, seed * 7919)
    s = (head + unit * (N // len(unit) + 2))[:N]
    return s


USER = "In one short sentence, state which letter of the Greek alphabet follows gamma."
DOCS = [doc(i) for i in range(4)]

results = {"mode": MODE, "cases": {}}
for i, s in enumerate(DOCS):
    r = call(s, USER)
    results["cases"]["doc%d" % i] = r
    print("doc%d ptok=%s cached=%s tf_cached=%s sha=%s wall=%.2fs" % (
        i, r["prompt_tokens"], r["usage_cached"], r["tf_cached"], r["sha"], r["wall"]),
        flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print("wrote", OUT)
