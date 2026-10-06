"""P7.1.2 cost fitter: read TF_EXL3_COST_LOG JSONL, print verify[] (fwd ms per window width)
and a least-squares fit of the MTP arm's mtp_ms ~ a + b*(steps-1) + c*(backlog-1)."""

import json
import statistics
from collections import defaultdict

import numpy as np

rows_data = defaultdict(list)
m_data = []
n = 0
with open("/tmp/cost.jsonl") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n += 1
        rows_data[r["rows"]].append(r["fwd_ms"])
        if r["arm"] == "m":
            m_data.append((r["steps"], r["backlog"], r["mtp_ms"]))

print("total rounds:", n, "  m-arm:", len(m_data))
print("%4s %6s %12s %10s %10s" % ("rows", "count", "fwd_ms_mean", "fwd_min", "fwd_max"))
for rows in sorted(rows_data):
    v = rows_data[rows]
    print("%4d %6d %12.4f %10.4f %10.4f" % (rows, len(v), statistics.mean(v), min(v), max(v)))

if m_data:
    A = np.array([[1.0, s - 1, b - 1] for s, b, _ in m_data])
    y = np.array([m for _, _, m in m_data])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    print("\nmtp fit: mtp_ms = %.4f + %.4f*(steps-1) + %.4f*(backlog-1)" % tuple(coef))
    print("residual std: %.4f" % float(np.std(y - pred)))
    g = defaultdict(list)
    for s, b, m in m_data:
        g[(s, b)].append(m)
    print("%6s %8s %5s %14s" % ("steps", "backlog", "n", "mtp_ms_mean"))
    for k in sorted(g):
        print("%6d %8d %5d %14.3f" % (k[0], k[1], len(g[k]), statistics.mean(g[k])))
