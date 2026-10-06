#!/usr/bin/env bash
set -u
H=http://127.0.0.1:8080/health
M=Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
before=$(curl -s --max-time 10 $H | python3 -c 'import sys,json;print(json.load(sys.stdin)["lookup_drafted_total"])')
echo "lookup_drafted_total before=$before"
python3 /src/TensorFold/tools/p7_edit_bench.py http://127.0.0.1:8080 $M --reps 1 --label probe
after=$(curl -s --max-time 10 $H | python3 -c 'import sys,json;print(json.load(sys.stdin)["lookup_drafted_total"])')
echo "lookup_drafted_total after=$after"
