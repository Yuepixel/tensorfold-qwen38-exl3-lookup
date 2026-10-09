#!/bin/sh
# F6b (patch 0013): cold-decode before/after profile.
#   Before: serve TensorFold v0.6.5 + 0001-0012
#   After : serve TensorFold v0.6.5 + 0001-0013
# Server up on 127.0.0.1:8080. The recapture signal is the solo-round fwd timings
# (~0.18 s per graph capture before 0013; none after); see README 5.6.
set -e
URL=${1:-http://127.0.0.1:8080}
MODEL=${2:-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5}
LABEL=${3:-f6b}
python3 tools/p7_profile.py "$URL" "$MODEL" --label "$LABEL" \
  --classes chat,code,edit,continue --streams 1 --rounds 4 --tokens 128 --unique
