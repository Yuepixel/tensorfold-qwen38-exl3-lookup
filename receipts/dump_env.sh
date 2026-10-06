#!/usr/bin/env bash
set -u
P=$(ps -eo pid,args | grep -F 'tensorfold serve' | grep -v grep | awk '{print $1}' | head -1)
echo "PID=$P"
if [ -n "$P" ]; then
  tr '\0' '\n' < /proc/$P/environ | grep -iE 'TF_EXL3|LOOKUP' || echo NO_LOOKUP_ENV
fi
echo '---health---'
curl -s --max-time 10 http://127.0.0.1:8080/health
echo
