#!/usr/bin/env bash
set -u
python3 - <<'PY'
import importlib
import tensorfold
for name in ("tensorfold",
             "tensorfold.families.qwen4_exp.cuda.engine",
             "tensorfold.families.qwen4_exp.cuda.decode",
             "tensorfold.families.qwen4_exp.cuda.multi"):
    try:
        m = importlib.import_module(name)
        print(name, "->", m.__file__)
    except Exception as e:
        print(name, "ERR", e)
try:
    import tensorfold.families.qwen4_exp.cuda.lookup as L
    print("lookup import OK ->", L.__file__)
except Exception as e:
    print("lookup import FAIL:", e)
import inspect
src = inspect.getsource(importlib.import_module("tensorfold.families.qwen4_exp.cuda.engine"))
print("engine has build_lookup:", "build_lookup" in src)
PY
pip show tensorfold 2>/dev/null | grep -iE 'Location|Version|Editable' || echo NO_PIP_INFO
ls -la /usr/local/lib/python3*/dist-packages/ 2>/dev/null | grep -i tensorfold || true
