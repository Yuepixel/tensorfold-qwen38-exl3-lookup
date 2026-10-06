#!/usr/bin/env python3
"""P7.1.2 field probe: one drafted request, dump the response blocks' keys and values."""
import json
import urllib.request

BASE = "http://127.0.0.1:8080"
MODEL = "Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5"
body = {
    "model": MODEL, "max_tokens": 48, "temperature": 0, "ignore_eos": True, "draft": True,
    "messages": [{"role": "user", "content": "用中文简要解释什么是 GPU 并行计算，一段话，不超过 80 字。"}],
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(BASE + "/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=180).read())
for k, v in d.items():
    if k == "choices":
        v = [{"finish_reason": c.get("finish_reason"),
              "message_keys": sorted((c.get("message") or {}).keys())} for c in v]
    print(k, "=", json.dumps(v, ensure_ascii=False)[:1200])
