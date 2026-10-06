"""Concurrent P7.1 class bench: K simultaneous prompts of one class share each round's forward
(the batched MultiDecoder path the P7.1.1 lookup wiring targets). Reports per-stream and mean decode
tok/s from the server's own tensorfold.decode_s. Compare TF_EXL3_LOOKUP=l7:2 vs "" (MTP-only).

  python3 tools/p7_conc_bench.py http://127.0.0.1:8080 MODEL --class code --streams 2 --rounds 6 --label l7:2
"""

import argparse
import json
import statistics
import threading
import urllib.request

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


def one(base, model, prompt, tokens, out, idx, barrier):
    body = {"model": model, "max_tokens": tokens, "temperature": 0, "ignore_eos": True,
            "messages": [{"role": "user", "content": prompt}], "draft": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    barrier.wait()
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read())
    tf = data["tensorfold"]
    n = data["usage"]["completion_tokens"]
    out[idx] = {"tps": (n - 1) / tf["decode_s"] if tf["decode_s"] > 0 else 0.0, "n": n,
                "decode_s": tf["decode_s"], "rounds": tf["rounds"],
                "ld": tf.get("lookup_drafted"), "la": tf.get("lookup_accepted")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--class", dest="klass", default="code", choices=sorted(CLASSES))
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--streams", type=int, default=2)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--label", default="")
    a = p.parse_args()
    prompt = CLASSES[a.klass]
    means = []
    for r in range(a.rounds):
        out = {}
        barrier = threading.Barrier(a.streams)
        threads = [threading.Thread(target=one, args=(a.base, a.model, prompt, a.tokens, out, i, barrier))
                   for i in range(a.streams)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        vals = [out[i] for i in sorted(out)]
        mean = statistics.mean(v["tps"] for v in vals)
        means.append(mean)
        print("round", r, a.label, "perstream=", [round(v["tps"], 1) for v in vals],
              "mean=%.1f" % mean, "ld=", [v["ld"] for v in vals], "la=", [v["la"] for v in vals],
              "rounds=", [v["rounds"] for v in vals])
    steady = means[1:] if len(means) > 1 else means
    print("LABEL", a.label, "CLASS", a.klass, "OVERALL_MEAN=%.1f" % statistics.mean(means),
          "STEADY_MEAN=%.1f" % statistics.mean(steady))


main()
