# Qwen3.8-Flash-Next on TensorFold v0.6.5 (Python) — EXL3 3.05 bpw + prompt-lookup (CUDA)

[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![TensorFold](https://img.shields.io/badge/TensorFold-0.6.5-orange)](https://github.com/ashhart/TensorFold)
[![GPU](https://img.shields.io/badge/NVIDIA-DGX%20Spark%20(GB10)-76B900)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![patches](https://img.shields.io/badge/patches-14-informational)](patches)

An **unofficial fork/branch of [TensorFold](https://github.com/ashhart/TensorFold) `v0.6.5`**:
an EXL3 3.05 bpw stack plus a **prompt-lookup (suffix) draft arm** for
**Qwen3.8-Flash-Next** on the CUDA path — 14 patches, the benchmark fixtures, and
byte-exact `token_sha` receipts covering the **EXL3 prompt-GEMM prefill work**, the
**prompt-lookup drafter**, and the **vision / video / image-history prefix-cache**
additions.

It began as the reference material for [TensorFold issue #444](https://github.com/ashhart/TensorFold/issues/444)
(now closed; that thread links back here) and is kept as an independent, runnable
branch of the Python/EXL3 line.

- **Engine:** TensorFold `0.6.5` (`v0.6.5` tag) + 14 patches below
- **Model:** `Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5` (EXL3 3.05 bpw, group-32, `h5_ng5` pack — the current weights)
- **Weight packs:** tuned on the 3.05 bpw pack above; the `0013` (F6b) decode receipt was captured while a `Lygodactylus` 4.05 bpw uncensored pack was served (2026-10-08 → 2026-10-10). §5.6 reports both packs side by side.
- **Hardware:** NVIDIA DGX Spark (GB10), CPU/GPU unified 128 GB
- **Modalities:** text, image, and video; multi-turn image chats keep a cached prefix
- **Policy under test:** forced `l7:2` (baseline, `MAX_DRAFTS=7, MIN_MATCH=2`) vs experimental cost gate (`auto`) — see below
- **Shareable copy:** [gist.github.com/Yuepixel/fdbc20aa54fa37f382455d679452d012](https://gist.github.com/Yuepixel/fdbc20aa54fa37f382455d679452d012)

The whole point of the receipts: **every draft arm must be byte-exact with the
serial (no-draft) decode.** For all fixtures, `token_sha` is identical across
serial / MTP-only / lookup / auto, single-stream and parallel, and a mid-flight
cancellation leaves the next request unchanged.

---

## Results at a glance

| | |
|---|---|
| **Prefill** (cold, patched vs `v0.6.5`) | **~670–800 → 1696–1777 t/s (≈2.4×)**, bit-identical |
| **Decode** (prompt-lookup drafter, verbatim loads) | **+10 % … +44 %** (neutral on free chat/code) |
| **Vision / video** | native image + video input (PyAV), multi-image, OCR-verified |
| **Image multi-turn cache** | turn-2 `cached_tokens` **0 → 2514 / 2550**, byte-identical to a cold re-prefill |
| **Determinism** | `token_sha` identical serial / MTP-only / lookup / auto, single & parallel |
| **Cold lone-stream decode (0013)** | graph-recapture rounds **191/888 → 0**; cold decode **21–44 → 33–123 t/s** (4.05 bpw uncensored) / **37.7–131.9 t/s** (current 3.05 bpw) |

The prefill lift is patches `0001–0004` (a rebase of
[#212](https://github.com/ashhart/TensorFold/pull/212) onto `v0.6.5`) plus `0007`
(a port of [#283](https://github.com/ashhart/TensorFold/pull/283), the width-gated
two-chunk ring, another **+13–15 %**); the drafter is `0005`/`0006`. It is the most
broadly useful part of this stack and is orthogonal to the drafter — full numbers in
[§5](#5-results). Patches `0008`–`0012` add native **vision**, **video + multi-image**,
and the **image-history prefix cache** for multi-turn image chats — see §5.4–§5.5.
`0013` — our own fix, not a port — removes the cold-decode graph-recapture penalty on a
lone stream (§5.6), and `0014` keeps a system-block checkpoint so long shared prefixes
stop re-prefilling (§5.7).

### 🤖 Deploy it with one link (for AI agents)

Hand a coding agent (Claude Code, Cursor, opencode, Codex, …) this file and it has
everything needed to build and serve the stack on a DGX Spark:

> Fetch **https://raw.githubusercontent.com/Yuepixel/tensorfold-qwen38-exl3-lookup/main/RECIPE.md**
> and follow it to install and run Qwen3.8-Flash-Next (EXL3 3.05 bpw) on a DGX Spark
> (GB10) — TensorFold v0.6.5 (Python engine line) + 14 patches.

[`RECIPE.md`](RECIPE.md) is self-contained: prerequisites, exact commits, the patch
apply, the CUDA build, the weight fetch, the serve command, and the acceptance check.

---

## 1. Patches

Applied in order on top of `v0.6.5` (tag `p7.1-batch-lookup-v1` = `0001`–`0006`; tag `p8-ring-v1` = `0001`–`0007`; tags `p9-vision-v1` / `p10-vision-v1` = through `0011`; tag `p11-image-prefix-v1` = `0001`–`0012`; `0013` and `0014` sit on top of that (no tag)):

| # | Patch | Area |
|---|-------|------|
| 0001 | `perf(cuda): EXL3 prompt experts decode each weight tile once for up to 64 pairs` | EXL3 prompt GEMM |
| 0002 | `Flash Next's EXL3 routed windows take a whole 2048-row prompt chunk` | EXL3 routed windows |
| 0003 | `EXL3 packs' fp16 matmuls sum K slices in one program` | EXL3 packed matmul |
| 0004 | `EXL3 prompt GEMM tiles (128,64,8,3,8)` | EXL3 prompt GEMM |
| 0005 | `qwen4_exp: add prompt-lookup (suffix) draft arm wired into mtp_decode` | lookup arm |
| 0006 | `qwen4_exp: wire prompt-lookup into the batched decoder (per-stream backlog)` | lookup batching |
| 0007 | `perf(cuda): EXL3 prompt experts keep two trellis-word chunks in flight for gate\|up below 4 bits` | EXL3 prompt GEMM |
| 0008 | `vision: exl3_convert supports inline vision towers and BF16 inputs` (port of #229) | vision / EXL3 convert |
| 0009 | `vision: shadow verification + image/regression scripts` | vision bring-up |
| 0010 | `vision: enable native image input in production` (`run_serve.sh --vision`) | vision serve |
| 0011 | `run_serve.sh: TENSORFOLD_VISION_MAX_IMAGES / TENSORFOLD_VISION_IMAGE_TOKENS knobs` | vision knobs |
| 0012 | `fix(flash next cuda): image prompts resume and keep prompt states, matched on the images' pixels` (cherry-pick of #263) | image prefix cache |
| 0013 | `fix(flash next cuda): keep the solo graph slot hot on a cold prefix miss (F6b)` (ours) | decode / solo graph slot |
| 0014 | `feat(flash next cuda): keep a state at the last 2048-row boundary before a system block's end` (port of `grearjake-star`'s `sys-checkpoint` @ `b4a9993`) | long-context prefix reuse |

```sh
git checkout v0.6.5
git am patches/*.patch        # or: git apply patches/*.patch
```

Combined diffstat (`v0.6.5..p11-image-prefix`): **53 files, +2625 / −104**. `0013` adds one file, +8 / −6 (`multi_solo.py`); `0014` touches 4 files, +190 / −6 (`markers.py`, `engine.py`, two tests).
(`0001`–`0007` alone are 34 files, +2019 / −40.) A squashed view of the first seven is in
`receipts/all-changes.diff`.

Touched areas:

```
src/tensorfold/cuda/exl3/{experts.cpp,experts.cu,experts.py,experts_cb0.cu,experts_cb1.cu,experts_cb2.cu,experts_prompt.cuh,prefill.py}
src/tensorfold/cuda/{health.py,streams.py}
src/tensorfold/families/qwen4_exp/cuda/{decode.py,engine.py,exl3_mm.py,exl3_pack.py,forward.py,lookup.py,multi.py,multi_fill.py,multi_solo.py,image_rows.py,state.py}
src/tensorfold/vision/{exl3_convert.py,qwen_checkpoint.py}
src/tensorfold/server/metrics.py
tests/cuda/{test_exl3_prompt_experts.py,test_qwen4_exp_lookup.py,test_flashnext_vision.py}
tests/{test_flashnext_image_keys.py,test_flashnext_absolute_grow_host.py,test_vision_exl3_convert.py}
tools/p7_conc_bench.py, tools/p7_conc_edit.py, tools/p7_cost_fit.py, tools/p7_edit_bench.py,
tools/p7_probe.py, tools/p7_profile.py, tools/prefill_cold.py
run_serve.sh, run_serve_vision.sh
```

### Where the lookup arm lives

- `families/qwen4_exp/cuda/lookup.py` (new, 370 LOC, pure Python): `SuffixIndex`
  (reverse longest-match on an n-gram suffix, `n = min(MIN_MATCH, 8)`) + `Lookup.plan()`
  (band gating, EOS truncation, optional cost-gated depth). Ported from
  `jayleaton/glm53-tensorfold-spark` `patches/0020` (`glm5_next/cuda/lookup.py`).
- `decode.py`: `_mtp_lookup_decode()` (batches lookup rows into a backlog, absorbs
  them on the next MTP round) — this is the **per-stream backlog** handling.
- `engine.py` `_decode()`: builds the lookup from the prefill rows when `self.depth > 0`.
- `multi.py` `_draft_all()`: the batched path (≥2 concurrent streams) keeps its own
  `lookups/back/arm` dicts keyed by stream id.

---

## 2. Running the server

`run_serve.sh` in this tree uses `TF_EXL3_LOOKUP` (default `l7:2`) and serves native
vision by default (`TF_EXL3_VISION`, with `TENSORFOLD_VISION_MAX_IMAGES` /
`TENSORFOLD_VISION_IMAGE_TOKENS` — see §5.4):

```sh
# baseline forced l7:2 (default)
exec tensorfold serve /models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 --host 0.0.0.0 --port 8080

# MTP-only (disable lookup): TF_EXL3_LOOKUP must be set to empty, not unset
TF_EXL3_LOOKUP= tensorfold serve /models/... --host 0.0.0.0 --port 8080

# experimental cost gate
TF_EXL3_LOOKUP=auto tensorfold serve /models/... --host 0.0.0.0 --port 8080
```

> **Note:** the script expands `${TF_EXL3_LOOKUP-l7:2}` (single dash), so *unsetting*
> the var still yields `l7:2`. To get MTP-only you must export it **empty**
> (`TF_EXL3_LOOKUP=`).

### ⚠️ Caveat: `--parallel` + single stream does not reach the lookup arm

With `--parallel N`, a **single** in-flight request is served by the solo decoder
(`multi_solo.py::_solo_round`), which does **not** thread `lookup` into
`mtp_decode`. The lookup wiring is only exercised by:

- the **no-scheduler** path (`engine._decode`) — i.e. **without** `--parallel`, or
- the **batched** path (`multi._draft_all`) — i.e. **≥2 concurrent streams**.

So a lone request against a `--parallel` server is effectively MTP-only
(`lookup_drafted == 0`). This is the reason the same config can show `ld=0`
single-stream and `ld>0` at 2 streams. Relevant if the native port routes solo
requests separately. (The `run_serve.sh` committed upstream has no `--parallel`.)

---

## 3. Fixtures

Every case below is a plain `/v1/chat/completions` call with `temperature: 0`,
`max_tokens: 96`, `chat_template_kwargs.enable_thinking = false`, and
`draft: true|false`. The generator is `receipts/collect.py` (stdlib only).

```json
{
  "model": "Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5",
  "messages": [{"role": "user", "content": "<prompt>"}],
  "temperature": 0,
  "max_tokens": 96,
  "draft": true,
  "stream": false,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

7 prompts, chosen to hit different draft behaviour:

| id | intent | why |
|----|--------|-----|
| `prose` | short factual, 2 tokens out | lookup never fires (no repetition) |
| `code` | write `fib(n)` | moderate continuation overlap |
| `repeat` | strict `apple` pattern | early EOS |
| `chat` | short Chinese paragraph | lookup drafted but mostly rejected |
| `edit_full` | type-annotate a 15-fn file, echo all | long verbatim echo → high accept |
| `edit_continue` | echo file then append `double()` | verbatim echo + divergence at the end |
| `pattern_novel` | continue a colour cycle, then diverge + `DONE` | **lookup rounds → MTP round** (the return path) |

Exact prompt texts are in `receipts/collect.py` (`PROMPTS`).

---

## 4. Byte-exact receipts (`token_sha`)

SHA returned in the `tensorfold` block. All values are the first 12 hex chars.

### 4.1 Single stream (no `--parallel`)

`serial == drafted == MTP-only == auto` for every prompt:

| prompt | `token_sha` | completion_tokens |
|--------|-------------|-------------------|
| prose | `f4cf4bac607d` | 2 |
| code | `2deda2799112` | 74 |
| repeat | `aef89122efe1` | 2 |
| chat | `35ac773b2d02` | 55 |
| edit_full | `98f603d270f7` | 96 |
| edit_continue | `fb7342c804b3` | 96 |
| pattern_novel | `d4e923c62b39` | 40 |

Lookup activity (`ld` = `lookup_drafted`, `la` = `lookup_accepted`), policy `l7:2`:

| prompt | ld | la | rounds (l7:2 vs MTP-only) |
|--------|----|----|---------------------------|
| code | 7 | 2 | 13 vs 73 |
| chat | 7 | 0 | 46 vs 54 |
| edit_full | 85 | 69 | 15 vs 95 |
| edit_continue | 84 | 84 | 12 vs 95 |
| pattern_novel | 31 | 31 | 6 vs 39 |
| prose / repeat | 0 | 0 | 1 vs 1 |

### 4.2 `auto` (experimental cost gate)

Same hashes. The gate is conservative: only `edit_continue` fired
(`ld=13/la=13`); every other prompt stayed at `ld=0`. This is why `auto` is
**experimental** and the default is forced `l7:2` — the cost model is not
calibrated across prompt shapes (also see `tools/p7_cost_fit.py`).

### 4.3 Parallel (`--parallel 2`, `k=2` concurrent → batched path)

Every stream's sha equals the single-stream sha (`parallel_match: true` for all):

| prompt | sha set | `ld` per stream |
|--------|---------|-----------------|
| code | `{2deda2799112}` | 6, 6 |
| chat | `{35ac773b2d02}` | 6, 6 |
| edit_full | `{98f603d270f7}` | 48, 36 |
| edit_continue | `{fb7342c804b3}` | 48, 48 |
| pattern_novel | `{d4e923c62b39}` | 20, 20 |
| prose | `{f4cf4bac607d}` | 0, 0 |
| repeat | `{aef89122efe1}` | 0, 0 |

MTP-only at 2 streams: identical sha set, all `ld=0`.

### 4.4 Cancellation

`collect.py cancel`: start a `stream=true` request (`max_tokens 512`), read 5 SSE
lines, abort. A subsequent `repeat` request — serial **and** drafted — returns
the same sha as before the abort (`ref_match: true`), and `/health` is clean
(`requests_running: 0`, `decoding: 0`, `prefilling: 0`). Receipts:
`receipt_cancel_lookup-l7-2.json`, `receipt_cancel_mtponly.json`.

### 4.5 lookup → MTP return

`pattern_novel` is built for this: a repeating colour cycle (lookup rounds) that
must then diverge into a novel token + `DONE` (MTP rounds). `ld=31/la=31`,
`rounds=6` vs MTP-only `39`, sha unchanged — the lookup arm hands control back
to MTP cleanly.

### 4.6 Cold lone-stream byte-exactness under F6b (`0013`)

F6b changes the *scheduling* of the solo graph slot, not the math: on a cold
(prefix-missing) request the slot keeps its captured graphs instead of being swapped.
Verified byte-exact: 4 unique-prefix fixtures, `temperature 0` — a serial run
(`--concurrency 1 --no-draft`) and a cold, prefix-missing parallel run
(`--concurrency 2`, which exercises the F6b path) return the **same `token_sha`** for
every fixture. Receipt: `receipt_f6b_lookup-l7-2.json`; reproduce with
`receipts/f6b_before_after.sh` (see §5.6).

---

## 5. Results

### 5.1 Prefill — the big one

The CUDA prompt-GEMM work (`0001`–`0004`, a rebase of
[#212](https://github.com/ashhart/TensorFold/pull/212) onto `v0.6.5`, plus `0007`, a
port of [#283](https://github.com/ashhart/TensorFold/pull/283)) is what lifts
EXL3 off the floor. Cold prefill, `tools/prefill_cold.py`, `temperature 0`, random
nonce prefix so it always cache-misses:

| prompt tokens | 2048 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---|---|
| `v0.6.5` without the fix (the #258 level) | ~670–800 across the board | | | | |
| `v0.6.5` + patches `0001`–`0004` | **1385** | **1528** | **1549** | **1537** | **1498** |
| + patch `0007` (ring, all seven) | **1696** | **1771** | **1777** | **1758** | **1706** |

≈**2.4×**, output bit-identical, decode untouched; 256k needle TTFT ≈168 s. The ring
keeps the next chunk of trellis words in flight while the current one decodes; it is
**width-gated** to `gate|up` below 4 bits, so it fires on this 3.05 bpw pack (K2 = 6)
and stays off on 4-bit packs (K2 = 8). This is orthogonal to the drafter and is the
reusable artifact, since #212 is closed/unmerged and the Python line is frozen (#286).
Reproduce:

```sh
python3 tools/prefill_cold.py build /models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 prompts.json
python3 tools/prefill_cold.py run  http://127.0.0.1:8080 Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 prompts.json prefill.json
```

### 5.2 Decode — prompt-lookup drafter (single stream, no `--parallel`)

`decode_tps = completion_tokens / decode_s`, `l7:2` vs MTP-only:

| prompt | MTP-only tps | l7:2 tps | Δ |
|--------|--------------|----------|---|
| code | 112.6 | 110.2 | −2.1% |
| chat | 34.9 | 34.6 | −0.9% |
| edit_full | 116.5 | 128.5 | +10.3% |
| edit_continue | 122.0 | 160.3 | +31.4% |
| pattern_novel | 105.3 | 151.3 | +43.7% |
| serial (no draft) | ~36 | — | — |

Lookup wins where the model echoes verbatim (`edit*`, `pattern_novel`) and is
roughly neutral on free chat/code.

### 5.3 Dual stream (`--parallel 2`, batched path)

Per-stream tps, `l7:2` vs MTP-only:

| load | single MTP → lookup | dual MTP → lookup |
|------|---------------------|-------------------|
| continue | 115.4 → 160.2 (+38.8%) | 115.1 → 159.5 (+38.6%) |
| edit | 116.4 → 131.0 (+12.5%) | 115.4 → 130.8 (+13.3%) |
| chat | 41.6 → 61.0 (+46.6%) | 37.7 → 59.4 (+57.6%) |
| code | 113.5 → 109.7 (−3.3%) | 111.1 → 107.1 (−3.6%) |

Dual per-stream ≈ single, so aggregate throughput scales almost linearly. Raw rows:
`receipt_parallel_*.json`.

### 5.4 Vision, video, and multi-image (`0008`–`0011`)

Patch `0008` ports upstream [#229](https://github.com/ashhart/TensorFold/pull/229)
(inline vision towers, BF16 inputs) onto the EXL3 convert path; `0009`/`0010` bring the
tower up and enable it in the production serve script; `0011` adds the two knobs. The
vision weights ride a **FP16 sidecar** next to the 3.05 bpw pack (the EXL3 checkpoint is
untouched). Images arrive as data URLs; the FrameAV/PyAV path (`av 19.0.1`) also decodes
**video** frames. Knobs (served via `run_serve.sh`):

| var | default | meaning |
|---|---|---|
| `TENSORFOLD_VISION_MAX_IMAGES` | `4` | images (or frame groups) accepted per request |
| `TENSORFOLD_VISION_IMAGE_TOKENS` | `4096` | token budget per image |

These are the patch defaults; the **reference deployment raises them to `12` / `49152`**
(via `run_serve.sh`) so a multi-image request fits more image tokens.

Verified OCR/description on single- and multi-image requests, a video clip transcribed
to its on-screen text, and mixed text/image requests — all correct; `--parallel 2`
concurrent image streams are byte-equal. (Regression scripts: `run_serve_vision.sh`,
`p9_vision_test*.py`.)

### 5.5 Image-history prefix cache (`0012`) — multi-turn images stop re-prefilling

Patch `0012` cherry-picks upstream [#263](https://github.com/ashhart/TensorFold/pull/263),
which fixes open [#414](https://github.com/ashhart/TensorFold/issues/414): with one image
in an agent chat, **every later turn came back `cached_tokens=0`** and re-prefilled the
whole prompt. The fix keys each image placeholder row on the image's own pixels (a
negative row id derived from the content hash), so a resumed turn matches and keeps the
cached prefix instead of rebuilding it.

Before → after (2-turn chat, ~2.5k prompt, `temperature 0`):

| turn | `cached_tokens` | `prompt_tokens` | wall |
|---|---|---|---|
| image, turn 1 | 0 | 2515 | 2.87 s |
| image, turn 2 — with `0012` | **2514** | 2550 | 0.24 s |
| image, turn 2 — before | 0 | 2550 | 1.53 s |

The resumed turn's output is **byte-identical to a cold re-prefill**; the text-only
baseline hash is unchanged; three concurrent image streams under `--parallel 2` return
byte-equal outputs; and the prefix cache composes with the `TF_EXL3_LOOKUP=l7:2` drafter.
Caveat: **video multi-turn cache reuse** flows through the same path but was not measured
on its own.

### 5.6 Cold lone-stream decode — graph-recapture removed (`0013`)

With `--parallel 2`, a lone request that misses the prefix cache used to take the bail
path in `_move_to_solo` (`self.solo.st = old`), which rebuilt an empty `Graphs` on the
fresh slot and **lazily recaptured every decoding round** (~0.18 s/round, ~21.5 % of
rounds). `0013` evicts/drops the kept prefix end so the request keeps the hot graph
slot instead. This is engine code and is **pack-agnostic**; it was measured on both the
3.05 bpw pack (current) and the 4.05 bpw uncensored pack (served 2026-10-08 → 2026-10-10).

**Under the 3.05 bpw pack** (`h5_ng5`, current weights; re-measured 2026-10-10).
`tools/p7_profile.py` (streams=1, unique prefixes), decode t/s with `0013`:

| class | cold | warm |
|---|---|---|
| chat | 37.7 | 38.2 |
| code | 102.2 | 105.7 |
| edit | 131.9 | 136.6 |
| continue | 119.7 | 124.0 |

Cold ≈ warm (within ~3 % per class): a cold prefix miss no longer pays the recapture
tax. A before-fix baseline was not separately re-run on this pack, but the recapture
mechanism and its removal are identical to the 4.05 bpw run below.

**Under the 4.05 bpw uncensored pack** (`Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-4bpw`,
the pack served 2026-10-08 → 2026-10-10), **before → after `0013`**:

| class | decode t/s before `0013` | after `0013` |
|---|---|---|
| chat | 21.0 | 33.5 |
| code | 38.3 | 111.5 |
| edit | 49.7 | 123.1 |
| continue | 44.1 | 116.7 |

Solo-round forward passes over 0.1 s: **191 / 888 → 0 / 888**. TTFT and prefill are
unchanged (the fix only touches decode rounds); warm-state decode also improves (`code`
94.8 → 103.0, `edit` 103.9 → 121.5, `continue` 107.6 → 126.1). Full receipt:
`receipt_f6b_lookup-l7-2.json` (both packs).

### 5.7 System-block checkpoint — long shared prefixes stop re-prefilling (`0014`)

A system block that only changes near its end (a date line, a memory section) used to be
prefilled whole again, because the shared block's state was kept only at the second
message's start. `0014` (a port of `grearjake-star`'s `sys-checkpoint` branch, upstream
commit `b4a9993`) keeps **one more state** at the last multiple of `TF_SYS_CHECKPOINT`
rows (default `2048`, a prompt pass; `0` off) at least `MIN_GAP` before that start, so a
13.1K block resumes at `12288`. It is one more stop of `snapshot_points`, kept/forked/
evicted within the same kept-state count. Pure Python — no CUDA recompile.

Micro A/B (single session, shared prefix): `cached_tokens` **0 → 8192**, prefill
**4.57 s → 0.38 s** (~12×). Six-session shared prefix: round-1 / total **41.56 s r1,
41.61 s r2 → 10.18 s (4.1×) / 3.22 s (12.9×)**. `token_sha` and `token_ids` are
**identical** off vs on — the reply is a fresh prefill's, token for token.

Receipts: `receipts/P20-bench-result-20261010.json` (A/B), `receipts/P20-tok-consistency-{on,off}.json` (token identity), the profiler trace `receipts/P20-profile-20261010.txt`, and notes in `receipts/P20-sys-checkpoint-notes.md`.

Known limitation: the checkpoint shares the same kept-state count, so under bounded
eviction it can be evicted before use — an intermittent cold miss (~1/6 serial, ~2–3/6
concurrent). Enlarging the keep (`TENSORFOLD_KEEP` 8/16/64) measured **no** change; a
GLM-style disk spill is the proposed next step.

---

## 6. Reproducing

```sh
# inside the tf-exl3 container, server up on 127.0.0.1:8080
python3 receipts/collect.py probe
python3 receipts/collect.py single   lookup-l7-2
python3 receipts/collect.py single   mtponly
python3 receipts/collect.py single   auto
python3 receipts/collect.py parallel lookup-l7-2 2
python3 receipts/collect.py parallel mtponly     2
python3 receipts/collect.py cancel   lookup-l7-2
python3 receipts/f6b_before_after.sh
```

Each command writes `receipt_<cmd>_<label>.json` next to itself.

---

## 7. Attribution

- prompt-lookup design ported from **`jayleaton/glm53-tensorfold-spark`**
  (`patches/0020`, `glm5_next/cuda/lookup.py`).
- The width-gated two-chunk ring (`0007`) ports **TensorFold PR
  [#283](https://github.com/ashhart/TensorFold/pull/283)** (`grearjake-star/ring-by-width`,
  design by Jürgen Schmied) onto `v0.6.5` + `0001`–`0006`.
- The inline-vision-tower convert change (`0008`) ports **TensorFold PR
  [#229](https://github.com/ashhart/TensorFold/pull/229)**.
- The image-history prefix cache (`0012`) cherry-picks **TensorFold PR
  [#263](https://github.com/ashhart/TensorFold/pull/263)** (by olexale), which addresses
  open issue [#414](https://github.com/ashhart/TensorFold/issues/414).
- `0013` (F6b, the solo-graph-slot cold-miss fix) is this project's own work — not a port
  or cherry-pick.
- The system-block checkpoint (`0014`) ports `grearjake-star`'s TensorFold branch
  `sys-checkpoint` (commit `b4a9993450f636b314485a07969600cc2be831a4`).
- Built on **TensorFold** by ashhart and contributors; EXL3 kernel work follows
  the existing Flash-Next CUDA/EXL3 path.
- Hardware/quant recipe: Qwen3.8-Flash-Next EXL3 3.05 bpw `h5_ng5` on a single
  DGX Spark (GB10).

## 8. License

This is an **independent, unofficial distribution** of patches against TensorFold
`v0.6.5`; it is not affiliated with or endorsed by the TensorFold project.

The repository as a whole — the patches, tools, receipt scripts, and documentation
— is distributed under the **Apache License, Version 2.0** ([`LICENSE`](LICENSE),
Copyright 2026 Yuepixel).

- Upstream TensorFold's Apache-2.0 license text is reproduced in
  [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt); its `NOTICE` and the
  third-party attribution notices it carries are reproduced in [`NOTICE`](NOTICE)
  and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- Code written before TensorFold 0.6.0 keeps its MIT notice in
  [`LICENSES/MIT.txt`](LICENSES/MIT.txt); third-party MIT components are listed in
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- Receipts are data; reuse freely.

## 9. 致谢 / Acknowledgments

谨向 TensorFold 的原作者 **ashhart** 致以最诚挚的谢意：是他开辟了这条路，没有他卓越的贡献，就没有我们后来的一切工作。感谢 **米亚（MiaAI-Lab）** 的指引——她在 X 上的内容推广与配方分享，让我们得以一次次继续向前。也感谢每一位补丁作者，我们已在他们的作品处一一署名：正是在他们工作的基础上，我们才能在 TensorFold v0.6.5 之上做出这一系列改进。由衷感谢你们。

We owe a most sincere thank-you to **ashhart**, the original author of TensorFold: he blazed this trail, and without his outstanding contribution none of the work that followed would exist. Our gratitude to **Mia (MiaAI-Lab)** for the guidance — her promotion on X and the recipes she shared are what let us keep moving forward. And our thanks to every patch author, each credited where their work appears: it is on their shoulders that we built this series of improvements on top of TensorFold v0.6.5. Thank you, sincerely.

---

*This article was completed with the assistance of DeepSeek AI.*
