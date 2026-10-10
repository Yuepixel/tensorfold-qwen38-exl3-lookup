# P21 · Disk prefix spill (`0015`) — kept prefixes survive a restart

> Built on `0014` (system-block checkpoint). `0014` removed most repeat prefill on a
> live server; its known limit is that the checkpoint competes for the same bounded
> in-memory kept-state count, so under steady/concurrent traffic it can still be
> evicted before use. `0015` answers that with a disk tier.
> Source tags: `[manual]` a human ran the command · `[auto]` produced by unattended
> runs · `[inferred]` read out of the code/docs · `[unverified]` a guess.

## 1. What it does

When a kept prefix is **evicted** from VRAM, `0015` writes it to disk; a later request
that shares it resumes the **longest** stored prefix instead of re-prefilling. It is
pure Python — no CUDA recompile — and it reuses the existing kept-prefix resume path,
so the reply is byte-identical to a fresh prefill.

Storage (one prefix = two files, named by the SHA-256 of the token ids):

- `<sha256[:32]>.safetensors` — the tensors past `pos`: `rec` / `conv` / `ple` tail,
  and every attention layer's `k` / `v` (plus `ks` / `vs` when the KV cache is
  quantized), the index cache and the pooled rows; plus the MTP rows when present.
- `<sha256[:32]>.json` — `{ids, pos, mtp_len, kv_dtype, ratio, layers, quantized, mtp,
  ple, tail, saved}`.

Flow: on eviction the GPU rows are copied to host memory **synchronously** (before the
slot is shrunk), then written by a background daemon thread (atomic `tmp` + `os.replace`,
tensors first then the sidecar). `trim()` deletes the oldest files past the byte cap;
`close()` (clean shutdown) flushes the queue. Resume copies the stored rows into the
fresh slot, validates `kv_dtype` / `ratio` / `pos ≤ capacity`, and hands back the same
`{state, tail}` the in-memory path uses.

Scope: **text only**, CUDA `--parallel` path only, **default off**.

## 2. Env knobs

| var | default | meaning |
|---|---|---|
| `TF_SPILL_GIB` | `0` (off) | disk cap in GiB (`int(gib * 2**30)`); `0` disables the whole tier |
| `TF_SPILL_DIR` | `~/.cache/tensorfold/prefix-spill` | root; a `<model>-<hash>` subdir is used per model |
| `TF_SPILL_MIN_TOKENS` | `8192` | don't bother storing prefixes shorter than this |

The served container sets `TF_SPILL_GIB=8` / `TF_SPILL_MIN_TOKENS=2048` in its
`run_serve.sh` by default.

## 3. Verdict — honest scope

On the **hot path it changes nothing**: the kept prefix is already in VRAM, so an
in-process hit is identical to `0014`. The value is a **cold restart** — after the
server is restarted (RAM cache gone), a repeated prefix resumes from disk instead of
being prefilled again. Small prefixes (`< TF_SPILL_MIN_TOKENS`) are never stored.

## 4. A/B — one GB10, `temperature 0`

`receipts/p21_spill_ab.py`: 4 distinct long single-turn chats; `warm` with a fresh spill
dir, then a **cold restart** with the dir intact.

- warm (all distinct prefixes): every request `cached_tokens = 0`; **1 kept prefix**
  ended up on disk (`pos = 4096`, `mtp_len = 4095`, `kv = bf16`, 12 layers, ≈230 MB).
- cold (RAM empty, dir kept): the matching prefix came back with
  **`cached_tokens = 4096`**, and its `token_sha` equalled the warm reference — the
  reply is byte-for-byte a fresh prefill's.

| scenario | `cached_tokens` | wall |
|---|---|---|
| fresh ~6k prefix (miss) | 0 | 3.43 s |
| same prefix after a cold restart | **4096** | 2.12 s |

16 / 20 token ids identical across the boundary; `token_sha` equal. Raw:
`receipts/P21-ab-warm-20261010.json`, `receipts/P21-ab-cold-20261010.json`.

## 5. Stress — 10 sessions, cold restart

`receipts/p21_stress.py burn` sends 10 distinct ~6k prefixes cold (empty RAM, empty
dir): all `cached = 0`, **4 prefixes spilled** (`pos = 4096`), `total_wall = 41.12 s`.
Then a **cold restart** (dir kept) and `resend` the same 10:

| run | hits | total wall |
|---|---|---|
| burn (all miss) | 0/10 | 41.12 s |
| resend (cold restart) | **4/10** (`cached = 4096`) | 36.66 s |

Hit requests ran ≈2.1 s vs ≈3.4 s for the misses; **misses were unchanged** (no
regression). Raw: `receipts/P21-stress-burn-20261010.json`,
`receipts/P21-stress-resend-20261010.json`.

With `TF_SPILL_MIN_TOKENS=2048` the whole ~6k prefix is stored (`pos ≈ 5954`, ≈285 MB),
so a cold restart skips nearly the entire prefill.

## 6. Token consistency

`token_sha` (and the returned `token_ids`) are identical warm vs cold and spill-on vs
spill-off — the resumed reply is a fresh prefill's, token for token. Unit tests
`tests/cuda/test_flashnext_spill.py` (round-trip resume for bf16 and int8; eviction →
spill → a cold `MultiDecoder` resumes with `cached > 0`) pass alongside the `0014`
tests (8 passed).

## 7. Reproduce

```sh
# in the served container, server up on 127.0.0.1:8080
python3 receipts/p21_spill_ab.py warm receipts/P21-ab-warm-20261010.json
# restart the process (RAM empty), keep the spill dir
python3 receipts/p21_spill_ab.py cold receipts/P21-ab-cold-20261010.json
python3 receipts/p21_stress.py burn   receipts/P21-stress-burn-20261010.json 10
# restart again, keep the spill dir
python3 receipts/p21_stress.py resend receipts/P21-stress-resend-20261010.json 10
python3 receipts/p21_meta.py   # inspect the spill dir
```

## 8. Provenance

- `0015` is this project's own work (not a port or cherry-pick). The design follows the
  GLM spill discussion in TensorFold issue #373 / PRs #423, #427 (closed, unmerged) and
  the `gin'iro`/MiaAI-Lab GLM recipe patch `0088-glm-spill-tier`, adapted to the
  Flash-Next CUDA `qwen4_exp` kept-prefix path.
- Upstream references: [`ashhart/TensorFold#373`](https://github.com/ashhart/TensorFold/issues/373),
  [`#423`](https://github.com/ashhart/TensorFold/pull/423),
  [`#427`](https://github.com/ashhart/TensorFold/pull/427).

## 9. Fix (`0018`) — the eviction hot path no longer spikes TTFT

`0015` shipped with a latent hot-path cost: when a kept prefix was **evicted**, `_collect`
copied the rows to host with a **pageable** `.to("cpu")`, which blocks as the prefix grows.
On a monotonic ~16k→94k growth load the per-eviction `gap = ttft − prefill` rose to
**7–16 s** (non-eviction requests stayed at 0.05 s). The spike is **not** disk I/O (writes
are on the background thread) nor D2H bandwidth (a pinned copy of 2 GiB is ≈0.036 s): it is
a fresh `cudaHostAlloc` on every eviction.

`0018` (ours, P21c/P21d):

- **P21c** stages the copies through **pinned** host buffers (`non_blocking` D2H + one
  `synchronize`), pooled by size, so an eviction reuses a buffer instead of re-paying the
  cold allocation. Per-eviction gap drops **7–16 s → ≤1 s** (rest 0.05 s).
- **P21d** stops writing the `pooled` / `mtp.pooled` blocks for **text** prefixes: they are a
  deterministic function of the index cache `ikc` (the pool kernel preserves a block's bits
  when recomputed), so `restore` re-runs `qsa_pool` through a `MultiDecoder.repool` hook.
  Image prefixes keep their pooled rows (image rotary). The file shrinks by the pooled bytes.

Fix A/B (`receipts/P21c-d-fix-20261011.json`), same growth load, `gap` in seconds:

| load | base (`0015`) | fixed (`0018`) |
|---|---|---|
| turn 8 (~78k prompt) | 7.35 | 0.28 |
| turn 9 (~86k) | 7.69 | 0.22 |
| turn 10 (~94k) | 15.79 | 0.98 |
| non-eviction turns | 0.05–0.06 | 0.05 |

Payload (text prefix, `insp`): ≈31.4k tokens **1084.2 → 1056.4 MB**, ≈47.1k tokens
**1566.7 → 1526.8 MB** (−2.5%). Cold resume stays **bit-identical** (`token_sha`): stress
burn 10 → cold restart → resend 10 resumed **4/10** from disk. Unit tests
`test_flashnext_spill.py` (bf16 + int8 round-trip) pass.

The served container keeps spill **on** (`TF_SPILL_GIB=8`); after `0018` the eviction cost is
bounded (~1 s worst case on a monotonic 94k load) instead of growing without bound.

---

*This article was completed with the assistance of DeepSeek AI.*