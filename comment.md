Update — vision, video, multi-image, an image-history prefix cache, and a disk prefix spill (and its eviction fix).

Since the `0001`–`0007` update, the same EXL3/CUDA build also does native **image + video** input and, the one that matters for multi-turn, stops **image chats from re-prefilling every turn**. New patches are `0008`–`0012` in this repo (github.com/Yuepixel/tensorfold-qwen38-exl3-lookup), all `git am`-able on top of `0007` (`v0.6.5` + the seven you already have). Full numbers: `README.md` §5.4–5.5. The repo is now an unofficial, license-compliant fork — upstream `NOTICE` / `THIRD_PARTY_NOTICES.md` / Apache-2.0 ship alongside (§4 kept), no endorsement implied.

### 1. Vision — native image input (`0008`–`0010`)

Ports upstream #229 (inline vision towers, BF16 input) onto the EXL3 path and enables it in the production serve script. The vision weights ride a FP16 sidecar next to the 3.05 bpw pack, so the EXL3 checkpoint itself is untouched. Images arrive as data URLs; verified with OCR and shape/colour probes.

### 2. Video + multi-image (`0011`)

The same image stack (PyAV `av 19.0.1`) handles video frames and multiple images per request, gated by two knobs:

- `TENSORFOLD_VISION_MAX_IMAGES` (patch default 4; reference serve 12)
- `TENSORFOLD_VISION_IMAGE_TOKENS` (patch default 4096; reference serve 49152)

Verified: single- and multi-image OCR, a video clip transcribed to its on-screen text, and mixed text/image requests — all correct, and concurrency-safe.

### 3. Image-history prefix cache (`0012`) — the headline

Cherry-pick of **#263**, which fixes open **#414**: with one image in an agent chat, every later turn came back `cached_tokens=0` and re-prefilled the whole prompt. The fix keys each image placeholder row on the image's own pixels (a negative row id derived from the content hash), so a resumed turn matches and keeps the cached prefix instead of rebuilding it.

Before → after (2-turn chat, ~2.5k prompt, greedy, `temperature 0`):

| turn | `cached_tokens` | `prompt_tokens` | wall |
|---|---|---|---|
| image, turn 1 | 0 | 2515 | 2.87 s |
| image, turn 2 — with #263 | **2514** | 2550 | 0.24 s |
| image, turn 2 — before #263 | 0 | 2550 | 1.53 s |

Byte-exactness holds throughout: a resumed turn's output is **byte-identical to a cold re-prefill**, the text-only baseline hash is unchanged, and under `--parallel 2` three concurrent image streams return byte-equal outputs.

### 4. Cold lone-stream decode (`0013`)

One more, in this repo only (`0013`, ours — not a port or cherry-pick). With `--parallel 2`,
a lone request that misses the prefix cache used to bail in `_move_to_solo` and rebuild an
empty `Graphs` on the fresh slot, **lazily recapturing every decoding round** (~0.18 s
each, ~21 % of rounds). `0013` evicts/drops the kept prefix end so the request keeps the
hot graph slot. Cold decode goes **21–44 → 33–123 t/s** (chat/code/edit/continue),
recapture rounds **191/888 → 0**, TTFT/prefill unchanged, and serial-vs-parallel cold
output stays byte-exact. Numbers: `README.md` §5.6; patch `0013`; receipt
`receipt_f6b_lookup-l7-2.json`.

### 5. Disk prefix spill (`0015` + fix `0018`) — kept prefixes survive a restart

`0015` (ours) writes an evicted kept prefix to disk and resumes the **longest** stored prefix on a later miss, so a **cold restart** stops re-prefilling repeated prompts. It is *not* a hot-path speedup — the kept prefix already lives in VRAM; the value is surviving a restart. Honest scope: **off by default** (`TF_SPILL_GIB`), `--parallel` only, text only. A/B (one GB10, `temperature 0`): a re-sent ~6k prefix after a cold restart came back `cached_tokens` **0 → 4096** with an **identical `token_sha`** (wall 3.43 → 2.12 s); under a 10-session stress, **4/10** prefixes resumed from disk after a cold restart, with no regression on the misses. Numbers: `README.md` §5.8; notes `receipts/P21-prefix-spill-notes.md`. The follow-up `0018` (ours) fixes the spill's **eviction hot path**: it stages the copies through pooled **pinned** host buffers and stops writing the pooled index rows for text prefixes (recomputed on restore), taking the per-eviction `gap = ttft − prefill` from **7–16 s → ≤1 s** and shrinking the file ~2.5%, with the same byte-exact resume (`receipts/P21c-d-fix-20261011.json`).

### 6. Off-theme additions (`0016`–`0017`)

Two further patches ship in this repo, unrelated to the Qwen3.8-Flash-Next EXL3 stack above: `0016` (ours) adds a **compressed-tensors NVFP4** read path for `Qwen3.6-35B-A3B` routed experts (MoE) on the CUDA backend, and `0017` (ours) adds a generic **GGUF reader** with on-the-fly CUDA dequant — a dense `qwen3_5` path plus grouped `qwen3_5_moe` expert kernels. Both are additive; neither changes the EXL3 path or any number in this post. Both are **experimental — read + run only, no performance work, not benchmarked.**

### Still to polish

- **Video multi-turn cache reuse** is not separately measured — it flows through the same path as images, but I did not test it on its own.
- The lookup **cost gate** stays experimental; the default remains forced `l7:2`.
- As you noted, the Python line is frozen (#286), so this stack is CUDA/Python reference material for the native port. #263 is closed/unmerged, so the image-cache fix would need folding in there too, ideally next to the #212 prefill work.

Credits: #263 by @olexale; #229 and the vision-tower work upstream; the lookup arm is a port of @jayleaton's suffix drafter; the ring is @jschmied's design via #283.

This article was completed with the assistance of DeepSeek AI.
