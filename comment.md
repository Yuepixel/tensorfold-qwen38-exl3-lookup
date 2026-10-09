Update — vision, video, multi-image, and an image-history prefix cache: the last of the stack.

Since the `0001`–`0007` update, the same EXL3/CUDA build also does native **image + video** input and, the one that matters for multi-turn, stops **image chats from re-prefilling every turn**. New patches are `0008`–`0012` in this repo (github.com/Yuepixel/tensorfold-qwen38-exl3-lookup), all `git am`-able on top of `0007` (`v0.6.5` + the seven you already have). Full numbers: `README.md` §5.4–5.5.

### 1. Vision — native image input (`0008`–`0010`)

Ports upstream #229 (inline vision towers, BF16 input) onto the EXL3 path and enables it in the production serve script. The vision weights ride a FP16 sidecar next to the 3.05 bpw pack, so the EXL3 checkpoint itself is untouched. Images arrive as data URLs; verified with OCR and shape/colour probes.

### 2. Video + multi-image (`0011`)

The same image stack (PyAV `av 19.0.1`) handles video frames and multiple images per request, gated by two knobs:

- `TENSORFOLD_VISION_MAX_IMAGES` (default 4)
- `TENSORFOLD_VISION_IMAGE_TOKENS` (default 4096)

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

### Still to polish

- **Video multi-turn cache reuse** is not separately measured — it flows through the same path as images, but I did not test it on its own.
- The lookup **cost gate** stays experimental; the default remains forced `l7:2`.
- As you noted, the Python line is frozen (#286), so this stack is CUDA/Python reference material for the native port. #263 is closed/unmerged, so the image-cache fix would need folding in there too, ideally next to the #212 prefill work.

Credits: #263 by @olexale; #229 and the vision-tower work upstream; the lookup arm is a port of @jayleaton's suffix drafter; the ring is @jschmied's design via #283.

This article was completed with the assistance of DeepSeek AI.
