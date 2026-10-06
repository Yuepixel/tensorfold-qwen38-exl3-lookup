# RECIPE — one-shot deploy (AI-executable)

> **For an AI coding agent.** This file is self-contained. If a user handed you the
> link to this file and asked you to "install it", follow the steps below end to end,
> report what you ran, and stop and ask if a step cannot be completed.

Goal: build **TensorFold `v0.6.5` + 12 patches** and serve **Qwen3.8-Flash-Next
(EXL3 3.05 bpw)** on a single **DGX Spark (GB10)**, giving **~2.4× EXL3 prefill**
(≈1700 t/s vs ~700), the **prompt-lookup drafter** on top, plus **native image/video**
input and an **image-history prefix cache** for multi-turn image chats, with byte-exact
output. Full numbers and receipts: [`README.md`](README.md).

---

## 0. Prerequisites

| | |
|---|---|
| GPU | NVIDIA **DGX Spark (GB10)**, `sm_121`, ≥96 GiB unified memory |
| OS | Linux. Reference build ran inside `nvcr.io/nvidia/pytorch:26.07-py3` (CUDA 13, PyTorch ≥2.7) |
| Python | 3.11+; the container ships one |
| Disk | ~90 GiB free for the weights + build |
| Net | GitHub reachable for the engine + patches; the model is on Hugging Face |

If the card is not a GB10, this still builds, but the performance targets in §5 are
the GB10 ones.

---

## 1. Engine at the exact base commit

The patches are cut against the **frozen Python engine** and only apply cleanly on
this tag:

```sh
git clone https://github.com/ashhart/TensorFold
cd TensorFold
git checkout v0.6.5        # = 609ca419abecebdc5a059498a613680bd3aa847f
```

## 2. Apply the patches

```sh
git clone https://github.com/Yuepixel/tensorfold-qwen38-exl3-lookup
git -C TensorFold am /path/to/tensorfold-qwen38-exl3-lookup/patches/*.patch
# fallback if `am` complains: for p in .../patches/*.patch; do git -C TensorFold apply "$p"; done
```

Twelve patches, in order: `0001`–`0004` are the EXL3 prompt-GEMM work (the prefill
lift), `0005`/`0006` add the prompt-lookup drafter, `0007` adds the width-gated
two-chunk ring (another +13–15 % prefill; port of PR #283). `0008`–`0011` add native
**image + video + multi-image** input (the vision tower rides a FP16 sidecar next to the
pack — port of PR #229), and `0012` adds the **image-history prefix cache** (cherry-pick
of PR #263, fixes #414). No conflicts expected.

## 3. Build the CUDA extension

```sh
cd TensorFold
pip install -e . --no-build-isolation
# if the arch check complains: export TORCH_CUDA_ARCH_LIST=12.1
```

Sanity check it compiled in:

```sh
python3 -c "import tensorfold; print(tensorfold.__file__)"
python3 -m pytest tests/cuda/test_exl3_prompt_experts.py -q   # expect: 15 passed
```

## 4. Fetch the weights

EXL3 3.05 bpw, group-32, `h5_ng5` pack (~79 GiB) — repo `turboderp/Qwen3.8-Flash-Next-exl3`,
**revision `3.05bpw_h5_ng5`** (commit `69e33439`):

```sh
# direct (if reachable)
huggingface-cli download turboderp/Qwen3.8-Flash-Next-exl3 \
  --revision 3.05bpw_h5_ng5 \
  --local-dir /models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5

# or via the mirror
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download turboderp/Qwen3.8-Flash-Next-exl3 \
  --revision 3.05bpw_h5_ng5 \
  --local-dir /models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
```

## 5. Serve

```sh
# baseline (lookup ON, forced l7:2)
TF_EXL3_LOOKUP=l7:2 tensorfold serve \
  /models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 \
  --host 0.0.0.0 --port 8080 --parallel 2

# MTP-only (disable the drafter) — the var must be exported EMPTY, not unset:
#   TF_EXL3_LOOKUP= tensorfold serve /models/... --host 0.0.0.0 --port 8080
```

Notes:

- Native **image/video** input needs the FP16 vision sidecar
  (`TENSORFOLD_VISION_WEIGHTS=<.../vision-f16-Qwen3.8-Flash-Next-exl3-3.05bpw.safetensors>`)
  and the `--vision` flag; tune with `TENSORFOLD_VISION_MAX_IMAGES` (default 4) and
  `TENSORFOLD_VISION_IMAGE_TOKENS` (default 4096) — see README §5.4.
- Removing `--parallel 2` puts a lone request on the no-scheduler path, which also
  reaches the lookup arm; with `--parallel` a *single* in-flight request is served
  MTP-only (see README §2).
- The EXL3 pack rejects `--tp 2`; this recipe is single-GPU.

## 6. Acceptance (do not skip)

Byte-exact presence and the prefill lift are both checkable. Server up on
`127.0.0.1:8080`:

```sh
# (a) prefill — build prompts once (needs the tokenizer), then run
cd /path/to/tensorfold-qwen38-exl3-lookup
python3 tools/prefill_cold.py build /models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 prompts.json
python3 tools/prefill_cold.py run  http://127.0.0.1:8080 Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 prompts.json prefill.json

# (b) byte-exact receipts — compare token_sha to README §4
python3 receipts/collect.py probe
python3 receipts/collect.py single lookup-l7-2
python3 receipts/collect.py single mtponly
```

**Pass criteria**

| check | expected |
|---|---|
| Cold prefill (2048/8192/16384/32768/65536) | ≈ 1696 / 1771 / 1777 / 1758 / 1706 t/s (~2.4× the ~670–800 baseline) |
| `token_sha` (7 fixtures) | identical to README §4.1: `f4cf4bac607d`, `2deda2799112`, `aef89122efe1`, `35ac773b2d02`, `98f603d270f7`, `fb7342c804b3`, `d4e923c62b39` |
| `serial == MTP-only == lookup` | same hash for every fixture |

If prefill lands near ~700 rather than ~1500, patches `0001`–`0004` did not build —
rerun §3 and confirm `pytest tests/cuda/test_exl3_prompt_experts.py` passes.

## 7. If something fails

| symptom | fix |
|---|---|
| `git am` rejects a patch | confirm you are on `v0.6.5` (`609ca41`); otherwise fall back to `git apply` |
| import fails after build | rebuild with `--no-build-isolation`; check `TORCH_CUDA_ARCH_LIST=12.1` |
| load error on the pack | verify the revision is `3.05bpw_h5_ng5` and all shards downloaded |
| prefill still ~700 | the prompt-GEMM patches are not in the build (see §6) |

---

*This article was completed with the assistance of DeepSeek AI.*
