# Receipts — patches `0016` / `0017` (off-theme additions)

These two patches are additive and unrelated to the Qwen3.8-Flash-Next EXL3 stack
documented in `README.md`. They ship in this repo because it is the fork's
upstream-facing branch. Both are **experimental — read + run only**: they can load and
serve the two model families below, but **no performance work was done and nothing is
benchmarked**; no speed or quality claims are made for them.

## `0016` — `qwen3_5_moe: read compressed-tensors NVFP4 routed experts + MTP`

- Adds a ModelOpt / compressed-tensors **NVFP4** read path for `qwen3_5_moe` routed
  experts (and the MTP head) on the CUDA backend.
- Scope: 8 files, +229 / −29.
- Result: `Qwen3.6-35B-A3B` NVFP4 served on a single DGX Spark (CUDA), coherent text.

## `0017` — `GGUF reader and CUDA on-the-fly dequant (dense + grouped MoE experts)`

- New `tensorfold/gguf/` package: GGUF parser (`reader.py`), block dequant
  (`dequant.py` — Q8_0 / Q4_K / Q5_K / Q6_K, bit-exact against the `gguf` PyPI
  reference), HF name mapping (`hf.py`), and a tokenizer synthesised from GGUF
  metadata.
- CUDA on-the-fly dequant: a fused `ggml` matmul for dense `qwen3_5`, plus grouped
  `ggml_gate_up` / `ggml_down` expert kernels for `qwen3_5_moe` (the router and the
  shared expert are handled alongside the routed experts).
- Fixes the GDN value-head tiling order (llama.cpp `tiled` → grouped) for `qwen3_5`.
- Scope: 27 files, +1562 / −23.
- Result: `Qwen3.5-0.8B` Q4_K_M and `Qwen3.6-35B-A3B` UD-Q4_K_M both served on a
  single DGX Spark (CUDA), coherent and correct (e.g. `391 + 9 = 400`).

## How to apply

`git am patches/*.patch` on top of `v0.6.5` (`609ca41`). Patches `0016` and `0017`
also apply cleanly on their own on a stock `v0.6.5`.

This article was completed with the assistance of DeepSeek AI.
