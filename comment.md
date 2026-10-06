Published everything you asked for here:

**https://github.com/Yuepixel/tensorfold-qwen38-exl3-lookup**

Contents:
- **`patches/0001..0006.patch`** — `git am`-able on top of `v0.6.5`; combined diffstat 34 files, +2002/−40. 0001–0004 are the EXL3 prompt-GEMM work; 0005 adds the prompt-lookup (suffix) draft arm wired into `mtp_decode`; 0006 wires it into the **batched** decoder with the per-stream backlog handling. `receipts/all-changes.diff` is the squashed view.
- **`tools/`** — the benchmark fixtures (`p7_edit_bench.py`, `p7_conc_bench.py`, `p7_conc_edit.py`, `p7_cost_fit.py`, `p7_probe.py`, `p7_profile.py`, `prefill_cold.py`).
- **`receipts/collect.py`** — the fixture generator (stdlib only); exact prompt texts live in its `PROMPTS`, payload fixed at `temperature:0, max_tokens:96, chat_template_kwargs.enable_thinking:false`.
- **`receipts/receipt_*.json`** — the byte-exact `token_sha` receipts.

Setup: TensorFold `0.6.5` + these patches, model `Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5`, single DGX Spark (GB10).

**Baseline vs experimental.** Forced `l7:2` (`MAX_DRAFTS=7, MIN_MATCH=2`) is the shipped baseline. `auto` (the cost gate) is **experimental**: it is not calibrated across prompt shapes — across our 7 fixtures it only fired on one (`edit_continue`, ld 13/la 13) and stayed MTP-only everywhere else. Kept gated, default remains forced `l7:2`.

**Byte-exactness.** For all 7 fixtures `token_sha` is identical across serial / MTP-only / lookup / auto:
`prose f4cf4bac607d`, `code 2deda2799112`, `repeat aef89122efe1`, `chat 35ac773b2d02`, `edit_full 98f603d270f7`, `edit_continue fb7342c804b3`, `pattern_novel d4e923c62b39`.
- **Parallel** (`--parallel 2`, k=2, batched path): every stream's sha equals the single-stream sha.
- **Cancellation**: abort a `stream:true` request after 5 SSE lines → the next request (serial and drafted) returns the same sha; `/health` clean.
- **lookup → MTP return**: covered by `pattern_novel` (repeat span → must diverge + `DONE`); ld 31/la 31, rounds 6 vs 39 MTP-only, same sha.

**One caveat worth knowing for the native port:** under `--parallel`, a *single* in-flight request is served by the solo decoder (`multi_solo.py::_solo_round`), which does **not** thread `lookup` into `mtp_decode`. Lookup only runs in the no-scheduler path (`engine._decode`, i.e. without `--parallel`) or the batched path (`multi._draft_all`, ≥2 concurrent streams). So a lone request against a `--parallel` server is effectively MTP-only. The committed `run_serve.sh` has no `--parallel`.

Reference single-stream decode (tps, l7:2 vs MTP-only): `edit_continue` 122.0→160.3 (+31%), `pattern_novel` 105.3→151.3 (+44%), `edit_full` 116.5→128.5 (+10%), chat/code ~flat.

The lookup arm is a port of `jayleaton/glm53-tensorfold-spark` `patches/0020` (`glm5_next/cuda/lookup.py`); credited in the README. Shout if you want anything reshaped for the port.
