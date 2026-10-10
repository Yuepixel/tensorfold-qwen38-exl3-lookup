# P20 · 系统块检查点（TF_SYS_CHECKPOINT）——上游移植 + GB10 A/B

> 承 P18（注入块冻结）/ P17（预填充基线负结果）。P20 = 给现役 CUDA `qwen4_exp` 引擎加「system 块结束前的检查点」，治多会话/agent 流量里 **system 块尾部一变就整段冷 prefill** 的问题。
> 来源标注：`[手动]` 人手实测 ｜ `[自主]` 系统自主运行 ｜ `[推断]` 由代码/文档读出 ｜ `[未验证]` 猜测。

## 1. 背景：P20 原定两杠杆被证伪

用户原任务给两杠杆治多会话反复冷 prefill：`--checkpoint-slots`（内存多槽）+ `--spill-gib`（磁盘 spill）。

**证伪** `[推断]+[手动]`：在 v0.6.5 现役 CUDA EXL3 路径上二者均为**静默 no-op**：
- 服务端 `CheckpointStore`/spill 只接在 MLX/通用 `ChatApp`（`server/app.py`，由 `cli.py` `_serve_mlx` 构造）；CUDA 走 `_serve_cuda` → `cuda/server.py:App`，其 `__init__` **不收** `checkpoint_slots`/`spill_bytes`。
- `cli.py:269-270` 的 `--checkpoint-slots` 传递被 `CUDA_CHECKPOINT_SLOTS` 门控，`qwen4_exp` 未声明（仅 `qwen3_5`/27B 声明）⇒ 静默忽略。
- 上游 `README.md:135-136`：`--checkpoint-slots`=Both（CUDA 仅 27B）、`--spill-gib`=**MLX only**。
- 真机只读探针（container `tf-exl3`）：`grep CUDA_CHECKPOINT_SLOTS families` 只命中 `qwen3_5`；`grep spill cuda/server.py` 空。

→ 盘 spill 对本路径**任何现成件都没有**（D 探测结论）。

## 2. 上游情报（D：联网 + GitHub）

| 候选 | 出处 | 内容 | 对 qwen4_exp CUDA |
|---|---|---|---|
| **① 系统块检查点（采用）** | `grearjake-star/TensorFold` 分支 `sys-checkpoint` @ `b4a9993`（parent `609ca41` = 我们 base） | `TF_SYS_CHECKPOINT`（默认 2048 行，0=关）：在 system 块结束前的最后一个倍数行边界多保一个状态 | ✅ 纯内存、正建在我们 base 上、直接可摘 |
| ② message-start keeps + fork lanes | 上游 **PR #163**（已并入 v0.6.5） | 给 `qwen4_exp/cuda` 加 message-start 快照 + fork | ✅ **已在我们树里**（`tests/cuda/test_flashnext_prompt_cache.py` 存在） |
| ③ `--checkpoint-slots` for Flash Next CUDA | 上游 **PR #302** | 把 checkpoint-slots 接到 CUDA | ❌ **draft、未合**（`merged:false`） |
| ④ CUDA 盘 spill（GLM） | 上游 **PR #423** / #427（均未合，closes #373 仍 open）；MiaAI-Lab v1.9 PR#78 patch `0088-glm-spill-tier`；JSpark3 v2.0.2 | per-rank `{key}.rank{r}.safetensors`、最长前缀续、关机 flush | ❌ 全是 **GLM**，`qwen4_exp` 无现成件 |
| ⑤ Zig 原生线 | `zig-flashnext` @ `78ee0f4` | Flash-Next prompt 复用**仅 Metal**；CUDA 后端「尚未接入 server」、无盘 spill | ❌ |

→ 结论：**qwen4_exp/Flash-Next 的 CUDA 盘 spill 上游/fork/Zig 全无**；内存多状态有现成捷径（①）。

## 3. 采用方案与落地

- **移植** `[手动]`：`git apply` 上游 patch（经 SMB 中转；host 取 `api.github.com` 的 `application/vnd.github.patch`，因 GB10 直连 github 超时）。`apply --check` 4 文件全过；应用后 `markers.py`/`prefixes.py` 与 609ca41 完全一致（干净），`engine.py` 与我们改动区（L403+ P7 lookup）相距 ~240 行，**零冲突**。
- **GB10 提交** `[手动]`：分支 `p9-vision` **`78f72c7`**（4 文件 +190/−6）：`src/tensorfold/cuda/markers.py`、`src/tensorfold/families/qwen4_exp/cuda/engine.py`、`tests/cuda/test_flashnext_sys_checkpoint.py`、`tests/test_cuda_sys_checkpoint.py`。
- **机制** `[推断]`：`snapshot_points(openers, assistant, checkpoint=N)` 在原有「第二消息起点 + 末个 assistant 起点」之外，多返回 `(第二消息起点 − MIN_GAP)//N*N` 这个检查点（`points.checkpoint`）；engine `__init__` 传 `checkpoint_rows()`（env `TF_SYS_CHECKPOINT`，默认 2048）。它作为 `snapshot_points` 的**另一个 stop**，被同样 keep/fork/evict（占用相同的 kept 状态数）。
- **默认开**：未设 env = 2048。`TF_SYS_CHECKPOINT=0` = 关。**run_serve.sh 无需改**（纯 Python，改完重启即生效，免 CUDA 重编）。

## 4. GB10 A/B（每侧各自全新重启，冷态）

模型 `Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5`，容器 `tf-exl3`，`--parallel 2`。脚本：`tools/p20_sysckpt_micro.py`、`tools/p20_sysckpt_multi.py`。

**微测**（~8.5k system 块，A/B 尾 8 token 不同）：

| | cached | prefill_s | wall |
|---|---|---|---|
| OFF（`TF_SYS_CHECKPOINT=0`）B | 0 | 4.57 | 4.64s |
| ON（默认 2048）B | **8192** | **0.38** | 0.45s |

（B2 完全重复两态均 cached=8482，`[手动]` 无回归。）

**多会话**（6 会话共享 ~13k 前缀，各带不同尾）：

| | r1 cached | r1 total wall | r2 cached | r2 total wall |
|---|---|---|---|---|
| OFF | 0/12 | 41.56s | 0/12 | 41.61s |
| ON | 11/12 | **10.18s (4.1×)** | 12/12 | **3.22s (12.9×)** |

`[手动]`：OFF 下多会话**连精确重复都命中 0**——kept 状态被 keep 上限驱逐（并发互踩）；ON 后命中率 0/12 → 11/12 → 12/12。内存：重启后 available ~50 GiB，无异常增长。

## 5. 与 P18 的关系

P18 的客户端方案（opencode 插件 `freezeInjection`）冻结注入块，使 system 块**会话内稳定**；本检查点解决的是**块尾变化**（日期行 / 记忆段更新 / 换会话时注入内容不同）时仍能续算——对应 P18 遗留的「追加式注入」痛点，且改在**引擎侧**（对所有客户端生效）。

## 6. 未决 / 注意

- `[手动]` 上游 commit 自述「resume 恢复的是 kept 前缀 ⇒ 回复与全新 prefill 逐 token 相同」**已在本机复验通过**：`tools/p20_token_consistency.py`（请求带 `return_token_ids: true`，greedy `temperature=0`，4 个「同 system 块 + 不同尾」prompt）在 OFF（`TF_SYS_CHECKPOINT=0`，4×full prefill）与 ON（case0 cold + case1–3 `cached=8192`）两侧，**每 case 的 `token_sha` 与 `token_ids` 数组逐项全等**。结果 `refs/P20-tok-consistency-{off,on}.json`。
- 该杠杆只对「system 块尾部变化」生效；若现网 miss 另有主因，收益有限。真实 agent 负载的聚合 `cache_read` 改善**未单独长跑测量** `[未验证]`。
- 盘 spill（GLM 设计）仍未移植 — 若将来要，按 PR #423 的 per-rank safetensors + 最长前缀续搬进 `qwen4_exp/cuda`。

## 7. 复现：间歇冷 miss 的根因 = 检查点与 message-start 状态共用 kept 计数（2026-10-10 `[手动]`）

- 背景：P20 长跑采集端到端自测时，4 个「同大 system 块、尾不同」请求命中呈间歇（`0/8192/0/8192`），非 A/B 里的干净 12/12，遂单独复现。
- 探针 `tools/p20_repro.py`（SMB `/home/yueyue/p20_repro.py`）：同 system 块（`"lorem ipsum…elit "*950` ≈ 51300 字符）+ 各自不同尾；`temperature=0 max_tokens=16 return_token_ids`；模式 `seq`（顺序）/`conc`（多线程并发），参数 n。命中读响应顶层 `tensorfold.cached`，检查点落点本机 = **7644**（system 块约 7900 tok）。
- 冷重启后实测 `(tag:cached)`：
  - **并发 6（冷）**：`0:0, 1:7644, 2:7644, 3:0, 4:0, 5:7644` → 3 冷 3 中。
  - **并发 6（热，紧接着再跑）**：`0:7644, 1:0, 2:0, 3:7644, 4:7644, 5:7644` → 4 中 2 冷。
  - **串行 6**：`0–4` 全中（7644，其一 7660），`5:0`（`prefill=3.91s`）→ 5 中 1 冷。
  - **串行 12**：仅 `5:0` 冷，其余全中，且 `6–11` 又全中 → **非单调容量到期，而是「周期性单次驱逐后立即重建」**。
- 根因（读补丁自述）：检查点是 `snapshot_points` 的**额外一个 stop**，因此「**kept / forked / evicted 都走 message-start 状态的同一 kept-state 计数**」（`P20-sys-checkpoint-b4a9993.patch` commit message 原文："...within the same kept-state count"）。即检查点与 per-turn 状态争夺同一**有界槽位**；持续新请求（尤其 `--parallel 2` 双 lane）把它挤出 → 回归冷 prefill。
- 结论：P20 **净收益仍大**（绝大多数请求命中），但**并发/持续流量下非 100%**；这正是 P21（盘 spill / 更大多槽）要兜的场景。廉价候选「调大 `keep`」**已实测为阴性**（见下条）。
- **keep 调参阴性（2026-10-10 `[手动]`）**：将 `qwen4_exp/cuda/engine.py` 的 `KEEP`/`KEEP_SERIAL` 改为 env 可调（`TENSORFOLD_KEEP` / `TENSORFOLD_KEEP_SERIAL`，默认不变），冷起分别以 `KEEP=8 / 16 / 64` 跑同序 4 组（conc6 冷 / conc6 热 / seq6 / seq12），结果**逐组完全相同**（conc6 冷 3 中 3 冷、热 4 中 2 冷、seq6 5 中 1 冷、seq12 11 中 1 冷，冷点固定在某一索引）。容器内 `/proc/<serve_pid>/environ` 复核 `TENSORFOLD_KEEP=64` 确已注入 ⇒ **真阴性**：间歇冷 miss 与 kept-count 上限**无关**（非 `prefixes.py:53` 的 FIFO 计数驱逐），更像**内存压力型** `_evict_kept`（`multi.py:154`）或双 lane 槽位回收。env 化改动**已回滚**，GB10 树保持 `78f72c7`。

## 8. P20 成绩单 / 基线（2026-10-10，全 `[手动]`）

环境：现役默认态（`TF_SYS_CHECKPOINT=2048` 开、`keep=8`、`--parallel 2`、HEAD `78f72c7`），GB10 冷态；`free -g` 起测前 = used 72 / available 49 GiB。脚本 `tools/p20_bench.py`，原始 JSON `refs/P20-bench-result-20261010.json`。

**冷 prefill 曲线（单流、全新内容、cached=0）**

| prompt tok | prefill_s | 折算 tok/s |
|---|---|---|
| 1581 | 1.005 | 1573 |
| 6243 | 3.423 | 1824 |
| 12388 | 6.544 | 1893 |
| 24728 | 13.042 | 1896 |

**decode — 四类负载**（`tools/p7_profile.py <base> <model> --rounds 4 --tokens 128`，`--unique`=冷；t/s = 128/decode_s）

| 类 | 冷 s1 | 温 s1 | 冷 s2 | 温 s2 |
|---|---|---|---|---|
| chat | 37.7 | 38.2 | 46.1 | 55.3 |
| code | 115.5 | 106.8 | 82.9 | 92.2 |
| edit | 131.6 | 135.9 | 90.9 | 99.9 |
| continue | 122.1 | 124.9 | 111.4 | 120.4 |

（原始日志 `refs/P20-profile-20261010.txt`，驱动脚本 `tools/p20_profile.sh`；与 P19 同口径表 `refs/P19-F6b适用3.05bpw验证.md` §2 逐格吻合 ⇒ **decode 未受 P20 影响**。）

**decode — 单条散文 prompt**（`tools/p20_bench.py`）：244 completion / `decode_s` 7.29 = **≈33.5 tok/s**（单流）；MTP `accepted 149/231` = 64.5%，rounds 94，`lookup=0`。散文无可复现后缀、吃不到 lookup，即上表 **chat 档**（故单值 33.5 并非低，而是 chat 口径）。

**系统块检查点复用（微测 ~8.5k system 块、尾不同）**

| case | cached | prefill_s | wall |
|---|---|---|---|
| A 冷 | 0 | 4.545 | 4.61s |
| B 换尾 | 8192 | 0.367 | 0.44s（≈12.4×） |
| B2 完全重复 | 8482 | 0.043 | 0.10s |

**多会话（6 会话共享 ~13k 前缀、尾不同）**

| | cached | total wall |
|---|---|---|
| r1 | conv0 8192；conv1–5 12288 | 5.89s |
| r2 | 全 12288 | 3.75s |

（对照 P20 A/B 冷起 OFF：41.56 / 41.61s。）

**并发 vs 串行**（`tools/p20_repro.py`，同 system 块 ~51300 字符 + 不同尾）

- 串行 6：`0:冷, 1–5:7644` → 1 冷 5 中。
- 并发 6：`0:冷, 1/2/3/5:7644, 4:冷` → 2 冷 4 中（并发下间歇驱逐再现，与 §7 一致）。

→ 作为后续优化（P21 盘 spill 等）的**同口径基线**。

## 9. 出处

- 上游 commit：https://github.com/grearjake-star/TensorFold/commit/b4a9993450f636b314485a07969600cc2be831a4
- 上游 issue #169 / #163（CUDA 不保状态）；issue #414（含图 agent 聊天每轮重 prefill）；PR #302（checkpoint-slots，draft 未合）；PR #423/#427 + issue #373（GLM 盘 spill，未合）
- 本地补丁归档：`refs/P20-sys-checkpoint-b4a9993.patch`
